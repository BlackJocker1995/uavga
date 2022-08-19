import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from Cptool.config import toolConfig
from Cptool.gaMavlink import GaMavlinkAPM
from Cptool.mavtool import load_param, read_range_from_dict
from uavga.problem import ProblemGA


def test1():
    para_dict = load_param()
    ranges = read_range_from_dict(para_dict)
    units = ranges[:, 1] - ranges[:, 0]

    base_data = pd.read_csv(f'result/{toolConfig.MODE}/params_test1.csv')
    test_date_drop_result = base_data.drop('result', 1).to_numpy() / units

    # 认为都是positive

    search_data = pd.read_csv(f'result/{toolConfig.MODE}/params_test2.csv')
    search_data_drop_result = search_data.drop('result', 1).to_numpy() / units

    similar_mat = cosine_similarity(search_data_drop_result, test_date_drop_result)

    max_similar_value = similar_mat.max(axis=0)
    max_similar_index = similar_mat.argmax(axis=0)
    in_index = max_similar_index[max_similar_value > 0.97]

    in_index = np.unique(in_index)
    search_sim_base = base_data.iloc[in_index]
    search_out_base = base_data.drop(in_index)


    TP = search_sim_base[search_sim_base['result'] != 'pass']
    TN = search_out_base[search_out_base['result'] == 'pass']
    FP = search_sim_base[search_sim_base['result'] == 'pass']
    FN = search_out_base[search_out_base['result'] != 'pass']

    print(f'TP: {TP.shape[0] / search_sim_base.shape[0]}')
    print(f'TN: {TN.shape[0] / search_out_base.shape[0]}')


    print()


def test2():
    data = pd.read_csv(f'result/{toolConfig.MODE}/params.csv')
    data_drop_result = data.drop('result', 1)
    tandf = data.drop_duplicates(data_drop_result.columns)
    tandf.to_csv('result/params1.csv', index=False)


def test3():
    candidate_vars = []
    candidate_objs = []
    # pd.set_option('precision', 6)
    with open(f'result/{toolConfig.MODE}/pop.pkl', 'rb') as f:
        obj_populations = pickle.load(f)
    for pop in obj_populations:
        pop_v = pop.ObjV
        pop_p = pop.Phen

        candidate_var_index = np.unique(pop_p, axis=0, return_index=True)[1]

        pop_v = pop_v[candidate_var_index]
        pop_p = pop_p[candidate_var_index]

        candidate = [-1] * pop_v

        candidate_index = np.argsort(candidate.reshape(-1))
        pop_v = pop_v[candidate_index].reshape((-1, 1))
        pop_p = pop_p[candidate_index].reshape((-1, 20))

        candidate_var = pop_v[:min(4, len(pop_v))]
        candidate_obj = pop_p[:min(4, len(pop_p))]
        candidate_obj = ProblemGA.reasonable_range_static(candidate_obj)

        candidate_vars.extend(candidate_var)
        candidate_objs.extend(candidate_obj)
    candidate_vars = np.array(candidate_vars, dtype=float).round(8)
    candidate_objs = np.array(candidate_objs, dtype=float).round(8)


    ver_data = pd.read_csv(f'result/{toolConfig.MODE}params.csv').round(8)
    candidate = pd.DataFrame(candidate_objs, columns=ver_data.columns.drop('result'))
    candidate['score'] = candidate_vars

    out = pd.merge(candidate, ver_data)
    # how='outer'
    out.to_csv(f'result/{toolConfig.MODE}/merge.csv')

    print()


def test4():
    data = pd.read_csv(f'result/{toolConfig.MODE}/params.csv')
    for th in np.arange(1, 6, 0.05):
        indata = data[data['score'] >= th]
        outdata = data[data['score'] < th]

        bad_ratio = indata[indata["result"]!="pass"].shape[0] / indata.shape[0]
        if outdata.shape[0] == 0:
            good_ratio = 0
        else:
            good_ratio = outdata[outdata["result"]=="pass"].shape[0] / outdata.shape[0]
        print(f'TH: {round(th,4)} TP : {bad_ratio}   ---   TN : {good_ratio}')
    # PX4:1.2 Ardupilot: 1.4


if __name__ == '__main__':
    toolConfig.select_mode("PX4")
    test4()


