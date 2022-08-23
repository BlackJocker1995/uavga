import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from Cptool.config import toolConfig
from Cptool.gaMavlink import GaMavlinkAPM
from Cptool.mavtool import load_param, read_range_from_dict, get_default_values
from range.rangeproblem import GARangeProblem
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
    popO = pd.read_csv(f'result/{toolConfig.MODE}/Population Info/ObjV.csv', header=None).to_numpy()
    popP = pd.read_csv(f'result/{toolConfig.MODE}/Population Info/Phen.csv', header=None).to_numpy()

    rate = popO[:, 1]
    # sort
    candidate_index = np.argsort(rate)

    popP = popP[candidate_index]
    popO = popO[candidate_index]

    item = []

    name = toolConfig.PARAM.copy()
    name_down = [it + "down" for it in name]
    name_up = [it + "up" for it in name]
    column = []
    for a, b in zip(name_down, name_up):
        column.append(a)
        column.append(b)
    column.append('num')
    column.append('rate')

    # retrans to raw value
    for index in range(popP.shape[0]):
        pop_p = popP[index, :]
        raw_pop = GARangeProblem.reasonable_range_static(pop_p)
        # add rate and overrate
        pop = np.r_[raw_pop, popO[index]]
        item.append(pop)
    item = pd.DataFrame(item, columns=column)
    item.to_csv(f'result/{toolConfig.MODE}/merge.csv', index=False)


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


def test5():
    items = pd.read_csv(f'result/{toolConfig.MODE}/merge.csv')
    np_items = items.to_numpy()[:, :-2]
    up = np_items[:, 1::2]
    lower = np_items[:, 0::2]
    now_range = up - lower

    para_dict = load_param()
    default_value = read_range_from_dict(para_dict)
    deup = default_value[:, 1::2]
    delower = default_value[:, 0::2]
    default_range = deup - delower

    default_range = np.repeat(default_range.reshape((1,-1)), now_range.shape[0], axis=0)

    reduce = (default_range - now_range) / default_range

    reduce = pd.DataFrame(reduce, columns=toolConfig.PARAM)

    reduce.to_csv(f'result/{toolConfig.MODE}/range.csv', index=False)

if __name__ == '__main__':
    toolConfig.select_mode("PX4")
    test5()


