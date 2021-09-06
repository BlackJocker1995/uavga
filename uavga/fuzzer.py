import colorsys
import pickle
import random
import time

import numpy as np
import pandas as pd
from pymavlink import mavutil, mavwp
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA

from uavga.uavgeat import UAVProblem
from uavga.uavgen import UAVGA
import ModelFit.config
import matplotlib.pyplot as plt


class LGFuzzer(object):
    def __init__(self, ga_params, model_file, model_trans, model_csv):
        """

        :param ga_params:需要参加fuzzing的param
        :param model_file: lstm模型文件
        :param model_trans: lstm模型的归一化文件
        :param model_csv: 数据文件
        """
        # 参加ga的param
        self.ga = UAVGA(ga_params)
        self.ga.set_trans(model_trans)
        self.ga.set_model(model_file)

        # read csv
        data = pd.read_csv(model_csv, header=0, index_col=0)
        self.csv_data = data

    def random_choie_meanshift(self, segment_csv, rate=0.25):
        data_class = segment_csv.reshape(
                (-1, segment_csv.shape[1] * segment_csv.shape[2]))



        bandwidth = estimate_bandwidth(data_class, quantile=rate)
        clf = MeanShift(bandwidth=bandwidth, bin_seeding=True)

        clf.fit(data_class)
        predicted = clf.labels_
        print(f'Meanshift class: {max(predicted)}')
        # -------------
        c = list(map(lambda x: color(tuple(x)), ncolors(max(predicted) + 1)))
        #c = np.random.rand(max(predicted) + 1, 1)
        #c = list(map(lambda x: color(tuple(x)), ncolors(max(predicted) + 1)))

        colors = [c[i] for i in predicted]

        pca = PCA(n_components=2, svd_solver='arpack')
        show = pca.fit_transform(data_class)

        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(show[:, 0], show[:, 1], show[:, 2], c=colors, s=5)
        plt.scatter(show[:, 0], show[:, 1], c=colors, s=5)

        plt.show()
        # -------------------------
        out = []
        for i in range(max(predicted)):
            index = np.where(predicted == i)[0]
            col_index = np.random.choice(index, min(index.shape[0], 10))
            select = segment_csv[col_index]
            out.extend(select)
        out = np.array(out)
        return out

    def run(self, num=0, meanshift=False):
        """
        开始fuzzing搜索
        :param num: 返回的候选的个数
        :return:
        """
        segment_csv = self.split_segment()
        if num != 0 and not meanshift:
            index = np.random.choice(np.arange(segment_csv.shape[0]), num)
            segment_csv = segment_csv[index, :, :]
        elif meanshift:
            segment_csv = self.random_choie_meanshift(segment_csv)

        obj_population = []  # 种群

        for i, context in enumerate(segment_csv):
            self.ga.uavproblem.context_value = context
            self.ga.run()
            obj_population.append(self.ga.population)
            print(f'------------------- {i+1} / {segment_csv.shape[0]} -----------------')
        with open('result/pop.pkl','wb') as f:
            pickle.dump(obj_population, f)

    def split_segment(self):
        tmp = self.csv_data.to_numpy()[:, :ModelFit.config.CONTEXT_LEN]
        return np.array(np.array_split(tmp, tmp.shape[0] // (ModelFit.config.INPUT_LEN + 1), axis=0))

    @staticmethod
    def return_best_n_gen(n=1):
        candidate_vars = []
        candidate_objs = []

        with open('result/pop.pkl','rb') as f:
            obj_populations = pickle.load(f)
        for pop in obj_populations:
            pop_v = pop.ObjV
            pop_p = pop.Phen

            candidate_var_index = np.unique(pop_p, axis=0, return_index=True)[1]

            pop_v = pop_v[candidate_var_index]
            pop_p = pop_p[candidate_var_index]

            candidate = [-1] * pop_v

            candidate_index = np.argsort(candidate.reshape(-1))
            pop_v = pop_v[candidate_index].reshape((-1,1))
            pop_p = pop_p[candidate_index].reshape((-1,20))

            if n != 0:
                candidate_var = pop_v[:min(n, len(pop_v))]
                candidate_obj = pop_p[:min(n, len(pop_p))]
            candidate_obj = UAVProblem.reasonable_range_static(candidate_obj)

            candidate_vars.extend(candidate_var)
            candidate_objs.extend(candidate_obj)

        return candidate_vars, candidate_objs

    @staticmethod
    def return_random_n_gen(n=1):
        candidate_vars = []
        candidate_objs = []

        with open('result/pop.pkl', 'rb') as f:
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

            if n != 0:
                candidate_var = pop_v[[1,200,500]]
                candidate_obj = pop_p[[1,200,500],:]
            candidate_obj = UAVProblem.reasonable_range_static(candidate_obj)

            candidate_vars.extend(candidate_var)
            candidate_objs.extend(candidate_obj)
        return candidate_vars, candidate_objs


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)








