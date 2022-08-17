# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import pandas as pd

from Cptool.config import toolConfig


class RangeProblem:
    def __init__(self):
        self.param_bounds = None
        self.step = None

    def init_bounds_and_step(self, param_bounds, step):
        self.param_bounds = param_bounds
        # step
        self.step = step

    def param_value2step(self, configuration):
        np_config = np.ceil(configuration / self.step)
        np_config = np_config * self.step
        np_config = pd.DataFrame([np_config.tolist()], columns=toolConfig.PARAM)
        return np_config.iloc[0].to_dict()

    def function(self, configuration):
        pass


class GARangeProblem(RangeProblem, ea.Problem):
    def __init__(self, name, M, maxormins, Dim,
                 varTypes, lb, ub, lbin, ubin, result_data):
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)
        self.data = result_data

    def aimFunc(self, pop):
        # 得到决策变量矩阵
        x = pop.Phen
        x = self.reasonable_range(x)

        button = x[:, ::2]
        top = x[:, 1::2]

        score_rate = np.zeros(x.shape[0])
        score_len = np.zeros(x.shape[0])
        for i, t, b in zip(range(x.shape[0]), top, button):
            rate, length = self.satisfy_range(t, b)
            score_rate[i] = rate
            score_len[i] = length

        # 计算目标函数值，赋值给pop种群对象的ObjV属性
        f2 = score_rate.reshape((-1, 1))
        # f2 = ((top - button) * self.step).sum(axis=1).reshape((-1, 1))
        f1 = score_len.reshape((-1, 1))
        pop.ObjV = np.hstack([f1, f2])

    def reasonable_range(self, param):
        return param * np.repeat(self.step, 2)

    def satisfy_range(self, top, button):
        values = self.data.values[:, :-1]

        to_top = (top - values).min(axis=1)
        to_button = (values - button).min(axis=1)

        index = np.where((to_top >= 0) & (to_button >= 0))[0]
        if len(index) == 0:
            return 0, len(index)
        satisfy_value = self.data.iloc[index]
        pass_index = satisfy_value.result == 'pass'
        pass_rate = pass_index.values.sum() / satisfy_value.shape[0]
        print(f'include num: {len(index)}   rate: {pass_rate}')
        return pass_rate, len(index)
