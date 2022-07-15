# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import pickle
import pandas as pd

from Cptool.gaMavlink import GaMavlinkAPM


class ANAProblem(ea.Problem):

    def __init__(self, param_choice, para_dict, result_data):
        self.para_dict = para_dict
        self.sub_parr_dict = para_dict[param_choice]

        # 范围
        value_range = np.array([self.sub_parr_dict[it]['range'] for it in list(self.sub_parr_dict)])
        # 步长
        self.step = np.array([self.sub_parr_dict[it]['step'] for it in list(self.sub_parr_dict)])
        self.data = result_data

        default = np.array([self.sub_parr_dict[it]['default'] for it in list(self.sub_parr_dict)]) / self.step

        name = 'ANAProblem'  # 初始化name（函数名称，可以随意设置）boundary
        M = 2  # 初始化M（目标维数）
        maxormins = [-1, -1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = value_range.shape[0] * 2  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        self.lb = np.repeat(value_range[:, 0] / self.step, 2)  # 决策变量下界
        self.lb[1::2] = default
        self.ub = np.repeat(value_range[:, 1] / self.step, 2)  # 决策变量上界
        self.ub[::2] = default
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, self.lb, self.ub, lbin, ubin)

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
