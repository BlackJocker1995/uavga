# -*- coding: utf-8 -*-
import pickle

import geatpy as ea
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model

from ModelFit.config import mlConfig
import ModelFit.config
from Cptool.gaMavlink import GaMavlink
from Cptool.config import toolConfig

class UAVProblem(ea.Problem):
    def __init__(self, param_choice, para_dict):
        self.para_dict = para_dict
        self.sub_parr_dict = para_dict[param_choice]

        # sub 的数据
        self.step_unit = GaMavlink.read_unit_from_dict(self.sub_parr_dict)
        sub_value_range = GaMavlink.read_range_from_dict(self.sub_parr_dict)
        # general 的数据 装欢成lstm输入的长度
        default_value = GaMavlink.get_default_values(para_dict).loc[['default']]
        self.segment_default_value = pd.DataFrame(default_value, dtype=np.float)

        name = 'UAVProblem'  # 初始化name（函数名称，可以随意设置）boundary
        M = 1 # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = sub_value_range.shape[0] # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        self.lb = sub_value_range[:, 0] / self.step_unit  # 决策变量下界
        self.ub = sub_value_range[:, 1] / self.step_unit  # 决策变量上界
        lbin = [0] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [0] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, self.lb, self.ub, lbin, ubin)

        self.predict_module = None
        self.trans = None
        self.context_value = None

        # 参与fuzzing的param的数量
        self.m = len(toolConfig.PARAM)
        self.params = param_choice

    def aimFunc(self, pop):
        # 得到决策变量矩阵
        x = pop.Phen
        x = self.reasonable_range(x)

        # 替换param
        x = x.reshape(-1, 20)


        default_param_value = self.segment_default_value
        default_param_value = pd.concat([default_param_value] * x.shape[0])
        default_param_value[self.params] = x
        default_param_value = pd.concat([default_param_value] * (mlConfig.INPUT_LEN + 1))

        segment_param = default_param_value
        segment_param = segment_param.to_numpy().reshape((x.shape[0], mlConfig.INPUT_LEN + 1, -1))

        # sensor 数据
        segment_x = self.context_value
        segment_x = np.repeat(segment_x, x.shape[0])
        segment_x = segment_x.reshape((x.shape[0], mlConfig.INPUT_LEN + 1, -1))

        merge_value = np.concatenate([segment_x, segment_param], -1)
        # 先变二位归一化，然后再变回三维
        merge_value = self.trans.transform(merge_value.reshape(-1, merge_value.shape[2])).reshape(
            (-1, mlConfig.INPUT_LEN + 1, merge_value.shape[2]))

        sensor_x_param = merge_value[:, :mlConfig.INPUT_LEN, :]
        sensor_y = merge_value[:, mlConfig.INPUT_LEN, :mlConfig.OUTPUT_DATA_LEN]

        predict_x = self.predict_module.predict(sensor_x_param)

        # calculate prediction score
        score = self.evaluate(predict_x, sensor_y)

        score = score.reshape((-1, 1))

        pop.ObjV = score


    # def individualFunc(self, x):
    #     # 替换param
    #     x = x.reshape(-1, 20)
    #
    #     default_param_value = self.segment_default_value
    #     default_param_value = pd.concat([default_param_value] * x.shape[0])
    #     default_param_value[self.params] = x
    #     segment_param = default_param_value.loc[default_param_value.index.repeat(ModelFit.config.INPUT_LEN + 1)]
    #     segment_param = segment_param.to_numpy().reshape((x.shape[0], ModelFit.config.INPUT_LEN + 1, -1))
    #
    #
    #     # sensor 数据
    #     segment_x = self.context_value
    #     segment_x = np.repeat(segment_x, x.shape[0])
    #     segment_x = segment_x.reshape((x.shape[0], ModelFit.config.INPUT_LEN + 1, -1))
    #
    #     merge_value = np.concatenate([segment_x, segment_param ], -1)
    #     # 先变二位归一化，然后再变回三维
    #     merge_value = self.trans.transform(merge_value.reshape(-1, merge_value.shape[2])).reshape(
    #         (-1, ModelFit.config.INPUT_LEN + 1, merge_value.shape[2]))
    #
    #     sensor_x_param = merge_value[:, :ModelFit.config.INPUT_LEN, :]
    #     sensor_y = merge_value[:, ModelFit.config.INPUT_LEN, :ModelFit.config.OUTPUT_DATA_LEN]
    #
    #     predict_x = self.predict_module.predict(sensor_x_param)
    #
    #     # calculate prediction score
    #     score = self.evaluate(predict_x, sensor_y)
    #
    #     # score.reshape((-1, 1))
    #
    #     return score

    def reasonable_range(self, param):
        """
        还原数据
        :param param:
        :return:
        """
        return param * self.step_unit

    def set_model(self, filename):
        """
        设置预测模型
        :param filename:
        :return:
        """
        self.predict_module = load_model(filename)

    def set_trans(self, filename):
        with open(filename, 'rb') as f:
            self.trans = pickle.load(f)

    def evaluate(self, x, y):
        return np.abs(x - y).sum(axis=1)

    @staticmethod
    def reasonable_range_static(param):
        """
        还原数据
        :param param:
        :return:
        """
        para_dict = GaMavlink.load_param()
        step_unit = GaMavlink.read_unit_from_dict(para_dict)
        return param * step_unit