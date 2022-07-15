# -*- coding: utf-8 -*-
import pickle

import geatpy as ea
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model


from Cptool.gaMavlink import GaMavlinkAPM
from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM


class Problem:
    def __init__(self):
        self.status_data: pd.DataFrame = None
        self.predictor: CyLSTM = None
        self.param_bounds = None
        self.step = None

    def init_status(self, status_data):
        order = toolConfig.STATUS_ORDER.copy()
        order = order.remove("TimeS")
        self.status_data = pd.DataFrame(status_data, columns=order)

    def init_predictor(self, predictor):
        self.predictor = predictor
        self.predictor.read_trans()

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


class ProblemGA(Problem, ea.Problem):
    def __init__(self, name, M, maxormins, Dim,
                 varTypes, lb, ub, lbin, ubin):
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        # 得到决策变量矩阵
        x = pop.Phen
        x = self.reasonable_range(x)

        # repeat data
        repeat_status = pd.concat([self.status_data] * x.shape[0]).reset_index(drop=True)
        repeat_param = pd.DataFrame(np.repeat(x.values, self.status_data.shape[0], axis=0), columns=x.columns)
        repeat_status[toolConfig.PARAM] = repeat_param

        status_step = self.status_data.shape[0]
        feature_step = self.status_data.shape[0] - toolConfig.INPUT_LEN
        feature = pd.DataFrame()
        for i in range(x.shape[0]):
            status_index = i * status_step
            feature_index = i * feature_step
            tmp_status = repeat_status.iloc[status_index:status_index+status_step]
            # status data to feature data
            tmp_feature_data = self.predictor.status2feature(tmp_status)
            feature = pd.concat([feature, tmp_feature_data])
        # create predicted status of this status patch
        feature_x, feature_y = self.predictor.data_split(feature)
        # Predict
        predicted_feature = self.predictor.predict_feature(feature_x)
        # reshape to 3D (x number, status patch)
        predicted_feature = predicted_feature.reshape((x.shape[0], -1, predicted_feature.shape[-1]))
        feature_y = feature_y.reshape((x.shape[0], -1, predicted_feature.shape[-1]))
        # deviation loss
        patch_array_loss = self.predictor.cal_patch_deviation(predicted_feature, feature_y)

        pop.ObjV = patch_array_loss.reshape((-1, 1))

    def reasonable_range(self, param):
        """
        还原数据
        :param param:
        :return:
        """
        np_config = param * self.step
        np_config = pd.DataFrame(np_config, columns=toolConfig.PARAM)
        return np_config

    def evaluate(self, x, y):
        return np.abs(x - y).sum(axis=1)

    @staticmethod
    def reasonable_range_static(param):
        """
        还原数据
        :param param:
        :return:
        """
        para_dict = GaMavlinkAPM.load_param()
        step_unit = GaMavlinkAPM.read_unit_from_dict(para_dict)
        return param * step_unit