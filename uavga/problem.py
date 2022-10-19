# -*- coding: utf-8 -*-
import pickle

import geatpy as ea
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model


from Cptool.gaMavlink import GaMavlinkAPM
from Cptool.config import toolConfig
from Cptool.mavtool import min_max_scaler_param, load_param, read_unit_from_dict, pad_configuration_default_value
from ModelFit.approximate import CyLSTM


class Problem:
    def __init__(self):
        self.status_data: pd.DataFrame = None
        self.predictor: CyLSTM = None
        self.param_bounds = None
        self.step = None

    def init_status(self, status_data):
        self.status_data = status_data

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
        x = self.reasonable_range(x).to_numpy()
        # Padding the parameters if the configuration do not apply all parameters.
        if x.shape[1] != load_param().shape[1]:
            x = pad_configuration_default_value(x)

        param = min_max_scaler_param(x)

        # Statue change
        status_data = self.status_data.reshape((1, self.status_data.shape[0],
                                                -1, toolConfig.DATA_LEN))[:, :, :, :toolConfig.STATUS_LEN]
        param = param.reshape((param.shape[0], 1, 1, -1))
        # repeat data
        repeat_status = np.repeat(status_data, param.shape[0], axis=0)
        repeat_param = np.repeat(param, repeat_status.shape[2], axis=2)
        repeat_param = np.repeat(repeat_param, repeat_status.shape[1], axis=1)

        # Merge
        merge_data = np.c_[repeat_status, repeat_param]
        # Reshape
        merge_data = merge_data.reshape((merge_data.shape[0], merge_data.shape[1], -1)).astype(np.float)

        # create predicted status of this status patch
        feature_x, feature_y = self.predictor.data_split_3d(merge_data)
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
        para_dict = load_param()
        step_unit = read_unit_from_dict(para_dict)
        return param * step_unit


class ProblemGAOld(Problem, ea.Problem):
    def __init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin):
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)
        super().__init__()

    def aimFunc(self, pop):
        # 得到决策变量矩阵
        x = pop.Phen
        x = self.reasonable_range(x).to_numpy()
        param = min_max_scaler_param(x)

        # Statue change
        status_data = self.status_data.reshape((1, 1, -1, toolConfig.DATA_LEN))[:, :, :, :toolConfig.STATUS_LEN]
        param = param.reshape((param.shape[0], 1, 1, -1))
        # repeat data
        repeat_status = np.repeat(status_data, param.shape[0], axis=0)
        repeat_param = np.repeat(param, repeat_status.shape[2], axis=2)
        repeat_param = np.repeat(repeat_param, repeat_status.shape[1], axis=1)

        # Merge
        merge_data = np.c_[repeat_status, repeat_param]
        # Reshape
        merge_data = merge_data.reshape((merge_data.shape[0], merge_data.shape[1], -1)).astype(np.float)

        # create predicted status of this status patch
        feature_x, feature_y = self.predictor.data_split_3d(merge_data)
        # Predict
        predicted_feature = self.predictor.predict_feature(feature_x)
        # reshape to 3D (x number, status patch)
        predicted_feature = predicted_feature.reshape((x.shape[0], -1, predicted_feature.shape[-1]))
        feature_y = feature_y.reshape((x.shape[0], -1, predicted_feature.shape[-1]))
        # deviation loss
        patch_array_loss = self.predictor.cal_deviation_old(predicted_feature, feature_y)

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
        para_dict = load_param()
        step_unit = read_unit_from_dict(para_dict)
        return param * step_unit