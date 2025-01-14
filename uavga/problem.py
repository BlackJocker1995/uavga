# -*- coding: utf-8 -*-
import pickle

import geatpy as ea
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model
from abc import ABC, abstractmethod

from Cptool.gaMavlink import GaMavlinkAPM
from Cptool.config import toolConfig
from Cptool.mavtool import min_max_scaler_param, load_param, read_unit_from_dict, pad_configuration_default_value, \
    select_sub_dict
from ModelFit.approximate import CyLSTM


class BaseProblem(ABC):
    def __init__(self):
        self.status_data = None
        self.predictor = None
        self.param_bounds = None
        self.step = None

    @abstractmethod
    def evaluate(self, x, y):
        pass

    def param_value2step(self, configuration):
        np_config = np.ceil(configuration / self.step) * self.step
        return pd.DataFrame([np_config.tolist()], 
                          columns=toolConfig.PARAM).iloc[0].to_dict()


class ProblemGA(BaseProblem, ea.Problem):
    def __init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin):
        BaseProblem.__init__(self)
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                          varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        x = self._preprocess_population(pop)
        predicted_feature, feature_y = self._get_predictions(x)
        pop.ObjV = self._calculate_loss(predicted_feature, feature_y)

    def _preprocess_population(self, pop):
        """Preprocess population data"""
        x = pop.Phen
        x = self.reasonable_range(x).to_numpy()
        if x.shape[1] != load_param().shape[1]:
            x = pad_configuration_default_value(x)
        return x

    def _get_predictions(self, x):
        """Get model predictions"""
        param = min_max_scaler_param(x)
        merge_data = self._prepare_merge_data(param)
        feature_x, feature_y = self.predictor.data_split_3d(merge_data)
        predicted_feature = self.predictor.predict_feature(feature_x)
        
        dims = (x.shape[0], -1, predicted_feature.shape[-1])
        return (predicted_feature.reshape(dims), 
                feature_y.reshape(dims))

    def _prepare_merge_data(self, param):
        """Prepare merged data for prediction"""
        status = self.status_data.reshape((1, self.status_data.shape[0],
                                         -1, toolConfig.DATA_LEN))
        status = status[:, :, :, :toolConfig.STATUS_LEN]
        param = param.reshape((param.shape[0], 1, 1, -1))
        
        repeat_status = np.repeat(status, param.shape[0], axis=0)
        repeat_param = np.repeat(param, repeat_status.shape[2], axis=2)
        repeat_param = np.repeat(repeat_param, repeat_status.shape[1], axis=1)
        
        merge_data = np.c_[repeat_status, repeat_param]
        return merge_data.reshape((merge_data.shape[0], 
                                 merge_data.shape[1], -1)).astype(np.float)

    def _calculate_loss(self, predicted, actual):
        """Calculate prediction loss"""
        return self.predictor.cal_patch_deviation(predicted, actual).reshape((-1, 1))

    def reasonable_range(self, param):
        """
        Restore data to original scale
        Args:
            param: Input parameters
        Returns:
            Scaled parameters as DataFrame
        """
        np_config = param * self.step
        np_config = pd.DataFrame(np_config, columns=toolConfig.PARAM_PART)
        return np_config

    @staticmethod
    def reasonable_range_static(param):
        """
        Restore data to original scale using static method
        Args:
            param: Input parameters
        Returns:
            Scaled parameters
        """
        para_dict = load_param()
        param_choice_dict = select_sub_dict(para_dict, toolConfig.PARAM_PART)
        step_unit = read_unit_from_dict(param_choice_dict)
        return param * step_unit


class ProblemGAOld(BaseProblem, ea.Problem):
    def __init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin):
        BaseProblem.__init__(self)
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)

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
        Restore data to original scale
        Args:
            param: Input parameters
        Returns:
            Scaled parameters as DataFrame
        """
        np_config = param * self.step
        np_config = pd.DataFrame(np_config, columns=toolConfig.PARAM)
        return np_config

    @staticmethod
    def reasonable_range_static(param):
        """
        Restore data to original scale using static method
        Args:
            param: Input parameters
        Returns:
            Scaled parameters
        """
        para_dict = load_param()
        param_choice_dict = select_sub_dict(para_dict, toolConfig.PARAM_PART)
        step_unit = read_unit_from_dict(param_choice_dict)
        return param * step_unit