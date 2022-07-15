import json
import os

import numpy as np
import pandas as pd
from pymavlink import mavutil, mavwp
from Cptool.config import toolConfig
import sys, select, os
import datetime
from timeit import default_timer as timer
import signal

def load_param() -> json:
    """
    load parameter we want to fuzzing
    :return:
    """
    if toolConfig.MODE == 'Ardupilot':
        path = 'Cptool/param_ardu.json'
    elif toolConfig.MODE == 'PX4':
        path = 'Cptool/param_px4.json'
    with open(path, 'r') as f:
        return pd.DataFrame(json.loads(f.read()))


def get_default_values(para_dict):
    return para_dict.loc[['default']]


def select_sub_dict(para_dict, param_choice):
    return para_dict[param_choice]


def read_range_from_dict(para_dict):
    return np.array(para_dict.loc['range'].to_list())


def read_unit_from_dict(para_dict):
    return para_dict.loc['step'].to_numpy()


# Log analysis function
def read_path_specified_file(log_path, exe):
    """
        :param log_path:
        :param exe:
        :return:
        """
    file_list = []
    for filename in os.listdir(log_path):
        if filename.endswith(f'.{exe}'):
            file_list.append(filename)
    file_list.sort()
    return file_list


def rename_bin(log_path, ranges):
    file_list = read_path_specified_file(log_path, 'BIN')
    # 列出文件夹内所有.BIN结尾的文件并排序
    for file, num in zip(file_list, range(ranges[0], ranges[1])):
        name, _ = file.split('.')
        os.rename(f"{log_path}/{file}", f"{log_path}/{str(num).zfill(8)}.BIN")


def min_max_scaler_param(param_value):
    para_dict = load_param()
    participle_param = toolConfig.PARAM
    param_choice_dict = select_sub_dict(para_dict, participle_param)

    param_bounds = read_range_from_dict(param_choice_dict)
    lb = param_bounds[:, 0]
    ub = param_bounds[:, 1]
    param_value = (param_value - lb) / (ub-lb)
    return param_value


def return_min_max_scaler_param(param_value):
    param = load_param()
    param_bounds = read_range_from_dict(param)
    lb = param_bounds[:, 0]
    ub = param_bounds[:, 1]
    param_value = (param_value * (ub-lb)) + lb
    return param_value


def min_max_scaler(trans, values):
    status_value = values[:, :toolConfig.STATUS_LEN]
    param_value = values[:, toolConfig.STATUS_LEN:]

    param_value = min_max_scaler_param(param_value)

    status_value = trans.transform(status_value)

    return np.c_[status_value, param_value]


def return_min_max_scaler(trans, values):
    status_value = values[:, :toolConfig.STATUS_LEN]
    param_value = values[:, toolConfig.STATUS_LEN:]

    param_value = return_min_max_scaler_param(param_value)

    status_value = trans.transform(status_value)

    return np.c_[status_value, param_value]
