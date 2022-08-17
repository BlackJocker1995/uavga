import os

import pandas as pd
import numpy as np


def satisfy_range(panda_m, top, button):
    values = panda_m.values[:, :-1]

    to_top = (top - values).min(axis=1)
    to_button = (values - button).min(axis=1)

    index = np.where((to_top >= 0) & (to_button >= 0))[0]
    satisfy_value = panda_m.iloc[index]
    pass_index = satisfy_value.result == 'pass'
    pass_rate = pass_index.values.sum() / satisfy_value.shape[0]
    print(f'rate: {pass_rate}')


def best_summary(UavConfig):
    objV = 'E:/program/uavga/Result/ObjV.csv'
    phen = 'E:/program/uavga/Result/Phen.csv'

    pd_objV = pd.read_csv(objV,names=['num','rate'])
    pd_phen = pd.read_csv(phen)

    pd_objV['sort_num'] = pd_objV['rate'].rank(ascending=0, method='dense')
    index = pd_objV.sort_num  < 10
    pd_choice_phen = pd_phen[index]


    param_choice_dict = {key: value for key, value in UavConfig.param_dict.items() if key in UavConfig.get_participate_param()}
    # 步长 扩充
    step = np.array([param_choice_dict[it]['step'] for it in list(param_choice_dict)])
    step = step.repeat(2)
    # 还原
    pd_choice_phen = (pd_choice_phen * step)

    # 求众数
    bincout = pd_choice_phen.mode().to_numpy()[0, :]
    print(bincout)
    return bincout





