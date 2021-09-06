# coding: utf-8
import numpy as np
import pandas


def read_invalid_csv(file):
    return pandas.read_csv(file, index_col=0)


def cal_coverage(range_up, range_down, invalid_config):
    invalid_config = invalid_config[invalid_config['result'] != 'pass']

    data = invalid_config.to_numpy()[:, :6]
    over_up = data < range_up
    over_down = data > range_down

    reduce:np.array = over_down * over_up
    reduce = np.prod(reduce,axis=1)

    avoid_ratio = np.sum(reduce==0) / len(reduce)
    print(avoid_ratio)
    print(f'Avoid Invalid {np.sum(reduce==0)} / {len(reduce)}')

    not_avoid_invalid = invalid_config[reduce!=0]

    return avoid_ratio


def cal_invalidRvalid(range_up, range_down, invalid_config):
    # invalid_config = invalid_config[invalid_config['result'] != 'pass']

    data = invalid_config.to_numpy()[:, :6]
    over_up = data <= range_up
    over_down = data >= range_down

    reduce:np.array = over_down * over_up
    reduce = np.prod(reduce,axis=1)

    invalid_config = invalid_config[reduce==1]
    invalid = invalid_config[invalid_config['result'] != 'pass']
    valid = invalid_config[invalid_config['result'] == 'pass']

    print(f'Invalid / Valid / Covered - {len(invalid)} / {len(valid)} / {len(invalid_config)}')


if __name__ == '__main__':
    # GA
    # range_up = [2.1, 0.4, 0.9, 0.1,2000,8000]
    # range_down = [0.6,-1,-0.7, -0.8,300,1000]

    # range_up = [4.2, 2.1, 1.3, 3, 1950, 4650]
    # range_down = [1.9, -0.5, -0.7, -0.1, 50, 1100]

    # M
    # range_up = [6.0, 4.1, 3.2, 1.3, 2000, 8000]
    # range_down = [0.6, -1, -0.7, -0.8, 300, 1000]

    # one
    range_up = [6.0, 5.0, 5.0, 5.0, 2000, 8000]
    range_down = [0.1, -4.7, -5.0, -5.0, 50, 1000]

    invalid_config = read_invalid_csv(r'E:\program\uavga\out\ardupilot\6.csv')
    cal_invalidRvalid(range_up, range_down, invalid_config)
