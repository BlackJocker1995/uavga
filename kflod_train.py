import pandas as pd

import ModelFit.config
from ModelFit.approximate import CyLSTM

"""
训练LSTM的模型
"""
if __name__ == '__main__':
    lstm = CyLSTM(100, 1024)
    lstm.run_5flow_test(f"./log/Ardupilot/csv/train.csv", cuda=True)
    test = pd.read_csv(f"./log/Ardupilot/csv/test.csv", header=0, index_col=0)
    lstm.test_kfold(f"model/Ardupilot/{ModelFit.config.INPUT_LEN}", test, 5)