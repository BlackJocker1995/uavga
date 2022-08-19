import pandas as pd

from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM

"""
训练LSTM的模型
"""
if __name__ == '__main__':
    toolConfig.select_mode("PX4")

    lstm = CyLSTM(100, 1024)
    test = pd.read_csv(f"model/{toolConfig.MODE}/{toolConfig.INPUT_LEN}_{toolConfig.OUTPUT_LEN}/features_test.csv", header=0)
    lstm.read_model()
    lstm.test(test)