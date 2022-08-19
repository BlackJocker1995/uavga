import pandas as pd

from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM

"""
Train LSTM Model
"""
if __name__ == '__main__':
    lstm = CyLSTM(100, 512)
    # read
    feature = pd.read_csv(f"model/{toolConfig.MODE}/{toolConfig.INPUT_LEN}_{toolConfig.OUTPUT_LEN}/features_train.csv")
    # Train
    lstm.train(feature, cuda=True)
