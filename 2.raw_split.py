from sklearn.model_selection import train_test_split
import numpy as np
from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM

"""
Train LSTM Model
"""
if __name__ == '__main__':
    pd_csv = CyLSTM.merge_file_data(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_changed/csv")
    index = np.arange(0, pd_csv.shape[0])
    train, test = train_test_split(index, test_size=0.1, shuffle=True, random_state=2022)
    train_data = pd_csv.iloc[train]
    test_data = pd_csv.iloc[test]
    test_data.to_csv(f"model/{toolConfig.MODE}/raw_test.csv", index=False)
