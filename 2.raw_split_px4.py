import pickle

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split
import numpy as np
from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM, Modeling

"""
Train LSTM Model
"""
if __name__ == '__main__':
    toolConfig.select_mode("PX4")

    pd_csv = CyLSTM.merge_file_data(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/ulg_changed/csv")

    np_data = pd_csv.to_numpy()[:, :toolConfig.STATUS_LEN]

    trans = Modeling.load_trans()
    np_data = trans.transform(np_data)

    np_data = np_data[:-(np_data.shape[0] % (toolConfig.SEGMENT_LEN + 1)), :]
    # Split
    np_data = np.array(np.array_split(np_data, np_data.shape[0] // (toolConfig.SEGMENT_LEN + 1), axis=0))

    index = np.arange(0, np_data.shape[0])
    train, test = train_test_split(index, test_size=0.1, shuffle=True, random_state=2022)
    test_data = np_data[test]

    with open(f"model/{toolConfig.MODE}/raw_test.pkl", "wb") as f:
        pickle.dump(test_data, f)
