from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM

"""
Train LSTM Model
"""
if __name__ == '__main__':
    # pd_csv = CyLSTM.merge_file_data(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_changed/csv")
    # CyLSTM.fit_trans(pd_csv)

    lstm = CyLSTM(100, 512)
    feature = lstm.extract_feature(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_changed/csv")
    # Save
    feature.to_csv(f"model/{toolConfig.MODE}/features.csv", index=False)