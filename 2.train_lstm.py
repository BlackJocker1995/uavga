from ModelFit.approximate import CyLSTM, CyTCN

"""
Train LSTM Model
"""
if __name__ == '__main__':
    CyTCN.fit_trans(f"./log/Ardupilot/csv/train.csv")
    lstm = CyLSTM(200, 512)
    lstm.run(f"./log/Ardupilot/csv/train.csv", cuda=True)