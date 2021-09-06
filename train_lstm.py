from ModelFit.approximate import CyLSTM, CyTCN

"""
训练LSTM的模型
"""
if __name__ == '__main__':
    CyTCN.fit_trans(f"./log/Ardupilot/csv/train.csv")
    lstm = CyLSTM(200, 512)
    lstm.run(f"./log/Ardupilot/csv/train.csv", cuda=True)