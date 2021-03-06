import pandas as pd

from ModelFit.approximate import CyLSTM

"""
生成对比
"""
if __name__ == '__main__':
    lstm = CyLSTM(100, 512)
    lstm.read_model()
    normal = pd.read_csv(f"./log/Ardupilot/csv/test.csv", index_col=0)
    lstm.test_cmp_draw(normal, "predict&true", num=150)
    #lstm.test_cmp_draw(normal, "Real", num=50)