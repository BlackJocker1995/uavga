import pandas as pd

from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM, CyTCN

"""
生成对比
"""
if __name__ == '__main__':
    toolConfig.select_mode("PX4")
    lstm = CyLSTM(100, 512)
    lstm.read_model()
    normal = pd.read_csv(f"csv/csv/4.csv", index_col=0).iloc[:400]
    lstm.test_cmp_draw(normal, "predict&true", num=300)
    #lstm.test_cmp_draw(normal, "Real", num=50)