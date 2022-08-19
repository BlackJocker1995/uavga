import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Cptool.config import toolConfig

if __name__ == '__main__':
    toolConfig.select_mode("PX4")
    feature = pd.read_csv(f"model/{toolConfig.MODE}/{toolConfig.INPUT_LEN}_{toolConfig.OUTPUT_LEN}/features.csv")
    index = np.arange(0, feature.shape[0])
    train, test = train_test_split(index, test_size=0.1, shuffle=True, random_state=2022)
    train_data = feature.iloc[train]
    test_data = feature.iloc[test]
    train_data.to_csv(f"model/{toolConfig.MODE}/{toolConfig.INPUT_LEN}_{toolConfig.OUTPUT_LEN}/features_train.csv", index=False)
    test_data.to_csv(f"model/{toolConfig.MODE}/{toolConfig.INPUT_LEN}_{toolConfig.OUTPUT_LEN}/features_test.csv", index=False)
