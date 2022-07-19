import pickle

from Cptool.config import toolConfig
from uavga.fuzzer import run_fuzzing

if __name__ == '__main__':
    with open(f"model/{toolConfig.MODE}/raw_test.pkl", 'rb') as f:
        np_data = pickle.load(f)
    run_fuzzing(np_data)
