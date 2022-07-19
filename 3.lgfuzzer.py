import argparse
import csv
import os
import pickle
import time

import numpy as np
import pandas as pd

import Cptool
import ModelFit
from Cptool.config import toolConfig
from Cptool.gaSimManager import GaSimManager

# from Cptool.gaSimManager import GaSimManager
from uavga.fuzzer import run_fuzzing

if __name__ == '__main__':
    with open(f"model/{toolConfig.MODE}/raw_test.pkl", 'rb') as f:
        np_data = pickle.load(f)
    run_fuzzing(np_data)
