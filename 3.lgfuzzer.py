import argparse
import csv
import os
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
    csv_data = pd.read_csv(f"model/{toolConfig.MODE}/raw_test.csv")
    run_fuzzing(csv_data, num=100)
