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
from uavga.fuzzer import LGFuzzer

# from Cptool.gaSimManager import GaSimManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--device', dest='device', type=str, help='Name of the candidate')
    args = parser.parse_args()
    device = args.device

    if device is None:
        device = 0
    print(device)

    # The parameters you want to fuzzing, they must be corresponding to the predictor had.
    param = [
        "PSC_POSXY_P",
        "PSC_VELXY_P",
        "PSC_POSZ_P",
        "ATC_ANG_RLL_P",
        "ATC_RAT_RLL_I",
        "ATC_RAT_RLL_D",
        "ATC_RAT_RLL_P",
        "ATC_ANG_PIT_P",
        "ATC_RAT_PIT_P",
        "ATC_RAT_PIT_I",
        "ATC_RAT_PIT_D",
        "ATC_ANG_YAW_P",
        "ATC_RAT_YAW_P",
        "ATC_RAT_YAW_I",
        "ATC_RAT_YAW_D",
        "WPNAV_SPEED",
        "WPNAV_SPEED_UP",
        "WPNAV_SPEED_DN",
        "WPNAV_ACCEL",
        "ANGLE_MAX"
    ]
    # Initialize the fuzzer
    lgfuizzer = LGFuzzer(param, f'model/{toolConfig.MODE}/{toolConfig.INPUT_LEN}/lstm.h5',
                         f'model/{toolConfig.MODE}//trans.pkl',
                         f"log/{toolConfig.MODE}/csv/test.csv")
    # Run the fuzzing test
    lgfuizzer.run(num=3, meanshift=True)