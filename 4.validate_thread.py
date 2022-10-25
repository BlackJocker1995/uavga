import argparse
import csv
import logging
import os
import pickle
import subprocess
import time

import numpy as np
import pandas as pd

import Cptool
import ModelFit
from Cptool.config import toolConfig
from Cptool.gaMavlink import GaMavlinkAPM
from Cptool.gaSimManager import GaSimManager


# from Cptool.gaSimManager import GaSimManager
from uavga.fuzzer import return_random_n_gen, return_cluster_thres_gen

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--thread', dest='thread', type=int, help='Name of the candidate', default=1)
    args = parser.parse_args()
    thread = args.thread
    thread = int(thread)
    print(thread)

    for i in range(thread):
        cmd = f'gnome-terminal --tab --working-directory={os.getcwd()} -e "python3 {os.getcwd()}/4.validate.py --device {i}"'
        os.system(cmd)
