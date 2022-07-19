import argparse
import csv
import logging
import os
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
    parser.add_argument('--device', dest='device', type=str, help='Name of the candidate')
    args = parser.parse_args()
    device = args.device
    if device is None:
        device = 0
    print(device)

    # Get Fuzzing result and validate
    candidate_var, candidate_obj = return_cluster_thres_gen(0.35)
    candidate_obj = np.array(candidate_obj, dtype=float).round(8)
    candidate_var = np.array(candidate_var, dtype=float).round(8)

    # Simulator validation
    manager = GaSimManager(debug=toolConfig.DEBUG)

    i = 0
    # Random order
    rand_index = (np.arange(candidate_obj.shape[0]))
    np.random.shuffle(rand_index)
    candidate_obj = candidate_obj[rand_index]
    candidate_var = candidate_var[rand_index]

    # Loop to validate configurations with SITL simulator
    for index, vars, value_vector in zip(np.arange(candidate_obj.shape[0]), candidate_var, candidate_obj):
        print(f'======================={index} / {candidate_obj.shape[0]} ==========================')
        # if exist file, append new data in the end.
        if os.path.exists(f'result/{toolConfig.MODE}/params.csv'):
            while not os.access(f"result/{toolConfig.MODE}/params.csv", os.R_OK):
                continue
            data = pd.read_csv(f'result/{toolConfig.MODE}/params.csv')
            exit_data = data.drop(['score', 'result'], axis=1, inplace=False)
            # carry our simulation test
            if ((exit_data - value_vector).sum(axis=1).abs() < 0.00001).sum() > 0:
                continue

        configuration = pd.Series(value_vector, index=toolConfig.PARAM).to_dict()
        # start multiple SITL
        manager.start_multiple_sitl(device)
        manager.mav_monitor_init(GaMavlinkAPM, device)

        if not manager.mav_monitor_connect():
            manager.stop_sitl()
            continue

        manager.mav_monitor.set_mission("Cptool/fitCollection.txt", israndom=False)
        manager.mav_monitor.set_params(configuration)

        manager.mav_monitor.start_mission()

        result = manager.mav_monitor_error()

        # if the result have no instability, skip.
        if not os.path.exists(f'result/{toolConfig.MODE}/params.csv'):
            while not os.access(f"result/{toolConfig.MODE}/params.csv", os.W_OK):
                continue
            data = pd.DataFrame(columns=(toolConfig.PARAM + ['score', 'result']))
        else:
            while not os.access(f"result/{toolConfig.MODE}/params.csv", os.W_OK):
                continue
            # Add instability result
            tmp_row = value_vector.tolist()
            tmp_row.append(vars[0])
            tmp_row.append(result)

            # Write Row
            with open(f"result/{toolConfig.MODE}/params.csv", 'a+') as f:
                csv_file = csv.writer(f)
                csv_file.writerow(tmp_row)
                logging.debug("Write row to params.csv.")

        manager.stop_sitl()
        i += 1

    localtime = time.asctime(time.localtime(time.time()))
    # Mail notification plugin
    # send_mail(Cptool.config.AIRSIM_PATH, localtime)