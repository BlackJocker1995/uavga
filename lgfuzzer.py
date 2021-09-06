import argparse
import csv
import os
import time

import numpy as np
import pandas as pd

import Cptool.config
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

    param = [
        "PSC_POSXY_P",
        "PSC_VELXY_P",
        "PSC_POSZ_P",
        "ATC_ANG_RLL_P",
        "ATC_ANG_PIT_P",
        "ATC_ANG_YAW_P",
        "ATC_RAT_RLL_I",
        "ATC_RAT_RLL_D",
        "ATC_RAT_RLL_P",
        "ATC_RAT_PIT_P",
        "ATC_RAT_PIT_I",
        "ATC_RAT_PIT_D",
        "ATC_RAT_YAW_P",
        "ATC_RAT_YAW_I",
        "ATC_RAT_YAW_D",
        "WPNAV_SPEED",
        "WPNAV_SPEED_UP",
        "WPNAV_SPEED_DN",
        "WPNAV_ACCEL",
        "ANGLE_MAX",
    ]
    # lgfuizzer = LGFuzzer(param, f'model/{Cptool.config.MODE}/{ModelFit.config.INPUT_LEN}/lstm.h5',
    #                      f'model/{Cptool.config.MODE}//trans.pkl',
    #                      f"./log/{Cptool.config.MODE}/csv/train.csv")
    #
    # lgfuizzer.run(num=3, meanshift=True)




    candidate_var, candidate_obj = LGFuzzer.return_random_n_gen(5)
    candidate_obj = np.array(candidate_obj, dtype=float).round(8)
    candidate_var = np.array(candidate_var, dtype=float).round(8)

    manager = GaSimManager(debug=Cptool.config.DEBUG)

    results = []
    i = 0
    # 乱序
    rand_index = (np.arange(candidate_obj.shape[0]))
    np.random.shuffle(rand_index)
    candidate_obj = candidate_obj[rand_index]
    candidate_var = candidate_var[rand_index]

    for index, vars, value_vector in zip(np.arange(candidate_obj.shape[0]), candidate_var, candidate_obj):
        # if skip
        if os.path.exists(f'result/params.csv'):
            while not os.access(f"result/params.csv", os.R_OK):
                continue
            data = pd.read_csv(f'result/params.csv')
            exit_data = data.drop(['score', 'result'], axis=1, inplace=False)
        if ((exit_data - value_vector).sum(axis=1).abs() < 0.00001).sum() > 0:
            continue


        manager.start_multiple_sitl(device)
        manager.mav_monitor_init(device)

        if not manager.mav_monitor_connect():
            manager.stop_sitl()
            continue
        manager.mav_monitor_set_mission("Cptool/mission.txt", random=True)
        manager.mav_monitor_set_param(params=param, values=value_vector)

        print(f'======================={index} / {candidate_obj.shape[0]} ==========================')
        # manager.start_mav_monitor()
        manager.mav_monitor_start_mission()
        result = manager.mav_monitor_error()
        if result == 'skip':
            results.append(result)
        else:
            if not os.path.exists(f'result/params.csv'):
                while not os.access(f"result/params.csv", os.W_OK):
                    continue
                data = pd.DataFrame(columns=(Cptool.config.PARAM + ['score', 'result']))
            else:
                while not os.access(f"result/params.csv", os.W_OK):
                    continue
                tmp_row = value_vector.tolist()
                tmp_row.append(vars[0])
                tmp_row.append(result)

                # Write Row
                with open("result/params.csv", 'a+') as f:
                    csv_file = csv.writer(f)
                    csv_file.writerow(tmp_row)

        manager.stop_sitl()
        i += 1

    localtime = time.asctime(time.localtime(time.time()))
    # send_mail(Cptool.config.AIRSIM_PATH, localtime)