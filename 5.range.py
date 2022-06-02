from range.range import find_range
import pandas as pd

from range.rangegen import ANAGA

if __name__ == '__main__':
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

    result_data = pd.read_csv('result/params_test1.csv', header=0)
    ga = ANAGA(param, result_data)
    ga.run()