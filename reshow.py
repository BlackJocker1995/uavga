from config.ardupilot.ArdupilotConfig import ArdupilotConfig
from uavga.fuzzing import ArdupilotFuzzing

if __name__ == '__main__':
    a = [
    'PSC_POSXY_P',
    'PSC_VELXY_P',
    'PSC_POSZ_P',
    'ATC_ANG_RLL_P',
    'ATC_RAT_RLL_I',
    'ATC_RAT_RLL_D',
    'ATC_RAT_RLL_P',
    'ATC_ANG_PIT_P',
    'ATC_RAT_PIT_P',
    'ATC_RAT_PIT_I',
    'ATC_RAT_PIT_D',
    'ATC_ANG_YAW_P',
    'ATC_RAT_YAW_P',
    'ATC_RAT_YAW_I',
    'ATC_RAT_YAW_D',

    # 'INS_POS1_Z',
    # 'INS_POS2_Z',
    # 'INS_POS3_Z',

    'WPNAV_SPEED',
    'WPNAV_SPEED_UP',
    'WPNAV_SPEED_DN',
    'WPNAV_ACCEL'
    'ANGLE_MAX',
    ]
    v = [
        0.58,    2.4,    1.99,    6.8 ,   0.3 ,   0.048,    0.225 ,   0.09 ,   0.155 ,   0.77 ,   0.048 ,   3.49 ,   0.885 ,   0.77 ,0.001  ,  450 ,   500 ,   440   , 450  ,  5900

    ]
    ArdupilotFuzzing.re_show(ArdupilotConfig, v, speed=5)