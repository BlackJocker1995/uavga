# SITL Type PX4 and Ardupilot
# {'PX4','Ardupilot'}
MODE = 'Ardupilot'
# Simulation Type
# Ardupilot : ['Airsim', 'Morse', 'Gazebo', 'SITL']
# PX4 : ['Jmavsim']
SIM = 'SITL'
# Simulation Speed
SPEED = 10
# Output Debug Message
DEBUG = True

# Wind Speed range
WIND_RANGE = [8, 10.7]

# GUI Windows size
HEIGHT = 640
WEIGHT = 480

LIMIT_H = 50
LIMIT_L = 40

# Copter LOG Path
ARDUPILOT_LOG_PATH = '/media/rain/data'

# PX4 LOG Path
PX4_LOG_PATH = '/home/rain/PX4-Autopilot'

# Mavlink Part
LOG_MAP = ['IMU', 'ATT', 'RATE', 'PARM']
# LOG_MAP = ['ATT', 'RATE']

PARAM = [
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


