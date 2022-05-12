# coding:utf-8
import sys
import Cptool.config


class ToolConfig:
    class ConstError(PermissionError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("can't change const %s" % name)
        if not name.isupper():
            raise self.ConstCaseError('const name "%s" is not all uppercase' % name)
        self.__dict__[name] = value


toolConfig = ToolConfig()
# SITL Type PX4 and Ardupilot
# {'PX4','Ardupilot'}
toolConfig.MODE = 'Ardupilot'
# Simulation Type
# Ardupilot : ['Airsim', 'Morse', 'Gazebo', 'SITL']
# PX4 : ['Jmavsim']
toolConfig.SIM = 'SITL'
# Simulation Speed
toolConfig.SPEED = 10
# Output Debug Message
toolConfig.DEBUG = True
# Wind Speed range
toolConfig.WIND_RANGE = [8, 10.7]
# GUI Windows size
toolConfig.HEIGHT = 640
toolConfig.WEIGHT = 480
# Mission flight attitude range
toolConfig.LIMIT_H = 50
toolConfig.LIMIT_L = 40
# Copter LOG Path
toolConfig.ARDUPILOT_LOG_PATH = '/media/rain/data'
# PX4 LOG Path
toolConfig.PX4_LOG_PATH = '/home/rain/PX4-Autopilot'
# Mavlink Part
toolConfig.LOG_MAP = ['IMU', 'ATT', 'RATE', 'PARM']
# LOG_MAP = ['ATT', 'RATE']
toolConfig.PARAM = [
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
# LOG_MAP = ['ATT', 'RATE']
toolConfig.INPUT_LEN = 3
