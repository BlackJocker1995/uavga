# coding:utf-8
import json
import time

import pandas as pd


class ToolConfig:
    class ConstError(PermissionError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __init__(self):
        # Mode
        # {'PX4','Ardupilot'}
        self.__dict__["MODE"] = None

        # Simulation Speed
        self.__dict__["SPEED"] = 3
        # Flight home (None, AVC_plane)
        self.__dict__["HOME"] = "AVC_plane"  # "AVC_plane"
        # Output Debug Message
        self.__dict__["DEBUG"] = True
        # Wind Speed range
        self.__dict__["WIND_RANGE"] = [8, 10.7]
        # Airsim Windows size
        self.__dict__["HEIGHT"] = 640
        self.__dict__["WEIGHT"] = 480
        # Mission flight attitude range
        self.__dict__["LIMIT_H"] = 50
        self.__dict__["LIMIT_L"] = 40
        # Copter LOG Path
        self.__dict__["ARDUPILOT_LOG_PATH"] = '/media/rain/data'

        #--------PATH ---------#
        # STIL
        self.__dict__["SITL_PATH"] = "/home/rain/ardupilot/Tools/autotest/sim_vehicle.py"
        # Airsim
        self.__dict__["AIRSIM_PATH"] = "/media/rain/data/airsim/Africa_Savannah/LinuxNoEditor/Africa_001.sh"
        # self.__dict__["AIRSIM_PATH"] = "/media/rain/data/airsim/Blocks/LinuxNoEditor/Blocks.sh"
        # PX4 LOG Path
        self.__dict__["PX4_RUN_PATH"] = '/home/rain/PX4-Autopilot'
        # Jmavsim Path
        self.__dict__["JMAVSIM_PATH"] = "/home/rain/PX4-Autopilot/Tools/jmavsim_run.sh"
        # Morse path
        self.__dict__["MORSE_PATH"] = "/home/rain/ardupilot/libraries/SITL/examples/Morse/quadcopter.py"

        self.__dict__["CLUSTER_CHOICE_NUM"] = 10

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("can't change const %s" % name)
        if not name.isupper():
            raise self.ConstCaseError('const name "%s" is not all uppercase' % name)
        self.__dict__[name] = value

    def __getattr__(self, item):
        if self.__dict__["MODE"] is None:
            raise ValueError("Set config Mode at first!")
        return self.__dict__[item]

    def select_mode(self, mode):
        if mode not in ["Ardupilot", "PX4"]:
            raise ValueError("Bad mode")
        # Change Mode
        self.__dict__["MODE"] = mode

        if mode == "Ardupilot":
            # Simulation Type
            # Ardupilot : ['Airsim', 'Morse', 'Gazebo', 'SITL']
            self.__dict__["SIM"] = "SITL"  # "Jmavsim"

            # Mavlink Part
            self.__dict__["LOG_MAP"] = ['IMU', 'ATT', 'RATE', 'PARM', 'VIBE', "MAG"]  # "POS"
            # Online Mavlink Part
            self.__dict__["OL_LOG_MAP"] = ['ATTITUDE', 'RAW_IMU', 'VIBRATION']  # 'GLOBAL_POSITION_INT'
            # Status Order
            self.__dict__["STATUS_ORDER"] = ['TimeS', 'Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw',
                                             # 'Lat', 'Lng', 'Alt',
                                             'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ',
                                             'MagX', 'MagY', 'MagZ', 'VibeX', 'VibeY', 'VibeZ']

            with open('Cptool/param_ardu.json', 'r') as f:
                param_name = pd.DataFrame(json.loads(f.read())).columns.tolist()
            self.__dict__["PARAM"] = param_name

            self.__dict__["PARAM_PART"] = [
                "PSC_VELXY_P",
                "PSC_VELXY_I",
                "PSC_VELXY_D",
                "PSC_ACCZ_P",
                "PSC_ACCZ_I",
                "ATC_ANG_RLL_P",
                "ATC_RAT_RLL_P",
                "ATC_RAT_RLL_I",
                # "ATC_RAT_RLL_D",
                # "ATC_ANG_PIT_P",
                # "ATC_RAT_PIT_P",
                # "ATC_RAT_PIT_I",
                "ATC_RAT_PIT_D",
                "ATC_ANG_YAW_P",
                "ATC_RAT_YAW_P",
                "ATC_RAT_YAW_I",
                "ATC_RAT_YAW_D",
                "WPNAV_SPEED",
                "WPNAV_ACCEL",
                "ANGLE_MAX"
            ]

            # self.__dict__["PARAM"] = [
            #     "PSC_VELXY_P",
            #     "INS_POS1_Z",
            #     "INS_POS2_Z",
            #     "INS_POS3_Z",
            #     "WPNAV_SPEED",
            #     "ANGLE_MAX"
            # ]
        elif mode == "PX4":
            # PX4 : ['Jmavsim']
            self.__dict__["SIM"] = "Jmavsim"  # "Jmavsim"

            now = time.localtime()
            now_time = time.strftime("%Y-%m-%d", now)
            # File path
            self.__dict__["PX4_LOG_PATH"] = f"{self.__dict__['PX4_RUN_PATH']}/build/px4_sitl_default/logs/{now_time}"
            # Status Order
            self.__dict__["STATUS_ORDER"] = ['TimeS', 'Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw',
                                             'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ',
                                             'MagX', 'MagY', 'MagZ', 'VibeX', 'VibeY', 'VibeZ']

            with open('Cptool/param_px4.json', 'r') as f:
                param_name = pd.DataFrame(json.loads(f.read())).columns.tolist()
            self.__dict__["PARAM"] = param_name

            # TODO: px4 data
            self.__dict__["PARAM_PART"] = [
                "MC_ROLL_P",
                "MC_PITCH_P",
                "MC_YAW_P",
                "MC_YAW_WEIGHT",
                "MPC_XY_P",
                "MPC_Z_P",
                "MC_PITCHRATE_P",
                "MC_ROLLRATE_P",
                "MC_YAWRATE_P",
                "MPC_TILTMAX_AIR",
                "MIS_YAW_ERR",
                "MPC_Z_VEL_MAX_DN",
                "MPC_Z_VEL_MAX_UP",
                "MPC_TKO_SPEED"
            ]

        if len(self.__dict__["PARAM_PART"]) == len(self.__dict__["PARAM"]):
            self.__dict__["EXE"] = ""
        else:
            self.__dict__["EXE"] = len(self.__dict__["PARAM_PART"])

        ######################
        # Model Config       #
        ######################
        # Status length
        self.__dict__["STATUS_LEN"] = len(self.__dict__["STATUS_ORDER"]) - 1

        # Parameter length
        self.__dict__["PARAM_LEN"] = len(self.__dict__["PARAM"])

        # Predictor input vector length
        self.__dict__["INPUT_LEN"] = 4
        # Predictor output vector length
        self.__dict__["OUTPUT_LEN"] = 1

        # input data entry length
        self.__dict__["DATA_LEN"] = self.__dict__["STATUS_LEN"] + len(toolConfig.PARAM)

        # Whole predictor input length
        self.__dict__["INPUT_DATA_LEN"] = self.__dict__["DATA_LEN"] * self.__dict__["INPUT_LEN"]

        # Whole predictor output length
        self.__dict__["OUTPUT_DATA_LEN"] = self.__dict__["STATUS_LEN"] * self.__dict__["OUTPUT_LEN"]

        # Vector length of a segment
        self.__dict__["SEGMENT_LEN"] = 10 + self.__dict__["INPUT_LEN"]

        # transform values
        self.__dict__["RETRANS"] = True


toolConfig = ToolConfig()
toolConfig.select_mode("Ardupilot")

# toolConfig.PARAM = [
#     "PSC_POSXY_P",
#     "PSC_VELXY_P",
#     "PSC_POSZ_P",
#     "ATC_ANG_RLL_P",
#     "ATC_ANG_PIT_P",
#     "ATC_ANG_YAW_P",
#     "ATC_RAT_RLL_I",
#     "ATC_RAT_RLL_D",
#     "ATC_RAT_RLL_P",
#     "ATC_RAT_PIT_P",
#     "ATC_RAT_PIT_I",
#     "ATC_RAT_PIT_D",
#     "ATC_RAT_YAW_P",
#     "ATC_RAT_YAW_I",
#     "ATC_RAT_YAW_D",
#     "WPNAV_SPEED",
#     "WPNAV_SPEED_UP",
#     "WPNAV_SPEED_DN",
#     "WPNAV_ACCEL",
#     "ANGLE_MAX",
# ]
