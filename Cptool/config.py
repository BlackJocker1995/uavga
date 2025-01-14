# coding:utf-8
import json
import time
import yaml
import os

import pandas as pd


class ToolConfig:
    class ConstError(PermissionError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __init__(self):
        # Load YAML config with fallback to defaults
        self.yaml_config = self._load_yaml_config()
        self._init_defaults()

    def _load_yaml_config(self):
        """Load YAML config with fallback to empty dict"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Warning: Could not load config.yaml ({str(e)}), using defaults")
            return {}

    def _get_yaml_value(self, *keys, default=None):
        """Safely get nested YAML config value with fallback"""
        config = self.yaml_config
        for key in keys:
            if not isinstance(config, dict):
                return default
            config = config.get(key, default)
        return config

    def _init_defaults(self):
        """Initialize with YAML values or defaults"""
        self.__dict__["MODE"] = self._get_yaml_value('mode', default=None)
        self.__dict__["SPEED"] = self._get_yaml_value('simulation', 'speed', default=3)
        self.__dict__["HOME"] = self._get_yaml_value('simulation', 'home', default="AVC_plane")
        self.__dict__["DEBUG"] = self._get_yaml_value('debug', default=True)
        self.__dict__["WIND_RANGE"] = self._get_yaml_value('wind_range', default=[8, 10.7])
        self.__dict__["HEIGHT"] = self._get_yaml_value('window', 'height', default=640)
        self.__dict__["WEIGHT"] = self._get_yaml_value('window', 'weight', default=480)
        self.__dict__["LIMIT_H"] = self._get_yaml_value('flight', 'limit_h', default=50)
        self.__dict__["LIMIT_L"] = self._get_yaml_value('flight', 'limit_l', default=40)
        self.__dict__["ARDUPILOT_LOG_PATH"] = self._get_yaml_value('paths', 'ardupilot_log_path', default='/media/rain/data')
        self.__dict__["SITL_PATH"] = self._get_yaml_value('paths', 'sitl_path', default="/home/rain/ardupilot/Tools/autotest/sim_vehicle.py")
        self.__dict__["AIRSIM_PATH"] = self._get_yaml_value('paths', 'airsim_path', default="/media/rain/data/airsim/Africa_Savannah/LinuxNoEditor/Africa_001.sh")
        self.__dict__["PX4_RUN_PATH"] = self._get_yaml_value('paths', 'px4_run_path', default='/home/rain/PX4-Autopilot')
        self.__dict__["JMAVSIM_PATH"] = self._get_yaml_value('paths', 'jmavsim_path', default="/home/rain/PX4-Autopilot/Tools/jmavsim_run.sh")
        self.__dict__["MORSE_PATH"] = self._get_yaml_value('paths', 'morse_path', default="/home/rain/ardupilot/libraries/SITL/examples/Morse/quadcopter.py")
        self.__dict__["CLUSTER_CHOICE_NUM"] = self._get_yaml_value('cluster_choice_num', default=10)

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
                "ATC_RAT_RLL_D",
                "ATC_ANG_PIT_P",
                "ATC_RAT_PIT_P",
                "ATC_RAT_PIT_I",
                "ATC_RAT_PIT_D",
                "ATC_ANG_YAW_P",
                "ATC_RAT_YAW_P",
                "ATC_RAT_YAW_I",
                # "ATC_RAT_YAW_D",
                # "WPNAV_SPEED",
                # "WPNAV_ACCEL",
                # "ANGLE_MAX"
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

    def get(self, key, default=None):
        """Safe config getter with default value"""
        return self.__dict__.get(key, default)

    def validate_config(self):
        """Validate critical configuration values"""
        required = ['MODE', 'SITL_PATH', 'PARAM']
        for key in required:
            if not self.__dict__.get(key):
                raise ValueError(f"Missing required config: {key}")

        if self.__dict__["MODE"] not in ["Ardupilot", "PX4"]:
            raise ValueError("Invalid MODE - must be 'Ardupilot' or 'PX4'")

        # Validate paths exist
        paths = ['SITL_PATH', 'PX4_RUN_PATH', 'ARDUPILOT_LOG_PATH']
        for path_key in paths:
            path = self.__dict__.get(path_key)
            if path and not os.path.exists(path):
                print(f"Warning: Path does not exist: {path_key}={path}")


toolConfig = ToolConfig()
toolConfig.select_mode("ArduPilot")