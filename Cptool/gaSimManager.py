"""
SimManager Version: 1.2
"""
import logging
import math
import multiprocessing
import os
import re
import sys
import time
from typing import Type

import numpy as np
import pexpect
from numpy.dual import norm
from pexpect import spawn
from pymavlink import mavextra, mavwp

from Cptool.gaMavlink import GaMavlinkAPM, DroneMavlink
from Cptool.config import toolConfig


class GaSimManager(object):

    def __init__(self, debug: bool = False):
        self._sim_task = None
        self._sitl_task = None
        self.sim_monitor = None
        self.mav_monitor = None
        self._even = None
        self.sim_msg_queue = multiprocessing.Queue()
        self.mav_msg_queue = multiprocessing.Queue()

        if debug:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.INFO)

    def start_sim(self):
        """
        启动AIRSIM_PATH目录下的Airsim模拟器
        :return:
        """
        # Airsim
        cmd = None
        if toolConfig.SIM == 'Airsim':
            cmd = f'gnome-terminal -- {toolConfig.AIRSIM_PATH} ' \
                  f'-ResX={toolConfig.HEIGHT} -ResY={toolConfig.WEIGHT} -windowed'
        if toolConfig.SIM == 'Jmavsim':
            cmd = f'gnome-terminal -- bash /home/rain/PX4-Autopilot/Tools/jmavsim_run.sh'
        if toolConfig.SIM == 'Morse':
            cmd = f'gnome-terminal -- morse run /home/rain/ardupilot/libraries/SITL/examples/Morse/quadcopter.py'
        if toolConfig.SIM == 'Gazebo':
            cmd = f'gnome-terminal -- gazebo --verbose worlds/iris_arducopter_runway.world'
        if cmd is None:
            raise ValueError('Not support mode')
        logging.info(f'Start Simulator {toolConfig.SIM}')
        self._sim_task = pexpect.spawn(cmd, cwd='/home/rain/')

    def start_sitl(self):
        """
        start the simulator
        :return:
        """
        if os.path.exists(f"{toolConfig.ARDUPILOT_LOG_PATH}/eeprom.bin") and toolConfig.MODE == "Ardupilot":
            os.remove(f"{toolConfig.ARDUPILOT_LOG_PATH}/eeprom.bin")
        if os.path.exists(f"{toolConfig.ARDUPILOT_LOG_PATH}/mav.parm") and toolConfig.MODE == "Ardupilot":
            os.remove(f"{toolConfig.ARDUPILOT_LOG_PATH}/mav.parm")
        if os.path.exists(f"{toolConfig.PX4_RUN_PATH}/build/px4_sitl_default/tmp/rootfs/eeprom/parameters_10016") \
                and toolConfig.MODE == "PX4":
            os.remove(f"{toolConfig.PX4_RUN_PATH}/build/px4_sitl_default/tmp/rootfs/eeprom/parameters_10016")

        cmd = None
        if toolConfig.MODE == 'Ardupilot':
            if toolConfig.SIM == 'Airsim':
                if toolConfig.HOME is not None:
                    cmd = f"python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py -v ArduCopter " \
                          f"--location={toolConfig.HOME}" \
                          f" -f airsim-copter --out=127.0.0.1:14550 --out=127.0.0.1:14540 " \
                          f" -S {toolConfig.SPEED}"
                else:
                    cmd = f"python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py -v ArduCopter -f airsim-copter " \
                          f"--out=127.0.0.1:14550 --out=127.0.0.1:14540 -S {toolConfig.SPEED}"
            if toolConfig.SIM == 'Morse':
                cmd = f"python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py -v ArduCopter --model morse-quad " \
                      f"--add-param-file=/home/rain/ardupilot/libraries/SITL/examples/Morse/quadcopter.parm  " \
                      f"--out=127.0.0.1:14550 -S {toolConfig.SPEED}"
            if toolConfig.SIM == 'Gazebo':
                cmd = f'python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py -f gazebo-iris -v ArduCopter ' \
                      f'--out=127.0.0.1:14550 -S {toolConfig.SPEED}'
            if toolConfig.SIM == 'SITL':
                if toolConfig.HOME is not None:
                    cmd = f"python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py --location={toolConfig.HOME} " \
                          f"--out=127.0.0.1:14550 --out=127.0.0.1:14540 -v ArduCopter -w -S {toolConfig.SPEED} "
                else:
                    cmd = f"python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py " \
                          f"--out=127.0.0.1:14550 --out=127.0.0.1:14540 -v ArduCopter -w -S {toolConfig.SPEED} "
            self._sitl_task = pexpect.spawn(cmd, cwd=toolConfig.ARDUPILOT_LOG_PATH, timeout=30, encoding='utf-8')

        if toolConfig.MODE == 'PX4':
            if toolConfig.HOME is None:
                pre_argv = f"PX4_HOME_LAT=-35.362758 " \
                           f"PX4_HOME_LON=149.165135 " \
                           f"PX4_HOME_ALT=583.730592 " \
                           f"PX4_SIM_SPEED_FACTOR={toolConfig.SPEED}"
            else:
                pre_argv = f"PX4_HOME_LAT=40.072842 " \
                           f"PX4_HOME_LON=-105.230575 " \
                           f"PX4_HOME_ALT=0.000000 " \
                           f"PX4_SIM_SPEED_FACTOR={toolConfig.SPEED}"

            if toolConfig.SIM == 'Airsim':
                cmd = f'make {pre_argv} px4_sitl_default none_iris'
            if toolConfig.SIM == 'Jmavsim':
                cmd = f'make {pre_argv} px4_sitl_default jmavsim'

            self._sitl_task = pexpect.spawn(cmd, cwd=toolConfig.PX4_RUN_PATH, timeout=30, encoding='utf-8')
        logging.info(f"Start {toolConfig.MODE} --> [{toolConfig.SIM}]")
        if cmd is None:
            raise ValueError('Not support mode or simulator')

    def start_multiple_sitl(self, drone_i=0):
        """
        start multiple simulators (not support PX4 now)
        :param drone_i:
        :return:
        """
        if os.path.exists(f"{toolConfig.ARDUPILOT_LOG_PATH}/eeprom.bin"):
            os.remove(f"{toolConfig.ARDUPILOT_LOG_PATH}/eeprom.bin")
        if os.path.exists(f"{toolConfig.ARDUPILOT_LOG_PATH}/mav.parm"):
            os.remove(f"{toolConfig.ARDUPILOT_LOG_PATH}/mav.parm")

        cmd = f"python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py --location=AVC_plane " \
                  f"--out=127.0.0.1:1455{drone_i} --out=127.0.0.1:1454{drone_i} " \
                  f"-v ArduCopter -w -S {toolConfig.SPEED} --instance {drone_i}"

        self._sitl_task = (pexpect.spawn(cmd, cwd=toolConfig.ARDUPILOT_LOG_PATH, timeout=30, encoding='utf-8'))
        logging.info(f"Start {toolConfig.MODE} --> [{toolConfig.SIM} - {drone_i}]")

    def mav_monitor_init(self, mavlink_class: Type[DroneMavlink] = DroneMavlink, drone_i=0):
        """
        初始化SITL在环
        :return:
        """
        self.mav_monitor = mavlink_class(14540 + int(drone_i),
                                         recv_msg_queue=self.sim_msg_queue,
                                         send_msg_queue=self.mav_msg_queue)
        self.mav_monitor.connect()
        if toolConfig.MODE == 'Ardupilot':
            if self.mav_monitor.ready2fly():
                return True
        elif toolConfig.MODE == 'PX4':
            while True:
                line = self._sitl_task.readline()
                if 'notify' in line:
                    # Disable the fail warning and return
                    self._sitl_task.send("param set NAV_RCL_ACT 0 \n")
                    time.sleep(0.1)
                    self._sitl_task.send("param set NAV_DLL_ACT 0 \n")
                    return True

    def mav_monitor_error(self):
        """
        monitor error during the flight
        :return:
        """
        logging.info(f'Start error monitor.')
        # Setting
        mission_time_out_th = 180
        result = 'pass'
        # Waypoint
        loader = mavwp.MAVWPLoader()
        loader.load('Cptool/fitCollection.txt')
        #
        lpoint1 = loader.wpoints[-1]
        lpoint1 = np.array([lpoint1.x, lpoint1.y])
        lpoint2 = loader.wpoints[2]
        lpoint2 = np.array([lpoint2.x, lpoint2.y])
        pre_location = loader.wpoints[0]
        # logger
        small_move_num = 0
        deviation_num = 0
        # Flag
        start_check = False

        start_time = time.time()
        while True:
            msg = self.mav_monitor.get_msg(["STATUSTEXT", "GLOBAL_POSITION_INT"])
            if msg is None:
                continue
            # System status message
            if msg.get_type() == "STATUSTEXT":
                line = msg.text
                if msg.severity == 6:
                    if "Disarming" in line:
                        # if successful landed, break the loop and return true
                        logging.info(f"Successful break the loop.")
                        return True
                    elif "Mission:" in line:
                        # Update Current mission
                        mission_task = re.findall('Mission: ([0-9]+) [WP|Land]', line)
                        if len(mission_task) > 0:
                            mission_task = int(mission_task[0])
                            lpoint1 = lpoint2
                            lpoint2 = loader.wpoints[mission_task]
                            lpoint2 = np.array([lpoint2.x, lpoint2.y])
                            # Switch Flag as mission before are not moving
                            if mission_task == 2:
                                start_check = True

                elif msg.severity == 2 or msg.severity == 0:
                    # Appear error, break loop and return false
                    if "SIM Hit ground at" in line:
                        result = 'crash'
                        break
                    elif "Potential Thrust Loss" in line:
                        result = 'Thrust Loss'
                        break
                    elif "Crash" in line:
                        result = 'crash'
                        break
                    elif "PreArm" in line:
                        result = 'PreArm Failed'
                        break
                # elif msg.severity == 2
            elif msg.get_type() == "GLOBAL_POSITION_INT":
                # Check deviation
                position_lat = msg.lat * 1.0e-7
                position_lon = msg.lon * 1.0e-7
                position = (position_lon, position_lat)
                # Calculate distance
                moving_dis = mavextra.distance_lat_lon(pre_location.x, pre_location.y,
                                                       position_lat, position_lon)
                # Update position
                pre_location.x = position_lat
                pre_location.y = position_lon

                if start_check:
                    # Is small move?
                    if moving_dis < 1:
                        small_move_num += 1
                    else:
                        small_move_num = 0

                    # Point2line distance
                    if (lpoint2 - lpoint1).sum() == 0:
                        deviation_dis = np.abs(np.cross(lpoint1 - lpoint2,
                                                        lpoint1 - position)) / norm(lpoint2 - lpoint1)
                    else:
                        deviation_dis = 0

                    # Is deviation ?
                    if deviation_dis > 10:
                        deviation_num += 1
                    else:
                        deviation_num = 0

                    # Threshold; Judgement
                    # Timeout
                    if small_move_num > 10:
                        time_index = True
                    # deviation
                    if deviation_num > 3:
                        result = 'deviation'
                        break

            # Timeout Check if stack at one point
            mid_point_time = time.time()
            if (mid_point_time - start_time) > mission_time_out_th:
                result = 'timeout'
                break

        logging.info(f"Monitor result: {result}")
        return result

    def mav_monitor_connect(self):
        """
        mavlink connect
        :return:
        """
        return self.mav_monitor.connect()

    def mav_monitor_set_mission(self, mission_file, random: bool = False):
        """
        set mission
        :param mission_file: file path
        :param random:
        :return:
        """
        return self.mav_monitor.set_mission(mission_file, random)

    def mav_monitor_set_param(self, params, values):
        """
        set drone configuration
        :return:
        """
        for param, value in zip(params, values):
            self.mav_monitor.set_param(param, value)

    def mav_monitor_get_param(self, param):
        """
        get drone configuration
        :return:
        """
        return self.mav_monitor.get_param(param)

    def mav_monitor_start_mission(self):
        """
        start mission
        :return:
        """
        self.mav_monitor.start_mission()

    def start_mav_monitor(self):
        """
        start monitor
        :return:
        """
        self.mav_monitor.start()

    def sim_close_msg(self):
        pass

    def stop_sitl(self):
        """
        stop the simulator
        :return:
        """
        self._sitl_task.sendcontrol('c')
        while True:
            line = self._sitl_task.readline()
            if not line:
                break
        self._sitl_task.close(force=True)
        logging.info('Stop SITL task.')
        self.sim_close_msg()
        logging.debug('Send mavclosed to Airsim.')

    def get_mav_monitor(self):
        return self.mav_monitor

    def sitl_task(self) -> spawn:
        return self._sitl_task