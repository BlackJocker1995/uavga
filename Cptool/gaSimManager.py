"""
SimManager Version: 4.0 22-10-24
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
from pymavlink import mavextra, mavwp, mavutil

from Cptool.gaMavlink import GaMavlinkAPM, DroneMavlink
from Cptool.config import toolConfig
from Cptool.mavtool import Location


class SimManager(object):

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

    """
    Base Function
    """
    def start_sim(self):
        """
        start simulator
        :return:
        """
        # Airsim
        cmd = None
        if toolConfig.SIM == 'Airsim':
            cmd = f'gnome-terminal -- {toolConfig.AIRSIM_PATH} ' \
                  f'-ResX={toolConfig.HEIGHT} -ResY={toolConfig.WEIGHT} -windowed'
        if toolConfig.SIM == 'Jmavsim':
            cmd = f'gnome-terminal -- bash {toolConfig.JMAVSIM_PATH}'
        if toolConfig.SIM == 'Morse':
            cmd = f'gnome-terminal -- morse run {toolConfig.MORSE_PATH}'
        if toolConfig.SIM == 'Gazebo':
            cmd = f'gnome-terminal -- gazebo --verbose worlds/iris_arducopter_runway.world'
        if cmd is None:
            raise ValueError('Not support mode')
        logging.info(f'Start Simulator {toolConfig.SIM}')
        self._sim_task = pexpect.spawn(cmd)

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
                    cmd = f"python3 {toolConfig.SITL_PATH} -v ArduCopter " \
                          f"--location={toolConfig.HOME}" \
                          f" -f airsim-copter --out=127.0.0.1:14550 --out=127.0.0.1:14540 " \
                          f" -S {toolConfig.SPEED}"
                else:
                    cmd = f"python3 {toolConfig.SITL_PATH} -v ArduCopter -f airsim-copter " \
                          f"--out=127.0.0.1:14550 --out=127.0.0.1:14540 -S {toolConfig.SPEED}"
            if toolConfig.SIM == 'Morse':
                cmd = f"python3 {toolConfig.SITL_PATH}  -v ArduCopter --model morse-quad " \
                      f"--add-param-file=/home/rain/ardupilot/libraries/SITL/examples/Morse/quadcopter.parm  " \
                      f"--out=127.0.0.1:14550 -S {toolConfig.SPEED}"
            if toolConfig.SIM == 'Gazebo':
                cmd = f'python3 {toolConfig.SITL_PATH} -f gazebo-iris -v ArduCopter ' \
                      f'--out=127.0.0.1:14550 -S {toolConfig.SPEED}'
            if toolConfig.SIM == 'SITL':
                if toolConfig.HOME is not None:
                    cmd = f"python3 {toolConfig.SITL_PATH}  --location={toolConfig.HOME} " \
                          f"--out=127.0.0.1:14550 --out=127.0.0.1:14540 -v ArduCopter -w -S {toolConfig.SPEED} "
                else:
                    cmd = f"python3 {toolConfig.SITL_PATH}  " \
                          f"--out=127.0.0.1:14550 --out=127.0.0.1:14540 -v ArduCopter -w -S {toolConfig.SPEED} "
            self._sitl_task = pexpect.spawn(cmd, cwd=toolConfig.ARDUPILOT_LOG_PATH, timeout=30, encoding='utf-8')

        if toolConfig.MODE == 'PX4':
            if toolConfig.HOME is None:
                pre_argv = f"HEADLESS=1 " \
                           f"PX4_HOME_LAT=-35.362758 " \
                           f"PX4_HOME_LON=149.165135 " \
                           f"PX4_HOME_ALT=583.730592 " \
                           f"PX4_SIM_SPEED_FACTOR={toolConfig.SPEED}"
            else:
                pre_argv = f"HEADLESS=1 " \
                           f"PX4_HOME_LAT=40.072842 " \
                           f"PX4_HOME_LON=-105.230575 " \
                           f"PX4_HOME_ALT=0.000000 " \
                           f"PX4_SIM_SPEED_FACTOR={toolConfig.SPEED}"

            if toolConfig.SIM == 'Airsim':
                cmd = f'make {pre_argv} px4_sitl none_iris'
            if toolConfig.SIM == 'Jmavsim':
                cmd = f'make {pre_argv} px4_sitl jmavsim'

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
        if os.path.exists(f"{toolConfig.PX4_RUN_PATH}/build/px4_sitl_default/tmp/rootfs/eeprom/parameters_10016") \
                and toolConfig.MODE == "PX4":
            os.remove(f"{toolConfig.PX4_RUN_PATH}/build/px4_sitl_default/tmp/rootfs/eeprom/parameters_10016")

        if toolConfig.HOME is not None:
            cmd = f"python3 {toolConfig.SITL_PATH} --location={toolConfig.HOME} " \
                  f"--out=127.0.0.1:1455{drone_i} --out=127.0.0.1:1454{drone_i} " \
                  f"-v ArduCopter -w -S {toolConfig.SPEED} --instance {drone_i}"
        else:
            cmd = f"python3 {toolConfig.SITL_PATH} " \
                  f"--out=127.0.0.1:1455{drone_i} --out=127.0.0.1:1454{drone_i} " \
                  f"-v ArduCopter -w -S {toolConfig.SPEED} --instance {drone_i}"

        self._sitl_task = (pexpect.spawn(cmd, cwd=toolConfig.ARDUPILOT_LOG_PATH, timeout=30, encoding='utf-8'))
        logging.info(f"Start {toolConfig.MODE} --> [{toolConfig.SIM} - {drone_i}]")

    def mav_monitor_init(self, mavlink_class: Type[DroneMavlink] = DroneMavlink, drone_i=0):
        """
        initial SITL monitor
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
                    time.sleep(0.1)
                    # Enable detector
                    self._sitl_task.send("param set CBRK_FLIGHTTERM 0 \n")
                    return True

    def sim_monitor_init(self, simulator_class):
        """
        init airsim monitor
        :return:
        """
        self.sim_monitor = simulator_class(recv_msg_queue=self.mav_msg_queue, send_msg_queue=self.sim_msg_queue)
        time.sleep(3)

    def start_mav_monitor(self):
        """
        start monitor
        :return:
        """
        self.mav_monitor.start()

    def start_sim_monitor(self):
        """
        启动Airsim监控进程
        :return:
        """
        self.sim_monitor.start()

    """
    Mavlink Operation
    """

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
        logging.debug('Send mavclosed to Airsim.')

    """
    Other get/set
    """
    def get_mav_monitor(self):
        return self.mav_monitor

    def sitl_task(self) -> spawn:
        return self._sitl_task


class GaSimManager(SimManager):
    def __init__(self, debug: bool = False):
        super(GaSimManager, self).__init__(debug)

    """
    Advanced Function
    """
    def mav_monitor_error(self):
        """
        monitor error during the flight
        :return:
        """
        logging.info(f'Start error monitor.')
        # Setting
        mission_time_out_th = 240
        result = 'pass'
        # Waypoint
        loader = mavwp.MAVWPLoader()
        if toolConfig.MODE == "PX4":
            loader.load('Cptool/fitCollection_px4.txt')
        else:
            loader.load('Cptool/fitCollection.txt')
        #
        lpoint1 = Location(loader.wpoints[0])
        lpoint2 = Location(loader.wpoints[1])
        pre_location = Location(loader.wpoints[0])
        # logger
        small_move_num = 0
        deviation_num = 0
        low_lat_num = 0
        # Flag
        start_check = False
        current_mission = 0
        pre_alt = 0
        last_time = 0

        start_time = time.time()
        while True:
            # time.sleep(0.1)
            if toolConfig.MODE == "PX4":
                self.mav_monitor.gcs_msg_request()
            status_message = self.mav_monitor.get_msg(["STATUSTEXT"])
            position_msg = self.mav_monitor.get_msg(["GLOBAL_POSITION_INT", "MISSION_CURRENT"])

            # System status message
            if status_message is not None and status_message.get_type() == "STATUSTEXT":
                line = status_message.text
                # print(status_message)
                if status_message.severity == 6:
                    if "Disarming" in line or "landed" in line or "Landing" in line:
                        # if successful landed, break the loop and return true
                        logging.info(f"Successful break the loop.")
                        break
                    if "preflight disarming" in line:
                        result = 'PreArm Failed'
                        break
                elif status_message.severity == 2 or status_message.severity == 0:
                    # Appear error, break loop and return false
                    if "SIM Hit ground" in line:
                        result = 'crash'
                        break
                    elif "Potential Thrust Loss" in line:
                        result = 'Thrust Loss'
                        break
                    elif "Crash" in line \
                            or "Failsafe enabled: no global position" in line \
                            or "failure detected" in line:
                        result = 'crash'
                        break
                    elif "PreArm" in line or "speed has been constrained by max speed" in line:
                        result = 'PreArm Failed'
                        break

            if position_msg is not None and position_msg.get_type() == "MISSION_CURRENT":
                # print(position_msg)
                if int(position_msg.seq) != current_mission and int(position_msg.seq) != 6:
                    logging.debug(f"Mission change {current_mission} -> {position_msg.seq}")
                    lpoint1 = Location(loader.wpoints[current_mission])
                    lpoint2 = Location(loader.wpoints[position_msg.seq])
                    # Start Check
                    if int(position_msg.seq) == 1:
                        start_check = True
                    current_mission = int(position_msg.seq)
                    if toolConfig.MODE == "PX4" and int(position_msg.seq) == 5:
                        start_check = False
            elif position_msg is not None and position_msg.get_type() == "GLOBAL_POSITION_INT":
                # print(position_msg)
                # Check deviation
                position_lat = position_msg.lat * 1.0e-7
                position_lon = position_msg.lon * 1.0e-7
                alt = position_msg.relative_alt / 1000
                time_usec = position_msg.time_boot_ms * 1e-6
                position = Location(position_lat, position_lon, time_usec)

                # Calculate distance
                moving_dis = Location.distance(pre_location, position)
                time_step = position.timeS - pre_location.timeS
                alt_change = abs(pre_alt- alt)
                # Update position
                pre_location.x = position_lat
                pre_location.y = position_lon
                pre_alt = alt

                if start_check:
                    if alt < 1:
                        low_lat_num += 1
                    else:
                        small_move_num = 0

                    velocity = moving_dis / time_step
                    # logging.debug(f"Velocity {velocity}.")
                    # Is small move?
                    # logging.debug(f"alt_change {alt_change}.")
                    if velocity < 1 and alt_change < 0.1 and small_move_num != 0:
                        logging.debug(f"Small moving {small_move_num}, num++, num now - {small_move_num}.")
                        small_move_num += 1
                    else:
                        small_move_num = 0

                    # Point2line distance
                    a = Location.distance(position, lpoint1)
                    b = Location.distance(position, lpoint2)
                    c = Location.distance(lpoint1, lpoint2)

                    if c != 0:
                        p = (a + b + c) / 2
                        deviation_dis = 2 * math.sqrt(p * (p - a) * (p - b) * (p - c) + 0.01) / c
                    else:
                        deviation_dis = 0
                    # Is deviation ?
                    # logging.debug(f"Point2line distance {deviation_dis}.")
                    if deviation_dis > 10:
                        # logging.debug(f"Deviation {deviation_dis}, num++, num now - {deviation_num}.")
                        deviation_num += 1
                    else:
                        deviation_num = 0

                    # deviation
                    if deviation_num > 3:
                        result = 'deviation'
                        break
                    # Threshold; Judgement
                    # Timeout
                    if small_move_num > 10:
                        result = 'timeout'
                        break
                # ============================ #

            # Timeout Check if stack at one point
            mid_point_time = time.time()
            last_time = mid_point_time
            if (mid_point_time - start_time) > mission_time_out_th:
                result = 'timeout'
                break

        logging.info(f"Monitor result: {result}")
        return result
