import logging
import math
import multiprocessing
import os
import re
import sys
import threading
import time

import eventlet
import pexpect
from pexpect import spawn
from pymavlink import mavextra, mavwp

from Cptool.gaMavlink import GaMavlink
from Cptool.config import toolConfig

class GaSimManager(object):

    def __init__(self, debug: bool = False):
        self._sitl_task = None
        self._mav_monitor: GaMavlink = None
        self._even = None
        self.msg_queue = multiprocessing.Queue()


        if debug:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.INFO)

    def start_sitl(self):
        """
        start the simulator
        :return:
        """
        if os.path.exists(f"{toolConfig.ARDUPILOT_LOG_PATH}/eeprom.bin"):
            os.remove(f"{toolConfig.ARDUPILOT_LOG_PATH}/eeprom.bin")
        if os.path.exists(f"{toolConfig.ARDUPILOT_LOG_PATH}/mav.parm"):
            os.remove(f"{toolConfig.ARDUPILOT_LOG_PATH}/mav.parm")

        cmd = None
        if toolConfig.MODE == 'Ardupilot':
            if toolConfig.SIM == 'SITL':

                cmd = f"python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py --location=AVC_plane " \
                      f"--out=127.0.0.1:14550 --out=127.0.0.1:14540 -v ArduCopter -w -S {toolConfig.SPEED} "
            if toolConfig.SIM == 'Airsim':
                cmd = f"python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py -v ArduCopter -f airsim-copter " \
                      f"--out=127.0.0.1:14550 --out=127.0.0.1:14540 -S {toolConfig.SPEED}"

            self._sitl_task = pexpect.spawn(cmd, cwd=toolConfig.ARDUPILOT_LOG_PATH, timeout=30, encoding='utf-8')
        logging.info(f"Start {toolConfig.MODE} --> [{toolConfig.SIM}]")
        if cmd is None:
            raise ValueError('Not support mode or simulator')

    def start_multiple_sitl(self, drone_i=0):
        """
        start multiple simulators
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

    def mav_monitor_init(self, drone_i=0):
        """
        init the SITL simulator
        :return:
        """
        if toolConfig.MODE == 'Ardupilot':
            while True:
                line = self._sitl_task.readline()
                if line.startswith('APM: EKF2 IMU0 is using GPS'):
                    break
                # if line.startswith('APM: GPS 1: detected as'):
                #     break
        elif toolConfig.MODE == 'PX4':
            while True:
                line = self._sitl_task.readline()
                if 'notify negative' in line:
                    break
        self._mav_monitor = GaMavlink(14540 + int(drone_i), self.msg_queue)

    def mav_monitor_error(self):
        """
        monitor error during the flight
        :return:
        """
        logging.info(f'Start error monitor.')

        time_out = 180
        loader = mavwp.MAVWPLoader()
        loader.load('Cptool/mission.txt')
        result = 'pass'
        line_point1 = loader.wpoints[-1]
        line_point2 = loader.wpoints[2]
        pre_location = loader.wpoints[0]
        th = threading.Thread(target=threading_send, args=(self._sitl_task,))
        th.start()
        index = 0
        dis_index = 0
        more_dis = 0
        time_index = False
        pre_dis = 0

        start_time = time.time()
        while True:
            msg = self._mav_monitor.get_msg(['STATUSTEXT'])
            if msg is not None:
                msg = msg.to_dict()
                if msg['severity'] == 0:
                    line = msg['text']
                    if line.startswith('Crash: Disarming'):
                        result = 'crash'
                        break
                    if line.startswith('Potential Thrust Loss'):
                        result = 'Thrust Loss'
                        break
                if msg['severity'] == 2:
                    line = msg['text']
                    if line.startswith('PreArm: Check ACRO_BAL_ROLL/PITCH'):
                        result = 'PreArm Failed'
                        break

            mid_point_time = time.time()
            if (mid_point_time - start_time) > time_out:
                if time_index == True:
                    result = 'pass'
                    break
                else:
                    result = 'timeout'
                    break
            try:
                line = self._sitl_task.readline()
            except TimeoutError:
                result = 'TimeoutError'
                break
            except Exception as e:
                logging.debug(f'{line} -- {e}')
                sys.exit(1)
            if line == 'DISARMED\r\n':
                logging.debug('DISARMED')
                break
            mission_task = re.findall('APM: Mission: ([0-9]+) [WP|Land]', line)
            if len(mission_task) > 0:
                if int(mission_task[0]) != 2:
                    mission_task = int(mission_task[0])
                    line_point1 = line_point2
                    line_point2 = loader.wpoints[mission_task]

            candicate = re.findall('lat : ([0-9]+), lon : ([-0-9]+)', line)
            if len(candicate) == 1:
                position = candicate[0]
                real_x = int(position[0]) * 1.0e-7
                real_y = int(position[1]) * 1.0e-7

                moving_dis = mavextra.distance_lat_lon(pre_location.x, pre_location.y, real_x, real_y)
                pre_location.x = real_x
                pre_location.y = real_y
                # long time
                if moving_dis < 1:
                    index += 1
                    if index > 100:
                        time_index = True

                a = mavextra.distance_lat_lon(real_x, real_y, line_point1.x, line_point1.y)
                b = mavextra.distance_lat_lon(real_x, real_y, line_point2.x, line_point2.y)
                c = mavextra.distance_lat_lon(line_point1.x, line_point1.y, line_point2.x, line_point2.y)

                if c != 0:

                    p = (a + b + c) / 2

                    dis = 2 * math.sqrt(p * (p - a) * (p - b) * (p - c) + 0.01) / c

                    # dis = a + b - c
                    if dis > 10:
                        more_dis += 1
                        if more_dis > 3:
                            result = 'deviation'
                            # print(f'dis  {dis}')
                            break
                    if dis > 0.5 and dis > pre_dis:
                        dis_index += 1
                        if dis_index > 5:
                            result = 'deviation'
                            # print(f'dis  {dis}')
                            break
                        pre_dis = dis
                    else:
                        dis_index = 0

        th.do_run = False
        th.join()
        logging.info(result)
        return result

    def mav_monitor_connect(self):
        """
        mavlink connect
        :return:
        """
        return self._mav_monitor.connect()

    def mav_monitor_set_mission(self, mission_file, random: bool = False):
        """
        set mission
        :param mission_file: file path
        :param random:
        :return:
        """
        return self._mav_monitor.set_mission(mission_file, random)

    def mav_monitor_set_param(self, params, values):
        """
        set drone configuration
        :return:
        """
        for param, value in zip(params, values):
            self._mav_monitor.set_param(param, value)

    def mav_monitor_get_param(self, param):
        """
        get drone configuration
        :return:
        """
        return self._mav_monitor.get_param(param)

    def mav_monitor_start_mission(self):
        """
        start mission
        :return:
        """
        self._mav_monitor.start_mission()

    def start_mav_monitor(self):
        """
        start monitor
        :return:
        """
        self._mav_monitor.start()

    def stop_sitl(self):
        """
        stop the simulator
        :return:
        """
        self._sitl_task.sendcontrol('c')
        time.sleep(1)
        eventlet.monkey_patch()  # 必须加这条代码
        with eventlet.Timeout(5, False):  # 设置超时时间为2秒
            while True:
                line = self._sitl_task.readline()
                if line.startswith('SIM_VEHICLE: Killing tasks'):
                    break
        logging.info('Stop SITL task.')
        self._sitl_task.close(force=True)
        while not self._sitl_task.isalive:
            continue
        logging.debug('Send mavclosed to Airsim.')

    def get_mav_monitor(self):
        return self._mav_monitor

    def sitl_task(self) -> spawn:
        return self._sitl_task


def threading_send(task):
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        task.send('status GLOBAL_POSITION_INT \n')
        time.sleep(0.1)