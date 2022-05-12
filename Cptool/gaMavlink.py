import json
import logging
import multiprocessing
import os
import random
import shutil

import numpy as np
import pandas as pd
import ray
from pymavlink import mavutil, mavwp
from pymavlink.mavutil import mavserial
from pyulog import ULog

from Cptool.config import toolConfig


class GaMavlink(multiprocessing.Process):
    """
    Mainly responsible for initiating the communication link to interact with UAV
    """

    def __init__(self, port, msg_queue):
        super(GaMavlink, self).__init__()
        self.msg_queue = msg_queue
        self._master: mavserial = None
        self._port = port

    def connect(self):
        """
        Connect drone
        :return:
        """
        self._master = mavutil.mavlink_connection('udp:0.0.0.0:{}'.format(self._port))
        try:
            self._master.wait_heartbeat(timeout=30)
        except TimeoutError:
            return False
        logging.info("Heartbeat from system (system %u component %u)" % (
            self._master.target_system, self._master.target_system))
        return True

    def set_mission(self, mission_file, random: bool, timeout=30) -> bool:
        """
        Set mission
        :param random: Out of order
        :param mission_file: mission file
        :param timeout:
        :return: success
        """
        if not self._master:
            logging.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')

        loader = mavwp.MAVWPLoader()
        loader.load(mission_file)
        logging.debug(f"Load mission file {mission_file}")

        # clear the waypoint
        self._master.waypoint_clear_all_send()
        # Pop home wp if mode is PX4
        if toolConfig.MODE == 'PX4':
            loader = self.trans_wp2px4(loader)
        # send the waypoint count
        self._master.waypoint_count_send(loader.count())
        seq_list = [True] * loader.count()
        try:
            # looping to send each waypoint information
            while True in seq_list:
                msg = self._master.recv_match(type=['MISSION_REQUEST'], blocking=True,
                                              timeout=timeout)
                if msg is not None and seq_list[msg.seq] is True:
                    self._master.mav.send(loader.wp(msg.seq))
                    seq_list[msg.seq] = False
                    logging.debug(f'Sending waypoint {msg.seq}')
            mission_ack_msg = self._master.recv_match(type=['MISSION_ACK'], blocking=True, timeout=timeout)
            logging.info('Upload mission finish.')
        except TimeoutError:
            logging.warning('Upload mission timeout!')
            return False
        return True

    def start_mission(self):
        """
        Arm and start the flight
        :return:
        """
        if not self._master:
            logging.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')
        # self._master.set_mode_loiter()
        self._master.arducopter_arm()
        self._master.set_mode_auto()

    def set_param(self, param: str, value: float) -> None:
        """
        set a value of specific parameter
        :param param: name of the parameter
        :param value: float value want to set
        """
        if not self._master:
            raise ValueError('Connect at first!')

        self._master.param_set_send(param, value)
        self.get_param(param)

    def set_params(self, params_dict: dict) -> None:
        """
        set multiple parameter
        :param params_dict: a dict consist of {parameter:values}...
        """
        for param, value in params_dict.items():
            self.set_param(param, value)

    def get_param(self, param: str) -> float:
        """
        get current value of a parameter.
        :param param: name
        :return: value of parameter
        """
        self._master.param_fetch_one(param)
        while True:
            message = self._master.recv_match(type=['PARAM_VALUE', 'PARM'], blocking=True).to_dict()
            if message['param_id'] == param:
                logging.debug('name: %s\t value: %f' % (message['param_id'], message['param_value']))
                break
        return message['param_value']

    def get_msg(self, type, block=False):
        """
        receive the mavlink message
        :param type:
        :param block:
        :return:
        """
        msg = self._master.recv_match(type=type, blocking=block)
        return msg

    def set_mode(self, mode:str):
        """
        Set flight mode
        :param mode: string type of a mode, it will be convert to an int values.
        :return:
        """
        if not self._master:
            logging.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')
        mode_id = self._master.mode_mapping()[mode]

        self._master.mav.set_mode_send(self._master.target_system,
                                       mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                       mode_id)
        while True:
            message = self._master.recv_match(type='COMMAND_ACK', blocking=True).to_dict()
            if message['command'] == mavutil.mavlink.MAVLINK_MSG_ID_SET_MODE:
                logging.debug(f'Mode: {mode} Set successful')
                break

    @staticmethod
    def log_extract_apm(msg: dict):
        """
        parse the msg of mavlink
        :param msg:
        :return:
        """
        out = None
        if msg['mavpackettype'] == 'ATT':
            if len(toolConfig.LOG_MAP):
                out = {
                    'TimeS': msg['TimeUS'] / 1000000,
                    'Roll': msg['Roll'],
                    'Pitch': msg['Pitch'],
                    'Yaw': (msg['Yaw'] + 180) % 360 - 180,
                }
        elif msg['mavpackettype'] == 'RATE':
            out = {
                'TimeS': msg['TimeUS'] / 1000000,
                'RateRoll': msg['R'],
                'RatePitch': msg['P'],
                'RateYaw': msg['Y'],
            }
        elif msg['mavpackettype'] == 'IMU':
            out = {
                'TimeS': msg['TimeUS'] / 1000000,
                'AccX': msg['AccX'],
                'AccY': msg['AccY'],
                'AccZ': msg['AccZ'],
                'GyrX': msg['GyrX'],
                'GyrY': msg['GyrY'],
                'GyrZ': msg['GyrZ'],
            }
        elif msg['mavpackettype'] == 'PARM':
            out = {
                'TimeS': msg['TimeUS'] / 1000000,
                msg['Name']: msg['Value']
            }
        return out

    @staticmethod
    def log_extract_px4(msg):
        """
        parse the on-board log message of px4
        :param msg:
        :return:
        """
        msg.drop(['label'], axis=1, inplace=True)
        msg = msg.fillna(0)
        msg = msg.sum()
        msg[['roll', 'pitch', 'yaw']] = msg[['roll', 'pitch', 'yaw']] * 180
        msg[['roll_body', 'pitch_body', 'yaw_body']] = msg[['roll_body', 'pitch_body', 'yaw_body']] * 180
        return msg.to_dict()

    @staticmethod
    def extract_from_log_file(log_file):
        """
        extract log message form a bin file.
        :param log_file:
        :return:
        """
        accept_item = toolConfig.LOG_MAP

        logs = mavutil.mavlink_connection(log_file)
        att = []
        rate = []
        imu = []
        parm = []
        accpet_param = GaMavlink.load_param().columns.to_list()

        while True:
            msg = logs.recv_match(type=accept_item)
            if msg is None:
                break
            msg = msg.to_dict()

            # 剔除IMU1的情况，只要IMU0
            # if msg['mavpackettype'] == 'IMU' and msg['I'] == 0:
            if msg['mavpackettype'] == 'ATT':
                att.append(GaMavlink.log_extract_apm(msg))
            elif msg['mavpackettype'] == 'RATE':
                rate.append(GaMavlink.log_extract_apm(msg))
            elif msg['mavpackettype'] == 'IMU': #and msg['I'] == 0:
                imu.append(GaMavlink.log_extract_apm(msg))
            elif msg['mavpackettype'] == 'PARM' and msg['Name'] in accpet_param:
                parm.append(GaMavlink.log_extract_apm(msg))

        att = pd.DataFrame(att)
        rate = pd.DataFrame(rate)
        acc = pd.DataFrame(imu)
        parm = pd.DataFrame(parm)
        parm.fillna(method='ffill', inplace=True)
        parm.dropna(inplace=True)
        parm.drop_duplicates(GaMavlink.load_param().columns.to_list(), 'first', inplace=True)
        parm = parm[['TimeS'] + toolConfig.PARAM]


        # 进行采样，统一刷新率
        att['TimeS'] = att['TimeS'].round(1)
        att.drop_duplicates('TimeS', keep='first', inplace=True)

        rate['TimeS'] = rate['TimeS'].round(1)
        rate.drop_duplicates('TimeS', keep='first', inplace=True)

        acc['TimeS'] = acc['TimeS'].round(1)
        acc.drop_duplicates('TimeS', keep='first', inplace=True)

        parm['TimeS'] = parm['TimeS'].round(1)
        parm.drop_duplicates('TimeS', keep='last', inplace=True)

        # 合数据
        out = pd.merge(att, acc, on='TimeS')
        out = pd.merge(out, rate, on='TimeS')

        out.dropna(inplace=True)

        # 加入configuration
        out = pd.merge(out, parm, on='TimeS', how='outer')
        out.fillna(method='ffill', inplace=True)

        attitude_name = ['TimeS', 'Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw',
                         'AccX',    'AccY'    ,'AccZ'    ,'GyrX'  ,  'GyrY'  ,  'GyrZ'
                         ]
        other_name = out.keys().difference(attitude_name)
        attitude_name.extend(other_name.tolist())

        out = out[attitude_name]
        return out

    @staticmethod
    def read_path_specified_file(log_path, exe):
        """
        :param log_path:
        :param exe:
        :return:
        """
        file_list = []
        for filename in os.listdir(log_path):
            if filename.endswith(f'.{exe}'):
                file_list.append(filename)
        file_list.sort()
        return file_list

    @staticmethod
    def extract_from_log_path(log_path, threat=None):
        """
        extract and convert bin file to csv
        :param log_path:
        :param threat: multiple threat
        :return:
        """

        file_list = GaMavlink.read_path_specified_file(log_path, 'BIN')
        if not os.path.exists(f"{log_path}/csv"):
            os.makedirs(f"{log_path}/csv")

        # 列出文件夹内所有.BIN结尾的文件并排序
        for file in file_list:
            name, _ = file.split('.')
            csv_data = GaMavlink.extract_from_log_file(log_path + f'/{file}')
            csv_data.to_csv(f'{log_path}/csv/{name}.csv', index=False)

    @staticmethod
    @ray.remote
    def extract_from_log_path_threat(log_path, file_list):
        for file in file_list:
            name, _ = file.split('.')
            # if os.path.exists(f'{log_path}/csv/{name}.csv'):
            #     continue
            csv_data = GaMavlink.extract_from_log_file(log_path + f'/{file}')
            # 只保留一位小数
            # csv_data = csv_data[['TimeS', 'Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw',
            #                      'DesRoll', 'DesPitch', 'DesYaw', 'DesRateRoll', 'DesRatePitch', 'DesRateYaw']]
            csv_data.to_csv(f'{log_path}/csv/{name}.csv', index=False)
            print(f"\r{log_path} Process: {name}")
        if os.path.exists(f'{log_path}/mark.pkl'):
            shutil.copyfile(f'{log_path}/mark.pkl', f'{log_path}/csv/mark.pkl')
        return True

    @staticmethod
    def extract_from_ulog(log_file):
        """
        extract and convert ulog file to csv
        :param log_path:
        :return:
        """
        # load ulog
        ulog = ULog(log_file)
        att = pd.DataFrame(ulog.get_dataset('vehicle_attitude_setpoint').data)
        rate = pd.DataFrame(ulog.get_dataset('vehicle_rates_setpoint').data)
        acc = pd.DataFrame(ulog.get_dataset('vehicle_acceleration').data)

        # 给标记
        att = att[['timestamp', 'roll_body', 'pitch_body', 'yaw_body']]
        att['label'] = np.zeros(len(att))
        rate = rate[['timestamp', 'roll', 'pitch', 'yaw']]
        rate['label'] = np.zeros(len(rate)) + 1
        acc = acc[['timestamp', 'xyz[0]', 'xyz[1]', 'xyz[2]']]
        acc['label'] = np.zeros(len(acc)) + 2

        # 合并到一个表中
        array = att.append(rate, ignore_index=True)
        array = array.append(acc, ignore_index=True)
        array = array.sort_values(by='timestamp').reset_index(drop=True)

        # 找出重复的index
        pre = array['label'].to_numpy()[:-1]
        next = array['label'].to_numpy()[1:]
        # 去重
        array = array.iloc[:-1][(pre - next) != 0]
        label = array['label'].to_numpy()

        data = []
        for i in range(len(label) - 2):
            if label[i:i + 3].sum() == 3:
                out = GaMavlink.log_extract_px4(array.iloc[i:i + 3])
                data.append(out)
        data = pd.DataFrame(data, columns=['timestamp', 'xyz[0]', 'xyz[1]', 'xyz[2]',
                                           'roll_body', 'pitch_body', 'pitch_body',
                                           'roll', 'pitch', 'yaw'])
        data.rename(columns={
            'timestamp': 'TimeS',
            'xyz[0]': 'AccX',
            'xyz[1]': 'AccY',
            'xyz[2]': 'AccZ',
            'roll_body': 'Roll',
            'pitch_body': 'Pitch',
            'yaw_body': 'Yaw',
            'roll': 'RateRoll',
            'pitch': 'RatePitch',
            'yaw': 'RateYaw',
        }, inplace=True)
        return data

    @staticmethod
    def load_param() -> json:
        """
        load parameter we want to fuzzing
        :return:
        """
        if toolConfig.MODE == 'Ardupilot':
            path = 'Cptool/param_ardu.json'
        elif toolConfig.MODE == 'PX4':
            path = 'Cptool/param_px4.json'
        with open(path, 'r') as f:
            return pd.DataFrame(json.loads(f.read()))

    @staticmethod
    def random_param_value(param_json: dict):
        """
        random create the value
        :param param_json:
        :return:
        """
        out = {}
        for name, item in param_json.items():
            range = item['range']
            step = item['step']
            random_sample = random.randrange(range[0], range[1], step)
            out[name] = random_sample
        return out

    @staticmethod
    def get_default_values(para_dict):
        return para_dict.loc[['default']]

    @staticmethod
    def select_sub_dict(para_dict, param_choice):
        return para_dict[param_choice]

    @staticmethod
    def read_range_from_dict(para_dict):
        return np.array(para_dict.loc['range'].to_list())

    @staticmethod
    def read_unit_from_dict(para_dict):
        return para_dict.loc['step'].to_numpy()

    def run(self):
        """
        loop check
        :return:
        """

        while True:
            msg = self._master.recv_match(type=['STATUSTEXT'], blocking=False)
            if msg is not None:
                msg = msg.to_dict()
                # print(msg2)
                if msg['severity'] in [0, 2]:
                    # self.send_msg_queue.put('crash')
                    logging.info('ArduCopter detect Crash.')
                    self.msg_queue.put('error')
                    break
