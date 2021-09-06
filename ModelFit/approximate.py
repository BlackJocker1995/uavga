# -*- coding:utf-8 -*-
import logging
import os, sys
import pickle
import time
from abc import abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, concatenate, Conv1D, add, Activation, Input, Flatten, Embedding, RepeatVector
from keras.layers import LSTM
from keras.models import Sequential
from tcn import TCN, tcn_full_summary
from sklearn import model_selection
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

import Cptool.config
import ModelFit.config


class Modeling(object):
    def __init__(self, resize: bool = True, debug: bool = False):
        self._model: Sequential = None
        self._trans: MinMaxScaler = None
        if Cptool.config.MODE == 'Ardupilot':
            self._uav_class = 'Ardupilot'
        elif Cptool.config.MODE == 'PX4':
            self._uav_class = 'PX4'
        self._resize = resize

        if debug:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.INFO)

    def _cs_to_sl(self, values):

        # ensure all data is float
        values = values.astype('float32')

        # normalize features

        if ModelFit.config.RETRANS:
            trans = self.load_trans()

            values = trans.transform(values)

        # frame as supervised learning
        reframed = self._series_to_supervised(values, ModelFit.config.INPUT_LEN, True)

        if not os.path.exists('model/{}/{}'.format(self._uav_class, ModelFit.config.INPUT_LEN)):
            os.makedirs('model/{}/{}'.format(self._uav_class, ModelFit.config.INPUT_LEN))

        return reframed

    def _series_to_supervised(self, data, n_in=1, dropnan=True):
        # convert series to supervised learning
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, 1):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def _train_valid_split(self, values):
        # split into train and test sets
        X, Y = self._data_split(values)
        train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.2, random_state=0)

        logging.info(f"Shape: {train_X.shape}, {train_Y.shape}, {valid_X.shape}, {valid_Y.shape}")

        return train_X, train_Y, valid_X, valid_Y

    def _test_split(self, values):
        X, Y = self._data_split(values)
        logging.info(f"Shape: {X.shape}, {Y.shape}")
        return X, Y

    @abstractmethod
    def _data_split(self, value):
        pass

    @abstractmethod
    def _fit_network(self, train_X, train_Y, valid_X, valid_Y, num=None):
        return None

    @abstractmethod
    def _build_model(self, train_shape: np.shape):
        return None

    def set_model(self, path):
        local = os.getcwd()
        self._model = load_model(f"{local}/{path}")

    def run(self, train_filename, cuda: bool = False):
        if not cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # load dataset
        dataset = pd.read_csv(train_filename, header=0, index_col=0)

        values = dataset.values

        values = self._cs_to_sl(values)
        train_X, train_y, valid_X, valid_y = self._train_valid_split(values)
        model = self._fit_network(train_X, train_y, valid_X, valid_y)
        self._model = model

    def predict(self, values):
        if self._model is None:
            logging.warning('Model is not trained!')
            raise ValueError('Train or load model at first')

        predict_X = self._model.predict(values)
        # 数据还原
        if ModelFit.config.RETRANS:
            trans = self.load_trans()
            predict_X = trans.inverse_transform(predict_X)

        return predict_X

    def test_cmp_draw(self, test, cmp_name, num=150, exec='pdf'):
        if self._model is None:
            logging.warning('Model is not trained!')
            raise ValueError('Train or load model at first')
        if not os.path.exists(f'{os.getcwd()}/fig/{Cptool.config.MODE}/{ModelFit.config.INPUT_LEN}/{cmp_name}'):
            os.makedirs(f'{os.getcwd()}/fig/{Cptool.config.MODE}/{ModelFit.config.INPUT_LEN}/{cmp_name}')

        values = self._cs_to_sl(test)
        X, Y = self._data_split(values)

        predict_y = self._model.predict(X)
        # if self._resize:
        #     scalar = self.load_trans()
        #     predict_y = scalar.inverse_transform(predict_y)
        #     Y = scalar.inverse_transform(Y)

        if X.shape[0] > num:
            col = self._systematicSampling(X, num)
            #col = np.arange(200, 400)
            predict_y = predict_y[col, :]
            test = Y[col, :]
        else:
            test = Y
        # 'AccX', 'AccY', 'AccZ',
        for name, i in zip(['Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw',], range(6)):
            x = predict_y[:,  i]
            y = test[:,  i]

            fig = plt.figure(figsize=(8, 5))
            ax1 = plt.subplot()

            ax2 = ax1.twinx()

            # if name == 'Yaw':
            #     x += 3.3
            # if name == 'Pitch':
            #     x += 0.07
            # if name == 'Roll':
            #     x += 0.11
            # if name == 'RatePitch':
            #     x += 0.18
            # if name == 'RateRoll':
            #     x += 0.15
            # if name == 'RateYaw':
            #     x += 0.5

            ax1.plot(x, '-', label='Predicted', linewidth=2)
            ax1.plot(y, '--', label='Real', linewidth=2)
            if name in ['AccX', 'AccY', 'AccZ']:
                ax1.set_ylabel(f'{name} (m/s/s)', fontsize=18)
            if name in ['RateRoll', 'RatePitch', 'RateYaw']:
                ax1.set_ylabel(f'{name} (deg/s)', fontsize=18)
            if name in ['Roll', 'Pitch', 'Yaw']:
                ax1.set_ylabel(f'{name} (deg)', fontsize=18)
            ax2.bar(np.arange(len(x)), np.abs(x - y), label='Error')
            ax2.set_ylim([0, 10 * np.max(np.abs(x - y))])
            if name in ['AccX', 'AccY', 'AccZ']:
                ax2.set_ylabel('Error (m/s/s)', fontsize=18)
            if name in ['RateRoll', 'RatePitch', 'RateYaw']:
                ax2.set_ylabel('Error (deg/s)', fontsize=18)
            if name in ['Roll', 'Pitch', 'Yaw']:
                ax2.set_ylabel('Error (deg)', fontsize=18)

            fig.legend(loc='upper center', ncol=3, fontsize='18')
            plt.setp(ax1.get_xticklabels(), fontsize=18)
            plt.setp(ax2.get_yticklabels(), fontsize=18)
            plt.setp(ax1.get_yticklabels(), fontsize=18)

            plt.margins(0, 0)
            # plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(f'{os.getcwd()}/fig/{Cptool.config.MODE}/{ModelFit.config.INPUT_LEN}/{cmp_name}/{name.lower()}.{exec}')
            # plt.show()
            plt.clf()

    def test_feature_draw(self, X, Y, cmp_name, exec='pdf'):
        if self._model is None:
            logging.warning('Model is not trained!')
            raise ValueError('Train or load model at first')
        if not os.path.exists(f'{os.getcwd()}/fig/{Cptool.config.MODE}/{ModelFit.config.INPUT_LEN}/{cmp_name}'):
            os.makedirs(f'{os.getcwd()}/fig/{Cptool.config.MODE}/{ModelFit.config.INPUT_LEN}/{cmp_name}')

        for name, i in zip(['AccX', 'AccY', 'AccZ', 'Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw',], range(9)):
            x = X[:, i]
            y = Y[:, i]

            fig = plt.figure(figsize=(8, 4.8))
            ax1 = plt.subplot()

            ax2 = ax1.twinx()

            # if name == 'Pitch':
            #     x += 4
            ax1.plot(x, '-', label='Predicted', linewidth=2)
            ax1.plot(y, '--', label='Real', linewidth=2)
            if name in ['AccX', 'AccY', 'AccZ']:
                ax1.set_ylabel(f'{name} (m/s/s)', fontsize=18)
            if name in ['RateRoll', 'RatePitch', 'RateYaw']:
                ax1.set_ylabel(f'{name} (deg/s)', fontsize=18)
            if name in ['Roll', 'Pitch', 'Yaw']:
                ax1.set_ylabel(f'{name} (deg)', fontsize=18)
            ax2.bar(np.arange(len(x)), np.abs(x - y), label='Error')
            ax2.set_ylim([0, 10 * np.max(np.abs(x - y))])
            if name in ['AccX', 'AccY', 'AccZ']:
                ax2.set_ylabel('Error (m/s/s)', fontsize=18)
            if name in ['RateRoll', 'RatePitch', 'RateYaw']:
                ax2.set_ylabel('Error (deg/s)', fontsize=18)
            if name in ['Roll', 'Pitch', 'Yaw']:
                ax2.set_ylabel('Error (deg)', fontsize=18)

            fig.legend(loc='upper center', ncol=3, fontsize='18')

            plt.margins(0, 0)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(f'{os.getcwd()}/fig/{Cptool.config.MODE}/{ModelFit.config.INPUT_LEN}/{cmp_name}/{name.lower()}.{exec}', dpi=300)
            # plt.show()
            plt.clf()

    def run_5flow_test(self, train_filename, cuda: bool = False):
        if not cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # load dataset
        dataset = pd.read_csv(train_filename, header=0, index_col=0)

        values = dataset.values

        values = self._cs_to_sl(values)

        X, Y = self._data_split(values)

        for i in range(5):
            _, X_other, _, y_other = train_test_split(X, Y, test_size=0.2, random_state=5 + i,
                                                                  shuffle=False, stratify=None)
            X_valid, X_test, y_valid, y_test = train_test_split(X_other, y_other, test_size=0.2, random_state=5 + i,
                                                                shuffle=True, stratify=None)

            self._fit_network(X_valid, y_valid, X_test, y_test, i)

    def test(self, test_data, cuda: bool = False):
        if self._model is None:
            logging.warning('Model is not trained!')
            raise ValueError('Train or load model at first')

        if not cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # load dataset
        values = test_data.values
        values = self._cs_to_sl(values)
        test_X, test_Y = self._test_split(values)

        start = time.time()
        self._model.predict(test_X)
        end = time.time()
        logging.info("time cost:%.4f s" % (end - start))

        score = self._model.evaluate(test_X, test_Y, batch_size=256, verbose=1)
        logging.info(f"{score[1]}")
        return score[1]

    def test_kfold(self, model_path, test_data, k, cuda: bool = False):
        scores = []
        for i in range(k):
            try:
                self.set_model(f"{model_path}/lstm{i}.h5")
            except Exception as e:
                logging.warning('Model is not trained!')
                raise ValueError('Train or load model at first')

            score = self.test(test_data, cuda)
            scores.append(score)
        logging.info(f"test scores: {scores}")

    @staticmethod
    def _systematicSampling(dataMat, number):
        length = len(dataMat)
        k = length // number
        out = range(length)
        out_index = out[:length:k]
        return out_index

    @staticmethod
    def fit_trans(train_filename):
        if not os.path.exists(f'model/{Cptool.config.MODE}/{ModelFit.config.INPUT_LEN}'):
            os.makedirs(f'model/{Cptool.config.MODE}/{ModelFit.config.INPUT_LEN}')

        # load dataset
        dataset = pd.read_csv(train_filename, header=0, index_col=0)

        values = dataset.values

        trans = MinMaxScaler(feature_range=(0, 1))
        trans.fit(values)

        with open(f'model/{Cptool.config.MODE}/trans.pkl', 'wb') as f:
            pickle.dump(trans, f)

    @staticmethod
    def load_trans():
        # load dataset
        with open(f'model/{Cptool.config.MODE}/trans.pkl', 'rb') as f:
            trans = pickle.load(f)
        return trans


class CyLSTM(Modeling):
    def __init__(self, epochs: int, batch_size: int,debug: bool = False):
        super(CyLSTM, self).__init__(batch_size, debug)

        # param
        self.epochs = epochs
        self.batch_size: int = batch_size

    def _data_split(self, value):
        values = value.values

        # split into input and outputs
        X, Y = values[:,
               : -ModelFit.config.DATA_LEN], \
               values[:, -ModelFit.config.DATA_LEN: -ModelFit.config.DATA_LEN + ModelFit.config.OUTPUT_DATA_LEN]

        # reshape input to be 3D [samples, timesteps, features]
        X = X.reshape((X.shape[0], ModelFit.config.INPUT_LEN, ModelFit.config.DATA_LEN))
        Y = Y.reshape((Y.shape[0], ModelFit.config.OUTPUT_DATA_LEN))

        return X, Y

    def _fit_network(self, train_X, train_Y, valid_X, valid_Y, num=None):
        if not self.epochs:
            raise ValueError('set LSTM param at first!')

        model = self._build_model(train_X.shape)
        # fit network
        history = model.fit(train_X, train_Y,
                            epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(valid_X, valid_Y),
                            verbose=2,
                            shuffle=True)

        if num is not None:
            model.save(f'model/{self._uav_class}/{ModelFit.config.INPUT_LEN}/lstm{num}.h5')
            plt.plot(history.history['loss'], label=f'train-{num}')
            plt.plot(history.history['val_loss'], label=f'validation-{num}')
        else:
            model.save(f'model/{self._uav_class}/{ModelFit.config.INPUT_LEN}/lstm.h5')
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='validation')
        # plot history
        axis_font = {'size': '18'}
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Epochs Time', fontsize=18)
        plt.legend(prop=axis_font)
        # plt.show()
        plt.savefig(f'model/{self._uav_class}/{ModelFit.config.INPUT_LEN}/loss.pdf')
        return model

    def _build_model(self, train_shape: np.shape):
        model = Sequential()
        model.add(LSTM(128, input_shape=(train_shape[1], train_shape[2])))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(ModelFit.config.OUTPUT_DATA_LEN))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mse'])
        model.summary()

        return model

    def read_model(self):
        self._model = load_model(f'model/{Cptool.config.MODE}/{ModelFit.config.INPUT_LEN}/lstm.h5')


class CyTCN(Modeling):
    def __init__(self, epochs: int, batch_size: int, debug: bool = False):
        super(CyTCN, self).__init__(debug)

        # param
        self.epochs = epochs
        self.batch_size: int = batch_size

    def _data_split(self, value):
        values = value.values

        # split into input and outputs
        X, Y = values[:, :-ModelFit.config.DATA_LEN], \
               values[:, ModelFit.config.INPUT_LEN * ModelFit.config.DATA_LEN :
                         ModelFit.config.INPUT_LEN * ModelFit.config.DATA_LEN + ModelFit.config.OUTPUT_DATA_LEN]

        # reshape input to be 3D [samples, timesteps, features]
        X = X.reshape((X.shape[0], ModelFit.config.INPUT_LEN, ModelFit.config.DATA_LEN))
        Y = Y.reshape((Y.shape[0], 1, ModelFit.config.OUTPUT_DATA_LEN))

        return X, Y

    def _fit_network(self, train_X, train_Y, valid_X, valid_Y, num=None):
        if not self.epochs:
            raise ValueError('set TCN param at first!')

        model = self._build_model(train_X.shape)
        # fit network
        history = model.fit(train_X, train_Y,
                            epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(valid_X, valid_Y),
                            verbose=2,
                            shuffle=False)

        if num is not None:
            model.save(f'model/{self._uav_class}/{ModelFit.config.INPUT_LEN}/tcn{num}.h5')
            plt.plot(history.history['loss'], label=f'train-{num}')
            plt.plot(history.history['val_loss'], label=f'validation-{num}')
        else:
            model.save(f'model/{self._uav_class}/{ModelFit.config.INPUT_LEN}/tcn.h5')
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='validation')
        # plot history
        axis_font = {'size': '18'}
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Epochs Time', fontsize=18)
        plt.legend(prop=axis_font)
        # plt.show()
        plt.savefig(f'model/{self._uav_class}/{ModelFit.config.INPUT_LEN}/tcn_loss.pdf')
        return model

    def _build_model(self, train_shape: np.shape):
        # 创建模型
        model = Sequential(
            layers=[
                TCN(input_shape=(train_shape[1], train_shape[2])),  # output.shape = (batch, 64)
                RepeatVector(1),  # output.shape = (batch, output_timesteps, 64)
                Dense(ModelFit.config.OUTPUT_DATA_LEN)  # output.shape = (batch, output_timesteps, output_dim)
            ]
        )
        model.compile(loss="mse",
                      optimizer="Adam", metrics=["accuracy"])  # 配置
        model.summary()

        return model

    def read_model(self):
        self._model = load_model(f'model/{Cptool.config.MODE}/{ModelFit.config.INPUT_LEN}/tcn.h5',
                                 custom_objects={"TCN": TCN})




