import os
import random
from typing import List

import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb
from anomaly.detector.generators.DataGenerator import DataGenerator
from keras import Sequential, Input, Model
from keras import regularizers
from keras.api.layers import TimeDistributed, LSTM, RepeatVector, Dense, Dropout
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.layers import BatchNormalization, Concatenate
from keras.src.optimizers import Adam

from anomaly.detector.metrics.Metrics import Metrics
from anomaly.detector.parts.CompositeStreamDetector import DetectorWithModel


class BehaviorDetector(DetectorWithModel):
    """
    A behavior detector that uses a sequence-to-sequence LSTM model to detect anomalies in time series data.

    Attributes:
        model (Sequential): The LSTM model used for anomaly detection.
    """

    def __init__(self,
                 metrics_count,
                 path="model.h5",
                 trained=False,
                 data_len=100,
                 lstm_size=128,
                 dropout_rate=0.2,
                 mult=1,
                 shift=10,
                 anomaly_metric_name="origin",
                 model_type=1,
                 logger_level="INFO"):
        super().__init__(trained, path, logger_level=logger_level)
        self.data_len = data_len
        self.trained = False
        self.anomaly_metric_name = anomaly_metric_name
        self.mult = mult
        self.shift = shift
        self.metrics_count = metrics_count
        self.model = getattr(self, f"_build_model_{model_type}")(lstm_size, data_len, dropout_rate)
        self.xgb_model = self._xgb_regressor_model()
        self.load_model()
        self.model_type = model_type
        self.logger_level = logger_level
        self.logger.info("Behavior detector successfully init.")
        self.first_layer_model = self._build_model_1(lstm_size, data_len, dropout_rate) if model_type == 3 else None


    def detect(self, metrics: Metrics) -> List[float]:
        """
        if series_length > data_len in will be cut off
        :param metrics:
        :return: measure of anomaly [0,1] for each point
        """
        if not self.trained:
            raise BrokenPipeError("Detector not trained to use.")
        if metrics.series_length() < self.data_len:
            raise IndexError(f"Wait metric size {self.data_len}, but income {metrics.series_length()}.")
        self.logger.debug("Income metrics: %s", metrics)
        income_data_ = self._prepare_data(metrics)
        self.logger.debug("incoming data: %s", income_data_)
        if self.model_type == 3:
            xgb_predicted = self.xgb_model.predict(income_data_.reshape(1, -1))
            predicted_values_ = self.model.predict([income_data_, xgb_predicted])
        else:
            predicted_values_ = self.model.predict(np.array(income_data_))
        self.logger.debug("Predicted values is %s", predicted_values_)
        reconstruction_error_ = self._reconstruct_error(income_data_, predicted_values_)
        self.logger.debug("Reconstruct errors is %s", reconstruction_error_)
        anomaly_scores = map(lambda x: 1 - 1 / (1 + self.mult * x), reconstruction_error_)
        return list(anomaly_scores)

    def _reconstruct_error(self, income_data_, predicted_values_):
        return np.max(np.abs(predicted_values_ - income_data_)
                      .reshape(self.data_len, self.metrics_count), axis=1)

    def _prepare_data(self, metrics):
        metrics = metrics.copy_cut_off(self.data_len, True)
        income_data_ = metrics.to_train_matrix(normalized=True)
        income_data_ = np.expand_dims(income_data_, axis=0)
        return income_data_

    def train(self, metrics: Metrics, epochs=10, batch_size=1, random_state=33):
        """
        Can use to train and retrain model.

        Parameters:
        :param random_state: 33
        :param metrics: income metrics (can't change in this detector)
        :param epochs: number epochs to train, default = 10
        :param batch_size: model fit batch_size
        :return:
        """
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        random.seed(random_state)

        len_ = len(metrics.series)
        if self.metrics_count != len_:
            self.logger.error("Wrong count of income metrics wait %s, but get %s", self.metrics_count, len_)
            raise IndexError(f"Wrong count of income metrics wait {self.metrics_count}, but get {len_}")
        self.logger.debug("Income metrics to train %s", metrics)
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        data_gen = DataGenerator(metrics,
                               self.data_len,
                               self.shift,
                               batch_size,
                               self.metrics_count,
                               logger_level=self.logger_level)
        if self.model_type == 3:
            if os.path.exists('model_checkpoint.h5'):
                checkpoint_cb = ModelCheckpoint('model_checkpoint.h5', save_best_only=True)
            self.first_layer_model.fit(data_gen, epochs=epochs, verbose=0, callbacks=[early_stopping], )
            income_data_ = self._prepare_data(metrics)
            predicted_values_ = self.first_layer_model.predict(np.array(income_data_))
            reconstruction_error_ = self._reconstruct_error(income_data_, predicted_values_)
            flatted_data_ = income_data_.reshape(income_data_.shape[0], -1)
            self.xgb_model.fit(flatted_data_, reconstruction_error_)
            xgb_predictions = self.xgb_model.predict(flatted_data_)
            second_data_generator_ = DataGenerator(metrics,
                          self.data_len,
                          self.shift,
                          batch_size,
                          self.metrics_count,
                          xgb_predictions=xgb_predictions,
                          logger_level=self.logger_level)
            self.model.fit(second_data_generator_, epochs=epochs, verbose=0, callbacks=[early_stopping], )#, checkpoint_cb
        else:
            self.model.fit(data_gen, epochs=epochs, verbose=0, callbacks=[early_stopping], )
        self.save_model()
        #joblib.dump(self.xgb_model, 'xgb_model.pkl')
        self.trained = True

    def _build_model_1(self, lstm_size, data_len, dropout_rate):
        model = Sequential()
        model.add(LSTM(lstm_size, activation='relu', return_sequences=True,
                       input_shape=(data_len, self.metrics_count),
                       kernel_regularizer=regularizers.l1_l2(l1=0.01,
                                                             l2=0.001)))  # , kernel_initializer='glorot_uniform'
        model.add(LSTM(lstm_size // 2, activation='relu', return_sequences=False,
                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001)))
        model.add(Dropout(dropout_rate))
        model.add(RepeatVector(self.data_len))
        model.add(LSTM(lstm_size, activation='relu', return_sequences=True,
                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001)))
        model.add(LSTM(lstm_size // 2, activation='relu', return_sequences=True,
                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001)))
        model.add(Dropout(dropout_rate))
        model.add(TimeDistributed(Dense(self.metrics_count)))
        model.compile(optimizer=Adam(learning_rate=0.0005, clipvalue=1.0), loss='mse')
        return model

    # add batchnormalization
    def _build_model_2(self, lstm_size, data_len, dropout_rate):
        model = Sequential()
        model.add(LSTM(lstm_size, activation='relu', return_sequences=True,
                       input_shape=(data_len, self.metrics_count),
                       kernel_regularizer=regularizers.l1_l2(l1=0.01,
                                                             l2=0.001)))  # , kernel_initializer='glorot_uniform'
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        model.add(LSTM(lstm_size // 2, activation='relu', return_sequences=False,
                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001)))
        model.add(RepeatVector(self.data_len))
        model.add(LSTM(lstm_size // 2, activation='relu', return_sequences=True,
                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001)))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        model.add(LSTM(lstm_size, activation='relu', return_sequences=True,
                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001)))
        model.add(TimeDistributed(Dense(self.metrics_count)))
        model.compile(optimizer=Adam(learning_rate=0.0005, clipvalue=1.0), loss='mse')
        return model

    # попробовать добавить сверточные
    # add XGBRegressor, пока проблема с генератором
    def _build_model_3(self, lstm_size, data_len, dropout_rate):
        shape_ = (data_len, self.metrics_count)
        lstm_input_ = Input(shape=shape_, name='lstm_input')
        xgb_input = Input(shape=(data_len, 1), name='xgb_input')

        main_part_model = LSTM(lstm_size, activation='relu', return_sequences=True,
                               input_shape=(data_len, self.metrics_count),
                               kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001))(lstm_input_)
        main_part_model = Dropout(dropout_rate)(main_part_model)
        main_part_model = BatchNormalization()(main_part_model)

        main_part_model = LSTM(lstm_size // 2, activation='relu', return_sequences=False,
                               kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001))(main_part_model)

        main_part_model = RepeatVector(self.data_len)(main_part_model)

        main_part_model = LSTM(lstm_size // 2, activation='relu', return_sequences=True,
                               kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001))(main_part_model)
        main_part_model = Dropout(dropout_rate)(main_part_model)
        main_part_model = BatchNormalization()(main_part_model)

        main_part_model = LSTM(lstm_size, activation='relu', return_sequences=True,
                               kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.001))(main_part_model)

        combined = Concatenate(axis=-1)([main_part_model, xgb_input])

        output = TimeDistributed(Dense(self.metrics_count))(combined)

        model = Model(inputs=[lstm_input_, xgb_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0005, clipvalue=1.0), loss='mse')
        return model

    def __prepare_xgb_regressor_data(self):
        pass

    @staticmethod
    def _xgb_regressor_model():
        return xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1024, max_depth=8, learning_rate=0.001)
