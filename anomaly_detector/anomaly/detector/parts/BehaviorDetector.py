from typing import List

import keras
import numpy as np
from keras import Sequential
from keras.api.layers import TimeDistributed, LSTM, RepeatVector, Dense, Dropout
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam

from anomaly.detector.metrics.Metrics import Metrics
from anomaly.detector.parts.CompositeStreamDetector import Detector, DetectorWithModel
from anomaly.detector.parts.DataGenerator import DataGenerator


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
                 logger_level="INFO"):
        super().__init__(trained, path, logger_level=logger_level)
        self.data_len = data_len
        self.trained = False
        self.anomaly_metric_name = anomaly_metric_name
        self.mult = mult
        self.shift = shift
        self.metrics_count = metrics_count
        self.model = self._build_model(lstm_size, data_len, dropout_rate)
        self.load_model()
        self.logger.info("Behavior detector successfully init.")

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
        income_data_ = self._prepare_data(metrics)
        predicted_values_ = self.model.predict(np.array(income_data_))
        self.logger.debug("Predicted values is %s", predicted_values_)
        reconstruction_error = np.max(np.abs(predicted_values_ - income_data_).reshape(50, 4), axis=1)
        anomaly_scores = map(lambda x: 1 - 1 / (1 + self.mult * x), reconstruction_error)
        return list(anomaly_scores)

    def _prepare_data(self, metrics):
        metrics = metrics.copy_cut_off(self.data_len, True)
        income_data_ = metrics.to_train_matrix(normalized=True)
        income_data_ = np.expand_dims(income_data_, axis=0)
        return income_data_

    def train(self, metrics: Metrics, epochs=10, batch_size=1):
        """
        Can use to train and retrain model.

        Parameters:
        :param metrics: income metrics (can't change in this detector)
        :param epochs: number epochs to train, default = 10
        :param batch_size: model fit batch_size
        :return:
        """
        len_ = len(metrics.series)
        if self.metrics_count != len_:
            self.logger.error("Wrong count of income metrics wait %s, but get %s", self.metrics_count, len_)
            raise IndexError(f"Wrong count of income metrics wait {self.metrics_count}, but get {len_}")
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        #data_gen = self._windowed_data(metrics)
        #steps_per_epoch = (metrics.series_length() - self.shift) // self.data_len + 1
        data_gen = DataGenerator(metrics, self.data_len, self.shift, batch_size, self.metrics_count)

        self.model.fit(data_gen, epochs=epochs, verbose=0, callbacks=[early_stopping])
        self.trained = True

    def _build_model(self, lstm_size, data_len, dropout_rate):
        model = Sequential()
        model.add(LSTM(lstm_size, activation='relu', return_sequences=True, input_shape=(data_len, self.metrics_count)))
        model.add(LSTM(lstm_size // 2, activation='relu', return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(RepeatVector(self.data_len))
        model.add(LSTM(lstm_size, activation='relu', return_sequences=True))
        model.add(LSTM(lstm_size // 2, activation='relu', return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(TimeDistributed(Dense(self.metrics_count)))
        model.compile(optimizer='adam', loss='mse')
        return model

    def _windowed_data(self, metrics: Metrics):
        """
        A generator that yields batches of LSTM input sequences.
        All data is normalized in this method
        Yields:
        - A tuple(X, X) with shape (data_len, num_series).
        """
        train_data = np.array(metrics.to_train_matrix(normalized=True))
        num_windows = (train_data.shape[0] - self.shift) // self.data_len + 1

        for i in range(num_windows):
            start_idx = i * self.shift
            end_idx = start_idx + self.data_len
            window = train_data[start_idx:end_idx]
            if window.shape[0] == self.data_len:
                yield window, window

