import os

import numpy as np
from typing import List

from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam

from anomaly.detector.metrics.Metrics import Metrics
from anomaly.detector.parts.CompositeStreamDetector import Detector, DetectorWithModel
from keras import Sequential
from keras import models
from keras.api.layers import LSTM, Dense, Dropout

from anomaly.detector.parts.DataGenerator import DataGenerator
from keras import regularizers


# У AnomalyDetector и BehaviorDetector есть общая часть, необходим общий родитель
class AnomalyDetector(DetectorWithModel):
    ANOMALY_KEY = "anomaly"

    def __init__(self,
                 detectors_count,
                 anomaly_key,
                 lstm_size=128,
                 dropout_rate=0.2,
                 data_len=50,
                 epochs=15,
                 shift=10,
                 batch_size=1,
                 path="model.h5",
                 trained=False,
                 logger_level="INFO"):
        super().__init__(trained, path, logger_level=logger_level)
        self.data_len = data_len
        self.detectors_count = detectors_count
        self.model = self._init_model(lstm_size, dropout_rate)
        self.epochs = epochs
        self.shift = shift
        self.batch_size = batch_size
        self.load_model()
        self.anomaly_key = anomaly_key
        self.logger.info("Anomaly detector successfully init.")

    def _init_model(self, lstm_size, dropout_rate):
        model = Sequential()
        model.add(LSTM(lstm_size * 2,
                       activation='relu',
                       return_sequences=True,
                       input_shape=(self.data_len, self.detectors_count),
                       kernel_regularizer=regularizers.l1_l2(l1=0.02, l2=0.001)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_size, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.02, l2=0.001)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_size // 2, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.02, l2=0.001)))
        model.add(LSTM(lstm_size, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.02, l2=0.001)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.0005, clipvalue=1.0), loss='binary_crossentropy')
        return model

    def detect(self, metrics: Metrics) -> List[float]:
        if not self.trained:
            raise BrokenPipeError("Detector not trained to use.")
        if metrics.series_length() < self.data_len:
            raise IndexError(f"Wait metric size {self.data_len}, but income {metrics.series_length()}.")
        income_data_ = self._prepare_data(metrics)
        self.logger.debug("Anomaly detector income data after prepare %s", income_data_)
        predicted_values_ = self.model.predict(np.array(income_data_))
        self.logger.debug("Anomaly detector predicted values is %s", predicted_values_)
        return predicted_values_[0].flatten().tolist()

    def train(self, metrics: Metrics):
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        data_gen = DataGenerator(metrics,
                                 self.data_len,
                                 self.shift,
                                 self.batch_size,
                                 self.detectors_count,
                                 self.anomaly_key)
        self.model.fit(data_gen, epochs=self.epochs, verbose=0, callbacks=[early_stopping])
        self.trained = True
        self.save_model()

    def fine_tuning_checkpoint(self):
        pass

    def _prepare_data(self, metrics):
        metrics = metrics.copy_cut_off(self.data_len, True)
        income_data_ = metrics.to_train_matrix(normalized=True)
        income_data_ = np.expand_dims(income_data_, axis=0)
        return income_data_
