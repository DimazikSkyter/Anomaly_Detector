import dataclasses
import logging
import os
import random

import keras
import tensorflow as tf
from keras import (layers, models)
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Dict
from dataclasses import dataclass, field

import numpy as np
from keras.api.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from anomaly.detector.metrics.Metrics import Metrics


class Mode(Enum):
    TRAIN = auto()
    DETECT = auto()


class Detector(ABC):
    def __init__(self, logger_name=None, logger_level="INFO",
                 log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
        if logger_name is None:
            logger_name = self.__class__.__name__
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        ch = logging.StreamHandler()
        ch.setLevel(logger_level)
        formatter = logging.Formatter(log_format)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def get_class_name(self):
        return type(self).__name__

    def is_trained(self) -> bool:
        return True

    def can_trained(self) -> bool:
        return False

    @abstractmethod
    def detect(self, metrics: Metrics) -> List[float]:
        pass

    def sigma_normalize(self, data, k) -> List[float]:
        return [2 * (1 / (1 + np.exp(-k * x)) - 0.5) for x in data]

    @staticmethod
    def min_max_scaler(data) -> List[float]:
        min_ = np.min(data)
        max_ = np.max(data)
        return [(element - min_) / (max_ - min_) for element in data]


class DetectorWithModel(Detector, ABC):

    def __init__(self, trained, path, logger_name=None, logger_level="INFO",
                 log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
        super().__init__(logger_name, logger_level, log_format)
        self.trained = trained
        self.path = path

    @abstractmethod
    def train(self, metrics: Metrics):
        pass

    def save_model(self):
        """Save the model to the specified path."""
        if not self.trained:
            raise ValueError("Model is not trained yet. Train the model before saving.")
        self.model.save(self.path)
        self.logger.info(f"Model saved to {self.path}")

    def load_model(self):
        """Load the model from the specified path."""
        if not os.path.exists(self.path):
            self.logger.warn("Model file not found %s", self.path)
            return
        self.model = models.load_model(self.path)
        self.logger.info(f"Model loaded from {self.path}")
        self.trained = True


class CompositeDetector(Detector):
    def __init__(self, detectors, logger_level="INFO"):
        super().__init__(logger_level=logger_level)
        self.detectors = detectors
        self.iteration = 0

    def is_trained(self):
        return all(detector.is_trained() for detector in self.detectors)

    def detect(self, metrics: Metrics) -> Metrics:
        self.logger.info(f"Start detect in composite detector {self.iteration}")
        results = {detector.get_class_name(): detector.detect(metrics) for detector in self.detectors}
        self.logger.debug(f"Detectors result {results}")
        return Metrics([], metrics.series_length(), results, metrics.timestamps)

    def train(self, metrics: Metrics):
        for detector in filter(lambda d: d.can_trained(), self.detectors):
            self.logger.info(f"Start train detector {detector.get_class_name()}")
            detector.train(metrics)


class Resolver(keras.Model):
    def __init__(self, final_converter_units, lstm_units):
        super(Resolver, self).__init__()
        self.final_converter_units = final_converter_units
        self.conv1 = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.pool1 = layers.MaxPooling1D(pool_size=2, strides=2)
        self.conv2 = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.pool2 = layers.MaxPooling1D(pool_size=2, strides=2)
        self.conv3 = layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')
        self.pool3 = layers.MaxPooling1D(pool_size=2, strides=2)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(final_converter_units, activation='sigmoid')
        self.lstm_layer = LSTM(lstm_units, return_sequences=False)
        self.output_layer = layers.Dense(final_converter_units, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.lstm_layer(x)
        x = self.output_layer(x)
        return x


class CompositeStreamDetector:
    model = None
    mode: Mode = Mode.TRAIN
    train_count: int = 0

    def __init__(self, detectors, model, properties, data_max_size=50, anomaly_seria_start_len=2, threshold=0.5):
        self.metrics: Metrics = None
        self.timestamps = []
        self.data_max_size = data_max_size
        self.detectors: List[Detector] = detectors
        self.model = model
        self.none_updated = 0
        self.anomaly_seria_start_len = anomaly_seria_start_len
        self.threshold = threshold
        self.__init_properties(properties)

    def __init_properties(self, properties):
        self.properties = properties
        if 'batch_size' not in properties:
            self.properties['batch_size'] = 50
        if 'epochs' not in properties:
            self.properties['epochs'] = 10

    def pop(self, metrics: Metrics):
        if self.metrics:
            self.metrics.merge(metrics)
        else:
            metrics.copy_cut_off(self.data_max_size)
            self.metrics = metrics

        if self.none_updated == self.data_max_size / 2:
            self._retrain()
            self.none_updated = 0
        else:
            self.none_updated += self.metrics.get_last_updated()

    def detect(self):
        detectors_opinions = self.get_detectors_opinions()
        X = np.stack(detectors_opinions, axis=-1)
        result = self.model.predict(X)
        return self._result_transform(result)

    def _result_transform(self, result):
        index_of_start = self.find_anomaly_index(0, result, self.anomaly_seria_start_len, True)
        if index_of_start:
            index_of_finish = self.find_anomaly_index(index_of_start, result, self.anomaly_seria_start_len, False)
            return index_of_start, index_of_finish
        else:
            return None, None

    def get_detectors_opinions(self):
        return [detector.detect(self.metrics) for detector in self.detectors]

    def train(self, metrics: Metrics, anomaly_vector):
        epochs = self.properties['epochs']
        batch_size = self.batch_size['batch_size']

        X, y = self._prepare_data()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def find_anomaly_index(self, first_index, result, len_, is_start: bool):
        if first_index > len(result):
            return None

        increment = 0
        for shift_index in range(len(result) - first_index):
            value = result[shift_index + first_index]
            if (value > self.threshold) == is_start:
                increment += 1
            else:
                increment = 0
            if increment >= len_:
                return first_index + shift_index - len_
        return None

    def _prepare_data(self):
        self.metrics.series
        self.metric_converter

        opinions = self.get_detectors_opinions()  # detectors * metrics * values
        final_metric = self.metric_converter(opinions)
        final_metric_np = final_metric.numpy()

        return final_metric.reshape((final_metric.shape[0], final_metric.shape[1], 1))

    @classmethod
    def get_mode(cls):
        return cls.mode

    @classmethod
    def set_mode(cls, mode: Mode):
        cls.mode = mode

    @classmethod
    def get_model(cls):
        return cls.model
