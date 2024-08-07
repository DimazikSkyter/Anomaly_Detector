import logging
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List

import numpy as np
from keras import (models)

from anomaly.detector.metrics.Metrics import Metrics


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

    def has_model(self):
        return False

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

    def has_model(self):
        return True

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

    def save_model(self):
        for detector in self.detectors:
            if detector.has_model():
                detector.save_model()

    def load_model(self):
        for detector in self.detectors:
            if detector.has_model():
                detector.load_model()
