import asyncio
import logging
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

from anomaly.detector.clients.VictoriaMetricsClient import VictoriaMetricsClient
from anomaly.detector.parts.BehaviorDetector import BehaviorDetector
from anomaly.detector.parts.CusumDetector import CUSUMDetector
from anomaly.detector.parts.WindowedMadDetector import MADDetector


#Пока что все метрики участвуют во всех детекторах, однако это нужно переделать

@dataclass
class DetectorsProps:
    __metrics_count: int
    __data_len: int
    __behavior_detector_lstm_size: int
    __behavior_detector_dropout_rate: float
    __behavior_detector_train_shift: int
    __behavior_detector_anomaly_metric_name: str
    __cusum_threshold: float
    __cusum_drift: float
    __cusum_window: int
    __anomaly_detector_lstm_size: int

    def get_steps(self) -> int:
        return self.__metrics_count

    def set_steps(self, metrics_count: int) -> None:
        self.__metrics_count = metrics_count

    def get_data_len(self) -> int:
        return self.__data_len

    def set_data_len(self, data_len: int) -> None:
        self.__data_len = data_len

    def get_behavior_detector_lstm_size(self) -> int:
        return self.__behavior_detector_lstm_size

    def set_behavior_detector_lstm_size(self, behavior_detector_lstm_size: int) -> None:
        self.__behavior_detector_lstm_size = behavior_detector_lstm_size

    def get_behavior_detector_dropout_rate(self) -> float:
        return self.__behavior_detector_dropout_rate

    def set_behavior_detector_dropout_rate(self, behavior_detector_dropout_rate: float) -> None:
        self.__behavior_detector_dropout_rate = behavior_detector_dropout_rate

    def get_behavior_detector_train_shift(self) -> int:
        return self.__behavior_detector_train_shift

    def set_behavior_detector_train_shift(self, behavior_detector_train_shift: int) -> None:
        self.__behavior_detector_train_shift = behavior_detector_train_shift

    def get_behavior_detector_anomaly_metric_name(self) -> str:
            return self.__behavior_detector_anomaly_metric_name

    def set_behavior_detector_anomaly_metric_name(self, behavior_detector_anomaly_metric_name: str) -> None:
        self.__behavior_detector_anomaly_metric_name = behavior_detector_anomaly_metric_name

    def get_cusum_threshold(self) -> float:
            return self.__cusum_threshold

    def set_cusum_threshold(self, cusum_threshold: float) -> None:
        self.__cusum_threshold = cusum_threshold

    def get_cusum_drift(self) -> float:
        return self.__cusum_drift

    def set_cusum_drift(self, cusum_drift: float) -> None:
        self.__cusum_drift = cusum_drift

    def get_cusum_window(self) -> int:
        return self.__cusum_window

    def set_cusum_window(self, cusum_window: int) -> None:
        self.__cusum_window = cusum_window

    def get_anomaly_detector_lstm_size(self) -> int:
        return self.__anomaly_detector_lstm_size

    def set_anomaly_detector_lstm_size(self, anomaly_detector_lstm_size: int) -> None:
        self.__anomaly_detector_lstm_size = anomaly_detector_lstm_size

@dataclass
class StorageProps:
    __base_url: str
    __queries: list
    __steps: int

    def get_base_url(self) -> str:
        return self.__base_url

    def set_base_url(self, base_url: str) -> None:
        self.__base_url = base_url

    def get_queries(self) -> list:
        return self.__queries

    def set_queries(self, queries: list) -> None:
        self.__queries = queries

    def get_steps(self) -> int:
        return self.__steps

    def set_steps(self, steps: int) -> None:
        self.__steps = steps

@dataclass
class AppProps:
    __storage_props: StorageProps
    __detectors_props: DetectorsProps

    def get_storage_props(self) -> StorageProps:
        return self.__storage_props

    def set_storage_props(self, storage_props: StorageProps) -> None:
        self.__storage_props = storage_props

    def get_detectors_props(self) -> DetectorsProps:
        return self.__detectors_props

    def set_detectors_props(self, detectors_props: DetectorsProps) -> None:
        self.__detectors_props = detectors_props



# все метрики уже должны быть от 0 до 1
class App:
    def __int__(self, props: AppProps, logger_level="INFO"):
        self._init_logger(logger_level)
        self.logger_level = logger_level
        self.timeseries_storage_client = self._init_storage_client(props)
        self.composite_detector = self._init_composite_detector(props)

    async def start(self):
        try:
            while True:
                start_time = time.time()
                result = await self._launch_async_pipe()  # Wait for async_chain to complete
                print(f"Result: {result}")
                elapsed_time = time.time() - start_time
                sleep_duration = max(0, 15 - elapsed_time)
                await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            self.logger.error("Application stopped. Try to save current state.")

    def _init_storage_client(self, props):
        return VictoriaMetricsClient(props.get_base_url(),
                                     props.get_queries(),
                                     props.get_steps())

    def _init_logger(self, logger_level, log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
        logger_name = self.__class__.__name__
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)
        ch = logging.StreamHandler()
        ch.setLevel(logger_level)
        formatter = logging.Formatter(log_format)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def _init_composite_detector(self, props):
        self.logger.debug(f"Initialize detectors with parameters {props.get_detectors_props()}")
        behavior_detector_ = BehaviorDetector(props.get_detectors_props().get_steps(),
                                              props.get_detectors_props().get_data_len(),
                                              props.get_detectors_props().set_behavior_detector_lstm_size(),
                                              props.get_detectors_props().get_behavior_detector_dropout_rate(),
                                              1,
                                              props.get_detectors_props().get_behavior_detector_train_shift(),
                                              props.get_detectors_props().get_behavior_detector_anomaly_metric_name(),
                                              self.logger_level
                                              )
        cusum_detector = CUSUMDetector(props.get_detectors_props().get_cusum_threshold(),
                                       props.get_detectors_props().get_cusum_drift(),
                                       props.get_detectors_props().get_cusum_window(),
                                       self.logger_level)

        mad_detector = MADDetector()

    def _launch_async_pipe(self):
        pass
