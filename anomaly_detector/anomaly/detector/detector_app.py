import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List

from anomaly.detector.clients.StorageClients import VictoriaMetricsClient
from anomaly.detector.converter.MetricConverter import MetricConverter
from anomaly.detector.metrics.Metrics import Metrics
from anomaly.detector.parts.AnomalyDetector import AnomalyDetector
from anomaly.detector.parts.BehaviorDetector import BehaviorDetector
from anomaly.detector.parts.CompositeStreamDetector import CompositeDetector
from anomaly.detector.parts.CusumDetector import CUSUMDetector
from anomaly.detector.parts.MasterDataCorrelationDetector import CorrelationDetector
from anomaly.detector.parts.WindowedMadDetector import MADDetector


# Пока что все метрики участвуют во всех детекторах, однако это нужно переделать

@dataclass
class DetectorsProps:
    __data_len: int
    __mad_detector_window: int
    __behavior_detector_lstm_size: int
    __behavior_detector_dropout_rate: float
    __behavior_detector_train_shift: int
    __behavior_detector_anomaly_metric_name: str
    __behavior_detector_path: str
    __cusum_threshold: float
    __cusum_drift: float
    __cusum_window: int
    __anomaly_detector_lstm_size: int
    __anomaly_detector_dropout_rate: int
    __anomaly_detector_epochs: int
    __anomaly_detector_shift: int
    __anomaly_detector_path: str
    __corr_detector_ssa_window_size: int
    __corr_detector_ssa_group: List[List[int]]
    __corr_detector_cov_window_size: int
    __corr_detector_origin_metric_name: str

    def get_data_len(self) -> int:
        return self.__data_len

    def set_data_len(self, data_len: int) -> None:
        self.__data_len = data_len

    def get_mad_detector_window(self) -> int:
        return self.__mad_detector_window

    def set_mad_detector_window(self, mad_detector_window: int) -> None:
        self.__mad_detector_window = mad_detector_window

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

    def get_behavior_detector_path(self) -> str:
        return self.__behavior_detector_path

    def set_behavior_detector_path(self, behavior_detector_path: str) -> None:
        self.__behavior_detector_path = behavior_detector_path

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

    def get_anomaly_detector_dropout_rate(self) -> int:
        return self.__anomaly_detector_dropout_rate

    def set_anomaly_detector_dropout_rate(self, anomaly_detector_dropout_rate: int) -> None:
        self.__anomaly_detector_dropout_rate = anomaly_detector_dropout_rate

    def get_anomaly_detector_epochs(self) -> int:
        return self.__anomaly_detector_epochs

    def set_anomaly_detector_epochs(self, anomaly_detector_epochs: int) -> None:
        self.__anomaly_detector_epochs = anomaly_detector_epochs

    def get_anomaly_detector_shift(self) -> int:
        return self.__anomaly_detector_shift

    def set_anomaly_detector_shift(self, anomaly_detector_shift: int) -> None:
        self.__anomaly_detector_shift = anomaly_detector_shift

    def get_anomaly_detector_path(self) -> str:
        return self.__anomaly_detector_path

    def set_anomaly_detector_path(self, anomaly_detector_path: str) -> None:
        self.__anomaly_detector_path = anomaly_detector_path

    def get_corr_detector_ssa_window_size(self) -> int:
        return self.__corr_detector_ssa_window_size

    def set_corr_detector_ssa_window_size(self, corr_detector_ssa_window_size: int) -> None:
        self.__corr_detector_ssa_window_size = corr_detector_ssa_window_size

    def get_corr_detector_ssa_group(self) -> List[List[int]]:
        return self.__corr_detector_ssa_group

    def set_corr_detector_ssa_group(self, corr_detector_ssa_group: List[List[int]]) -> None:
        self.__corr_detector_ssa_group = corr_detector_ssa_group

    def get_corr_detector_cov_window_size(self) -> int:
        return self.__corr_detector_cov_window_size

    def set_corr_detector_cov_window_size(self, corr_detector_cov_window_size: int) -> None:
        self.__corr_detector_cov_window_size = corr_detector_cov_window_size

    def get_corr_detector_origin_metric_name(self) -> str:
        return self.__corr_detector_origin_metric_name

    def set_corr_detector_origin_metric_name(self, corr_detector_origin_metric_name: str) -> None:
        self.__corr_detector_origin_metric_name = corr_detector_origin_metric_name


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
class MetricsInfo:
    __metrics_names: List[str]

    def get_metrics_names(self) -> List[str]:
        return self.__metrics_names

    def set_metrics_name(self, metrics_names: List[str]) -> None:
        self.__metrics_names = metrics_names


class AppProps:
    __logger_level: str
    __detector_pacing_sec: int
    __storage_props: StorageProps
    __detectors_props: DetectorsProps
    __metrics_info: MetricsInfo

    def load_from_local(self):
        pass

    def update_from_env(self):
        pass

    def get_logger_level(self) -> str:
        return self.__logger_level

    def set_logger_level(self, logger_level: str) -> None:
        self.__logger_level = logger_level

    def get_detector_pacing_sec(self) -> int:
        return self.__detector_pacing_sec

    def set_detector_pacing_sec(self, detector_pacing_sec: int) -> None:
        self.__detector_pacing_sec = detector_pacing_sec

    def get_storage_props(self) -> StorageProps:
        return self.__storage_props

    def set_storage_props(self, storage_props: StorageProps) -> None:
        self.__storage_props = storage_props

    def get_detectors_props(self) -> DetectorsProps:
        return self.__detectors_props

    def set_detectors_props(self, detectors_props: DetectorsProps) -> None:
        self.__detectors_props = detectors_props

    def get_metrics_info(self) -> MetricsInfo:
        return self.__metrics_info

    def set_metrics_info(self, metrics_info: MetricsInfo) -> None:
        self.__metrics_info = metrics_info


# все метрики уже должны быть от 0 до 1
class App:
    def __init__(self, props: AppProps):
        self._init_logger(app_props.get_logger_level())
        self.props = props
        self.timeseries_storage_client = self._init_storage_client()
        self.composite_detector = self._init_composite_detector()
        self.anomaly_detector = self._init_anomaly_detector()
        self.converter = App._get_converter()

    async def start(self):
        try:
            while True:
                start_time = time.time()
                await self._launch_async_pipe()
                elapsed_time = time.time() - start_time
                sleep_duration = max(0., self.props.get_detector_pacing_sec() - elapsed_time)
                await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            self.logger.error("Application stopped. Try to save current state.")
            self.save_current_models_state()

    def save_current_models_state(self):
        pass

    def _init_storage_client(self):
        return VictoriaMetricsClient(
            self.props.get_storage_props().get_base_url(),
            self.props.get_storage_props().get_queries())

    def _init_logger(self, logger_level, log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
        logger_name = self.__class__.__name__
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)
        ch = logging.StreamHandler()
        ch.setLevel(logger_level)
        formatter = logging.Formatter(log_format)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def _init_composite_detector(self):
        self.logger.info(f"Initialize detectors with parameters {self.props.get_detectors_props()}")
        behavior_detector_ = BehaviorDetector(
            # Currently all detectors work with all metrics
            len(self.props.get_metrics_info().get_metrics_names()),
            self.props.get_detectors_props().get_behavior_detector_path(),
            False,
            self.props.get_detectors_props().get_data_len(),
            self.props.get_detectors_props().get_behavior_detector_lstm_size(),
            self.props.get_detectors_props().get_behavior_detector_dropout_rate(),
            1,
            self.props.get_detectors_props().get_behavior_detector_train_shift(),
            self.props.get_detectors_props().get_behavior_detector_anomaly_metric_name(),
            self.props.get_logger_level())

        correlation_detector_ = CorrelationDetector(
            self.props.get_detectors_props().get_corr_detector_ssa_window_size(),
            self.props.get_detectors_props().get_corr_detector_ssa_group(),
            self.props.get_detectors_props().get_corr_detector_cov_window_size(),
            self.props.get_detectors_props().get_corr_detector_origin_metric_name(),
            self.props.get_logger_level())

        cusum_detector_ = CUSUMDetector(
            self.props.get_detectors_props().get_cusum_threshold(),
            self.props.get_detectors_props().get_cusum_drift(),
            self.props.get_detectors_props().get_cusum_window(),
            self.props.get_logger_level())

        mad_detector_ = MADDetector(
            self.props.get_detectors_props().get_mad_detector_window(),
            self.props.get_detectors_props().get_data_len(),
            self.props.get_logger_level())

        return CompositeDetector([behavior_detector_,
                                  correlation_detector_,
                                  cusum_detector_,
                                  mad_detector_],
                                 self.props.get_logger_level())

    def _init_anomaly_detector(self) -> AnomalyDetector:
        return AnomalyDetector(4,
                               self.props.get_detectors_props().get_anomaly_detector_lstm_size(),
                               self.props.get_detectors_props().get_anomaly_detector_dropout_rate(),
                               self.props.get_detectors_props().get_data_len(),
                               self.props.get_detectors_props().get_anomaly_detector_epochs(),
                               self.props.get_detectors_props().get_anomaly_detector_shift)

    async def _launch_async_pipe(self):
        try:
            step = self.props.get_storage_props().get_steps()
            metrics_json = self.timeseries_storage_client.get_metrics(step)
            metrics: Metrics = self.converter.convert(metrics_json,
                                                      self.props.get_detectors_props().get_data_len(),
                                                      step)
            self.logger.debug(f"Metrics from storage {metrics}")
            result_metrics: Metrics = self.composite_detector.detect(metrics)
            self.logger.debug(f"Result metrics: {result_metrics}")
            final_result: List[float] = self.anomaly_detector.detect(result_metrics)
            self.logger.debug(f"Final result {final_result}")
        except Exception as e:
            self.logger.error("Catch exception while", exc_info=True)

    @staticmethod
    def _get_converter() -> MetricConverter:
        return MetricConverter()


app_props = AppProps()
app_props.load_from_local()
app_props.update_from_env()
App(app_props).start()
