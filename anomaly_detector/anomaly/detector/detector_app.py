import asyncio
import logging
import os
import time
import yaml
from dataclasses import dataclass
from typing import List

from flask import Flask, jsonify, request

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
    __steps: int

    def get_base_url(self) -> str:
        return self.__base_url

    def set_base_url(self, base_url: str) -> None:
        self.__base_url = base_url

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

    def load_from_local(self, file_path):
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)

        tmp_logger_level = config.get("logger_level")
        self.__logger_level = tmp_logger_level if tmp_logger_level else "INFO"

        detector_pacing_sec = config.get("detector_pacing_sec")
        self.__detector_pacing_sec = detector_pacing_sec if detector_pacing_sec else 30

        self.__storage_props = StorageProps(
            __base_url=config.get("storage_base_url"),
            __steps=config.get("storage_steps")
        )

        self.__detectors_props = DetectorsProps(
            __data_len=config.get("data_len"),
            __mad_detector_window=config.get("mad_detector_window"),
            __behavior_detector_lstm_size=config.get("behavior_detector_lstm_size"),
            __behavior_detector_dropout_rate=config.get("behavior_detector_dropout_rate"),
            __behavior_detector_train_shift=config.get("behavior_detector_train_shift"),
            __behavior_detector_anomaly_metric_name=config.get("behavior_detector_anomaly_metric_name"),
            __behavior_detector_path=config.get("behavior_detector_path"),
            __cusum_threshold=config.get("cusum_threshold"),
            __cusum_drift=config.get("cusum_drift"),
            __cusum_window=config.get("cusum_window"),
            __anomaly_detector_lstm_size=config.get("anomaly_detector_lstm_size"),
            __anomaly_detector_dropout_rate=config.get("anomaly_detector_dropout_rate"),
            __anomaly_detector_epochs=config.get("anomaly_detector_epochs"),
            __anomaly_detector_shift=config.get("anomaly_detector_shift"),
            __anomaly_detector_path=config.get("anomaly_detector_path"),
            __corr_detector_ssa_window_size=config.get("corr_detector_ssa_window_size"),
            __corr_detector_ssa_group=config.get("corr_detector_ssa_group"),
            __corr_detector_cov_window_size=config.get("corr_detector_cov_window_size"),
            __corr_detector_origin_metric_name=config.get("corr_detector_origin_metric_name")
        )

        self.__metrics_info = MetricsInfo(
            __metrics_names=config.get("metrics_info_names", [])
        )

    def update_from_env(self):
        self.__logger_level = str(os.getenv("LOGGER_LEVEL", self.__logger_level))
        self.__detector_pacing_sec = int(os.getenv("DETECTOR_PACING_SEC"), self.__detector_pacing_sec)

        self.__storage_props.set_steps(os.getenv("STORAGE_STEPS", self.__storage_props.get_steps()))
        self.__storage_props.set_base_url(os.getenv("STORAGE_BASE_URL", self.__storage_props.get_base_url()))

        self.__detectors_props.set_data_len(int(os.getenv('DATA_LEN', self.__detectors_props.get_data_len())))
        self.__detectors_props.set_mad_detector_window(
            int(os.getenv('MAD_DETECTOR_WINDOW',
                          self.__detectors_props.get_mad_detector_window())))
        self.__detectors_props.set_behavior_detector_lstm_size(
            int(os.getenv('BEHAVIOR_DETECTOR_LSTM_SIZE',
                          self.get_detectors_props().get_behavior_detector_lstm_size())))
        self.__detectors_props.set_behavior_detector_dropout_rate(
            float(os.getenv('BEHAVIOR_DETECTOR_DROPOUT_RATE',
                            self.__detectors_props.get_behavior_detector_dropout_rate())))
        self.__detectors_props.set_behavior_detector_train_shift(
            int(os.getenv('BEHAVIOR_DETECTOR_TRAIN_SHIFT',
                          self.__detectors_props.get_behavior_detector_train_shift())))
        self.__detectors_props.set_behavior_detector_anomaly_metric_name(
            os.getenv('BEHAVIOR_DETECTOR_ANOMALY_METRIC_NAME',
                      self.__detectors_props.get_behavior_detector_anomaly_metric_name()))
        self.__detectors_props.set_behavior_detector_path(
            os.getenv('BEHAVIOR_DETECTOR_PATH', self.__detectors_props.get_behavior_detector_path()))
        self.__detectors_props.set_cusum_threshold(os.getenv(
            'CUSUM_THRESHOLD',
            self.__detectors_props.get_cusum_threshold()))
        self.__detectors_props.set_cusum_drift(os.getenv('CUSUM_DRIFT', self.__detectors_props.get_cusum_drift()))
        self.__detectors_props.set_cusum_window(os.getenv('CUSUM_WINDOW', self.__detectors_props.get_cusum_window()))
        self.__detectors_props.set_anomaly_detector_lstm_size(
            os.getenv('ANOMALY_DETECTOR_LSTM_SIZE', self.__detectors_props.get_anomaly_detector_lstm_size()))
        self.__detectors_props.set_anomaly_detector_dropout_rate(
            os.getenv('ANOMALY_DETECTOR_DROPOUT_RATE', self.__detectors_props.get_anomaly_detector_dropout_rate()))
        self.__detectors_props.set_anomaly_detector_epochs(
            os.getenv('ANOMALY_DETECTOR_EPOCHS', self.__detectors_props.get_anomaly_detector_epochs()))
        self.__detectors_props.set_anomaly_detector_shift(
            os.getenv('ANOMALY_DETECTOR_SHIFT', self.__detectors_props.get_anomaly_detector_shift()))
        self.__detectors_props.set_anomaly_detector_path(
            os.getenv('ANOMALY_DETECTOR_PATH', self.__detectors_props.get_anomaly_detector_path()))
        self.__detectors_props.set_corr_detector_ssa_window_size(
            os.getenv('CORR_DETECTOR_SSA_WINDOW_SIZE', self.__detectors_props.get_corr_detector_ssa_window_size()))
        self.__detectors_props.set_corr_detector_ssa_group(
            os.getenv('CORR_DETECTOR_SSA_GROUP', self.__detectors_props.get_corr_detector_ssa_group()))
        self.__detectors_props.set_corr_detector_cov_window_size(
            os.getenv('CORR_DETECTOR_COV_WINDOW_SIZE', self.__detectors_props.get_corr_detector_cov_window_size()))
        self.__detectors_props.set_corr_detector_origin_metric_name(os.getenv(
            'CORR_DETECTOR_ORIGIN_METRIC_NAME', self.__detectors_props.get_corr_detector_origin_metric_name()))

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
    def __init__(self, app_props: AppProps, flask_app):
        self._init_logger(app_props_.get_logger_level())
        self.app_props = app_props
        self.timeseries_storage_client = self._init_storage_client()
        self.composite_detector = self._init_composite_detector()
        self.anomaly_detector = self._init_anomaly_detector()
        self.converter = App._get_converter()
        self.flask_app = flask_app
        self.final_result = []
        self.add_routes()
        self.is_work_mode = True #перенести в пропсы

    async def start(self):
        try:
            while True:
                start_time = time.time()
                if self.is_work_mode:
                    await self._launch_async_pipe()
                elapsed_time = time.time() - start_time
                sleep_duration = max(0., self.app_props.get_detector_pacing_sec() - elapsed_time)
                await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            self.logger.error("Application stopped. Try to save current state.")
            self.save_current_models_state()

    def save_current_models_state(self):
        self.composite_detector.save_model()
        self.anomaly_detector.save_model()

    def prometheus_actuator(self):
        return jsonify({
            'final_result': self.final_result[-1]
        })

    def retrain_endpoint(self):
        data_ = request.json
        period_ = data_.get('period')
        subperiods_ = data_.get('subperiods')
        self._retrain(period_, subperiods_)
        return jsonify({'status': 'retrain initiated'})

    def _retrain(self, period_, subperiods_):
        #Переводим work_mode в False
        #generate_metrics
        #get_data from this period
        #переводим work_mode в True
        pass

    def add_routes(self):
        self.flask_app.add_url_rule('/actuator/prometheus', 'prometheus_actuator', self.prometheus_actuator, methods=['GET'])
        self.flask_app.add_url_rule('/retrain', 'retrain_endpoint', self.retrain_endpoint, methods=['POST'])

    def _init_storage_client(self):
        return VictoriaMetricsClient(
            self.app_props.get_storage_props().get_base_url(),
            self.app_props.get_metrics_info().get_metrics_names())

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
        self.logger.info(f"Initialize detectors with parameters {self.app_props.get_detectors_props()}")
        behavior_detector_ = BehaviorDetector(
            # Currently all detectors work with all metrics
            len(self.app_props.get_metrics_info().get_metrics_names()),
            self.app_props.get_detectors_props().get_behavior_detector_path(),
            False,
            self.app_props.get_detectors_props().get_data_len(),
            self.app_props.get_detectors_props().get_behavior_detector_lstm_size(),
            self.app_props.get_detectors_props().get_behavior_detector_dropout_rate(),
            1,
            self.app_props.get_detectors_props().get_behavior_detector_train_shift(),
            self.app_props.get_detectors_props().get_behavior_detector_anomaly_metric_name(),
            self.app_props.get_logger_level())

        correlation_detector_ = CorrelationDetector(
            self.app_props.get_detectors_props().get_corr_detector_ssa_window_size(),
            self.app_props.get_detectors_props().get_corr_detector_ssa_group(),
            self.app_props.get_detectors_props().get_corr_detector_cov_window_size(),
            self.app_props.get_detectors_props().get_corr_detector_origin_metric_name(),
            self.app_props.get_logger_level())

        cusum_detector_ = CUSUMDetector(
            self.app_props.get_detectors_props().get_cusum_threshold(),
            self.app_props.get_detectors_props().get_cusum_drift(),
            self.app_props.get_detectors_props().get_cusum_window(),
            self.app_props.get_logger_level())

        mad_detector_ = MADDetector(
            self.app_props.get_detectors_props().get_mad_detector_window(),
            self.app_props.get_detectors_props().get_data_len(),
            self.app_props.get_logger_level())

        composite_detector_ = CompositeDetector([behavior_detector_,
                                                 correlation_detector_,
                                                 cusum_detector_,
                                                 mad_detector_],
                                                self.app_props.get_logger_level())
        composite_detector_.load_model()
        return composite_detector_

    def _init_anomaly_detector(self) -> AnomalyDetector:
        return AnomalyDetector(4,
                               self.app_props.get_detectors_props().get_anomaly_detector_lstm_size(),
                               self.app_props.get_detectors_props().get_anomaly_detector_dropout_rate(),
                               self.app_props.get_detectors_props().get_data_len(),
                               self.app_props.get_detectors_props().get_anomaly_detector_epochs(),
                               self.app_props.get_detectors_props().get_anomaly_detector_shift)

    async def _launch_async_pipe(self):
        try:
            step = self.app_props.get_storage_props().get_steps()
            metrics_json = self.timeseries_storage_client.get_metrics(step)
            metrics: Metrics = self.converter.convert(metrics_json,
                                                      self.app_props.get_detectors_props().get_data_len(),
                                                      step)
            self.logger.debug(f"Metrics from storage {metrics}")
            result_metrics: Metrics = self.composite_detector.detect(metrics)
            self.logger.debug(f"Result metrics: {result_metrics}")
            final_result: List[float] = self.anomaly_detector.detect(result_metrics)
            self.final_result = final_result
            self.logger.info(f"Final result {final_result}")
        except Exception as e:
            self.logger.error("Catch exception while", exc_info=True)

    @staticmethod
    def _get_converter() -> MetricConverter:
        return MetricConverter()


flask_app_ = Flask(__name__)
app_props_ = AppProps()
app_props_.load_from_local("application.yml")
app_props_.update_from_env()
App(app_props_, flask_app_).start()
