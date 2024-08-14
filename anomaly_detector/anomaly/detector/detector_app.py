import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import yaml
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
class StartStopPeriods:
    _start: str
    _stop: str

    @staticmethod
    def split(start, stop) -> List['StartStopPeriods']:
        start_dt_ = datetime.fromisoformat(start)
        stop_dt_ = datetime.fromisoformat(stop)

        periods_ = []
        current_start_ = start_dt_

        while current_start_ < stop_dt_:
            current_stop_ = min(current_start_ + timedelta(hours=8), stop_dt_)
            period_ = StartStopPeriods(
                _start=current_start_.isoformat(),
                _stop=current_stop_.isoformat()
            )
            periods_.append(period_)
            current_start_ = current_stop_
        return periods_


@dataclass
class DetectorsProps:
    _data_len: int
    _mad_detector_window: int
    _behavior_detector_lstm_size: int
    _behavior_detector_dropout_rate: float
    _behavior_detector_train_shift: int
    _behavior_detector_anomaly_metric_name: str
    _behavior_detector_path: str
    _cusum_threshold: float
    _cusum_drift: float
    _cusum_window: int
    _anomaly_detector_lstm_size: int
    _anomaly_detector_dropout_rate: int
    _anomaly_detector_epochs: int
    _anomaly_detector_shift: int
    _anomaly_detector_path: str
    _anomaly_detector_seria_name: str
    _corr_detector_ssa_window_size: int
    _corr_detector_ssa_group: List[List[int]]
    _corr_detector_cov_window_size: int
    _corr_detector_origin_metric_name: str

    def get_data_len(self) -> int:
        return self._data_len

    def set_data_len(self, data_len: int) -> None:
        self._data_len = data_len

    def get_mad_detector_window(self) -> int:
        return self._mad_detector_window

    def set_mad_detector_window(self, mad_detector_window: int) -> None:
        self._mad_detector_window = mad_detector_window

    def get_behavior_detector_lstm_size(self) -> int:
        return self._behavior_detector_lstm_size

    def set_behavior_detector_lstm_size(self, behavior_detector_lstm_size: int) -> None:
        self._behavior_detector_lstm_size = behavior_detector_lstm_size

    def get_behavior_detector_dropout_rate(self) -> float:
        return self._behavior_detector_dropout_rate

    def set_behavior_detector_dropout_rate(self, behavior_detector_dropout_rate: float) -> None:
        self._behavior_detector_dropout_rate = behavior_detector_dropout_rate

    def get_behavior_detector_train_shift(self) -> int:
        return self._behavior_detector_train_shift

    def set_behavior_detector_train_shift(self, behavior_detector_train_shift: int) -> None:
        self._behavior_detector_train_shift = behavior_detector_train_shift

    def get_behavior_detector_anomaly_metric_name(self) -> str:
        return self._behavior_detector_anomaly_metric_name

    def set_behavior_detector_anomaly_metric_name(self, behavior_detector_anomaly_metric_name: str) -> None:
        self._behavior_detector_anomaly_metric_name = behavior_detector_anomaly_metric_name

    def get_behavior_detector_path(self) -> str:
        return self._behavior_detector_path

    def set_behavior_detector_path(self, behavior_detector_path: str) -> None:
        self._behavior_detector_path = behavior_detector_path

    def get_cusum_threshold(self) -> float:
        return self._cusum_threshold

    def set_cusum_threshold(self, cusum_threshold: float) -> None:
        self._cusum_threshold = cusum_threshold

    def get_cusum_drift(self) -> float:
        return self._cusum_drift

    def set_cusum_drift(self, cusum_drift: float) -> None:
        self._cusum_drift = cusum_drift

    def get_cusum_window(self) -> int:
        return self._cusum_window

    def set_cusum_window(self, cusum_window: int) -> None:
        self._cusum_window = cusum_window

    def get_anomaly_detector_lstm_size(self) -> int:
        return self._anomaly_detector_lstm_size

    def set_anomaly_detector_lstm_size(self, anomaly_detector_lstm_size: int) -> None:
        self._anomaly_detector_lstm_size = anomaly_detector_lstm_size

    def get_anomaly_detector_dropout_rate(self) -> int:
        return self._anomaly_detector_dropout_rate

    def set_anomaly_detector_dropout_rate(self, anomaly_detector_dropout_rate: int) -> None:
        self._anomaly_detector_dropout_rate = anomaly_detector_dropout_rate

    def get_anomaly_detector_epochs(self) -> int:
        return self._anomaly_detector_epochs

    def set_anomaly_detector_epochs(self, anomaly_detector_epochs: int) -> None:
        self._anomaly_detector_epochs = anomaly_detector_epochs

    def get_anomaly_detector_shift(self) -> int:
        return self._anomaly_detector_shift

    def set_anomaly_detector_shift(self, anomaly_detector_shift: int) -> None:
        self._anomaly_detector_shift = anomaly_detector_shift

    def get_anomaly_detector_path(self) -> str:
        return self._anomaly_detector_path

    def set_anomaly_detector_path(self, anomaly_detector_path: str) -> None:
        self._anomaly_detector_path = anomaly_detector_path

    def get_anomaly_detector_seria_name(self) -> str:
        return self._anomaly_detector_seria_name

    def set_anomaly_detector_seria_name(self, anomaly_detector_seria_name: str) -> None:
        self._anomaly_detector_seria_name = anomaly_detector_seria_name

    def get_corr_detector_ssa_window_size(self) -> int:
        return self._corr_detector_ssa_window_size

    def set_corr_detector_ssa_window_size(self, corr_detector_ssa_window_size: int) -> None:
        self._corr_detector_ssa_window_size = corr_detector_ssa_window_size

    def get_corr_detector_ssa_group(self) -> List[List[int]]:
        return self._corr_detector_ssa_group

    def set_corr_detector_ssa_group(self, corr_detector_ssa_group: List[List[int]]) -> None:
        self._corr_detector_ssa_group = corr_detector_ssa_group

    def get_corr_detector_cov_window_size(self) -> int:
        return self._corr_detector_cov_window_size

    def set_corr_detector_cov_window_size(self, corr_detector_cov_window_size: int) -> None:
        self._corr_detector_cov_window_size = corr_detector_cov_window_size

    def get_corr_detector_origin_metric_name(self) -> str:
        return self._corr_detector_origin_metric_name

    def set_corr_detector_origin_metric_name(self, corr_detector_origin_metric_name: str) -> None:
        self._corr_detector_origin_metric_name = corr_detector_origin_metric_name


@dataclass
class StorageProps:
    _base_url: str
    _steps: int

    def get_base_url(self) -> str:
        return self._base_url

    def set_base_url(self, base_url: str) -> None:
        self._base_url = base_url

    def get_steps(self) -> int:
        return self._steps

    def set_steps(self, steps: int) -> None:
        self._steps = steps


@dataclass
class MetricsInfo:
    _metrics_names: List[str]

    def get_metrics_names(self) -> List[str]:
        return self._metrics_names

    def set_metrics_name(self, metrics_names: List[str]) -> None:
        self._metrics_names = metrics_names


class AppProps:
    _logger_level: str
    _detector_pacing_sec: int
    _storage_props: StorageProps
    _detectors_props: DetectorsProps
    _metrics_info: MetricsInfo

    def load_from_local(self, file_path):
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)

        tmp_logger_level = config.get("logger_level")
        self._logger_level = tmp_logger_level if tmp_logger_level else "INFO"

        detector_pacing_sec = config.get("detector_pacing_sec")
        self._detector_pacing_sec = detector_pacing_sec if detector_pacing_sec else 30

        self._storage_props = StorageProps(
            _base_url=config.get("storage_base_url"),
            _steps=config.get("storage_steps")
        )

        self._detectors_props = DetectorsProps(
            _data_len=config.get("data_len"),
            _mad_detector_window=config.get("mad_detector_window"),
            _behavior_detector_lstm_size=config.get("behavior_detector_lstm_size"),
            _behavior_detector_dropout_rate=config.get("behavior_detector_dropout_rate"),
            _behavior_detector_train_shift=config.get("behavior_detector_train_shift"),
            _behavior_detector_anomaly_metric_name=config.get("behavior_detector_anomaly_metric_name"),
            _behavior_detector_path=config.get("behavior_detector_path"),
            _cusum_threshold=config.get("cusum_threshold"),
            _cusum_drift=config.get("cusum_drift"),
            _cusum_window=config.get("cusum_window"),
            _anomaly_detector_lstm_size=config.get("anomaly_detector_lstm_size"),
            _anomaly_detector_dropout_rate=config.get("anomaly_detector_dropout_rate"),
            _anomaly_detector_epochs=config.get("anomaly_detector_epochs"),
            _anomaly_detector_shift=config.get("anomaly_detector_shift"),
            _anomaly_detector_path=config.get("anomaly_detector_path"),
            _corr_detector_ssa_window_size=config.get("corr_detector_ssa_window_size"),
            _corr_detector_ssa_group=config.get("corr_detector_ssa_group"),
            _corr_detector_cov_window_size=config.get("corr_detector_cov_window_size"),
            _corr_detector_origin_metric_name=config.get("corr_detector_origin_metric_name")
        )

        self._metrics_info = MetricsInfo(
            _metrics_names=config.get("metrics_info_names", [])
        )

    def update_from_env(self):
        self._logger_level = str(os.getenv("LOGGER_LEVEL", self._logger_level))
        self._detector_pacing_sec = int(os.getenv("DETECTOR_PACING_SEC", self._detector_pacing_sec))

        self._storage_props.set_steps(int(os.getenv("STORAGE_STEPS", self._storage_props.get_steps())))
        self._storage_props.set_base_url(os.getenv("STORAGE_BASE_URL", self._storage_props.get_base_url()))

        self._detectors_props.set_data_len(int(os.getenv('DATA_LEN', self._detectors_props.get_data_len())))
        self._detectors_props.set_mad_detector_window(
            int(os.getenv('MAD_DETECTOR_WINDOW', self._detectors_props.get_mad_detector_window())))
        self._detectors_props.set_behavior_detector_lstm_size(
            int(os.getenv('BEHAVIOR_DETECTOR_LSTM_SIZE', self._detectors_props.get_behavior_detector_lstm_size())))
        self._detectors_props.set_behavior_detector_dropout_rate(
            float(os.getenv('BEHAVIOR_DETECTOR_DROPOUT_RATE',
                            self._detectors_props.get_behavior_detector_dropout_rate())))
        self._detectors_props.set_behavior_detector_train_shift(
            int(os.getenv('BEHAVIOR_DETECTOR_TRAIN_SHIFT', self._detectors_props.get_behavior_detector_train_shift())))
        self._detectors_props.set_behavior_detector_anomaly_metric_name(
            os.getenv('BEHAVIOR_DETECTOR_ANOMALY_METRIC_NAME',
                      self._detectors_props.get_behavior_detector_anomaly_metric_name()))
        self._detectors_props.set_behavior_detector_path(
            os.getenv('BEHAVIOR_DETECTOR_PATH', self._detectors_props.get_behavior_detector_path()))
        self._detectors_props.set_cusum_threshold(
            float(os.getenv('CUSUM_THRESHOLD', self._detectors_props.get_cusum_threshold())))
        self._detectors_props.set_cusum_drift(
            float(os.getenv('CUSUM_DRIFT', self._detectors_props.get_cusum_drift())))
        self._detectors_props.set_cusum_window(
            int(os.getenv('CUSUM_WINDOW', self._detectors_props.get_cusum_window())))
        self._detectors_props.set_anomaly_detector_lstm_size(
            int(os.getenv('ANOMALY_DETECTOR_LSTM_SIZE', self._detectors_props.get_anomaly_detector_lstm_size())))
        self._detectors_props.set_anomaly_detector_dropout_rate(
            int(os.getenv('ANOMALY_DETECTOR_DROPOUT_RATE', self._detectors_props.get_anomaly_detector_dropout_rate())))
        self._detectors_props.set_anomaly_detector_epochs(
            int(os.getenv('ANOMALY_DETECTOR_EPOCHS', self._detectors_props.get_anomaly_detector_epochs())))
        self._detectors_props.set_anomaly_detector_shift(
            int(os.getenv('ANOMALY_DETECTOR_SHIFT', self._detectors_props.get_anomaly_detector_shift())))
        self._detectors_props.set_anomaly_detector_path(
            os.getenv('ANOMALY_DETECTOR_PATH', self._detectors_props.get_anomaly_detector_path()))
        self._detectors_props.set_corr_detector_ssa_window_size(
            int(os.getenv('CORR_DETECTOR_SSA_WINDOW_SIZE', self._detectors_props.get_corr_detector_ssa_window_size())))
        self._detectors_props.set_corr_detector_ssa_group(
            os.getenv('CORR_DETECTOR_SSA_GROUP', self._detectors_props.get_corr_detector_ssa_group()))
        self._detectors_props.set_corr_detector_cov_window_size(
            int(os.getenv('CORR_DETECTOR_COV_WINDOW_SIZE', self._detectors_props.get_corr_detector_cov_window_size())))
        self._detectors_props.set_corr_detector_origin_metric_name(
            os.getenv('CORR_DETECTOR_ORIGIN_METRIC_NAME', self._detectors_props.get_corr_detector_origin_metric_name()))

    def get_logger_level(self) -> str:
        return self._logger_level

    def set_logger_level(self, logger_level: str) -> None:
        self._logger_level = logger_level

    def get_detector_pacing_sec(self) -> int:
        return self._detector_pacing_sec

    def set_detector_pacing_sec(self, detector_pacing_sec: int) -> None:
        self._detector_pacing_sec = detector_pacing_sec

    def get_storage_props(self) -> StorageProps:
        return self._storage_props

    def set_storage_props(self, storage_props: StorageProps) -> None:
        self._storage_props = storage_props

    def get_detectors_props(self) -> DetectorsProps:
        return self._detectors_props

    def set_detectors_props(self, detectors_props: DetectorsProps) -> None:
        self._detectors_props = detectors_props

    def get_metrics_info(self) -> MetricsInfo:
        return self._metrics_info

    def set_metrics_info(self, metrics_info: MetricsInfo) -> None:
        self._metrics_info = metrics_info


# все метрики уже должны быть от 0 до 1
class App:
    def __init__(self, app_props: AppProps, flask_app):
        self._init_logger(app_props.get_logger_level())
        self.app_props = app_props
        self.timeseries_storage_client = self._init_storage_client()
        self.composite_detector = self._init_composite_detector()
        self.anomaly_detector = self._init_anomaly_detector()
        self.converter = App._get_converter()
        self.flask_app = flask_app
        self.final_result = []
        self.add_routes()
        self.is_work_mode = False

    async def start(self):
        try:
            while True:
                start_time = time.time()
                self.logger.debug(f"Start new cycle in {start_time}")
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
        try:
            period_ = data_.get('period')
            subperiods_ = data_.get('subperiods')
            if self._retrain(period_, subperiods_):
                return jsonify({'status': 'retrain initiated'})
            else:
                return jsonify({'status': 'nothing to train'})
        except Exception as e:
            self.logger.error(f"Catch error while trying to retrain model with income data {data_}", exc_info=True)
            return jsonify({'status': 'Failed to train'})

    def _retrain(self, period_, subperiods_):
        self.is_work_mode = False
        periods_: List[StartStopPeriods] = StartStopPeriods.split(period_['start'], period_['stop'])
        if len(periods_) < 1:
            self.logger.warning("Nothing to retrain in period %s and sub periods %s", period_, subperiods_)
            return False
        metrics_body_ = [self.timeseries_storage_client.get_metrics(period) for period in periods_]
        metrics_list_: List[Metrics] = [self.converter.convert(body,
                                                               100000,
                                                               self.app_props.get_storage_props().get_steps())
                                        for body in metrics_body_]
        metrics_: Metrics = self._union_metrics(metrics_list_)
        self.composite_detector.train(metrics_)
        split_metrics_: List[Metrics] = self.split_metrics(metrics_)
        split_composite_layer_result_: List[Metrics] = [self.composite_detector.detect(metrics)
                                                        for metrics in split_metrics_]
        self.add_anomaly_seria(split_composite_layer_result_, subperiods_)
        for composite_layer_result in split_composite_layer_result_:
            self.anomaly_detector.train(composite_layer_result)
        self.is_work_mode = True

    def _union_metrics(self, other_metrics: List[Metrics]):
        metrics: Metrics = other_metrics[0].copy_cut_off(is_copy=True)
        for metrics_ in other_metrics[1:]:
            metrics.union(metrics_)
        return metrics

    def split_metrics(self, metrics):
        data_len = self.app_props.get_detectors_props().get_data_len()
        parts = metrics.series_length() // data_len + 1
        return [metrics.get_part(part * data_len, min((part + 1) * data_len, metrics.series_length()))
                for part in range(parts)]

    def add_anomaly_seria(self, metrics_list: List[Metrics], subperiods):
        timestamps_periods_: List[List[float]] = [
            [datetime.timestamp(datetime.fromisoformat(subperiod['start'])),
             datetime.timestamp(datetime.fromisoformat(subperiod['stop']))]
            for subperiod in subperiods]
        for metrics in metrics_list:
            timestamps_ = metrics.timestamps
            len_ = len(timestamps_)
            anomaly_seria = [0] * len_
            for index in range(len_):
                timestamp_ = timestamps_[index]
                if any([timestamp_start_stop[0] < timestamp_ < timestamp_start_stop[1]
                        for timestamp_start_stop in timestamps_periods_]):
                    anomaly_seria[index] = 1
            metrics.add_seria(AnomalyDetector.ANOMALY_KEY, anomaly_seria)

    def add_routes(self):
        self.flask_app.add_url_rule('/actuator/prometheus', 'prometheus_actuator', self.prometheus_actuator,
                                    methods=['GET'])
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
                               self.app_props.get_detectors_props().get_anomaly_detector_seria_name(),
                               self.app_props.get_detectors_props().get_anomaly_detector_lstm_size(),
                               self.app_props.get_detectors_props().get_anomaly_detector_dropout_rate(),
                               self.app_props.get_detectors_props().get_data_len(),
                               self.app_props.get_detectors_props().get_anomaly_detector_epochs(),
                               self.app_props.get_detectors_props().get_anomaly_detector_shift,
                               1,
                               self.app_props.get_detectors_props().get_anomaly_detector_path())

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


async def main():
    flask_app_ = Flask(__name__)
    app_props_ = AppProps()
    app_props_.load_from_local("application.yml")
    app_props_.update_from_env()
    app = App(app_props_, flask_app_)
    await app.start()


if __name__ == "__main__":
    asyncio.run(main())
