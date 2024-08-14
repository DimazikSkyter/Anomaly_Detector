from typing import List

import numpy as np

from anomaly.detector.metrics.Metrics import Metrics, Metric
from anomaly.detector.parts.CompositeStreamDetector import Detector


class MADDetector(Detector):
    """
    Для каждого входящего временного ряда смотрит насколько его локальный MAD отличается от глобального
     на временном отрезке
    """

    def __init__(self, window=10, data_max_size=50, logger_level="INFO"):
        super().__init__(logger_level=logger_level)
        self.data_max_size = data_max_size
        self.mad_values = []
        self.window = window
        self.logger.info("MAD detector successfully init.")

    def detect(self, metrics: Metrics) -> List[float]:
        timeseries = self._prepare_data(metrics)
        result_mad = None
        for seria in timeseries:
            seria_mads = self._calc_mad_score(seria)
            if result_mad:
                for index in range(len(seria_mads)):
                    if result_mad[index] < seria_mads[index]:
                        result_mad[index] = seria_mads[index]
            else:
                result_mad = seria_mads
        return result_mad

    def _calc_mad_score(self, seria) -> List[float]:
        if min(seria) == max(seria):
            return [0.00000001] * len(seria)
        global_mad = MADDetector._calculate_mad_in_window(seria)
        local_mads = self._calc_local_mad_score(seria, global_mad)
        relative_deviation_mads = [np.abs(local_mad - global_mad) / global_mad for local_mad in local_mads]
        return self.sigma_normalize(relative_deviation_mads, 1)

    def _calc_local_mad_score(self, data, global_mad) -> List[float]:
        window = self.window
        mads = [global_mad] * (window - 1)
        for i in range(len(data) - window + 1):
            mad = MADDetector._calculate_mad_in_window(data[i: i + window])
            mads.append(mad)
        return mads

    def _prepare_data(self, metrics):
        self.logger.debug("TYPE  %s", metrics.series.values())
        return metrics.series.values()

    @staticmethod
    def _calculate_mad_in_window(data_window) -> float:
        mean = np.mean(data_window)
        mad = sum(np.abs(data_window - mean)) / len(data_window)
        return mad
