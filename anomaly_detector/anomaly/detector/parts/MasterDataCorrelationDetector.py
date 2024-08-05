from typing import List

import numpy as np
import pandas as pd
from pyts.decomposition import SingularSpectrumAnalysis

from anomaly.detector.metrics.Metrics import Metrics
from anomaly.detector.parts.CompositeStreamDetector import Detector


class CorrelationDetector(Detector):
    def __init__(self, ssa_window_size, ssa_group, cov_window_size, origin_metric_name, logger_level="INFO"):
        super().__init__(logger_level=logger_level)
        self.ssa = SingularSpectrumAnalysis(window_size=ssa_window_size, groups=ssa_group)
        self.cov_window_size = cov_window_size
        self.origin_metric_name = origin_metric_name

    def detect(self, metrics: Metrics) -> List[float]:
        origin_metric_name_ = metrics.series[self.origin_metric_name]
        origin_trend_ = pd.Series(self._extract_trend(origin_metric_name_))
        trends_ = [(key, pd.Series(self._extract_trend(metric))) for key, metric in metrics.series.items() if
                   key != self.origin_metric_name]
        return self._collapse_vectors(
            [self.cal_windowed_covariance(origin_trend_, metric_trend, key) for key, metric_trend in trends_])

    def _extract_trend(self, timeseries: List[float]) -> List[float]:
        timeseries_trend_ = self.ssa.fit_transform(np.array(timeseries).reshape(1, -1))[0]
        self.logger.debug(f"Extracted trend of income timeseries {timeseries} to {timeseries_trend_}")
        return self.min_max_scaler(timeseries_trend_)

    def cal_windowed_covariance(self, v1, v2, key):
        rolling_cov_ = v1.rolling(window=self.cov_window_size).corr(v2)
        self.logger.debug("Result covariance vector for original metric "
                          f"{self.origin_metric_name} and key {key} is {rolling_cov_}")
        filled_cov = rolling_cov_.fillna(1)
        return filled_cov

    @staticmethod
    def _collapse_vectors(cov_vectors_list):
        return [min(np.abs(values)) for values in zip(*cov_vectors_list)]
