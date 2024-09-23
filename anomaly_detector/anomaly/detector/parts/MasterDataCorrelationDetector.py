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
        self.logger.info("Correlation detector successfully init.")

    def detect(self, metrics: Metrics) -> List[float]:
        origin_metric_seria_ = metrics.series[self.origin_metric_name]
        origin_trend_ = pd.Series(self._extract_trend(self.origin_metric_name, origin_metric_seria_))
#        self.logger.debug("Original trend is {}", origin_trend_[self.origin_metric_name]) пофиксить
        trends_ = [(key, pd.Series(self._extract_trend(key, metric))) for key, metric in metrics.series.items() if
                   key != self.origin_metric_name]
        return self._collapse_vectors(
            [self._cal_windowed_correlation(origin_trend_, metric_trend, key) for key, metric_trend in trends_])

    def _extract_trend(self, metric_name, timeseries: List[float]) -> List[float]:
        timeseries_trend_ = self.ssa.fit_transform(np.array(timeseries).reshape(1, -1))[0]
        self.logger.debug("Extracted trend for %s of income timeseries %s to %s",
                          metric_name,
                          timeseries,
                          timeseries_trend_)
        return self.min_max_scaler(timeseries_trend_)

    def _cal_windowed_correlation(self, v1, v2, key):
        rolling_cov_ = v1.rolling(window=self.cov_window_size).corr(v2)
        self.logger.debug("Result covariance vector for original metric "
                          "%s and key %s is %s",
                          self.origin_metric_name,
                          key,
                          rolling_cov_)
        filled_cov_ = rolling_cov_.fillna(1)
        return filled_cov_

    @staticmethod
    def _collapse_vectors(cov_vectors_list):
        return [min(np.abs(values)) for values in zip(*cov_vectors_list)]
