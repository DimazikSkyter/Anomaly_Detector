import unittest

import random as ran

import numpy as np
from matplotlib import pyplot as plt

from anomaly.detector.metrics.Metrics import Metrics, Metric
from anomaly.detector.parts.MasterDataCorrelationDetector import CorrelationDetector


class MyTestCase(unittest.TestCase):
    def test_positive_correlation(self):
        origin_metric = "origin_metric"
        covariance_detector = CorrelationDetector(7, [[0]], 10,
                                                  origin_metric + '{tag1="tag1_value"}', logger_level="DEBUG")
        seria1 = self._generate_seria(100, 2, 10, 9)
        seria2 = self._generate_seria(100, 1.5, 6, 7)
        seria3 = self._generate_seria(100, 1.7, 15, 5)[::-1]
        timestamps = list(range(100))

        master_metric = Metric(None, origin_metric, {"tag1": "tag1_value"}, seria1, timestamps)
        metric_1 = Metric(None, "metric_ 1", {"tag1": "tag1_value"}, seria2, timestamps)
        metric_2 = Metric(None, "metric_2", {"tag1": "tag1_value"}, seria3, timestamps)
        metrics = Metrics([master_metric, metric_1, metric_2], 100)

        metrics.plot()

        cov_ = covariance_detector.detect(metrics)
        self._plot(timestamps, cov_)

    def test_negative_correlation(self):
        origin_metric = "origin_metric"
        covariance_detector = CorrelationDetector(7, [[0]], 10,
                                                  origin_metric + '{tag1="tag1_value"}', logger_level="DEBUG")
        seria1 = self._generate_seria(100, 2, 10, 9)
        seria2 = self._generate_seria(100, 1.5, 6, 7, True)
        seria3 = self._generate_seria(100, 1.7, 15, 5)
        timestamps = list(range(100))

        master_metric = Metric(None, origin_metric, {"tag1": "tag1_value"}, seria1, timestamps)
        metric_1 = Metric(None, "metric_ 1", {"tag1": "tag1_value"}, seria2, timestamps)
        metric_2 = Metric(None, "metric_2", {"tag1": "tag1_value"}, seria3, timestamps)
        metrics = Metrics([master_metric, metric_1, metric_2], 100)

        metrics.plot()

        cov_ = covariance_detector.detect(metrics)
        self._plot(timestamps, cov_)

    @staticmethod
    def _generate_seria(size, mult, amplitude, seasonal_period, is_xpow2=False):
        x = np.arange(size)  # independent variable
        if is_xpow2:
            x = (x - np.mean(x)) ** 2 / np.mean(x) ** 1
        trend = mult * x
        seasonal = amplitude * np.sin(2 * np.pi * x / seasonal_period)
        noise_std = 0.2 * amplitude  # 10% of the amplitude
        noise = np.random.normal(0, noise_std, size)
        return trend + seasonal + noise

    @staticmethod
    def _plot(x, y):
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, label='y = 2x + Seasonal + Noise')
        plt.xlabel('x')
        plt.ylim(0, 1.1)
        plt.ylabel('y')
        plt.title('Correlation vector')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
