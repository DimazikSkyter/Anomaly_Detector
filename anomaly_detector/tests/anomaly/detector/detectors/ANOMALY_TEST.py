import unittest
from typing import List, Dict, Set

import numpy as np
import random

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from anomaly.detector.metrics.Metrics import Metrics, Metric
from anomaly.detector.parts.AnomalyDetector import AnomalyDetector


class MyTestCase(unittest.TestCase):

    def test_1(self):
        len_ = 10000
        metric_len_ = 100
        anomaly_seria_name = "anomaly"
        anomaly_seria_fullname = anomaly_seria_name + "{}"

        f1 = self._func1(len_)
        f2 = self._func2(len_)
        f3 = self._func3(len_)

        s1: Set[int] = self._anomaly(0.7, f1)
        s2: Set[int] = self._anomaly(0.7, f2)
        s3: Set[int] = self._anomaly(0.7, f3)

        anomaly_ = self._generate_anomalies_in_series(len_, s1.union(s2).union(s3))

        anomaly_detector = AnomalyDetector(3, anomaly_seria_fullname, lstm_size=512, dropout_rate=0.4,
                                           data_len=metric_len_, window_step=1, metrics_values_max_size=10000, logger_level="DEBUG")

        metrics: List[Metrics] = self._prepare_data(f1, f2, f3, anomaly_, anomaly_seria_name, metric_len_)

        train_part_len_ = round(0.75 * len(metrics))
        test_part_len_ = len(metrics) - train_part_len_

        for i in range(train_part_len_):
            if 1 in metrics[i].series[anomaly_seria_fullname]:
                anomaly_detector.train(metrics[i])

        total_sum_ = 0
        anomaly_result_seria_ = [0] * round(0.75 * len_)
        is_first_ = True

        for i in range(test_part_len_ - 1):
            iteration_ = train_part_len_ + i
            cur_metrics_: Metrics = metrics[iteration_]
            anomaly_expected_seria_ = cur_metrics_.series.pop(anomaly_seria_fullname)
            result_: list[float] = anomaly_detector.detect(cur_metrics_)
            print(f"For iteration {iteration_}"
                  f"income seria '{cur_metrics_.series}'"
                  f" of test detect the result is '{result_}'"
                  f" and expected anomaly seria is '{anomaly_expected_seria_}'")
            sum_ = np.sum(np.abs(np.array(anomaly_expected_seria_) - np.array(result_)))
            total_sum_ += sum_
            print(f"The sum of iteration {iteration_} is '{sum_}'")
            if is_first_:
                anomaly_result_seria_.extend(result_)
                is_first_ = False
            else:
                anomaly_result_seria_.extend(result_[round(metric_len_ / 2):])

        print(f"The test is finish. Total sum is {total_sum_}")
        print(f"Len is {len(anomaly_result_seria_)}")
        self._plot_series(f1, f2, f3, anomaly_, anomaly_result_seria_)

    def _plot_series(self, f1, f2, f3, ae, ar):
        fig, ax = plt.subplots(2, 1, figsize=(20, 16))
        ax_f = ax[0]
        ax_a = ax[1]
        ax_f.plot(f1, label="f1")
        ax_f.plot(f2, label="f2")
        ax_f.plot(f3, label="f3")
        ax_f.legend()
        ax_f.set_title("Series")
        ax_a.plot(ae, linestyle="-", label="expected anomaly seria")
        ax_a.plot(ar, linestyle="-.", label="result anomaly seria")
        ax_a.set_title("Anomalies")
        ax_a.legend()
        plt.show()

    def test_generate_data(self):
        len_ = 10000
        f1 = self._func1(len_)
        f2 = self._func2(len_)
        f3 = self._func3(len_)

        s1: Set[int] = self._anomaly(0.7, f1)
        s2: Set[int] = self._anomaly(0.7, f2)
        s3: Set[int] = self._anomaly(0.7, f3)

        anomaly_ = self._generate_anomalies_in_series(len_, s1.union(s2).union(s3))

        fig, ax = plt.subplots(2, 1, figsize=(32, 16))
        ax[0].plot(f1, label="f1")
        ax[0].plot(f2, label="f2")
        ax[0].plot(f3, label="f3")
        ax[0].set_title("Series")
        ax[1].plot(anomaly_, label="anomaly")
        ax[1].set_title("Anomaly")
        plt.legend()
        plt.show()

    def _func1(self, len_: int):
        y1 = np.sin(np.array(range(0, len_, 1)) / 10)
        y2 = np.sin(np.array(range(10, len_ + 10, 1)) / 30)
        noise = np.random.normal(0, 1, len_)
        return np.round((y1 * 5 + y2 * 3 + noise + 20) / 100, 3)

    def _func2(self, len_: int):
        y1 = np.array(range(0, len_, 1)) / len_
        y2 = np.sin(np.array(range(10, len_ + 10, 1)) / 30)
        noise = np.random.normal(0, 1, len_)
        return np.round((((y1 * 7) ** 2 + y2 + noise + 4) / 100), 3)

    def _func3(self, len_: int):
        y1 = np.array(range(0, len_, 1)) / len_
        y2 = np.sin(np.array(range(10, len_ + 10, 1)) / 30)
        noise = np.random.normal(0, 1, len_)
        return np.round((2.73 ** (2 + 1.5 * y1) + 2 * y2 + noise) / 100, 3)

    def _anomaly(self, probability: float, timeseria: List[float]):
        not_ignoring = set()
        for key, value in self._anomalies_with_len().items():
            if random.choices([True, False], [probability, (1 - probability)]):
                self._merge_anomaly_in_ts(timeseria, key, value)
                not_ignoring.add(key)
        return not_ignoring

    def _generate_anomalies_in_series(self, len_, not_ignoring: Set[int]):
        anomaly_seria_ = [0] * len_
        for start_point, anomaly_len in self._anomalies_with_len().items():
            if start_point in not_ignoring:
                anomaly_seria_[start_point:start_point + anomaly_len] = [1] * anomaly_len
        return anomaly_seria_

    def _generate_anomaly(self, len_):
        x = random.randint(50, 90)
        values = [x] * len_
        return np.round(np.abs(np.array(values) + np.random.randint(-10, 10, len_)) / 100, 3)

    def _merge_anomaly_in_ts(self, timeseries, start_point, anomaly_len):
        new_values = self._generate_anomaly(anomaly_len)
        timeseries[start_point:start_point + anomaly_len] = new_values

    def _anomalies_with_len(self):
        return {200: 10, 1000: 25, 3200: 11, 4000: 20, 5523: 22, 5901: 15, 7200: 14, 8500: 15, 9500: 10}

    def _prepare_data(self, f1, f2, f3, anomaly_, anomaly_seria_name, expected_len=100):
        metrics_lst_ = []
        len_ = len(f1)
        for i in range(round(len_ * 2 / expected_len)):
            subseries_: List[Metric] = []
            start_point_ = i * expected_len // 2
            finish_point_ = (i + 2) * expected_len // 2
            timestamps_: list[int] = list(range(start_point_, finish_point_))

            f1_seria_ = Metric(None, "f1", {"method": "line"},
                               f1[start_point_:finish_point_], timestamps_)
            subseries_.append(f1_seria_)
            f2_seria_ = Metric(None, "f2", {"method": "polynom"},
                               f2[start_point_:finish_point_], timestamps_)
            subseries_.append(f2_seria_)
            f3_seria_ = Metric(None, "f3", {"method": "exp"},
                               f3[start_point_:finish_point_], timestamps_)
            subseries_.append(f3_seria_)
            anomaly_seria_ = Metric(None, anomaly_seria_name, {},
                                    anomaly_[start_point_:finish_point_], timestamps_)
            subseries_.append(anomaly_seria_)

            metrics_ = Metrics(subseries_, expected_len)
            metrics_lst_.append(metrics_)
        return metrics_lst_
