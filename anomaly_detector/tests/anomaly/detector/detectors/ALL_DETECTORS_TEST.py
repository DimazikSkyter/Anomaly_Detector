import unittest
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from anomaly.detector.converter.MetricConverter import MetricConverter
from anomaly.detector.metrics.Metrics import Metrics
from anomaly.detector.parts.AnomalyDetector import AnomalyDetector
from anomaly.detector.parts.BehaviorDetector import BehaviorDetector
from anomaly.detector.parts.CompositeStreamDetector import CompositeDetector
from anomaly.detector.parts.CusumDetector import CUSUMDetector
from anomaly.detector.parts.MasterDataCorrelationDetector import CorrelationDetector
from anomaly.detector.parts.WindowedMadDetector import MADDetector


class MyTestCase(unittest.TestCase):
    names = ["cpu_usage.csv", "resp_time.csv", "tps_finish.csv", "tps_master.csv"]

    def test_something(self):
        anomaly_seria_train = []
        anomaly_seria_test = []
        anomaly_detector = AnomalyDetector(4)
        behavior_detector = BehaviorDetector(4)
        cusum_detector = CUSUMDetector(1, 0.01, 10)
        correlation_detector = CorrelationDetector(7, [[0]], 10,
                                                   "")
        mad_detector = MADDetector()
        composite_detector = CompositeDetector([
            behavior_detector,
            cusum_detector,
            correlation_detector,
            mad_detector], logger_level="DEBUG")

        data: List[Metrics] = self._generate_data()

        split_index = int(0.75 * len(data))

        train = data[:split_index]
        test = data[split_index:]

        iteration = 0

        for data_train in train:
            composite_detector.train(data_train)
            composite_result_train: Metrics = composite_detector.detect(data_train)
            self._add_anomyly_seria(composite_result_train,
                                    anomaly_seria_train[iteration * data_train.series_length():
                                                        (iteration + 1) * data_train.series_length()])
            anomaly_detector.train(composite_result_train)

        result = []
        iteration = 0
        for data_test in test:
            composite_result_test: Metrics = composite_detector.detect(data_test)
            result_single: List[float] = anomaly_detector.detect(composite_result_test)
            print(f"Result single {result_single} for interation {iteration}")
            result += result_single
        score = np.sum(np.abs(np.array(anomaly_seria_test) - np.array(result))) / len(anomaly_seria_test)
        print(f"anomaly test wait: {anomaly_seria_test},"
              f" anomaly test result {result},"
              f" score = {score}")

        self.assertGreater(0.1, score)

    def _generate_data(self, data_len=100) -> List[Metrics]:
        data: List[Metrics] = []
        for name in self.names:
            df = pd.read_csv(name, sep=";", header=None)
            Metrics([], 50000, )
        return data

    def _add_anomyly_seria(self, composite_result, anomaly_seria):
        pass

    def test_plot_income_metrics(self):
        fig, ax_tuple = plt.subplots(len(self.names), 1, figsize=(60, 30))
        for index in range(len(self.names)):
            df = pd.read_csv(self.names[index], sep=";", header=None)
            ax = ax_tuple[index]
            ax.plot(df[[1]][37000:], label="value")
            ax.set_title(self.names[index])
            ax.set_xlabel("timestamp")
            ax.set_ylabel("Value")
            ax.legend()
        plt.show()

if __name__ == '__main__':
    unittest.main()
