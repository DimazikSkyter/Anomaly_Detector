import unittest
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from anomaly.detector.metrics.Metrics import Metrics
from anomaly.detector.parts.AnomalyDetector import AnomalyDetector
from anomaly.detector.parts.BehaviorDetector import BehaviorDetector
from anomaly.detector.parts.CompositeStreamDetector import CompositeDetector
from anomaly.detector.parts.CusumDetector import CUSUMDetector
from anomaly.detector.parts.MasterDataCorrelationDetector import CorrelationDetector
from anomaly.detector.parts.WindowedMadDetector import MADDetector


class MyTestCase(unittest.TestCase):
    names = ["cpu_usage.csv", "resp_time.csv", "tps_finish.csv", "tps_master.csv"]
    anomaly_seria_name = "anomaly.csv"

    def test_something(self):
        data_len = 100
        anomaly_seria: List[float] = self._generate_anomaly_seria()
        anomaly_detector = AnomalyDetector(4, self.anomaly_seria_name, lstm_size=512, dropout_rate=0.4,
                                           data_len=100, epochs=30, logger_level="DEBUG")
        behavior_detector = BehaviorDetector(4, lstm_size=1024, dropout_rate=0.4, data_len=100, logger_level="DEBUG")
        cusum_detector = CUSUMDetector(1, 0.01, 10, logger_level="ERROR")
        correlation_detector = CorrelationDetector(7, [[0]], 10,
                                                   "tps_master.csv", logger_level="ERROR")
        mad_detector = MADDetector()
        composite_detector = CompositeDetector([
            behavior_detector,
            cusum_detector,
            correlation_detector,
            mad_detector], logger_level="ERROR")

        data: List[Metrics] = self._generate_data(data_len, 300000)

        split_index = int(0.75 * len(data))

        train = data[:split_index]
        test = data[split_index:]

        iteration = 0

        for data_train in train:
            composite_detector.train(data_train)

        for data_train in train:
            composite_result_train: Metrics = composite_detector.detect(data_train)
            self._add_anomaly_seria(composite_result_train,
                                    anomaly_seria[int(iteration * data_len / 2):
                                                  int((iteration + 2) * data_len / 2)])
            anomaly_detector.train(composite_result_train)

        result = []
        iteration = 0
        for data_test in test:
            composite_result_test: Metrics = composite_detector.detect(data_test)
            result_single: List[float] = anomaly_detector.detect(composite_result_test)
            print(f"Result single {result_single} for interation {iteration}")
            if result:
                result += result_single[int(data_len / 2):]
            else:
                result += result_single
        anomaly_seria_test = anomaly_seria[int(len(train) * data_len / 2):]
        score = np.sum(np.abs(np.array(anomaly_seria_test) - np.array(result))) / len(anomaly_seria_test)
        print(f"anomaly test wait: {anomaly_seria_test},"
                           f" anomaly test result {result},"
                           f" score = {score}")

        plt.plot(result, label="result")
        plt.plot(anomaly_seria_test, label="expected")
        plt.legend()
        plt.show()
        self.assertGreater(0.1, score)

    def _generate_anomaly_seria(self) -> List[float]:
        df = pd.read_csv("anomaly.csv", sep=";", header=None)
        return df.iloc[:, 1].tolist()

    def _generate_data(self, data_len=100, max_size=1000) -> List[Metrics]:
        data: List[Metrics] = []

        for name in self.names:
            df = pd.read_csv(name, sep=";", header=None)
            size_of_data = len(df.index)
            size_of_data = size_of_data if size_of_data < max_size else max_size
            num_parts = round(size_of_data * 2 / data_len) - 1

            for index in range(num_parts):
                start_idx = int(index * data_len / 2)
                end_idx = int((data_len * (index + 2)) / 2)
                part = df.iloc[start_idx:end_idx]

                if index < len(data):
                    metrics = data[index]
                    metrics.series[name] = part.iloc[:, 1].tolist()
                else:
                    metrics = Metrics([], 100,
                                      {name: part.iloc[:, 1].tolist()}, part.iloc[:, 0].tolist())

                if index < len(data):
                    data[index] = metrics
                else:
                    data.append(metrics)
        return data

    def _add_anomaly_seria(self, composite_result: Metrics, anomaly_seria: List[float]):
        composite_result.series[self.anomaly_seria_name] = anomaly_seria

    def test_plot_income_metrics(self):
        fig, ax_tuple = plt.subplots(len(self.names), 1, figsize=(60, 30))
        for index in range(len(self.names)):
            df = pd.read_csv(self.names[index], sep=";", header=None)
            ax = ax_tuple[index]
            ax.plot(df[[1]][32500:], label="value")
            ax.set_title(self.names[index])
            ax.set_xlabel("timestamp")
            ax.set_ylabel("Value")
            ax.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
