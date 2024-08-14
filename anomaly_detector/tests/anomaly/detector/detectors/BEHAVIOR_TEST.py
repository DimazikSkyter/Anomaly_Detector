import unittest
import json
import pandas as pd
from typing import List

from matplotlib import pyplot as plt

from anomaly.detector.converter.MetricConverter import MetricConverter
from anomaly.detector.metrics.Metrics import Metrics, Metric
from anomaly.detector.parts.BehaviorDetector import BehaviorDetector


class BehaviorDetectorTests(unittest.TestCase):
    names = ["cpu_usage.csv", "resp_time.csv", "tps_finish.csv", "tps_master.csv"]

    def test_no_anomaly(self):
        with open("train.json", 'r') as file:
            json_str_train = file.read()
        with open("test_no_anomaly.json", 'r') as file_no_anomaly:
            json_str_test = file_no_anomaly.read()

        json_loads_train = json.loads(json_str_train)
        json_loads_test = json.loads(json_str_test)

        metrics_train: Metrics = MetricConverter.convert(json_loads_train, 1000, 20)

        beh_detector = BehaviorDetector(4, data_len=50, shift=5, lstm_size=256, mult=2, dropout_rate=0.3,
                                        logger_level="DEBUG")
        beh_detector.train(metrics_train, epochs=60)

        metrics_test: Metrics = MetricConverter.convert(json_loads_test, 50, 20)
        print(f"Metrics test len {metrics_test.series_length()}")
        result = beh_detector.detect(metrics_test)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 10))

        for key, value in metrics_train.series.items():
            ax1.plot(value, label=key)
        ax1.set_title('Train metric behavior no anomaly')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')
        ax1.legend()

        for key, value in metrics_test.series.items():
            ax2.plot(value, label=key)
        ax2.set_title('Test metric behavior no anomaly')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.legend()

        ax3.plot(result, label='Detector')
        ax3.set_title('Behavior detector no anomaly')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Value')
        ax3.set_ylim(0, 1.1)
        ax3.legend()

        plt.tight_layout()
        plt.show()

    def test_anomaly(self):
        import os

        print("Current working directory:", os.getcwd())

        with open("train.json", 'r') as file:
            json_str_train = file.read()
        with open("test_anomaly.json", 'r') as file_with_anomaly:
            json_str_test = file_with_anomaly.read()

        json_loads_train = json.loads(json_str_train)
        json_loads_test = json.loads(json_str_test)

        metrics_train: Metrics = MetricConverter.convert(json_loads_train, 1000, 20)

        beh_detector = BehaviorDetector(4, data_len=50, lstm_size=256, shift=5, mult=2, dropout_rate=0.3,
                                        logger_level="DEBUG")
        beh_detector.train(metrics_train, epochs=60)

        metrics_test: Metrics = MetricConverter.convert(json_loads_test, 50, 20)
        print(f"Metrics test len {metrics_test.series_length()}")
        result = beh_detector.detect(metrics_test)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 10))

        for key, value in metrics_train.series.items():
            ax1.plot(value, label=key)
        ax1.set_title('Train metric behavior with anomaly')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')
        ax1.legend()

        for key, value in metrics_test.series.items():
            ax2.plot(value, label=key)
        ax2.set_title('Behavior detector with anomaly')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.legend()

        ax3.plot(result, label='Detector')
        ax3.set_title('Behavior detector with anomaly')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Value')
        ax3.set_ylim(0, 1.1)
        ax3.legend()

        plt.tight_layout()
        plt.show()

    def test_with_data(self):
        data_len = 100
        behavior_detector = BehaviorDetector(4, data_len=100, lstm_size=512, dropout_rate=0.4, logger_level="DEBUG")
        data: List[Metrics] = self._generate_data(data_len, 5000)

        split_index = int(0.75 * len(data))

        train = data[:split_index]
        test = data[split_index:]

        iteration = 0
        result = []

        for data_train in train:
            behavior_detector.train(data_train, epochs=10)

        for test_data in test:
            result_single = behavior_detector.detect(test_data)
            if result:
                result += result_single[int(data_len / 2):]
            else:
                result += result_single

        print(f"diff {result}")

        plt.plot(result, label="result")
        plt.legend()
        plt.show()
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


if __name__ == '__main__':
    unittest.main()
