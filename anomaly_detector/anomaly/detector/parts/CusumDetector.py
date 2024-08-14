from collections import deque
from typing import List

from anomaly.detector.metrics.Metrics import Metrics
from anomaly.detector.parts.CompositeStreamDetector import Detector


class CUSUMDetector(Detector):
    def __init__(self, threshold, drift, window, logger_level="INFO"):
        """
        Initialize the CUSUM detector with specified threshold and drift parameters.

        Parameters:
        - threshold: float, decision interval.
        - drift: float, drift parameter.
        """
        super().__init__(logger_level)
        self.threshold = threshold
        self.drift = drift
        self.window = window
        self.logger.info("Cusum detector successfully init.")

    def detect(self, metrics: Metrics) -> List[float]:
        result = []
        for seria in metrics.series.values():
            single_result = self.detect_single_seria(seria)
            if result:
                for index in range(len(single_result)):
                    if result[index] < single_result[index]:
                        result[index] = single_result[index]
            else:
                result = single_result
        return result

    def detect_single_seria(self, data):
        """
        Detect anomalies in the data using the CUSUM method.

        Parameters:
        - data: array-like, data to detect anomalies in.

        Returns:
        - pos_cusum: array, positive CUSUM values.
        - neg_cusum: array, negative CUSUM values.
        - anomalies: list, indices of detected anomalies.
        """
        normalized_data = self.min_max_scaler(data)
        increment = 0
        decrement = 0
        delta_queue = deque(maxlen=4)
        delta_queue.append(0)
        delta_queue.append(0)
        delta_queue.append(0)
        delta_queue.append(0)
        window = self.window
        cusum_mult = []

        for i in range(0, len(normalized_data)):

            # pos_window = pos_cusum[max(0, i - 1 - window):i - 1]
            # sum_positive = sum(pos_window) / max(1, len(pos_window))
            #
            # neg_window = neg_cusum[max(0, i - 1 - window):i - 1]
            # sum_negative = sum(neg_window) / max(1, len(neg_window))

            previous_deltas = sum(list(delta_queue))
            delta = (normalized_data[i] - normalized_data[i - 1]) * 0.8
            delta += previous_deltas * 0.2
            delta_queue.append(delta)

            increment = max(0, increment + delta - self.drift)
            decrement = max(0, decrement - delta - self.drift)
            mult = increment * decrement * 25

            self.logger.debug(f"current mult %s", mult)
            cusum_mult.append(mult / self.threshold)

            if mult > self.threshold:
                increment = 0  # Reset after anomaly detection
                decrement = 0

        return self.sigma_normalize(cusum_mult, 1)
