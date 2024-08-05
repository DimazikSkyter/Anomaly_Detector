import unittest
import numpy as np
from matplotlib import pyplot as plt

from anomaly.detector.parts.CusumDetector import CUSUMDetector


class MyTestCase(unittest.TestCase):

    def test_short_anomaly(self):
        np.random.seed(43)
        threshold = 1
        drift = 0.01
        rand_data = np.random.rand(1000) * 2 - 1
        base_data = (list(range(100)) + [100] * 492 + [86, 76, 50, 85, 49, 52]
                     + [100] * 393 + list(range(90, 0, -10)))
        data = rand_data + base_data

        cusum_detector = CUSUMDetector(threshold, drift, 10)

        cusum_result = cusum_detector.detect_single_seria(data)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 10))

        ax1.plot(data, label='Data')
        ax1.set_title('Short test')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')
        ax1.legend()

        ax2.plot(cusum_result, label='Cusum detector')
        ax2.set_title('Short test')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def test_medium_anomaly(self):
        np.random.seed(43)
        threshold = 1
        drift = 0.01
        rand_data = np.random.rand(1000) * 2 - 1
        base_data = (list(range(100)) + [100] * 490
                     + [86, 76, 50, 53, 51, 52, 49, 52, 51, 55, 58, 51, 57, 60, 67, 71, 82, 95]
                     + [100] * 383 + list(range(90, 0, -10)))
        data = rand_data + base_data

        cusum_detector = CUSUMDetector(threshold, drift, 10)

        cusum_result = cusum_detector.detect_single_seria(data)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 10))

        ax1.plot(data, label='Data')
        ax1.set_title('Medium long anomaly')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')
        ax1.legend()

        ax2.plot(cusum_result, label='Cusum detector')
        ax2.set_title('Medium long anomaly')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def test_long_anomaly(self):
        np.random.seed(43)
        threshold = 1
        drift = 0.01
        rand_data = np.random.rand(1000) * 2 - 1
        base_data = (list(range(100)) + [100] * 490
                     + [86, 76, 50, 53, 51, 52, 49, 37, 28, 29, 40, 42, 39, 52, 61, 70, 74, 51, 57, 60, 67,
                        71, 82, 95, 84, 87, 92, 97] + [100] * 373 + list(range(90, 0, -10)))

        print(f"Long test len{base_data}")
        data = rand_data + base_data

        cusum_detector = CUSUMDetector(threshold, drift, 10)

        cusum_result = cusum_detector.detect_single_seria(data)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 10))

        ax1.plot(data, label='Data')
        ax1.set_title('Long test anomaly')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')
        ax1.legend()

        ax2.plot(cusum_result, label='Cusum detector')
        ax2.set_title('Long test anomaly')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def test_short_notdeep_anomaly(self):
        np.random.seed(43)
        threshold = 1
        drift = 0.01
        rand_data = np.random.rand(1000) * 2 - 1
        base_data = (list(range(100)) + [100] * 492 + [89, 97, 95, 94, 96, 99, 89, 97, 95, 94]
                     + [100] * 389 + list(range(90, 0, -10)))

        print(f"Long test len{base_data}")
        data = rand_data + base_data

        cusum_detector = CUSUMDetector(threshold, drift, 10)

        cusum_result = cusum_detector.detect_single_seria(data)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 10))

        ax1.plot(data, label='Data')
        ax1.set_title('Not deep test anomaly')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')
        ax1.legend()

        ax2.plot(cusum_result, label='Cusum detector')
        ax2.set_title('Not deep test anomaly')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.legend()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
