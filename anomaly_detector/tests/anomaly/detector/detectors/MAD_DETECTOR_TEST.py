import unittest
from numpy import random as ran
import matplotlib.pyplot as plt

from anomaly.detector.metrics.Metrics import Metric, Metrics
from anomaly.detector.parts.WindowedMadDetector import MADDetector


class MyTestCase(unittest.TestCase):
    def test_dispersion_change(self):
        mad_detector = MADDetector(data_max_size=100, window=20, logger_level="DEBUG")
        data = [ran.normal(scale=1 / 10, size=1)[0] for i in range(1000)] + [ran.normal(scale=(i + 10) / 100, size=1)[0]
                                                                             for i in range(1000)]
        metric = Metric(None,
                        "test_seria",
                        {'tag_key': 'tag_value'},
                        data,
                        list(range(len(data))))
        metrics = Metrics([metric], 2000)
        mads = []
        i = 0

        for point in range(1901):
            sub_range_metrics = metrics.sub_range_metrics(point, point + 100)
            first_list_el = list(sub_range_metrics.series.values())[0]
            # print(f"Current min {min(np.abs(first_list_el))} and max {max(np.abs(first_list_el))}")
            mad_array = mad_detector.detect(sub_range_metrics)
            if not mads:
                mads = mad_array
            else:
                mads.append(mad_array[-1])

        print(f"Full mad size {len(mads)} and it:\n{mads}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 10))

        ax1.plot(data, label='Data')
        ax1.set_title('Source data')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')
        ax1.legend()

        ax2.plot(mads, label='Mads')
        ax2.set_title('Mad reaction')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.legend()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
