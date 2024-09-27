import unittest
import itertools

from anomaly.detector.converter.MetricConverter import MetricConverter
from anomaly.detector.metrics.Metrics import Metrics, Metric


class TestMetricConverter(unittest.TestCase):
    def test_should_merge_in_asc_ordering_with_intersection(self):
        metric1_1_: Metric = Metric(None,
                                    "name",
                                    {"id": "1"},
                                    [11, 12, 13, 14, 15, 16, 17, 18, 19, 10],
                                    [100, 101, 102, 103, 104, 105, 106, 107, 110, 112])

        metric1_2_: Metric = Metric(None,
                                    "name",
                                    {"id": "2"},
                                    [21, 22, 23, 24, 25, 26, 27, 28, 29, 20],
                                    [100, 101, 102, 103, 104, 105, 106, 107, 110, 112])

        metric2_1_: Metric = Metric(None,
                                    "name",
                                    {"id": "1"},
                                    [31, 32, 33, 34, 35, 36, 37, 38, 39, 30],
                                    [110, 111, 112, 113, 114, 115, 116, 117, 118, 119])

        metric2_2_: Metric = Metric(None,
                                    "name",
                                    {"id": "2"},
                                    [41, 42, 43, 44, 45, 46, 47, 48, 49, 40],
                                    [110, 111, 112, 113, 114, 115, 116, 117, 118, 119])

        metrics1_ = Metrics([metric1_1_, metric1_2_])
        metrics2_ = Metrics([metric2_1_, metric2_2_])

        metrics_merged_ = metrics1_.merge_new(metrics2_)

        self.assertEqual(metrics_merged_.timestamps, [100, 101, 102, 103, 104, 105, 106, 107, 110, 111,
                                                      112, 113, 114, 115, 116, 117, 118, 119])
        self.assertEqual(metrics_merged_.series['name{id="1"}'], [11, 12, 13, 14, 15, 16, 17, 18, 19, 32,
                                                                  10, 34, 35, 36, 37, 38, 39, 30])
        self.assertEqual(metrics_merged_.series['name{id="2"}'], [21, 22, 23, 24, 25, 26, 27, 28, 29, 42,
                                                                  20, 44, 45, 46, 47, 48, 49, 40])

    def test_should_merge_in_desc_ordering(self):
        metric1_1_: Metric = Metric(None,
                                    "name",
                                    {"id": "1"},
                                    [11, 12, 13, 14, 15, 16, 17, 18, 19, 10],
                                    [100, 101, 102, 103, 104, 105, 106, 107, 110, 112])

        metric1_2_: Metric = Metric(None,
                                    "name",
                                    {"id": "2"},
                                    [21, 22, 23, 24, 25, 26, 27, 28, 29, 20],
                                    [100, 101, 102, 103, 104, 105, 106, 107, 110, 112])

        metric2_1_: Metric = Metric(None,
                                    "name",
                                    {"id": "1"},
                                    [31, 32, 33, 34, 35, 36, 37, 38, 39, 30],
                                    [110, 111, 112, 113, 114, 115, 116, 117, 118, 119])

        metric2_2_: Metric = Metric(None,
                                    "name",
                                    {"id": "2"},
                                    [41, 42, 43, 44, 45, 46, 47, 48, 49, 40],
                                    [110, 111, 112, 113, 114, 115, 116, 117, 118, 119])

        metrics1_ = Metrics([metric1_1_, metric1_2_])
        metrics2_ = Metrics([metric2_1_, metric2_2_])

        metrics_merged_ = metrics2_.merge_new(metrics1_)

        self.assertEqual(metrics_merged_.timestamps, [100, 101, 102, 103, 104, 105, 106, 107, 110, 111,
                                                      112, 113, 114, 115, 116, 117, 118, 119])
        self.assertEqual(metrics_merged_.series['name{id="1"}'], [11, 12, 13, 14, 15, 16, 17, 18, 31, 32,
                                                                  33, 34, 35, 36, 37, 38, 39, 30])
        self.assertEqual(metrics_merged_.series['name{id="2"}'], [21, 22, 23, 24, 25, 26, 27, 28, 41, 42,
                                                                  43, 44, 45, 46, 47, 48, 49, 40])


if __name__ == '__main__':
    unittest.main()
