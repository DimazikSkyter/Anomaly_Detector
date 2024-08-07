import unittest

from anomaly.detector.parts.AnomalyDetector import AnomalyDetector


class MyTestCase(unittest.TestCase):
    def test_something(self):
        anomaly_detector = AnomalyDetector(4)
        composite_detector = CompositeDetector()
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
