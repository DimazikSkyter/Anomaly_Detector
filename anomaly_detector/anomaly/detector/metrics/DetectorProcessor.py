from typing import List, Set, Dict


from anomaly.detector.converter.MetricConverter import MetricConverter
from anomaly.detector.metrics.Metrics import Metrics
from anomaly.detector.metrics.MetricsLoader import MetricsLoader
from anomaly.detector.parts.CompositeStreamDetector import Detector
from anomaly.detector.storage.StorageClients import StorageClient
from anomaly.detector.transformers.SeriesTransformer import Transformer


class MetricsProcessor:

    def __init__(self, storage_client: StorageClient, metrics_loader: MetricsLoader, transformers: Set[Transformer], detectors: Set[Detector]):
        self.loader = metrics_loader
        self.storage_client = storage_client
        self.transformers = transformers
        self.detectors = detectors
        self.detector_series = None

    def run(self):
        if self.loader.has_new():
            metrics: Metrics = self.loader.cached_metrics
            self._accept_transformers(metrics)
            self.detector_series: Dict[str, list[float]] = self._convert_to_detectors_series(metrics)
            converted_json = MetricConverter.to_json(Metrics(None, metrics.single_seria_max_size, self.detector_series, metrics.timestamps))
            self.storage_client.save_metrics(converted_json)

    def _accept_transformers(self, metrics) -> None:
        for transformer in self.transformers:
            transformer.transform(metrics)

    def _convert_to_detectors_series(self, metrics) -> Dict[str, list[float]]:
        return {detector.get_class_name(): detector.detect(metrics) for detector in self.detectors}

