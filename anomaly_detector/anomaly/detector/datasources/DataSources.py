import json
from enum import IntEnum
from typing import Optional, Dict, List

from anomaly.detector.datasources.DataSourcesClients import PrometheusClient, VictoriaMetricsClient, DataSourceClient
from anomaly.detector.detector_app import Datasource
from anomaly.detector.metrics.Metrics import Metric


class DatasourceType(IntEnum):
    PROMETHEUS = 0
    VICTORIA_METRICS = 1
    JFR = 2
    CSV = 3

    _CLIENT_MAP: ['DatasourceType', DataSourceClient] = {
        PROMETHEUS: PrometheusClient,
        VICTORIA_METRICS: VictoriaMetricsClient,
        JFR: None, #Todo единовременная загрузка
        CSV: None #Todo сделать стриминговое чтение из файла
    }

    def get_client(self):
        return self._CLIENT_MAP[self]

class DatasourceStatus(IntEnum):
    ANNOUNCED = 0
    ONLINE = 1
    OFFLINE = 2


class DataSource:
    """Description of datasource for upload or save time series"""
    def __init__(self, name: str, address: str, type: DatasourceType, metrics: Optional[List[Metric]]):
        self.name = name
        self.address = address
        self.type = type
        self.status = DatasourceStatus.ANNOUNCED
        self.client: DataSourceClient = type.get_client()(address)
        if metrics:
            self.metrics = metrics
        else:
            self.metrics = []

    def connect(self) -> DatasourceStatus:
        if self.client.create_connect():
            self.status = DatasourceStatus.ONLINE
        else:
            self.status = DatasourceStatus.OFFLINE
        return self.status

    def add_metric(self, metric: Metric):
        self.metrics.append(metric)

    def remove_metric(self, metric: Metric):
        self.metrics = [m for m in self.metrics if not m.is_partially_match(metric)]


class DatasourceManager:
    """
    Класс в который мы передаем метаданные подключения и описания источника данных
    Создается клиент
    Определяется список метрик
        self.datasources = [
            {"name": "Database1", "ip_port": "192.168.1.10:3306", "status": "Online"},
            {"name": "CacheServer", "ip_port": "192.168.1.11:6379", "status": "Offline"},
            {"name": "WebServer", "ip_port": "192.168.1.12:8080", "status": "Online"},
        ]
    """
    data_sources = {}

    def __init__(self, data_sources: Optional[Dict[str: DataSource]]):
        if data_sources:
            self.data_sources = data_sources

    def add_new_datasource(self, data_source: DataSource):
        self.data_sources[data_source.name] = data_source

    def remove_data_source(self, name):
        if name in self.data_sources:
            del self.data_sources[name]

    def connect(self):
        reconnected = {key: value.connect() for key, value in self.data_sources.items() if value.status != DatasourceStatus.ONLINE}
        return [key for key, value in reconnected if value.status != DatasourceStatus.ONLINE]

    def upload_metrics(self, data_source_name: str):
        pass

    def add_metrics(self):
        pass

    def replace_metrics(self):
        pass

    def print_status(self):
        return json.dumps([value for value in self.data_sources.values()])
