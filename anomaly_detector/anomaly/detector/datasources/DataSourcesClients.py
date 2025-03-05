import datetime
import time
import logging

from abc import ABC, abstractmethod
from http.client import responses
from typing import Dict

import requests


# todo добавить таймауты
class DataSourceClient(ABC):

    def __init__(self, base_url, logger_name, logger_level="INFO",
                 log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
        self.base_url = base_url

        if logger_name is None:
            logger_name = self.__class__.__name__
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        ch = logging.StreamHandler()
        ch.setLevel(logger_level)
        formatter = logging.Formatter(log_format)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    @abstractmethod
    def create_connect(self) -> bool:
        pass

    @abstractmethod
    def get_metric(self, metric: str, start: datetime, finish: datetime, step=30) -> Dict:
        pass

    @abstractmethod
    def get_metrics(self, step=30) -> Dict:
        pass

    def save_metrics(self, converted_json):
        self.logger.info(f"Try to save json of metrics to db {converted_json}.")
        response = requests.post(self.base_url, json = converted_json)
        self.logger.info(response.status_code)



class PrometheusClient(DataSourceClient):

    def get_metric(self, metric: str, start: datetime, finish: datetime, step=30) -> Dict:
        pass

    def get_metrics(self, metric: str, step=30) -> Dict:
        pass


class VictoriaMetricsClient(DataSourceClient):
    def __init__(self, base_url, queries):
        super().__init__()
        self.base_url = base_url
        self.queries = queries

    def get_metric(self, metric: str, start: datetime, finish: datetime, step=30) -> Dict:
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(minutes=240)
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())

        # combined_query = f"({{__name__=~\"{'|'.join(self.queries)}\"}})"

        results = {}
        for query in self.queries:
            url = f"{self.base_url}/api/v1/query_range"
            params = {
                'query': query,
                'start': start_timestamp,
                'end': end_timestamp,
                'step': step
            }

            response = requests.get(url, params=params)
            if response.status_code == 200:
                results[query] = response.json()
            else:
                response.raise_for_status()

        return results

    def get_metrics(self, step=30) -> Dict:
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(minutes=240)
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())

        # combined_query = f"({{__name__=~\"{'|'.join(self.queries)}\"}})"

        results = {}
        for query in self.queries:
            url = f"{self.base_url}/api/v1/query_range"
            params = {
                'query': query,
                'start': start_timestamp,
                'end': end_timestamp,
                'step': step
            }

            response = requests.get(url, params=params)
            if response.status_code == 200:
                results[query] = response.json()
            else:
                response.raise_for_status()

        return results

    def test_stream_metrics(self, interval=60):
        try:
            while True:
                metrics = self.get_metrics()
                print(metrics)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Stopped by user")


# Usage example:
if __name__ == "__main__":
    base_url = "http://localhost:8428"
    queries = ["rate(windows_physical_disk_idle_seconds_total[1m])", "rate(process_cpu_seconds_total[1m])"]

    vm_client = VictoriaMetricsClient(base_url, queries)
    vm_client.test_stream_metrics(interval=20)  # Fetch metrics every 60 seconds
