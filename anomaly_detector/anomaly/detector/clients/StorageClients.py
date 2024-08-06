from abc import ABC, abstractmethod
from typing import Dict

import requests
import time
import datetime


class Storagelient(ABC):

    @abstractmethod
    def get_metrics(self, step=30) -> Dict:
        pass


class PrometheusClient(Storagelient):

    def get_metrics(self, step=30) -> Dict:
        pass


class VictoriaMetricsClient(Storagelient):
    def __init__(self, base_url, queries):
        super().__init__()
        self.base_url = base_url
        self.queries = queries

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

    def stream_metrics(self, interval=60):
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
    vm_client.stream_metrics(interval=20)  # Fetch metrics every 60 seconds
