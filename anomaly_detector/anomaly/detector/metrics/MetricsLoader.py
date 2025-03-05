import asyncio
import json
import os
import zipfile
from datetime import datetime, timedelta
from multiprocessing import Value
from typing import List, Dict, Optional
from wsgiref.validate import validator

from anomaly.detector.storage.StorageClients import StorageClient
from anomaly.detector.converter.MetricConverter import MetricConverter
from anomaly.detector.metrics.Metrics import Metrics


class MetricsLoader:

    def __init__(self, database_client: StorageClient, config: Dict, metrics_names: List[str] = None):
        if config is None:
            raise ValueError("Config cannot ne None.")

        self.database_client = database_client
        self.names = metrics_names or []
        self.enable = True

        self.poller_interval = config.get('poller_interval', 5)
        self.max_values_per_seria = config.get('max_values_per_seria', 10000)
        self.step = config.get('step', 30);
        self.time_shift = config.get('time_shift', 60);
        self.storage_folder = config.get('storage_folder', './');
        self.cached_metrics = self._load()
        self._has_new_data_in_metrics = Value('b', int(False))

        asyncio.create_task(self.run_poll())

    def print_all_config(self):
        return {
            'poller_interval': self.poller_interval,
            'max_values_per_seria': self.max_values_per_seria,
            'step': self.step,
            'time_shift': self.time_shift,
            'storage_folder': self.storage_folder
        }

    def update_metrics_names(self, metrics_names: List[str]):
        if not self.names:
            self.names = metrics_names
        self.names += metrics_names

    async def run_poll(self):
        while True:
            if self.enable:
                self.poll()
            await asyncio.sleep(self.poller_interval)

    def has_new(self) -> bool:
        with self._has_new_data_in_metrics.get_lock():
            return bool(self._has_new_data_in_metrics.value)

    def poll(self):
        total_metrics: Optional[Metrics] = None
        for name in self.names:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=self.time_shift)
            json_data = self.database_client.get_metric(name, start_time, end_time, self.step)
            metrics: Metrics = MetricConverter.convert(json_data, self.max_values_per_seria, self.step)
            if total_metrics:
                total_metrics.series_join(metrics)
            else:
                total_metrics = metrics
        if total_metrics:
            self._sync_cache_and_save(total_metrics)

    def _sync_cache_and_save(self, metrics):
        if self.cached_metrics:
            self.cached_metrics = self.cached_metrics.merge_new(metrics)
        else:
            self.cached_metrics = metrics
        with self._has_new_data_in_metrics.get_lock():
            self._has_new_data_in_metrics.value = int(True)
        self._persist()

    def _persist(self):
        """Serializes cached_metrics to JSON and saves it in a ZIP archive."""
        if self.cached_metrics is None or self.cached_metrics.series_length() <= 0:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"Metrics_{timestamp}"
        json_filename = f"{base_filename}.json"
        zip_filename = f"{base_filename}.zip"

        json_data = self.cached_metrics.to_json()
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr(json_filename, json_data)

    def _load(self) -> Optional[Metrics]:
        """Loads the most recent Metrics snapshot from a ZIP archive."""
        zip_files = [f for f in os.listdir() if f.startswith("Metrics_") and f.endswith(".zip")]
        if not zip_files:
            return None

        zip_files.sort(reverse=True)
        latest_zip = zip_files[0]
        print(f"Loading from: {latest_zip}")

        with zipfile.ZipFile(latest_zip, 'r') as zipf:
            json_filename = latest_zip.replace(".zip", ".json")
            if json_filename in zipf.namelist():
                with zipf.open(json_filename) as json_file:
                    json_data = json_file.read().decode("utf-8")
                    metrics_dict = json.loads(json_data)
                    #todo поменять формирование объекта. Сделать конвертацию через MetricConverter
                    return Metrics(
                        None,
                        metrics_dict["single_seria_max_size"],
                        metrics_dict["series"],
                        metrics_dict["timestamps"]
                    )
        return None

    def get_metrics(self, metrics_names: List[str]):
        return self.cached_metrics.filter(metrics_names)
