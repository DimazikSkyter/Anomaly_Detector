from typing import List

from anomaly.detector.metrics.Metrics import Metrics, Metric

#todo split to prometheus and victoria metrics
class MetricConverter:

    @staticmethod
    def convert(json, seria_max_size, step) -> Metrics:
        total_metrics = []
        for key, value in json.items():
            metrics_json = value['data']['result']
            total_metrics += [Metric(metric, key) for metric in metrics_json]
        metrics = MetricConverter.alignment_timestamps(total_metrics, step)
        return Metrics(metrics, seria_max_size)

    @staticmethod
    def alignment_timestamps(metrics, step) -> List[Metric]:
        mins = [metric.timestamps[0] for metric in metrics]
        maxes = [metric.timestamps[-1] for metric in metrics]
        min_inf = min(mins)
        max_sup = min(maxes)
        return [MetricConverter._shortcut(metric, min_inf, max_sup, step) for metric in metrics]

    @staticmethod
    def _shortcut(metric, min_inf, max_sup, step) -> Metric:
        new_timestamps = []
        new_values = []
        index = 0
        for timestamp in range(min_inf, max_sup + step, step):
            new_timestamps.append(timestamp)
            tmp_timestamp = metric.timestamps[index]
            if tmp_timestamp <= timestamp:
                new_values.append(metric.values[index])
                index += 1
            else:
                new_values.append(None)

        metric.values = new_values
        metric.timestamps = new_timestamps
        return metric


    @staticmethod
    def to_json(metrics: Metrics):
        metrics_data = []

        for metric_name, values in metrics.series.items():
            if not values:
                continue

            name_parts = metric_name.split("{", 1)
            base_name = name_parts[0]
            tags_str = name_parts[1][:-1] if len(name_parts) > 1 else ""  # Remove trailing '}'

            tags = {}
            if tags_str:
                for tag in tags_str.split(","):
                    key, value = tag.split("=")
                    tags[key.strip()] = value.strip().strip('"')  # Remove extra quotes

            metric_entry = {
                "metric": {"__name__": base_name, **tags},
                "values": values,
                "timestamps": metrics.timestamps
            }
            metrics_data.append(metric_entry)

        return metrics_data