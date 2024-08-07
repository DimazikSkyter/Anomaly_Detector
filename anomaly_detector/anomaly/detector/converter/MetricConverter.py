from typing import List

from anomaly.detector.metrics.Metrics import Metrics, Metric


# добавить разделение путей по базам
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
