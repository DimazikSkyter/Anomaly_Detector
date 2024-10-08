from datetime import datetime
from typing import List, Dict

import matplotlib.dates as mdates
import numpy as np
from matplotlib import pyplot as plt


class Metric:
    name: str
    tags: Dict[str, str]
    values: List[float]
    timestamps: List[int]

    def __init__(self, json, name: str, tags: Dict[str, str] = None, values: List[float] = None,
                 timestamps: List[int] = None):
        self.name = name
        if json:
            self.tags = {key: value for key, value in json['metric'].items() if key != '__name__'}
            self.values, self.timestamps = split(json['values'])
        else:
            self.tags = tags
            self.values = values
            self.timestamps = timestamps

    def __str__(self):
        return (f"Metric(name={self.name}, tags={self.tags}, "
                f"values={self.values[:3]}{'...' if len(self.values) > 3 else ''}, "
                f"timestamps={self.timestamps[:3]}{'...' if len(self.timestamps) > 3 else ''})")

    def __repr__(self):
        return (f"Metric(name={self.name!r}, tags={self.tags!r}, "
                f"values={self.values!r}, timestamps={self.timestamps!r})")


class Metrics:
    single_seria_max_size: int
    series: Dict[str, List[float]]
    timestamps: List

    _series_length: int = 0
    _last_updated: int = 0

    def __init__(self, metrics: List[Metric], seria_max_size=50, series=None, timestamps=None):
        """
        :param metrics: base metrics must not be empty and can't change
        :param seria_max_size: current value size of metrics need to equal this value
        """
        self.single_seria_max_size = seria_max_size
        if metrics:
            self.timestamps = metrics[0].timestamps
            self.series = {self._union_with_tags(metric.name, metric.tags): metric.values for metric in metrics}
        elif series and timestamps:
            self.timestamps = timestamps
            self.series = series
        else:
            raise SyntaxError("Not constructor found for you data set, need metrics or series and timestamps.")

        self._validate_and_shortcut_list_sizes()
        self._validate_timestamp_size()

    def __str__(self):
        series_preview = {k: v[:3] for k, v in self.series.items()}
        return (f"Metrics(single_seria_max_size={self.single_seria_max_size}, "
                f"series={series_preview}, "
                f"timestamps={self.timestamps[:3]}{'...' if len(self.timestamps) > 3 else ''})")

    def __repr__(self):
        return (f"Metrics(single_seria_max_size={self.single_seria_max_size!r}, "
                f"series={self.series!r}, "
                f"timestamps={self.timestamps!r})")

    def _validate_and_shortcut_list_sizes(self):
        if self.series:
            list_lengths = [len(lst) for lst in self.series.values()]
            if len(set(list_lengths)) > 1:
                raise ValueError("All lists in the dictionary must have the same length")
            elif list_lengths[0] > self.single_seria_max_size:
                self._shortcut()
        else:
            raise ValueError("Incoming series dict is empty")

    def _validate_timestamp_size(self):
        if self.series_length() != len(self.timestamps):
            self.timestamps = self._shortcut_seria(self.timestamps, self.series_length())

    def _shortcut(self):
        for key, values in self.series.items():
            self.series[key] = self._shortcut_seria(values, self.single_seria_max_size)

    def series_length(self) -> int:
        if self._series_length == 0:
            self._series_length = len(self.series[list(self.series.keys())[0]])
        return self._series_length

    def merge_new(self, m2: 'Metrics', save_max_size=False, new_size=None):
        keys = self.series.keys()
        if check_any_key(keys, m2.series.keys()):
            grid_3d_cur_ts_ = [(ts, a, idx) for a in [0] for idx, ts in enumerate(self.timestamps)]
            grid_3d_m2_ts_ = [(ts, a, idx) for a in [1] for idx, ts in enumerate(m2.timestamps)]
            #it's considered that the values for the same TS for the same Metric but in diff Metrics are the same
            merged_grid_list_ = self.merge_grids_(grid_3d_cur_ts_, grid_3d_m2_ts_)
            final_ts_grid_ = sorted(merged_grid_list_, key=lambda x: x[0])
            final_size_ = len(final_ts_grid_)
            new_series_ = {}
            for key in keys:
                values = m2.series.get(key, None)
                if values is not None:
                    new_series_[key] = [self._get_current_metrics(self.series[key], values, metrics_resolver)[metrics_idx]
                                       for _, metrics_resolver, metrics_idx in final_ts_grid_]
                else:
                    new_series_[key] = [None] * final_size_
            timestamps_ = [x for x, _, _ in final_ts_grid_]
            if save_max_size:
                self._cut_new(new_series_, new_size)
                timestamps_ = timestamps_[:self.single_seria_max_size]
            return Metrics(None,
                           self.single_seria_max_size if save_max_size else final_size_,
                           new_series_,
                           timestamps_)
        else:
            raise KeyError(f"Nothing to merge! "
                           f"Current keys '{self.series.keys()}' and income keys '{m2.series.keys()}'")


    def merge(self, another: 'Metrics'):
        self.union(another)
        return self.copy_cut_off()

    def union(self, m2: 'Metrics'):
        keys = self.series.keys()

        if check_any_key(keys, m2.series.keys()):
            pads = m2.series_length()
            for key in keys:
                values = m2.series.get(key, None)
                if values is None:
                    values = [None] * pads
                self.series[key] += values
            self.timestamps += m2.timestamps
            self._series_length += pads

    def get_part(self, start_index, stop_index):
        seria_ = {key: value[start_index: stop_index] for key, value in self.series.items()}
        timestamps_ = self.timestamps[start_index:stop_index]
        return self._new_metrics(seria_, timestamps_)

    def copy_cut_off(self, necessarily=False, max_size=None, is_copy=False):
        if max_size:
            self.single_seria_max_size = max_size
        if self._series_length > self.single_seria_max_size:
            series = {key: value[(len(value) - self.single_seria_max_size):] for key, value in self.series.items()}
            new_timestamps = self.timestamps[:len(list(series.values())[0])]
            return self._new_metrics(series, new_timestamps) \
                if is_copy \
                else self._update_current(series, new_timestamps)
        if self._series_length == max_size:
            return self._new_metrics(self.series, self.timestamps) \
                if is_copy \
                else self
        if necessarily:
            raise IndexError("Wrong max_size error nothing to cut.")

    def to_train_matrix(self, exclude=None, normalized=False) -> List[List[float]]:
        if exclude is None:
            exclude = []
        matrix: List[List[float]] = [[] for _ in range(len(list(self.series.values())[0]))]
        for key, seria in self.series.items():
            if key in exclude:
                continue
            if normalized:
                seria = self._scaler(np.array(seria))
                seria = list(seria.reshape(-1))  # переделать это
            for index in range(len(seria)):
                matrix[index].append(seria[index])
        return matrix

    def sub_range_metrics(self, start_timestamp, end_timestamp) -> 'Metrics':
        start_index, end_index = self._compute_indexes(start_timestamp, end_timestamp, self.timestamps)
        new_series = {key: values[start_index: end_index] for key, values in self.series.items()}
        return Metrics([], self.single_seria_max_size, new_series, self.timestamps[start_index:end_index])

    def plot(self):
        plt.figure(figsize=(15, 8))
        for key, values in self.series.items():
            dates = [datetime.fromtimestamp(ts) for ts in self.timestamps]
            mean_ = np.mean(values)
            plt.plot(dates, [value - mean_ for value in values], label=key)
        ax = plt.gca()
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title('Metrics:')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()

        plt.legend()
        plt.show()

    def add_seria(self, key: str, seria: List[float]):
        if key in self.series.keys():
            raise KeyError(f"Key {key} already exists.")
        if seria is None:
            raise ValueError(f"Seria {key} can't be None.")
        expected_len = self.series_length()
        income_len = len(seria)
        if income_len != expected_len:
            raise ValueError(f"Wrong size of new seria {income_len}, expected {expected_len}.")
        self.series[key] = seria


    @staticmethod
    def _get_current_metrics(list1, list2, resolver):
        return list1 if resolver == 0 else list2

    @staticmethod
    def _scaler(data):
        # scaler_robust_ = RobustScaler()
        # data_robusted_ = scaler_robust_.fit_transform(data.reshape(-1, 1))
        # scaler_standard_ = StandardScaler()
        # data_standardized_ = scaler_standard_.fit_transform(data.reshape(-1, 1))
        # scaler_minmax_ = MinMaxScaler(feature_range=(0, 1))
        # data_min_max_ = scaler_minmax_.fit_transform(data_standardized_)
        mean_ = np.mean(data)
        return data if np.abs(mean_) < 0.01 else (data / mean_ - 1)

    @staticmethod
    def _union_with_tags(name, tags):
        tags_string_ = ", ".join([f'{key}="{value}"' for key, value in tags.items()]) if tags else ""
        return f'{name}{{{tags_string_}}}'

    @staticmethod
    def _shortcut_seria(values, single_seria_max_size):
        return values[len(values) - single_seria_max_size:]

    @staticmethod
    def _compute_indexes(start_timestamp, end_timestamp, timestamps):
        if end_timestamp <= start_timestamp:
            raise IndexError("Start timestamp is greater than end timestamp")
        if timestamps[len(timestamps) - 1] < start_timestamp or timestamps[0] > end_timestamp:
            raise IndexError("The range is out of timestamp array")

        start_index = [x[0] for x in enumerate(timestamps) if x[1] >= start_timestamp][0]
        end_index_array = [x[0] for x in enumerate(timestamps) if x[1] < end_timestamp]
        end_index = end_index_array[len(end_index_array) - 1]
        return start_index, end_index

    @staticmethod
    def merge_grids_(cur_ts_dict_, m2_ts_dict_):
        merged_grid_ = []
        ts1 = []
        for ts, a, idx in cur_ts_dict_:
            merged_grid_.append((ts, a, idx))
            ts1.append(ts)

        for ts, a, idx in m2_ts_dict_:
            if ts not in ts1:
                merged_grid_.append((ts, a, idx))
        return merged_grid_


    @classmethod
    def get_last_updated(cls):
        return cls._last_updated

    def _new_metrics(self, series, new_timestamps) -> 'Metrics':
        return Metrics([], self.single_seria_max_size, series, new_timestamps)

    def _update_current(self, series, new_timestamps) -> 'Metrics':
        self.series = series
        self.timestamps = new_timestamps
        self._series_length = self.single_seria_max_size
        return self

    def _cut_new(self, new_series_, new_size):
        for key, value in new_series_:
            new_series_[key] = new_series_[key][:new_size] \
                if new_size \
                else new_series_[key][:self.single_seria_max_size]


def split(values_timestamp_list):
    timestamps = [item[0] for item in values_timestamp_list]
    values = [float(item[1]) for item in values_timestamp_list]
    return values, timestamps


def check_any_key(keys1, keys2):
    return any(key in keys2 for key in keys1)
