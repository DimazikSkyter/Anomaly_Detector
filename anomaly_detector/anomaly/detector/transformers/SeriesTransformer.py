from abc import abstractmethod

import numpy as np

from anomaly.detector.metrics.Metrics import Metrics


class Transformer:

    @abstractmethod
    def transform(self, metrics: Metrics):
        pass

class NoneDataNeighbourFiller(Transformer):

    def transform(self, metrics: Metrics):
        for key, value in metrics.series:
            if any(x is None for x in value):
                metrics.series[key] =self._replace_none_with_avg(value)

    def _replace_none_with_avg(self, arr: list[float]) -> list[float]:
        arr = np.array([np.nan if x is None else x for x in arr], dtype=np.float64)
        none_mask = np.isnan(arr)

        if np.all(none_mask):
            return [0.0] * len(arr)

        left_shifted = np.empty_like(arr)
        right_shifted = np.empty_like(arr)

        left_shifted[1:] = arr[:-1]
        left_shifted[0] = np.nan

        right_shifted[:-1] = arr[1:]
        right_shifted[-1] = np.nan

        avg_neighbors = np.nanmean(np.stack([left_shifted, right_shifted]), axis=0)
        arr[none_mask] = avg_neighbors[none_mask]

        return arr.tolist()
