import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    def __init__(self, metrics, data_len, shift, batch_size, metrics_count, y_key=None):
        super().__init__()
        excluded = [y_key] if y_key else []
        self.metrics_count = metrics_count
        self.metrics = metrics
        self.data_len = data_len
        self.shift = shift
        self.batch_size = batch_size
        self.train_data = np.array(metrics.to_train_matrix(exclude=excluded, normalized=True))
        self.indices = np.arange((self.train_data.shape[0] - self.shift) // self.data_len + 1)
        self.on_epoch_end()
        self.y_key = y_key

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_indices)
        return X, y

    # shuffle only first index, array order will be safe
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices):
        """
        If y_key == None its return X itself with all series inside metrics.
        Else, if y_key != None it uses this column to 'y' array.
        :param batch_indices:
        :return:
        """
        X = np.empty((self.batch_size, self.data_len, self.metrics_count))
        y = np.empty((self.batch_size, self.data_len, self.metrics_count if self.y_key else 1))
        for i, idx in enumerate(batch_indices):
            start_idx = idx * self.shift
            end_idx = start_idx + self.data_len
            window = self.train_data[start_idx:end_idx]
            if window.shape[0] == self.data_len:
                X[i,] = window
                y[i,] = self.metrics.series[self.y_key] if self.y_key else window
        return X, y
