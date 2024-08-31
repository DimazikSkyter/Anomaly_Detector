import logging

import keras
import numpy as np
import tensorflow as tf
from anomaly.detector.metrics.Metrics import Metrics


# Разбить на два
class DataGenerator(keras.utils.Sequence):
    def __init__(self, metrics: Metrics, data_len, shift, batch_size, metrics_count, xgb_predictions=None, y_key=None,
                 logger_level="INFO",
                 log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
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
        self.xgb_predictions = xgb_predictions

        logger_name = self.__class__.__name__
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        ch = logging.StreamHandler()
        ch.setLevel(logger_level)
        formatter = logging.Formatter(log_format)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation_with_xgb(batch_indices) \
            if self.xgb_predictions is not None \
            else self.__data_generation(batch_indices)
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
        y = np.empty((self.batch_size, self.data_len, 1 if self.y_key else self.metrics_count))
        for i, idx in enumerate(batch_indices):
            start_idx = idx * self.shift
            end_idx = start_idx + self.data_len
            window = self.train_data[start_idx:end_idx]
            if window.shape[0] == self.data_len:
                X[i,] = window
                y[i,] = (np.array(self.metrics.series[self.y_key])
                         .reshape(y.shape[1], y.shape[2])) if self.y_key else window
        self.logger.debug("Outcome generator data X=\n%s\n,y=\n%s", X[0][:3], y[0][:3])
        return X, y

    def __data_generation_with_xgb(self, batch_indices):
        """
        If y_key == None its return X itself with all series inside metrics.
        Else, if y_key != None it uses this column to 'y' array.
        :param batch_indices:
        :return:
        """
        X_1 = np.empty((self.batch_size, self.data_len, self.metrics_count))
        X_2 = np.empty((self.batch_size, self.data_len, 1))
        y = np.empty((self.batch_size, self.data_len, 1 if self.y_key else self.metrics_count))
        for i, idx in enumerate(batch_indices):
            start_idx = idx * self.shift
            end_idx = start_idx + self.data_len
            window = self.train_data[start_idx:end_idx]
            # if window.shape[0] == self.data_len:
            # Correct shape handling for x_xgb
            x_xgb = self.xgb_predictions[idx].reshape(-1, 1)  # Reshape to (100, 1)

            # Ensure x_xgb matches the data_len (100)
            if x_xgb.shape[0] != self.data_len:
                x_xgb = np.resize(x_xgb, (self.data_len, 1))

            X_1[i,] = window
            X_2[i,] = x_xgb
            y[i,] = (np.array(self.metrics.series[self.y_key])
                     .reshape(y.shape[1], y.shape[2])) if self.y_key else window
        self.logger.debug("Outcome generator data X_1=\n%s\n,X_2=\n%s\n,y=\n%s", X_1[0][:3], X_2[0][:3], y[0][:3])
        X_1_tensor = tf.convert_to_tensor(X_1, dtype=tf.float32)
        X_2_tensor = tf.convert_to_tensor(X_2, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

        return (X_1_tensor, X_2_tensor), y_tensor
