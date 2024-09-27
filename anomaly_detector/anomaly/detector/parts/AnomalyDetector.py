from typing import List

import numpy as np
import math
from keras import Sequential
from keras import regularizers
from keras.api.layers import LSTM, Dense, Dropout
from keras.src.optimizers import SGD
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from xgboost import XGBRegressor

from anomaly.detector.metrics.Metrics import Metrics
from anomaly.detector.parts.CompositeStreamDetector import Detector


# У AnomalyDetector и BehaviorDetector есть общая часть, необходим общий родитель
class AnomalyDetector(Detector):
    ANOMALY_KEY = "anomaly"

    # todo удалить lstm_size и dropout_rate4
    def __init__(self,
                 detectors_count,
                 anomaly_result_key,
                 lstm_size=128,
                 dropout_rate=0.2,
                 window=10,
                 window_step=5,
                 data_len=50,
                 metrics_values_max_size=50000,
                 threshold_xgb=0.75,
                 threshold_iforest=0.01,
                 threshold_ocsvm=0,
                 window_ocsvm=3,
                 logger_level="INFO"):
        super().__init__(logger_level=logger_level)
        self.data_len = data_len
        self.window = window
        self.window_step = window_step
        self.detectors_count = detectors_count
        self.models = self._init_model()
        self.anomaly_result_key = anomaly_result_key
        self.metrics_values_max_size = metrics_values_max_size
        self.threshold_xgb = threshold_xgb
        self.threshold_iforest = threshold_iforest
        self.threshold_ocsvm = threshold_ocsvm
        self.window_ocsvm = window_ocsvm
        self.metrics = None
        self.logger.info("Anomaly detector successfully init.")

    def _init_model(self):
        xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        iforest = IsolationForest(contamination=0.1)
        ocsvm = OneClassSVM(kernel='rbf', nu=0.05)

        models_ansamble = {
            'xgb': xgb,
            'iforest': iforest,
            'ocsvm': ocsvm
        }
        return models_ansamble

    def detect(self, metrics: Metrics) -> List[float]:
        if metrics.series_length() < self.data_len:
            raise IndexError(f"Wait metric size {self.data_len}, but income {metrics.series_length()}.")
        prepared_data_ = np.array([value for key, value in metrics.series.items()]).T
        self.logger.debug("Anomaly detector income data after prepare %s", prepared_data_)
        predicted_values_ = self._predict(np.array(prepared_data_))
        self.logger.debug("Anomaly detector predicted values is %s", predicted_values_)
        return predicted_values_.tolist()

    def train(self, metrics: Metrics):
        self.logger.debug("Income metrics into train %s", metrics)
        self.metrics = self._merge_with_current_metrics(metrics)
        anomaly_result_ = np.array(self.metrics.series[self.anomaly_result_key])
        features = np.array(
            [value for key, value in self.metrics.series.items() if key != self.anomaly_result_key]).T

        #todo переделать на raise
        assert features.shape[0] == anomaly_result_.shape[0], "Mismatch in feature and target lengths"

        self.models['xgb'].fit(features, anomaly_result_)
        self.models['iforest'].fit(features)
        self.models['ocsvm'].fit(features, anomaly_result_)

    #todo перенести в datagenerator
    def fine_tuning_checkpoint(self):
        pass

    def _predict(self, income_data):
        xgb_prediction_ = np.where(np.array(self.models['xgb'].predict(income_data)) > self.threshold_xgb, 1, 0)
        iforest_prediction_ = np.where(np.array(self.models['iforest'].decision_function(income_data)) < self.threshold_iforest, 1, 0)
        ocsvm_prediction_ = np.where(np.array(self.oscvm_smooth_(self.models['ocsvm'].decision_function(income_data))) > self.threshold_ocsvm, 1, 0)
        combined_predictions = xgb_prediction_ + iforest_prediction_ + ocsvm_prediction_
        final_prediction_ = np.where(combined_predictions >= 2, 1, 0)
        return final_prediction_



    def _merge_with_current_metrics(self, metrics):
        if self.metrics is None:
            return metrics
        else:
            return self.metrics.merge_new(metrics, new_size=self.metrics_values_max_size)

    def _calc_steps(self, data_len) -> int:
        return math.ceil((data_len - self.window + 1) / self.window_step)

    def __depricated_model_init_model(self, lstm_size, dropout_rate):
        model = Sequential()
        model.add(LSTM(lstm_size * 2,
                       activation='relu',
                       return_sequences=True,
                       input_shape=(self.data_len, self.detectors_count),
                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.0005)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_size, activation='tanh', return_sequences=True,
                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.0005)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_size // 2, activation='tanh', return_sequences=True,
                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.0005)))
        model.add(LSTM(lstm_size, activation='tanh', return_sequences=True,
                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.0005)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9, clipvalue=1.0), loss='mse')
        return model


    def __depricated_predict(self, income_data):
        start_list_ = [0] * (self.window - self.window_step)
        xgb_seria_ = np.array(start_list_)
        iforest_seria_ = np.array(start_list_)
        ocsvm_seria_ = np.array(start_list_)

        for i in range(self._calc_steps(len(income_data))):
            #нужно если window_step > 1
            xgb_seria_ = np.concatenate((xgb_seria_, np.array([0] * (self.window_step - 1))))
            iforest_seria_ = np.concatenate((iforest_seria_, np.array([0] * (self.window_step - 1))))
            ocsvm_seria_ = np.concatenate((ocsvm_seria_, np.array([0] * (self.window_step - 1))))

            windowed_data_ = income_data[i * self.window_step:i * self.window_step + self.window]
            self.logger.debug("Start %s iteration for windowed data %s", i, windowed_data_)
            xgb_prediction_ = self.models['xgb'].predict(windowed_data_)
            self.logger.debug(f"XGB predictions in anomaly detector {xgb_prediction_}")
            xgb_prediction_ = round(np.max(xgb_prediction_), 3)
            xgb_seria_ = np.append(xgb_seria_, xgb_prediction_)
            iforest_prediction_ = -self.models['iforest'].decision_function(windowed_data_)
            print(f"Iforest prediction in anomaly detector {iforest_prediction_}")
            iforest_prediction_ = round(np.abs(np.max(iforest_prediction_)), 3)
            iforest_seria_ = np.append(iforest_seria_, iforest_prediction_)
            ocsvm_prediction_ = -self.models['ocsvm'].decision_function(windowed_data_)
            print(f"Oscvm prediction in anomaly detector {ocsvm_prediction_}")
            ocsvm_prediction_ = round(np.average(ocsvm_prediction_))
            ocsvm_seria_ = np.append(ocsvm_seria_, ocsvm_prediction_)

            # nn_score = self.nn_model.predict(X)
            # nn_score = self.scaler.transform(nn_score.reshape(-1, 1)).flatten()
        anomaly_predictions_ = xgb_seria_ + iforest_seria_ + ocsvm_seria_
        self.logger.debug("The final anomaly predictions is %s", anomaly_predictions_)
        return anomaly_predictions_

    def oscvm_smooth_(self, prediction):
        oscvm_prediction_ = []
        shift = self.window_ocsvm - 1 // 2
        len_ = len(prediction)
        for i in range(len_):
            if i < shift or i > (len_ - shift):
                oscvm_prediction_.append(1 if prediction[i] > self.threshold_ocsvm else 0)
            else:
                oscvm_prediction_.append(1 if np.sum(prediction[i - shift: i + shift]) > self.threshold_ocsvm else 0)
        return oscvm_prediction_
