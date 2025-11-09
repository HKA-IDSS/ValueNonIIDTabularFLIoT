from abc import ABC
from logging import INFO
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
import keras
from flwr.common import log


class Model(ABC):
    def get_model(self):
        raise NotImplementedError()

    def set_model(self, model):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def predict_proba(self, x):
        raise NotImplementedError()

    def load_model(self, route):
        raise NotImplementedError()

    # def fit(self):
    #     raise NotImplementedError()


class KerasModel(Model):
    ml_model: keras.Sequential

    def __init__(self, model=None):
        self.ml_model = model

    def get_model(self):
        return self.ml_model

    def fit(self, x_train, y_train, epochs=100, batch_size=64, callbacks=None, config: Dict = None):
        pass

    def set_model(self, model_weights):
        self.ml_model.set_weights(model_weights)

    def predict(self, x):
        return self.ml_model.predict(x, verbose=0)

    def predict_proba(self, x):
        return self.ml_model.predict(x, verbose=0)

    def load_model(self, route):
        self.ml_model = keras.models.load_model(route)


class MLPModel(KerasModel):

    def __init__(self, model=None):
        super().__init__(model)

    def fit(self, x_train, y_train, epochs=100, batch_size=64, callbacks=None, config: Dict = None):
        self.ml_model.fit(x_train,
                          y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          callbacks=callbacks,
                          validation_split=0.2,
                          verbose=1)


class LSTMModel(KerasModel):
    def __init__(self, model=None):
        super().__init__(model)

    def fit(self, x_train, y_train, epochs=100, batch_size=64, callbacks=None, config: Dict = None):
        self.ml_model.fit(x_train,
                          y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          callbacks=callbacks,
                          validation_split=0.2,
                          verbose=1)


class DeepModel(KerasModel):

    def __init__(self, model=None):
        super().__init__(model)

    def fit(self, x_train, y_train, epochs=100, batch_size=64, callbacks=None, config: Dict = None):
        if callbacks is None:
            callbacks = [keras.callbacks.EarlyStopping(patience=10)]
        self.ml_model.fit(x_train,
                          y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          callbacks=callbacks,
                          validation_split=0.2,
                          verbose=2)


class DecisionTree(Model):
    pass
    # def get_model(self):
    #     return self.ml_model
    #
    # def set_model(self, model):
    #     self.ml_model.load_model(bytearray(model))
    #
    # def predict(self, x):
    #     return self.ml_model.predict(x)


class XGBoostModel(DecisionTree):
    ml_model: xgb.Booster

    def __init__(self, model=None):
        if model is None:
            self.ml_model = xgb.Booster()
        else:
            self.ml_model = model

    def get_model(self):
        return self.ml_model

    def set_model(self, tensors: bytes):
        # if type(tensors) == bytes:
        self.ml_model.load_model(bytearray(tensors))
        # else:
        #     self.ml_model.load_model(bytearray(tensors[0]))

    def fit(self, parameters, x_train, y_train, x_test=None, y_test=None, num_local_rounds=500,
            early_stopping_rounds=None, previous_xgb_model=None):
        train_dmatrix = xgb.DMatrix(x_train, label=np.argmax(y_train, axis=1))
        evals = [(train_dmatrix, "train")]
        if x_test is not None:
            valid_dmatrix = xgb.DMatrix(x_test, label=np.argmax(y_test, axis=1))
            evals.append((valid_dmatrix, "validate"))
        if early_stopping_rounds is None:
            early_stopping_rounds = min(num_local_rounds - 1, 10)
        self.ml_model = xgb.train(
            parameters,
            train_dmatrix,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_local_rounds,
            xgb_model=previous_xgb_model
        )

    def predict_proba(self, x_test: pd.DataFrame):
        # log(INFO, "Predict proba")
        d_matrix = xgb.DMatrix(x_test)
        predictions = self.ml_model.predict(d_matrix)
        # log(INFO, f"Predictions: {predictions}")
        if predictions.ndim == 1:
            predictions = np.array([[1-p, p] for p in predictions])
        return predictions

    def load_model(self, route: str):
        self.ml_model.load_model(route)
