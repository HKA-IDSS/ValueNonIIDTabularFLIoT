import os
os.environ["KERAS_BACKEND"] = "torch"

from abc import ABC
from logging import INFO
from typing import Dict, Optional

import numpy as np
import torch
import xgboost as xgb
import keras
from torch import norm
from torch.utils.data import DataLoader

from util.Util import unflatten, flatten




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
    ml_model: keras.Model
    _optimizer: keras.optimizers.Optimizer
    _loss: keras.losses.Loss

    def __init__(self, model=None):
        self.ml_model = model

    def get_model(self):
        return self.ml_model

    def fit(self,
            train_dataloader: DataLoader,
            aggregation_method,
            epochs=1,
            batch_size=64,
            callbacks=None,
            server_cv=None,
            local_cv=None,
            config: Dict = None):
        pass

    def set_model(self, model_weights):
        self.ml_model.set_weights(model_weights)

    def predict(self, x):
        return self.ml_model.predict(x, verbose=0)

    # def predict_proba(self, x):
    #     return self.ml_model.predict(x, verbose=0)

    def load_model(self, route):
        self.ml_model = keras.models.load_model(route)


class MLPModel(KerasModel):

    def __init__(self, model=None):
        super().__init__(model)

    # def _train_one_epoch_scaffold(
    #         net: nn.Module,
    #         trainloader: DataLoader,
    #         device: torch.device,
    #         criterion: nn.Module,
    #         optimizer: ScaffoldOptimizer,
    #         server_cv: torch.Tensor,
    #         client_cv: torch.Tensor,
    # ) -> nn.Module:
    #     # pylint: disable=too-many-arguments
    #     """Train the network on the training set for one epoch."""
    #     for data, target in trainloader:
    #         data, target = data.to(device), target.to(device)
    #         optimizer.zero_grad()
    #         output = net(data)
    #         loss = criterion(output, target)
    #         loss.backward()
    #         optimizer.step_custom(server_cv, client_cv)
    #     return net

    def fit(self,
            train_dataloader: DataLoader,
            aggregation_method,
            epochs=1,
            batch_size=64,
            callbacks=None,
            server_cv=None,
            local_cv=None,
            config: Dict = None):

        global_model = keras.models.clone_model(self.ml_model)
        global_gradients = []
        for epoch in range(epochs):
            for step, (inputs, targets) in enumerate(train_dataloader):
                logits = self.ml_model(inputs)

                if aggregation_method == "FedProx":
                    # Compute proximal term
                    proximal_term = 0.0
                    for local_weights, global_weights in zip(self.ml_model.trainable_weights, global_model.trainable_weights):
                        proximal_term += (local_weights - global_weights).norm(2)

                    loss = self._loss(targets, logits) + (config['mu_prox'] / 2) * proximal_term
                else:
                    loss = self._loss(targets, logits)

                # Backward pass
                self.ml_model.zero_grad()
                trainable_weights = [v for v in self.ml_model.trainable_weights]

                # Call torch.Tensor.backward() on the loss to compute gradients
                # for the weights.
                loss.backward()
                gradients = [v.value.grad for v in trainable_weights]

                # global_gradients = global_gradients + gradients

                if aggregation_method == "Scaffold" and server_cv is not None and local_cv is not None:
                    gradients = [
                        g - s_c + l_c
                        for g, s_c, l_c in zip(gradients, server_cv, local_cv)
                    ]
                    # Update weights
                    # global_logits = global_model(inputs)
                    # local_cv = self._loss(targets, global_logits)
                with torch.no_grad():
                    self._optimizer.apply(gradients, trainable_weights)

                # Gradient Rewards part:
                model_difference = [(new_param - old_param)
                                    for old_param, new_param
                                    in zip(global_model.trainable_weights, self.ml_model.trainable_weights)]
                flattened = flatten(model_difference)
                norm_value = norm(flattened) + 1e-7
                gradient = unflatten(torch.multiply(torch.tensor(0.5), torch.div(flattened, norm_value)),
                                     model_difference)
                global_gradients.append(gradient)

        # SCAFFOLD: compute updated local control variate and its delta
        new_local_cv = None
        cv_delta = None
        if aggregation_method == "Scaffold" and server_cv is not None and local_cv is not None:
            # Option 2 from the paper: c_i^+ = c_i - c + (1/K*lr) * (x - y_i)
            # where x = global weights before training, y_i = local weights after
            K = epochs * len(train_dataloader)  # total number of local steps
            lr = self._optimizer.learning_rate  # Keras optimizer exposes this
            new_local_cv = [
                l_c - s_c + (1.0 / (K * float(lr))) * (old_w - new_w.value.detach())
                for l_c, s_c, old_w, new_w
                in zip(local_cv, server_cv, global_model.trainable_weights, self.ml_model.trainable_weights)
            ]
            cv_delta = [new_c - old_c for new_c, old_c in zip(new_local_cv, local_cv)]
        return global_gradients, new_local_cv, cv_delta


class LSTMModel(KerasModel):
    def __init__(self, model=None):
        super().__init__(model)

    def fit(self,
            x_train,
            y_train,
            aggregation_method,
            epochs=1,
            batch_size=64,
            callbacks=None,
            server_cv=None,
            local_cv=None,
            config: Dict = None):
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

    def fit(self,
            x_train,
            y_train,
            aggregation_method,
            epochs=1,
            batch_size=64,
            callbacks=None,
            config: Dict = None):
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

    def predict(self, d_matrix: xgb.DMatrix):
        # log(INFO, "Predict proba")
        # d_matrix = xgb.DMatrix(x_test)
        predictions = self.ml_model.predict(d_matrix)
        # log(INFO, f"Predictions: {predictions}")
        if predictions.ndim == 1:
            predictions = np.array([[1-p, p] for p in predictions])
        return predictions

    def load_model(self, route: str):
        self.ml_model.load_model(route)
