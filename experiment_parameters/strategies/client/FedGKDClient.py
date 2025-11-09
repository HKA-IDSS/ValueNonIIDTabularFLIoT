from logging import INFO
from typing import Tuple, Any

import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.common.logger import log
from pandas import DataFrame

from util.Util import obtain_prediction_from_ensemble


class FedGKDClient(fl.client.NumPyClient):
    model: Any
    x_train: DataFrame
    x_test: DataFrame
    y_train: DataFrame
    y_test: DataFrame
    batch_size: int

    def __init__(self, model, x_train, x_test, y_train, y_test, batch_size):
        self.model = model.get_model()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.model_type = model.get_model()
        self.global_model_ensemble = []

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config, global_logits=None) -> Tuple[Any, int, dict]:
        log(INFO, f"Received config: \n {config}")
        current_round = config["current_round"]

        labels, counts = np.unique(np.argmax(np.asarray(self.y_train), axis=1), return_counts=True)
        metrics = {}
        for label, count in zip(labels, counts):
            metrics["Label " + str(label)] = int(count)
        log(INFO, "Metrics: {}".format(metrics))

        self.model.set_weights(parameters)

        # The first round is a normal fit, because there is no global model in the first round. (The paper doesn't mention this step)
        if current_round == 1:
            self.model.fit(self.x_train, self.y_train, epochs=20, batch_size=self.batch_size)
        else:
            self.global_model_ensemble.append(parameters)
            self.model.set_weights(parameters)
            new_weights, loss, ce_loss, kl_loss = global_knowledge_distillation(
                global_ensemble_params=self.global_model_ensemble, local_model=self.model, \
                x_train=self.x_train, y_train=self.y_train, model_type=self.model_type, rounds=20,
                model_coefficient=0.2)
            self.model.set_weights(new_weights)

        return self.model.get_weights(), len(self.x_train), metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        metrics = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        loss, accuracy = metrics[0], metrics[1]
        return loss, len(self.x_test), {"accuracy": accuracy}


''' Global Knowledge Distillation

Global ensemble model is the average over the history of all global models.
The loss for training is calculated with CE Loss over true labels and the local model and the kd loss for the ensemble and local model predictions.
'''


def global_knowledge_distillation(global_ensemble_params, local_model, x_train: DataFrame, y_train: DataFrame,
                                  model_type, \
                                  model_coefficient: float = 0.5, rounds: int = 10):
    # Init requirements
    optimizer = tf.keras.optimizers.Adam()
    kd_loss_object = tf.keras.losses.KLDivergence()
    x_train_tensor = tf.convert_to_tensor(x_train)
    y_train_tensor = tf.convert_to_tensor(y_train)
    ce_loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Train Loop
    for learning_step in range(rounds):
        # Get prediction form global ensemble: Aggregation function: Average
        global_ensemble_predictions_logits_averaged = obtain_prediction_from_ensemble(model=model_type,
                                                                                      model_ensemble=global_ensemble_params,
                                                                                      unlabeled_synthetic_data=x_train_tensor)

        with tf.GradientTape() as tape:
            tape.watch(local_model.trainable_variables)
            client_predictions_logits = local_model(x_train_tensor)

            # KLD: First argument the server model, the second one is the client
            kl_loss = kd_loss_object(global_ensemble_predictions_logits_averaged, client_predictions_logits)
            # CE: Uses local model and y_train
            ce_loss = ce_loss_object(y_true=y_train_tensor, y_pred=client_predictions_logits)

            # n_k are all trainings sample, model_coefficient varies (0.2 for ResNet-8 and DistillBert(Values from paper))
            loss = ce_loss + (model_coefficient / (2 * len(x_train)) * kl_loss)

        # Apply gradients to local_model
        gradient = tape.gradient(loss, local_model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, local_model.trainable_variables))

    return local_model.get_weights(), loss, ce_loss, kl_loss
