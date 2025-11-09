""" Part of experimental FedLF strategy"""
from logging import INFO
from typing import Tuple, Any

import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.common.logger import log
from pandas import DataFrame


class FedLFClient(fl.client.NumPyClient):
    model: Any
    x_train: DataFrame
    x_test: DataFrame
    y_train: DataFrame
    y_test: DataFrame
    batch_size: int

    def __init__(self, model, x_train, x_test, y_train, y_test, batch_size):
        self.model = model.get_model()
        self.global_model = model.get_model()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.model_weights_history = []

    def get_parameters(self, config):
        return self.model.get_weights()

    # Here, the function belongs to the Tensorflow function fit. So, if implemented for tensorflow, just copy
    # and paste it here.
    def fit(self, parameters, config, global_logits=None) -> Tuple[Any, int, dict]:
        log(INFO, f"Received config: \n {config}")
        current_round = config["current_round"]
        labels, counts = np.unique(np.argmax(np.asarray(self.y_train), axis=1), return_counts=True)
        metrics = {}
        for label, count in zip(labels, counts):
            metrics["Label " + str(label)] = int(count)
        log(INFO, "Metrics: {}".format(metrics))

        # Perform: One round default training (Round starts with 1)
        if current_round == 1:
            log(INFO, f"Perform inital training")
            self.model.set_weights(parameters)
            self.model.fit(self.x_train, self.y_train, epochs=20, batch_size=self.batch_size)

        # Perform: Knowledge Distillation with global and local model
        else:
            log(INFO, f"Knowledge Distillation")
            self.local_knowledge_distillation(local_params=self.model_weights_history[current_round - 2],
                                              teacher_params=parameters, knowledge_steps=20)

        # Store current local model for next round
        self.model_weights_history.append(self.model.get_weights())
        return self.model.get_weights(), len(self.x_train), metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        metrics = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        loss, accuracy = metrics[0], metrics[1]
        return loss, len(self.x_test), {"accuracy": accuracy}

    ''' Perform local KD

    1.Set a global model with the received weights
    2.Distill knowledge to local model based on the local data predictions
    3.Perform one round of training on local data
    '''

    def local_knowledge_distillation(self, local_params, teacher_params, knowledge_steps: int) -> None:

        optimizer = tf.keras.optimizers.Adam()
        loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        converted_x_train = tf.convert_to_tensor(self.x_train)

        self.global_model.set_weights(teacher_params)
        self.model.set_weights(local_params)

        for _ in range(knowledge_steps):
            global_model_predictions = self.global_model(converted_x_train)

            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)
                client_predictions = self.model(converted_x_train)
                loss = loss_func(global_model_predictions, client_predictions)

            gradient = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
