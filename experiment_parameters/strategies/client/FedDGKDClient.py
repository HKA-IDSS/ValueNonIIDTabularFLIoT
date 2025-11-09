"""Experimental combination of DKD and GKD 

Remark: This strategy is just an experiment.
In the evaluation of the other experiments DKD performs in the most cases better than KD.
The reason for the combination is that GKD uses the loss of CE and KD to train the client model (student).
What happens if we replace the KD with DKD? Does the training and performance improve?
"""
from logging import INFO
from typing import Tuple, Any

import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.common.logger import log
from pandas import DataFrame

from util.Util import obtain_prediction_from_ensemble


class FedDGKDClient(fl.client.NumPyClient):
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

        # Perform: One round default training (Round starts with 1)
        if current_round == 1:
            self.model.fit(self.x_train, self.y_train, epochs=20, batch_size=self.batch_size)
        # Perform: Knowledge Distillation with global and local model
        else:
            log(INFO, "Decoupled Knowledge Distillation")
            self.global_model_ensemble.append(parameters)
            self.model.set_weights(parameters)
            new_weights, loss, ce_loss, dkd_loss = gkd_with_dkd(global_ensemble_params=self.global_model_ensemble,
                                                                local_model=self.model, \
                                                                x_train=self.x_train, y_train=self.y_train, beta=1,
                                                                alpha=8, temperature=4, model_type=self.model_type,
                                                                epochs=20)
            self.model.set_weights(new_weights)

        return self.model.get_weights(), len(self.x_train), metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        metrics = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        loss, accuracy = metrics[0], metrics[1]
        return loss, len(self.x_test), {"accuracy": accuracy}


''' Decoupled Knowledge Distillation

The general idea is to split the the prediction in targets (True labels: e.g. softmax value)
and non-targets (other labels: softmax values for them).

The loss will be calculated (DKD_Loss): (alpha * target_class_knowledge_distillation + beta * non_target_class_knowledge_distillation) * temperature**2
'''


def gkd_with_dkd(global_ensemble_params, local_model, x_train: DataFrame, y_train: DataFrame, \
                 beta: float, alpha: float, temperature: float, model_type, epochs: int):
    optimizer = tf.keras.optimizers.Adam()
    x_train_tensor = tf.convert_to_tensor(x_train)
    y_train_tensor = tf.convert_to_tensor(y_train)
    ce_loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    for _ in range(epochs):
        # Get prediction form global ensemble: Aggregation function: Average
        global_ensemble_predictions_logits_averaged = obtain_prediction_from_ensemble(model=model_type,
                                                                                      model_ensemble=global_ensemble_params,
                                                                                      unlabeled_synthetic_data=x_train_tensor)

        with tf.GradientTape() as tape:
            tape.watch(local_model.trainable_variables)
            client_predictions_logits = local_model(x_train_tensor)
            dkd_loss = decoupled_kd(logits_student=client_predictions_logits,
                                    logits_teacher=global_ensemble_predictions_logits_averaged, target=y_train_tensor,
                                    alpha=alpha, beta=beta, temperature=temperature)
            ce_loss = ce_loss_object(y_true=y_train_tensor, y_pred=client_predictions_logits)
            loss = ce_loss + dkd_loss

        # Apply gradients to local_model
        gradient = tape.gradient(loss, local_model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, local_model.trainable_variables))

    return local_model.get_weights(), loss, ce_loss, dkd_loss


''' Perform local KD
    
1.Set a global model with the received weights
2.Distill knowledge to local model based on the local data predictions
3.Perform one round of training on local data
'''


def decoupled_kd(logits_student, logits_teacher, target: DataFrame, alpha: float, beta: float, temperature: float):
    log(INFO, f"Calculating DKD loss...")
    kl_loss_object = tf.keras.losses.KLDivergence()

    pred_student = tf.nn.softmax(logits_student / temperature)
    pred_teacher = tf.nn.softmax(logits_teacher / temperature)

    bool_target_matrix = tf.cast(target, tf.bool)

    log(INFO, type(bool_target_matrix))

    pred_targets_student = pred_student[bool_target_matrix]
    pred_non_targets_student = pred_student[~bool_target_matrix]

    pred_targets_teacher = pred_teacher[bool_target_matrix]
    pred_non_targets_teacher = pred_teacher[~bool_target_matrix]

    pred_teacher_non_target = tf.nn.softmax(pred_non_targets_teacher / temperature)
    pred_student_non_target = tf.nn.softmax(pred_non_targets_student / temperature)

    # TCKD
    tckd_loss = (kl_loss_object(y_true=pred_targets_teacher, y_pred=tf.math.log(pred_targets_student)) + kl_loss_object(
        y_true=pred_teacher_non_target, y_pred=tf.math.log(pred_student_non_target)))

    # NCKD
    nckd_loss = (kl_loss_object(y_true=pred_teacher_non_target, y_pred=tf.math.log(pred_student_non_target)))

    dkd_loss = (alpha * tckd_loss + beta * nckd_loss) * (temperature ** 2)
    return dkd_loss
