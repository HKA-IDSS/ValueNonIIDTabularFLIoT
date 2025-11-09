import os
import pickle
from logging import INFO
from typing import Optional, Tuple, List, Dict

import keras
import numpy as np
import pandas as pd
import regex
import tensorflow as tf
import yaml
from flwr.common import NDArray, log
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from yaml.loader import SafeLoader

# from experiment_parameters.data_preparation.Dataset import directory_of_data


def arg_parser(file) -> Dict:
    with open(file) as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


def one_hot_encode_labels(y):
    y_reshaped = np.reshape(y.values, (-1, 1))
    encoder = OneHotEncoder().fit(y_reshaped)
    y_encoded = encoder.transform(y_reshaped)
    labels = encoder.get_feature_names_out()
    y_one_hot_encoded = pd.DataFrame(y_encoded.toarray(), index=y.index, columns=labels)
    return y_one_hot_encoded, labels


def return_to_single_label(y):
    y_to_one_label = np.argmax(np.asarray(y), axis=1)
    y = pd.DataFrame(y_to_one_label, columns=["label"])
    return y


def from_string_to_dict(string_to_parse):
    splits = regex.findall(r'(\w*:\[.*?\])|(\w*:\D?\d*\.?\d*e?-?\d\d?)', string_to_parse)
    splits = [''.join(regex_tuple) for regex_tuple in splits]
    splits = [tuple(key_value.split(':')) for key_value in splits]
    return {metric: value for metric, value in splits}


def get_test_data(directory_of_data):
    X_test, y_test = pd.DataFrame(), pd.DataFrame()

    for file in os.listdir(directory_of_data):
        if "test" in file:
            if "_X_" in file:
                X_test = pd.concat([X_test, pd.read_csv(directory_of_data + os.sep + file, index_col=0)])
            elif "_y_" in file:
                y_test = pd.concat([y_test, pd.read_csv(directory_of_data + os.sep + file, index_col=0)])

    X_test.sort_index(inplace=True)
    y_test.sort_index(inplace=True)
    return X_test, y_test

""" Evaluate ensemble model with given weights

Based on the weights of each model in the ensemble a new model will be created.
The evaluation will be performed on the given test dataset.
The resulting prediction is the average over each model.
All losses and the accuracy will be returned.
"""


def evaluate_synthetic_ensemble(dataset, model, ensemble: List[List[NDArray]]) -> Optional[Tuple[float, float]]:
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    x_test, y_test = dataset.get_test_data()

    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x_test)
    x_test = pd.DataFrame(x_scaled, columns=x_test.columns)
    x_test.fillna(0, inplace=True)

    y_reshaped = np.reshape(y_test.values, (-1, 1))
    encoder_result = OneHotEncoder().fit_transform(y_reshaped)
    y_one_hot_encoded = pd.DataFrame(encoder_result.toarray(), index=y_test.index)

    num_of_models = len(ensemble)
    ensemble_predictions = []
    for weights_of_model in ensemble:
        model.set_weights(weights_of_model)
        logits = model.predict(x_test)
        predictions = tf.nn.softmax(logits)
        ensemble_predictions.append(predictions)

    # TODO: Probably, this can be written in a better way.
    first_array = ensemble_predictions.pop(0)
    for remaining_array in ensemble_predictions:
        first_array = np.add(first_array, remaining_array)
    ensemble_predictions_averaged = np.divide(first_array, num_of_models)

    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(y_one_hot_encoded, ensemble_predictions_averaged).numpy()

    accuracy_metric = tf.keras.metrics.Accuracy()
    accuracy_metric.reset_state()
    accuracy_metric.update_state(np.argmax(np.asarray(y_one_hot_encoded), axis=1),
                                 np.argmax(np.asarray(ensemble_predictions_averaged), axis=1))
    accuracy = accuracy_metric.result().numpy()

    # log(INFO, "Evaluate function running")
    # model.set_weights(weights)  # Update model with the latest parameters
    # loss, accuracy = model.evaluate(x_test, y_one_hot_encoded)

    return loss, accuracy


"""Obtain prediction over ensemble model

For each client create a model with client weights and perform predictions
Average the predicitions and return them
"""


# def obtain_prediction_from_ensemble(model, model_ensemble: List[List[NDArray]], unlabeled_synthetic_data) -> NDArray:

def obtain_prediction_from_ensemble(model, model_ensemble, unlabeled_synthetic_data):
    num_of_models = len(model_ensemble)
    ensemble_predictions = []
    for weights_of_model in model_ensemble:
        model.set_weights(weights_of_model)
        logits = model.predict(unlabeled_synthetic_data)
        ensemble_predictions.append(logits)

    # TODO: Probably, this can be written in a better way.
    first_array = ensemble_predictions.pop(0)
    for remaining_array in ensemble_predictions:
        first_array = np.add(first_array, remaining_array)
    ensemble_predictions_averaged = np.divide(first_array, num_of_models)

    return ensemble_predictions_averaged


def save_data_on_pickle(path_file, data):
    file = open(path_file, 'wb')
    pickle.dump(data, file)
    file.close()


def load_data_from_pickle_file(file_path):
    file = open(file_path, 'rb')
    client_weights = pickle.load(file)
    file.close()
    return client_weights


def retrieve_gradient_from_dataset(model, X_data, y_data, metric):
    tensors = tf.convert_to_tensor(X_data.to_numpy())  # Now passing a list of tensors.
    # log(INFO, f"Tensors: {tensors}")
    with tf.GradientTape() as tape:
        recorded_model = model.get_model()
        predictions = recorded_model(tensors)
        # log(INFO, f"Predictions: {predictions}")
        if metric == "MAE":
            cce = keras.losses.MeanAbsoluteError()
        else:
            cce = keras.losses.CategoricalCrossentropy()
        loss = cce(y_data, predictions)
        # log(INFO, f"Loss: {loss}")

    # log(INFO, f"Trainable variables: {model.get_model().trainable_variables}")
    gradients = tape.gradient(loss, recorded_model.trainable_variables)
    # log(INFO, f"Gradients: {gradients}")

    return gradients


# def shapley_values_decentralized(server_round, client_weights):
#     config = {
#         "server_round": server_round,
#         "client_weights": client_weights
#     }
#
#     return config


"""Obtain server model

Returns: the server model with given weights
"""


def obtain_server_model(model_type, weights: List[NDArray]):
    model = model_type
    model.set_weights(weights)
    return model


if __name__ == "__main__":
    metric = 'SV_526283ec68ae4011a84156cb875adf9c'
    value = ('CrossEntropyLoss:0.08598822269603118,Accuracy:0.03999897530484682,F1Score:[0.015640382661589035, 0.20077469556912503],MCC:0.15639464215108562,F1ScoreMacro:0.10820753911535702,F1ScoreMicro:0.03999897530484682')

    print(from_string_to_dict(value))
