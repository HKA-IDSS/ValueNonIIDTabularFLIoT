import ast
import os
from logging import INFO
from typing import Tuple, Any, List

import flwr as fl
# Define Flower client
import keras
import numpy as np
from flwr.common.logger import log
from numpy import ndarray
from pandas import DataFrame

from Definitions import ROOT_DIR
from experiment_parameters.model_builder.Model import KerasModel
from experiment_parameters.model_builder.ModelBuilder import Director
from metrics.Evaluator import evaluator
from metrics.Metrics import return_default_dict_of_metrics, DictOfMetrics
from metrics.Shapley_Values import ShapleyValuesNN
from util.Util import save_data_on_pickle, load_data_from_pickle_file, retrieve_gradient_from_dataset


class FedAvgClient(fl.client.NumPyClient):
    _model: KerasModel
    _model_name: str
    _x_train: DataFrame
    _x_test: DataFrame
    _y_train: DataFrame
    _y_test: DataFrame
    _batch_size: int
    # _shapley_values: ShapleyGTG
    _shapley_values: ShapleyValuesNN
    _client_number: int
    _metric_list: list
    _last_round_result = DictOfMetrics

    # Keep initial parameters to initialize SV.
    initial_parameters: List[ndarray] = None

    def __init__(self, model, x_train, x_test, y_train, y_test, client_number, metrics, total_count):
        self._model_name = model
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        # self._batch_size = batch_size
        self._client_number = client_number
        self._metric_list = metrics

    def get_parameters(self, config):
        return self._model.get_model().get_weights()

    # Here, the function belongs to the Tensorflow function fit. So, if implemented for tensorflow, just copy
    # and paste it here.
    def fit(self, parameters, config, global_logits=None) -> Tuple[Any, int, dict]:
        shape: int
        try:
            shape = self._y_train.shape[1]
        except:
            shape = 1

        if config["server_round"] == 1:
            log(INFO, "Config: {}".format(config))
            director = Director()
            # if self._model_name == "mlp":
            self._model = director.create_mlp(self._x_train.shape[1], shape, config)
            # elif self._model_name == "tabnet":
            #     self._model = director.create_tabnet(self._x_train.shape[1], self._y_train.shape[1], config).get_model()
            # TabNet does not initialize parameters, so they must be initialized by training. They are
            # overwritten afterwards.
            # self._model.fit(self._x_train[:2], self._y_train[:2], epochs=1,
            #                 batch_size=1, verbose=0)
            self._batch_size = config["batch_size"]

        if self.initial_parameters is not None:
            self.initial_parameters = parameters
            self._model.set_model(self.initial_parameters)

        if config["server_round"] == 1:
            self._model.set_model(parameters)
            self._last_round_result = evaluator(self._x_test,
                                                self._y_test,
                                                self._model,
                                                self._metric_list)

        labels, counts = np.unique(np.argmax(np.asarray(self._y_train), axis=1), return_counts=True)
        label_names = self._y_train.columns
        metrics = {"client_number": self._client_number}
        for label, count in zip(labels, counts):
            metrics["Label " + str(label_names[label])] = int(count)

        # if global_logits is not None:
        log(INFO, "Labels: {}".format(metrics))

        my_callbacks = [
            keras.callbacks.EarlyStopping(patience=2),
        ]

        if config["compute_shapley_values"] == 1:
            if shape == 1:
                loss_metric = "MAE"
            else:
                loss_metric = "CELoss"
            gradients = retrieve_gradient_from_dataset(self._model, self._x_train, self._y_train, loss_metric)
            save_data_on_pickle(ROOT_DIR +
                                os.sep + "data" +
                                os.sep + "pickled_information" +
                                os.sep + f"gradients_{self._client_number}.pkl",
                                gradients)

        self._model.fit(self._x_train, self._y_train, epochs=5,
                        batch_size=self._batch_size,
                        callbacks=my_callbacks)

        return self._model.get_model().get_weights(), len(self._x_train), metrics

    def evaluate(self, parameters, config):
        self._model.set_model(parameters)
        round_result = evaluator(self._x_test, self._y_test, self._model, self._metric_list)
        metric_results = round_result.return_flower_dict_as_str()

        if "MAE" in self._metric_list:
            loss = round_result.get_value_of_metric("MAE")
        else:
            loss = round_result.get_value_of_metric("CrossEntropyLoss")

        if config["compute_shapley_values"] == 1:
            client_weights = load_data_from_pickle_file(ROOT_DIR +
                                                        os.sep + "data" +
                                                        os.sep + "pickled_information" +
                                                        os.sep + "model.pkl")
            # client_weights = ast.literal_eval(config["all_models_from_clients"])
            if config["server_round"] == 1:
                number_of_clients = len(client_weights)
                # self._shapley_values = ShapleyGTG(number_of_clients)
                self._shapley_values = ShapleyValuesNN(self._x_test,
                                                       self._y_test,
                                                       config["num_rounds"],
                                                       self._metric_list)
                list_of_initial_metrics = self._last_round_result
                self._shapley_values.set_last_round_results(list_of_initial_metrics)
                # for client_name
                client_number_dict = ast.literal_eval(config["client_cid_number"])
                # self._shapley_values.set_client_index_dictionary(client_weights.keys())
                self._shapley_values.set_client_index_dictionary(client_number_dict)

            # log(INFO, "Podria usar los Shapley Values")
            # Shapley GTG
            # self._shapley_values.shapley_values_calculation(self._x_test,
            #                                                 self._y_test,
            #                                                 self._model,
            #                                                 list(client_weights.keys()),
            #                                                 client_weights,
            #                                                 round_result,
            #                                                 config["server_round"],
            #                                                 self._metric_list)
            self._shapley_values.shapley_values_calculation(self._model,
                                                            list(client_weights.keys()),
                                                            client_weights,
                                                            config["server_round"])
            self._shapley_values.set_last_round_results(round_result)
            round_sv = self._shapley_values.get_round_shapley_values(config["server_round"])
            sv_round_result = {"SV_" + client_id: str(round_sv[client_id])
                               for client_id, _ in self._shapley_values.get_client_index_dictionary().items()}
            metric_results = metric_results | sv_round_result

            if config["last_round"] == 1:
                log(INFO, "Columns sorted: {}".format(list(self._y_test.columns)))
                labels, counts = np.unique(np.argmax(np.asarray(self._y_test), axis=1), return_counts=True)
                testing_labels = {label: count for label, count in zip(labels, counts)}
                log(INFO, "Testing labels: {}".format(testing_labels))
                log(INFO, "")
                log(INFO, "=" * 50)
                shapley_values_total = self._shapley_values.get_shapley_values()
                for training_round in shapley_values_total.keys():
                    log(INFO, "Shapley Values in local_round {}: {}"
                        .format(training_round,
                                {client: str(sv) for client, sv in shapley_values_total[training_round].items()}))
                log(INFO, "=" * 50)
                log(INFO, "=" * 50)
                shapley_values_total = self._shapley_values.get_shapley_values()
                log(INFO, f"Shapley values total: {shapley_values_total}")
                result_dictionary = {client_id: return_default_dict_of_metrics(self._metric_list, self._y_test.shape[1])
                                     for client_id in client_weights.keys()}
                for training_round in shapley_values_total.keys():
                    single_round_dict = shapley_values_total.get(training_round)
                    result_dictionary = {k: result_dictionary[k] + single_round_dict[k] for k in
                                         result_dictionary.keys()}
                # result_dictionary = {k: str(shapley_values_client) for k, shapley_values_client in
                #                      result_dictionary.items()}
                log(INFO, "Result_dictionary: {}".format(result_dictionary))
                # os.makedirs(ROOT_DIR + os.sep + "metrics" + os.sep + "pickled_information", exist_ok=True)
                # save_data_on_pickle(ROOT_DIR +
                #                     os.sep + "metrics" +
                #                     os.sep + "pickled_information" +
                #                     os.sep + "sv" + str(self._client_number) + ".pkl", shapley_values_total)
                log(INFO, "=" * 50)

        log(INFO, f"Round result: {metric_results}")

        return loss, len(self._x_test), metric_results
