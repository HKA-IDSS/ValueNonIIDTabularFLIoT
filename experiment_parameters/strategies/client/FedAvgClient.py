import ast
import os
import pickle
import time
from logging import INFO
from typing import Tuple, Any, List, Optional

import flwr as fl
# Define Flower client
import keras
import numpy as np
import pandas as pd
import torch
from flwr.common.logger import log
from numpy import ndarray
from pandas import DataFrame
from torch.utils.data import DataLoader

from Definitions import ROOT_DIR
from experiment_parameters.model_builder.Model import KerasModel
from experiment_parameters.model_builder.ModelBuilder import Director
from metrics.Evaluator import partial_computation, evaluator
# from metrics.Evaluator import evaluator
from metrics.Metrics import return_default_dict_of_metrics, DictOfMetrics, AggregatableMeasuresClassification, \
    AggregatableMeasures
from metrics.Shapley_Values import get_all_partial_aggregation_evaluation_results, get_clients_powerset, \
    ShapleyValuesNN, get_all_metrics_from_partial_results
# from metrics.Shapley_Values import ShapleyValuesNN
from util.Util import save_data_on_pickle, load_data_from_pickle_file


os.environ["KERAS_BACKEND"] = "torch"
# assert torch.cuda.is_available(), "A cuda device is required to run this tutorial"

class FedAvgClient(fl.client.NumPyClient):
    _model: KerasModel
    _model_name: str
    _x_train: DataFrame
    _x_test: DataFrame
    _y_train: DataFrame
    _y_test: np.ndarray
    _train_dataloader: DataLoader
    _test_dataloader: DataLoader
    _batch_size: int
    _shapley_values: ShapleyValuesNN
    _client_number: int
    _metric_list: list
    _last_round_partial_computations: AggregatableMeasures
    _local_cv: List[np.ndarray] = None
    _times_dataframe: DataFrame
    _route_name: str
    _columns: list[str]

    # Keep initial parameters to initialize SV.
    initial_parameters: List[ndarray] = None

    def __init__(self, model,
                 route_to_dataset,
                 x_train,
                 x_test,
                 y_train,
                 y_test,
                 client_number,
                 metrics,
                 local_training_method,
                 subpopulation_x_test_data = None
    ):
        self._model_name = model
        self._route_name = route_to_dataset
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test.to_numpy()
        self._client_number = client_number
        self._metric_list = metrics
        self._local_training_method = local_training_method
        self._times_dataframe = pd.DataFrame()
        self._columns = list(self._y_train.columns)

        if subpopulation_x_test_data is not None:
            # Slice down to exactly this client's rows, using the shared original index.
            self._x_test_subpop_values = subpopulation_x_test_data.loc[self._x_test.index]
        else:
            self._x_test_subpop_values = None

    def get_parameters(self, config):
        return self._model.get_model().get_weights()

    # Here, the function belongs to the Tensorflow function fit. So, if implemented for tensorflow, just copy
    # and paste it here.
    def fit(self, parameters, config, global_logits=None) -> Tuple[Any, int, dict]:
        log(INFO, "Fit in FedAvgClient")
        if not config["early_stop"]:
            if config["server_round"] == 1:
                shape: int
                try:
                    shape = self._y_train.shape[1]
                except:
                    shape = 1

                log(INFO, "Config: {}".format(config))
                director = Director()
                # if self._model_name == "mlp":
                self._model = director.create_mlp(self._x_train.shape[1], shape, config)
                # elif self._model_name == "tabnet":
                #     self._model = director.create_tabnet(self._x_train.shape[1], self._y_train.shape[1], config).get_model()

                if config["compute_shapley_values"] == 1:
                    self._times_dataframe = pd.DataFrame()

                if self._local_training_method == "scaffold":
                    for layer in self._model.get_model().layers:
                        self._local_cv.append(np.zeros((len(layer.trainable_weights),)))

                self._batch_size = config["batch_size"]
                self._mu_prox = config.get("mu_prox", 0)
                # Create DataLoaders for the Datasets
                train_dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(self._x_train.to_numpy()), torch.from_numpy(self._y_train.to_numpy())
                )
                self._train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True
                )

                test_dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(self._x_test.to_numpy())
                )
                self._test_dataloader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True
                )

            if self.initial_parameters is not None:
                self.initial_parameters = parameters
                self._model.set_model(self.initial_parameters)

            server_cv = None
            local_cv = None

            if config.get("aggregation_method") == "Scaffold":
                # Server sends its global c via config (as list of numpy arrays, pickled)
                server_cv = pickle.loads(config["server_cv"])
                server_cv = [np.zeros(torch.tensor(a).shape) for a in server_cv]

                # Initialize c_i to zeros on first round
                if self._local_cv is None:
                    self._local_cv = [np.zeros(torch.tensor(a).shape) for a in server_cv]
                local_cv = self._local_cv

            if config["server_round"] == 1:
                predictions = self._model.predict(self._test_dataloader)
                log(INFO, "Predictions shape: {}".format(predictions.shape))
                self._last_round_partial_computations = partial_computation(predictions,
                                                                            self._y_test,
                                                                            self._columns,
                                                                            self._x_test_subpop_values)
                self._last_round_result = get_all_metrics_from_partial_results(self._last_round_partial_computations,
                                                                                self._metric_list)

            labels, counts = np.unique(np.argmax(np.asarray(self._y_train), axis=1), return_counts=True)
            label_names = self._y_train.columns
            metrics = {"client_number": self._client_number}
            for label, count in zip(labels, counts):
                metrics["Label " + str(label_names[label])] = int(count)

            # if global_logits is not None:
            log(INFO, "Labels: {}".format(metrics))

            # my_callbacks = [
            #     keras.callbacks.EarlyStopping(patience=2),
            # ]
            if config["compute_shapley_values"] == 1:
                self._times_dataframe.loc[config["server_round"], "BeforeLocalTraining"] = time.time()

            # Added here, as it is not passed always
            config["mu_prox"] = self._mu_prox

            if self._local_training_method == "Scaffold":
                # Send config, so model knows it is fed_prox
                gradients, new_local_cv, cv_delta = self._model.fit(self._train_dataloader, epochs=1,
                                            batch_size=self._batch_size,
                                            callbacks=None,
                                            aggregation_method="Scaffold",
                                            server_cv=config["server_cv"],
                                            local_cv=self._local_cv,
                                            config=config)
                if new_local_cv is not None:
                    self._local_cv = new_local_cv  # persist c_i for next round

                # Send cv_delta to server so it can update c
                if cv_delta is not None:
                    metrics["cv_delta"] = pickle.dumps([t.numpy() for t in cv_delta])

            else:
                gradients, _, _ = self._model.fit(self._train_dataloader,
                                            self._local_training_method,
                                            epochs=1,
                                            batch_size=self._batch_size,
                                            callbacks=None,
                                            config=config)

            if config["compute_shapley_values"] == 1:
                self._times_dataframe.loc[config["server_round"], "AfterLocalTraining"] = time.time()
                # if shape == 1:
                #     loss_metric = "MAE"
                # else:
                #     loss_metric = "CELoss"
                # gradients = retrieve_gradient_from_dataset(self._model, self._x_train, self._y_train, loss_metric)
                # log(INFO, "Gradients: {}".format(gradients))
                save_data_on_pickle(ROOT_DIR +
                                    os.sep + "data" +
                                    os.sep + "pickled_information" +
                                    os.sep + f"gradients_{self._client_number}.pkl",
                                    gradients)

            return self._model.get_model().get_weights(), len(self._x_train), metrics

        else:
            return self._model.get_model().get_weights(), len(self._x_train), {}

    def evaluate(self, parameters, config):
        log(INFO, "Evaluate in FedAvgClient")
        if not config["early_stop"]:
            metrics_dict = {}
            if config["compute_shapley_values"] == 1:
                client_weights = load_data_from_pickle_file(ROOT_DIR +
                                                            os.sep + "data" +
                                                            os.sep + "pickled_information" +
                                                            os.sep + "model.pkl")
                # client_weights = ast.literal_eval(config["all_models_from_clients"])
                # Two different dictionaries are needed, for powerset and merging models.
                client_number_dict = ast.literal_eval(config["client_cid_number"])
                number_client_dict = ast.literal_eval(config["number_client_cid"])
                number_of_clients = len(client_weights)
                if config["server_round"] == 1:
                    self._shapley_values = ShapleyValuesNN(config["num_rounds"], self._metric_list, self._y_test.shape[1])
                    # list_of_initial_metrics = self._last_round_result
                    # self._shapley_values.set_last_round_results(list_of_initial_metrics)
                    # self._shapley_values.set_client_index_dictionary(client_weights.keys())
                    self._shapley_values.set_client_index_dictionary(client_number_dict)

                self._times_dataframe.loc[config["server_round"], "BeforeLocalPartialEvaluations"] = time.time()
                partial_computations_dictionary = get_all_partial_aggregation_evaluation_results(self._test_dataloader,
                                                                                                 self._y_test,
                                                                                                 self._model,
                                                                                                 self._columns,
                                                                                                 client_weights,
                                                                                                 number_client_dict,
                                                                                                 self._last_round_partial_computations,
                                                                                                 get_clients_powerset(client_number_dict),
                                                                                                 self._x_test_subpop_values)
                self._times_dataframe.loc[config["server_round"], "AfterLocalPartialEvaluations"] = time.time()
                metric_powerset = {k: get_all_metrics_from_partial_results(partial_computations_dictionary[k],
                                                                        self._metric_list)
                                   for k in partial_computations_dictionary.keys()}
                self._shapley_values.shapley_values_calculation(metric_powerset,
                                                                list(client_number_dict.keys()),
                                                                config["server_round"])
                metrics_dict["SV_partial_computations"] = pickle.dumps(partial_computations_dictionary)
                metrics_dict[f"SV_local_client"] = pickle.dumps(
                    self._shapley_values.get_round_shapley_values(config["server_round"])
                )
                # self._shapley_values.set_last_round_results(round_result)
                # round_sv = self._shapley_values.get_round_shapley_values(config["server_round"])
                # sv_round_result = {"SV_partial_computations" + client_id: str(round_sv[client_id])
                #                    for client_id, _ in self._shapley_values.get_client_index_dictionary().items()}

                # if config["last_round"] == 1:
                #     # log(INFO, "Columns sorted: {}".format(list(self._y_test.columns)))
                #     labels, counts = np.unique(np.argmax(np.asarray(self._y_test), axis=1), return_counts=True)
                #     testing_labels = {label: count for label, count in zip(labels, counts)}
                #     log(INFO, "Testing labels: {}".format(testing_labels))
                #     log(INFO, "")
                #     log(INFO, "=" * 50)
                #     shapley_values_total = self._shapley_values.get_shapley_values()
                #     for training_round in shapley_values_total.keys():
                #         log(INFO, "Shapley Values in local_round {}: {}"
                #             .format(training_round,
                #                     {client: str(sv) for client, sv in shapley_values_total[training_round].items()}))
                #     log(INFO, "=" * 50)
                #     log(INFO, "=" * 50)
                #     shapley_values_total = self._shapley_values.get_shapley_values()
                #     log(INFO, f"Shapley values total: {shapley_values_total}")
                #     result_dictionary = {client_id: return_default_dict_of_metrics(self._metric_list, self._y_test.shape[1])
                #                          for client_id in client_weights.keys()}
                #     for training_round in shapley_values_total.keys():
                #         single_round_dict = shapley_values_total.get(training_round)
                #         result_dictionary = {k: result_dictionary[k] + single_round_dict[k] for k in
                #                              result_dictionary.keys()}
                #     # result_dictionary = {k: str(shapley_values_client) for k, shapley_values_client in
                #     #                      result_dictionary.items()}
                #     log(INFO, "Result_dictionary: {}".format(result_dictionary))
                #     # os.makedirs(ROOT_DIR + os.sep + "metrics" + os.sep + "pickled_information", exist_ok=True)
                #     # save_data_on_pickle(ROOT_DIR +
                #     #                     os.sep + "metrics" +
                #     #                     os.sep + "pickled_information" +
                #     #                     os.sep + "sv" + str(self._client_number) + ".pkl", shapley_values_total)
                #     log(INFO, "=" * 50)
                #     times_route = config["time_results_path"]
                #     os.makedirs(str(times_route), exist_ok=True)
                #     self._times_dataframe.to_csv(str(times_route) +
                #                                  os.sep + str(self._client_number))

            # log(INFO, f"Round result: {metric_results}")
            self._times_dataframe.loc[config["server_round"], "BeforeEvaluation"] = time.time()
            self._model.set_model(parameters)

            last_round_predictions = self._model.predict(self._test_dataloader)
            self._last_round_partial_computations = partial_computation(last_round_predictions,
                                                                        self._y_test,
                                                                        self._columns,
                                                                        self._x_test_subpop_values)
            # log(INFO, f"Partial computations: {self._last_round_partial_computations}")
            metrics_dict["partial_evaluations"] = pickle.dumps(self._last_round_partial_computations)
            round_result = evaluator(self._last_round_partial_computations, self._metric_list)
            if "MSE" in self._metric_list:
                if self._x_test_subpop_values is not None:
                    loss = round_result.get_global().get_value_of_metric("MSE")
                else:
                    loss = round_result.get_value_of_metric("MSE")
            else:
                if self._x_test_subpop_values is not None:
                    loss = round_result.get_global().get_value_of_metric("CrossEntropyLoss")
                else:
                    loss = round_result.get_value_of_metric("CrossEntropyLoss")

            self._times_dataframe.loc[config["server_round"], "AfterEvaluation"] = time.time()

            if config.get("shapley_values", 0) == 1 and config.get("last_round", False):
                times_route = config["time_results_path"]
                os.makedirs(str(times_route), exist_ok=True)
                self._times_dataframe.to_csv(str(times_route) +
                                             os.sep + str(self._client_number))

            return loss, len(self._x_test), metrics_dict

        else:
            if config["compute_shapley_values"] == 1:
                times_route = config["time_results_path"]
                os.makedirs(str(times_route), exist_ok=True)
                self._times_dataframe.to_csv(str(times_route) +
                                             os.sep + str(self._client_number))
            return 10000000000.0, len(self._x_test), {}
