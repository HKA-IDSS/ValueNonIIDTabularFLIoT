import ast
import json
import math
import os
import pickle
import time
import torch
from torch.utils.data import DataLoader
from logging import WARNING, INFO
from typing import Callable, Dict, List, Optional, Tuple, Union

import joblib
import keras
import numpy as np
import optuna
import pandas as pd
from flwr.common import Parameters, Scalar, FitRes, NDArray, parameters_to_ndarrays, \
    ndarrays_to_parameters, EvaluateIns, EvaluateRes, FitIns
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import weighted_loss_avg

from Definitions import ROOT_DIR
from experiment_parameters.aggregation_processes.aggregate import aggregate_nn
from experiment_parameters.model_builder.Model import Model, KerasModel
from metrics.Evaluator import partial_computation, evaluator
from metrics.GradientRewards import GradientRewards
from metrics.Metrics import return_default_dict_of_metrics, AggregatableMeasures, return_default_partial_computations, \
    SubpopulationDict
from metrics.ResultManager import FlowerMetricManager, SVCompatibleFlowerMetricManager
from metrics.Shapley_Values import ShapleyValuesNN, get_all_metrics_from_partial_results, \
    get_all_partial_aggregation_evaluation_results, get_clients_powerset
from util.Util import save_data_on_pickle, get_test_data

DEPRECATION_WARNING = """
DEPRECATION WARNING: deprecated `eval_fn` return format

    loss, accuracy

move to

    loss, {"accuracy": accuracy}

instead. Note that compatibility with the deprecated return format will be
removed in a future release.
"""

DEPRECATION_WARNING_INITIAL_PARAMETERS = """
DEPRECATION WARNING: deprecated initial parameter type

    flwr.common.Weights (i.e., List[np.ndarray])

will be removed in a future update, move to

    flwr.common.Parameters

instead. Use

    parameters = flwr.common.weights_to_parameters(weights)

to easily transform `Weights` to `Parameters`.
"""

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_eval_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_eval_clients`.
"""


def save_data_on_joblib(path_file, data):
    # file = open(path_file, 'wb')
    joblib.dump(data, path_file)
    # file.close()


def save_model_with_tensorflow(path_file, model):
    model.save(path_file + ".keras")


class FedAvgRewritten(FedAvg):
    """Configurable fedavg strategy implementation."""
    _shapley_values: ShapleyValuesNN
    _shapley_values_decentralized: ShapleyValuesNN
    _gradient_rewards: GradientRewards
    _max_round: int
    _compute_shapley_values: bool
    _model_final_name: str
    _id_and_client_number: List[tuple]
    _dataset_metrics: Optional[Union[FlowerMetricManager, SVCompatibleFlowerMetricManager]]

    # _former

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
            self,
            max_round: int,
            strategy_aggregation,
            data_loader_evaluation,
            model: Model,
            final_training,
            shapley_values,
            gradient_rewards,
            metric_list,
            model_final_name,
            compute_shapley_values,
            result_path,
            experiment_name,
            study,
            trial,
            y_test=None,
            x_test_subpopulation_values=None,
            fraction_fit: float = 0.1,
            fraction_eval: float = 0.1,
            min_fit_clients: int = 2,
            min_eval_clients: int = 2,
            min_available_clients: int = 2,
            eval_fn: Optional[
                Callable[[NDArray], Optional[Tuple[float, Dict[str, Scalar]]]]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
    ) -> None:

        super().__init__()
        if (
                min_fit_clients > min_available_clients
                or min_eval_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters

        # Added attributes
        # self._evaluation_dataset = dataset_factory.get_dataset()
        self._strategy_aggregation = strategy_aggregation
        self._model: Model = model
        self._final_training = final_training
        self._gradient_rewards = gradient_rewards
        self._metric_list = metric_list
        self._clients_weights = None
        self._dataset_metrics = None
        self._result_path = result_path
        self._max_round = max_round
        self._compute_shapley_values = compute_shapley_values
        self._model_final_name = model_final_name
        self._experiment_name = experiment_name
        self._last_round_result = None
        self._this_round_result = None

        self._shapley_values = shapley_values

        # Information for evaluation purposes.
        self._total_test_samples = None
        self._samples_per_class = None
        self._data_validator = data_loader_evaluation
        self.y_test = y_test
        self.target_classes = None
        self.clients_data_size_dict = None
        self.clients_list: list
        self.study = study
        self.trial: optuna.Trial = trial

        self._shapley_values_decentralized = ShapleyValuesNN(self._max_round, self._metric_list, self.y_test.shape[1])

        # Early stop
        self._max_patience = 5
        self._current_patience = 0
        self._former_loss = 10000000000.0
        self._best_round = None
        self._early_stop = False
        self._best_model: keras.Model

        # Scaffold
        self._server_cv: Optional[List[np.ndarray]] = None

        # Times being monitored
        self._times_dataframe = pd.DataFrame()

        # Feature based mavericks
        self.x_test_subpopulation_values = x_test_subpopulation_values

    def _log_metrics(self, metrics, evaluator_label, server_round):
        """
        Writes `metrics` into self._dataset_metrics. Transparently handles
        both a bare DictOfMetrics (dataset-level result only) and a
        SubpopulationDict (dataset-level global entry, fanned out into
        add_result as before, PLUS one add_subpopulation_result call per
        subpopulation key).
        """
        if isinstance(metrics, SubpopulationDict):
            for metric, value in metrics.get_global_flower_dict().items():
                self._dataset_metrics.add_result(metric, evaluator_label, server_round, value)
            for subpopulation_key, subpop_dict_of_metrics in metrics.subpopulation_items():
                for metric, value in subpop_dict_of_metrics.return_flower_dict().items():
                    self._dataset_metrics.add_subpopulation_result(
                        subpopulation_key, metric, evaluator_label, server_round, value)
        else:
            for metric, value in metrics.return_flower_dict().items():
                self._dataset_metrics.add_result(metric, evaluator_label, server_round, value)

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Tuple[Optional[Parameters]]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        # self.initial_parameters  # Keeping initial parameters in memory
        return initial_parameters

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        log(INFO, "Evaluate()")
        self._times_dataframe.loc[server_round, "BeforeEvaluation"] = time.time()
        if self.evaluate_fn is None or self._early_stop:
            log(INFO, "No evaluation function provided")
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        if server_round != 0:
            eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        else:
            eval_res = None
        if eval_res is None:
            return None

        loss, pickled_data = eval_res
        partial_computation_result = pickle.loads(pickled_data["partial_computation_result"])
        metrics = evaluator(partial_computation_result, self._metric_list)
        self._this_round_result = partial_computation_result

        self._times_dataframe.loc[server_round, "AfterEvaluation"] = time.time()

        if loss > self._former_loss:
            self._current_patience += 1
            self._best_round = server_round - self._current_patience
        else:
            self._former_loss = loss
            self._best_round = server_round
            self._current_patience = 0

        if self._current_patience == self._max_patience:
            self._early_stop = True

        log(INFO, "Server round: " + str(server_round))
        if server_round != 0 and self._final_training == 1:
            self._log_metrics(metrics, "Global", server_round)   # CHANGED — was the inline for-loop

        if (server_round == self._max_round or self._early_stop) and self._final_training == 0:
            if server_round == self._max_round:
                log(INFO, "Final round is reached, telling by max round.")
                self.study.tell(self.trial, loss)
            else:
                log(INFO, "Early stop is reached, telling by early stop.")
                self.study.tell(self.trial, self._former_loss)

        # if server_round == self._max_round and self._final_training == 1:
        #     self._dataset_metrics.save_dataframes_as_csv(self._result_path)

        return loss, metrics.return_flower_dict()

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        log(INFO, "Configure fit()")
        config = {
            "server_round": server_round,
            "max_round": self._max_round,
        }
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        if self._compute_shapley_values:
            config["compute_shapley_values"] = 1
        else:
            config["compute_shapley_values"] = 0

        if self._early_stop:
            config["early_stop"] = True
        else:
            config["early_stop"] = False

        if self._strategy_aggregation == "Scaffold":
            if self._server_cv is not None:
                config["server_cv"] = pickle.dumps(self._server_cv)
            else:
                # First round: send zeros — clients will initialise their own c_i to zeros too
                # We don't know the shape yet, so send a sentinel
                config["server_cv"] = pickle.dumps(None)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results using weighted average.

        Args:
            rnd (int): _description_
            results (List[Tuple[ClientProxy, FitRes]]): _description_
            failures (List[BaseException]): _description_

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: _description_
        """
        log(INFO, "Aggregate fit()")
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if not self._early_stop:
            client_weights = dict()
            self._id_and_client_number: dict = {client.cid: fit_res.metrics.pop("client_number")
                                                for client, fit_res in results}
            log(INFO, f"Client id number: {self._id_and_client_number}")
            # clients_data_size_dict: dict = {client.cid: fit_res.num_examples for client, fit_res in results}
            self.target_classes = self.y_test.columns.to_list()
            self.clients_data_size_dict: dict = \
                {client.cid: [fit_res.metrics["Label " + class_name]
                              if "Label " + class_name in fit_res.metrics.keys() else 0
                              for class_name in self.target_classes]
                 for client, fit_res in results}
            # log(INFO, f"Clients data sizes: {self.clients_data_size_dict}")
            for client, fit_res in results:
                client_weights[client.cid] = (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)

            self.clients_list: list = sorted(list(self.clients_data_size_dict.keys()))

            # Convert results
            if server_round == 1 and self._final_training == 1:
                for client, number_assigned in self._id_and_client_number.items():
                    log(INFO, "Clients_id: {} and number assigned: {}".format(client, number_assigned))

                if self._compute_shapley_values:
                    log(INFO, "Creating datasets")
                    self._dataset_metrics = SVCompatibleFlowerMetricManager(
                        metric_list=self._metric_list,
                        client_list=list(self._id_and_client_number.values()),
                        number_of_rounds=self._max_round,
                        classes=self.target_classes
                    )
                else:
                    self._dataset_metrics = FlowerMetricManager(
                        metric_list=self._metric_list,
                        client_list=list(self._id_and_client_number.values()),
                        number_of_rounds=self._max_round,
                        classes=self.target_classes
                    )
                assert self._dataset_metrics is not None
                self._model.set_model(parameters_to_ndarrays(self.initial_parameters))
                predictions = self._model.predict(self._data_validator)
                partial_evaluations = partial_computation(predictions,
                                                          self.y_test,
                                                          self.target_classes,
                                                          subpopulation_values=self.x_test_subpopulation_values)  # NEW
                self._last_round_result = partial_evaluations
                partial_metrics = get_all_metrics_from_partial_results(partial_evaluations, self._metric_list)

                self._log_metrics(partial_metrics, "Global", server_round)  # CHANGED — was the inline for-loop
                # self._shapley_values.set_last_round_results(partial_metrics)

                # for metric, value in partial_metrics.return_flower_dict().items():
                #     self._dataset_metrics.add_result(metric, "Global", server_round, value)

                # Printing testing labels for comparison
                # log(INFO, "Columns sorted: {}".format(self.target_classes))
                labels, counts = np.unique(np.argmax(np.asarray(self.y_test), axis=1), return_counts=True)
                testing_labels = {label: count for label, count in zip(labels, counts)}
                # log(INFO, "Testing labels: {}".format(testing_labels))

            # We store here the weights, to then pass them to the clients.
            self._clients_weights = client_weights
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for client, fit_res in results
            ]

            new_model = aggregate_nn(weights_results)
            self._model.set_model(new_model)
            if self._current_patience == 0:
                self._best_model = keras.models.clone_model(self._model.get_model())

            if self._strategy_aggregation == "Scaffold":
                cv_deltas = []
                for client, fit_res in results:
                    if "cv_delta" in fit_res.metrics:
                        delta = pickle.loads(fit_res.metrics.pop("cv_delta"))
                        cv_deltas.append([np.array(d) for d in delta])

                if cv_deltas:
                    # Average the deltas across clients and add to server c
                    n = len(cv_deltas)
                    avg_delta = [
                        sum(cv_deltas[i][layer] for i in range(n)) / n
                        for layer in range(len(cv_deltas[0]))
                    ]
                    if self._server_cv is None:
                        self._server_cv = avg_delta
                    else:
                        self._server_cv = [c + d for c, d in zip(self._server_cv, avg_delta)]

            # if self._compute_shapley_values == 1 and server_round != 1:
            if self._compute_shapley_values:
                # partial_evaluations = partial_computation(self._data_validator,
                #                                           self._model,
                #                                           self.target_classes,
                #                                           self.y_test)
                # dict_of_metrics = get_all_metrics_from_partial_results(partial_evaluations, self._metric_list)
                # log(INFO, "Calculate Shapley Values")
                # Shapley Value computation.
                self._times_dataframe.loc[server_round, "BeforeSV"] = time.time()
                self._shapley_values.set_client_index_dictionary(self._id_and_client_number)
                self._shapley_values_decentralized.set_client_index_dictionary(self._id_and_client_number)
                clients_powerset = get_clients_powerset(self._id_and_client_number)
                powerset_partial_measures = get_all_partial_aggregation_evaluation_results(
                    self._data_validator,
                    self.y_test,
                    self._model,
                    self.target_classes,
                    self._clients_weights,
                    self._shapley_values.get_index_client_id_dictionary(),
                    self._last_round_result,
                    clients_powerset,
                    self.x_test_subpopulation_values
                )
                metric_dict_powerset = {
                    k: get_all_metrics_from_partial_results(powerset_partial_measures[k], self._metric_list) \
                    for k, v in powerset_partial_measures.items()
                }
                self._shapley_values.shapley_values_calculation(metric_dict_powerset,
                                                                self.clients_list,
                                                                server_round)
                log(INFO, f"Shapley values calculation: {self._shapley_values.get_round_shapley_values(server_round)}")
                log(INFO, f"After SV")
                self._times_dataframe.loc[server_round, "AfterSV"] = time.time()
                log(INFO, f"Before Gradient Rewards")
                self._times_dataframe.loc[server_round, "BeforeGradientRewards"] = time.time()
                self._gradient_rewards.set_client_index_dictionary(self._id_and_client_number)
                self._gradient_rewards.calculate_rewards(self._model, server_round, self.clients_list, self.clients_data_size_dict)
                self._times_dataframe.loc[server_round, "AfterGradientRewards"] = time.time()
                log(INFO, f"After Gradient Rewards")

                # self._gradient_rewards.get_rewards()

                # log(INFO, "Shapley Values: {}".format(self._shapley_values.get_shapley_values()))

                # Evaluation now receives the dataset, the model and the weights of the model.
                # The weights by themselves are insufficient, because they are simply a list.

                # self._accuracy_object.set_value(accuracy)
                # log(INFO, "Loss of global model: {}".format(list_of_metrics.get_value()["CrossEntropyLoss"].get_value()))
                # log(INFO, "Accuracy of global model: {}".format(list_of_metrics.get_value()["Accuracy"].get_value()))

        if server_round == self._max_round or self._early_stop:
            if self._compute_shapley_values:
                # for client, fit_res in sorted(results, key=lambda x: x[0].cid):
                #     log(INFO, "Training labels of client {}: {}".format(client.cid, fit_res.metrics))
                # log(INFO, "=" * 50)
                shapley_values_total = self._shapley_values.get_shapley_values()
                gradient_rewards_total = self._gradient_rewards.get_rewards()
                # log(INFO, f"Shapley values keys: {list(shapley_values_total.keys())}")
                for training_round, dict_sv in shapley_values_total.items():
                    for evaluated_client_id, sv in dict_sv.items():
                        evaluated_client = self._id_and_client_number[evaluated_client_id]
                        flower_dict_type = sv.return_flower_dict()
                        for metric, value in flower_dict_type.items():
                            # log(INFO, f"{metric}: {value}")
                            self._dataset_metrics.add_shapley_value(metric,
                                                                    "Centralized",
                                                                    evaluated_client,
                                                                    training_round,
                                                                    value)
                #     log(INFO, "Shapley Values in local_round {}: {}"
                #         .format(training_round,
                #                 {client: str(sv) for client, sv in shapley_values_total[training_round].items()}))
                # log(INFO, "=" * 50)
                #
                # log(INFO, "=" * 50)
                # result_dictionary = {client_id: return_default_dict_of_metrics(self._metric_list, self.y_test.shape[1])
                #                      for client_id in self.clients_list}
                # for training_round in shapley_values_total.keys():
                #     if training_round <= self._best_round:
                #         single_round_dict = shapley_values_total.get(training_round)
                #         log(INFO, f"Single round dict: {single_round_dict}")
                #         result_dictionary = {k: result_dictionary[k] + single_round_dict[k] for k in
                #                              result_dictionary.keys()}

                # result_dictionary = {k: str(shapley_values_client) for k, shapley_values_client in
                #                      result_dictionary.items()}
                # log(INFO, "Result_dictionary: {}".format(result_dictionary))

                for round, dict_sv in shapley_values_total.items():
                    for evaluated_client_id, sv in dict_sv.items():
                        evaluated_client = self._id_and_client_number[evaluated_client_id]
                        # flower_dict_type = sv.return_flower_dict_as_str()
                        # for metric in flower_dict_type:
                        #     # while type(flower_dict_type[metric] is str):
                        #     #     flower_dict_type[metric] = ast.literal_eval(flower_dict_type[metric])
                        #     if '[' in flower_dict_type[metric]:
                        #         flower_dict_type[metric] = [float(value)
                        #                                     for value
                        #                                     in ast.literal_eval(flower_dict_type[metric])]
                        #     else:
                        #         # log(INFO, "")
                        #         flower_dict_type[metric] = float(flower_dict_type[metric])
                        # for metric, value in flower_dict_type.items():
                        #     self._dataset_metrics.add_shapley_value(metric,
                        #                                             "Centralized",
                        #                                             evaluated_client,
                        #                                             round,
                        #                                             value)

                        # Cosine Similarity
                        gradient_rewards_round = gradient_rewards_total.get(round)
                        gradients_with_clients_to_id = {self._id_and_client_number[k]: gradient_rewards_round[k]
                                                        for k in gradient_rewards_total.get(round)}
                        self._dataset_metrics.add_shapley_value("CosineSimilarity",
                                                                "Centralized",
                                                                evaluated_client,
                                                                round,
                                                                gradients_with_clients_to_id[evaluated_client])
                # log(INFO, "=" * 50)
                # log(INFO, "\n")
                # log(INFO, "\n")
                # log(INFO, "\n")
                # log(INFO, "=" * 50)
                # log(INFO, "Gradient Rewards:")
                # log(INFO, "-" * 50)
                # gradient_rewards = self._gradient_rewards.get_rewards()
                # gradient_result_dictionary = {client_id: 0 for client_id in self.clients_list}
                # for training_round in gradient_rewards.keys():
                #     single_round_dict = gradient_rewards.get(training_round)
                #     log(INFO, f"Gradient rewards round {training_round}: {single_round_dict}")
                #     gradient_result_dictionary = {k: gradient_result_dictionary[k] + single_round_dict[k]
                #                                   for k in result_dictionary.keys()}
                # log(INFO, "-" * 50)
                # gradient_result_dictionary = {self._id_and_client_number[k]: gradient_result_dictionary[k]
                #                               for k in gradient_result_dictionary.keys()}
                # log(INFO, f"Aggregated gradient rewards: {gradient_result_dictionary}")
                # log(INFO, "=" * 50)

            # Saving model in pickle.
            if self._early_stop:
                save_model_with_tensorflow(ROOT_DIR +
                                           os.sep + "data" +
                                           os.sep + "global_model" +
                                           os.sep + self._model_final_name,
                                           self._best_model)
            else:
                model_to_save = self._model.get_model()
                save_model_with_tensorflow(ROOT_DIR +
                                           os.sep + "data" +
                                           os.sep + "global_model" +
                                           os.sep + self._model_final_name,
                                           model_to_save)

        if not self._early_stop:
            new_model_to_parameters = ndarrays_to_parameters(new_model)
            return new_model_to_parameters, {}
        else:
            return None, {}

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        log(INFO, "Configure Evaluate()")
        """Configure the next local_round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {
            "server_round": server_round,
        }
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            # config = self.on_evaluate_config_fn(server_round, self._clients_weights)
            config = self.on_evaluate_config_fn(server_round)
            if self._compute_shapley_values:
                config["last_round"] = self._max_round == server_round
                config["compute_shapley_values"] = 1
                config["client_cid_number"] = str(self._id_and_client_number)
                config["number_client_cid"] = str(self._shapley_values_decentralized.get_index_client_id_dictionary())
                # _, total_dataset_y = get_test_data()
                config["total_num_of_classes"] = self.y_test.shape[1]
                # TODO: Change from pickle to data for configurations.
                save_data_on_pickle(ROOT_DIR + os.sep + "data" + os.sep + "pickled_information" + os.sep + "model.pkl",
                                    self._clients_weights)
                config["time_results_path"] = (ROOT_DIR +
                                               os.sep + "results" +
                                               os.sep + "times" +
                                               os.sep + self._experiment_name)
            else:
                config["compute_shapley_values"] = 0

            if server_round == self._max_round:
                config["last_round"] = 1
            else:
                config["last_round"] = 0

        if self._early_stop:
            config["early_stop"] = True
        else:
            config["early_stop"] = False

        # log(INFO, "EvaluateIns")
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # log(INFO, "Ending configure evaluate")
        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def normalize_metric(self, metric, value, client_samples, total_samples):
        if type(value) is list:
            value = np.asarray(value)
            portion_of_samples = np.divide(client_samples, total_samples)
            return np.multiply(value, portion_of_samples)
        else:
            if metric == "RMSE":
                value = math.pow(value, 2)
            portion_of_samples = np.sum(client_samples) / np.sum(total_samples)
            return value * portion_of_samples

    def sum_metrics(self, first_value, second_value):
        if type(first_value) is list or type(first_value) is np.ndarray:
            return [x + y for x, y in zip(first_value, second_value)]
        else:
            return first_value + second_value

    # def sum_sv(self, first_client, second_client):
    #     # if type(first_value) is list:
    #     #     return [x + y for x, y in zip(first_value, second_value)]
    #     # elif type(first_value) is float or type(first_value) is int:
    #     #     return first_value + second_value

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        log(INFO, "Aggregate Evaluate()")
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if os.path.exists(ROOT_DIR + os.sep + "data" + os.sep + "pickled_information" + os.sep + "model.pkl"):
            os.remove(ROOT_DIR + os.sep + "data" + os.sep + "pickled_information" + os.sep + "model.pkl")

        if self._early_stop:
            loss_aggregated = 1000000000.0
            metrics_aggregated = {}
        else:
            # Aggregate loss
            loss_aggregated = weighted_loss_avg(
                [
                    (evaluate_res.num_examples, evaluate_res.loss)
                    for _, evaluate_res in results
                ]
            )
            # metrics_aggregated = {}
            # eval_metrics = {self._id_and_client_number[client.cid]: (res.num_examples, res.metrics)
            #                 for client, res in results}
            # log(INFO, f"Res metrics: {[res.metrics for client, res in results]}")

            metric_dictionary = {self._id_and_client_number[client.cid]: (res.num_examples, res.metrics)
                                 for client, res in results}

            list_partial_evaluations = {
                k: evaluator(pickle.loads(v[1]["partial_evaluations"], encoding="utf-8"), self._metric_list)
                for k, v in metric_dictionary.items()}

            if self._final_training == 1:
                for client, res in list_partial_evaluations.items():
                    self._log_metrics(res, client, server_round)  # CHANGED — was the inline for-loop



            # CHANGED: pick the right zero-element for the accumulator depending on
            # whether clients sent subpopulation-aware partials or plain ones.
            sample_partial_evaluation = pickle.loads(
                next(iter(metric_dictionary.values()))[1]["partial_evaluations"], encoding="utf-8")
            if isinstance(sample_partial_evaluation, SubpopulationDict):
                all_partial_evaluations = SubpopulationDict({})
            else:
                all_partial_evaluations: AggregatableMeasures = return_default_partial_computations(
                    list(self.y_test.columns))

            for res in metric_dictionary.values():
                partial_evaluation = pickle.loads(res[1]["partial_evaluations"], encoding="utf-8")
                all_partial_evaluations += partial_evaluation

            metrics_aggregated_obj = evaluator(all_partial_evaluations, self._metric_list)  # CHANGED (was inline)
            if self._final_training == 1:
                self._log_metrics(metrics_aggregated_obj, "Aggregated",
                                  server_round)  # CHANGED — was the inline for-loop
            metrics_aggregated = metrics_aggregated_obj.return_flower_dict()  # CHANGED — flattening moved after logging

            if self._compute_shapley_values:
                log(INFO, "Decentralized Shapley Values Calculation")
                if server_round == 1:
                    self._shapley_values_decentralized.set_client_index_dictionary(self._id_and_client_number)

                sample_sv_evaluation = pickle.loads(
                    next(iter(metric_dictionary.values()))[1]["SV_partial_computations"], encoding="utf-8")
                sample_group_value = next(iter(sample_sv_evaluation.values()))
                sv_group_zero = (SubpopulationDict({}) if isinstance(sample_group_value, SubpopulationDict)
                                 else return_default_partial_computations(self.target_classes))

                all_partial_sv_evaluations = {}
                for res in metric_dictionary.values():
                    partial_sv_evaluation = pickle.loads(res[1]["SV_partial_computations"], encoding="utf-8")
                    all_partial_sv_evaluations = {
                        k: all_partial_sv_evaluations.get(k, sv_group_zero) + partial_sv_evaluation.get(k,
                                                                                                        sv_group_zero)
                        for k in set(all_partial_sv_evaluations) | set(partial_sv_evaluation)
                    }

                decentralized_metrics = {
                    k: get_all_metrics_from_partial_results(all_partial_sv_evaluations[k], self._metric_list) \
                    for k, v in all_partial_sv_evaluations.items()
                }
                # decentralized_metrics = get_all_metrics_from_partial_results(metric_dict_powerset, self._metric_list)
                self._shapley_values_decentralized.shapley_values_calculation(
                    decentralized_metrics,
                    self.clients_list,
                    server_round
                )

                if self._this_round_result is not None:
                    self._last_round_result = self._this_round_result

                for client_number, res in metric_dictionary.items():
                    sv_evaluator = client_number
                    sv_from_client = pickle.loads(res[1]["SV_local_client"], encoding="utf-8")
                    for sv_client_evaluated, sv in sv_from_client.items():
                        sv_client_evaluated_number = self._id_and_client_number[sv_client_evaluated]
                        for metric, value in sv.return_flower_dict().items():
                            self._dataset_metrics.add_shapley_value(metric,
                                                                    sv_evaluator,
                                                                    sv_client_evaluated_number,
                                                                    server_round,
                                                                    value)

                # log(INFO, f"Aggregated_metrics: {aggregated_metrics_sv}")
                for evaluated_client, sv in self._shapley_values_decentralized.get_round_shapley_values(
                        server_round).items():
                    for metric, value in sv.return_flower_dict().items():
                        self._dataset_metrics.add_shapley_value(metric,
                                                                "Aggregated",
                                                                self._id_and_client_number[evaluated_client],
                                                                server_round,
                                                                value)

        if self._final_training == 1 and (server_round == self._max_round or self._early_stop):
            if self._early_stop:
                # log(INFO, f"Dataset metrics: {self._dataset_metrics.get_subpopulation_dataframe()}")
                # log(INFO, f"Dataset metrics: {self._dataset_metrics.get_global_dataframes()}")
                log(INFO, f"Best round: {self._best_round}")
                self._dataset_metrics.save_dataframes_as_csv(self._result_path, self._best_round)
                os.makedirs(ROOT_DIR +
                            os.sep + "results" +
                            os.sep + "times" +
                            os.sep + self._experiment_name, exist_ok=True)
                self._times_dataframe.loc[:self._best_round].to_csv(ROOT_DIR +
                                                                    os.sep + "results" +
                                                                    os.sep + "times" +
                                                                    os.sep + self._experiment_name +
                                                                    os.sep + "server")
            else:
                # log(INFO, f"Dataset metrics: {self._dataset_metrics.get_subpopulation_dataframe()}")
                # log(INFO, f"Dataset metrics: {self._dataset_metrics.get_global_dataframes()}")
                log(INFO, f"Best round: {self._max_round}")
                self._dataset_metrics.save_dataframes_as_csv(self._result_path, self._max_round)
                os.makedirs(ROOT_DIR +
                            os.sep + "results" +
                            os.sep + "times" +
                            os.sep + self._experiment_name, exist_ok=True)
                self._times_dataframe.loc[:self._max_round].to_csv(ROOT_DIR +
                                                                   os.sep + "results" +
                                                                   os.sep + "times" +
                                                                   os.sep + self._experiment_name +
                                                                   os.sep + "server")

        return loss_aggregated, metrics_aggregated
