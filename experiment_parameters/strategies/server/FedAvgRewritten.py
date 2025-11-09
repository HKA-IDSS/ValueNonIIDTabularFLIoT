import ast
import json
import math
import os
from functools import reduce
from logging import WARNING, INFO
from typing import Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from flwr.common import Parameters, Scalar, FitRes, NDArray, parameters_to_ndarrays, \
    ndarrays_to_parameters, EvaluateIns, EvaluateRes, FitIns
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import weighted_loss_avg

from Definitions import ROOT_DIR
from experiment_parameters.aggregation_processes.aggregate import aggregate_nn
from experiment_parameters.model_builder.Model import Model
from metrics.Evaluator import evaluator
from metrics.GradientRewards import GradientRewards
from metrics.Metrics import return_default_dict_of_metrics
from metrics.ResultManager import FlowerMetricManager, SVCompatibleFlowerMetricManager
from metrics.Shapley_Values import ShapleyValuesNN
from util.Util import save_data_on_pickle, from_string_to_dict, get_test_data

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
    _gradient_rewards: GradientRewards
    _max_round: int
    _compute_shapley_values: bool
    _model_final_name: str
    _id_and_client_number: List[tuple]
    _dataset_metrics: FlowerMetricManager | SVCompatibleFlowerMetricManager

    # _former

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
            self,
            max_round: int,
            directory_of_data,
            model: Model,
            final_training,
            shapley_values,
            gradient_rewards,
            metric_list,
            model_final_name,
            compute_shapley_values,
            result_path,
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
        self._model: Model = model
        self._final_training = final_training
        self._shapley_values = shapley_values
        self._gradient_rewards = gradient_rewards
        self._metric_list = metric_list
        self._clients_weights = None
        self._max_round = max_round
        self._compute_shapley_values = compute_shapley_values
        self._model_final_name = model_final_name
        self._result_path = result_path

        # Information for evaluation purposes.
        self._total_test_samples = None
        self._samples_per_class = None
        self.x_test, self.y_test = get_test_data(directory_of_data)
        self.target_classes = None
        self.clients_data_size_dict = None

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
        if self.evaluate_fn is None:
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

        loss, metrics = eval_res

        if server_round != 0 and self._final_training == 1:
            for metric, value in metrics.items():
                self._dataset_metrics.add_result(metric, "Global", server_round, value)

        # if server_round == self._max_round and self._final_training == 1:
        #     self._dataset_metrics.save_dataframes_as_csv(self._result_path)

        return loss, metrics

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {"server_round": server_round}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        if self._compute_shapley_values:
            config["compute_shapley_values"] = 1
        else:
            config["compute_shapley_values"] = 0

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
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
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
        clients_list: list = sorted(list(self.clients_data_size_dict.keys()))
        for client, fit_res in results:
            client_weights[client.cid] = (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)

        if server_round == 1 and self._final_training == 1:
            # for client, number_assigned in self._id_and_client_number.items():
            #     log(INFO, "Clients_id: {} and number assigned: {}".format(client, number_assigned))

            if self._compute_shapley_values == 1:
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
            self._model.set_model(parameters_to_ndarrays(self.initial_parameters))
            list_of_metrics = evaluator(self.x_test,
                                        self.y_test,
                                        self._model,
                                        self._metric_list)
            self._shapley_values.set_last_round_results(list_of_metrics)

            for metric, value in list_of_metrics.return_flower_dict_as_str().items():
                self._dataset_metrics.add_result(metric, "Global", server_round, value)

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
        list_of_metrics = evaluator(self.x_test, self.y_test, self._model, self._metric_list)

        # if self._compute_shapley_values == 1 and server_round != 1:
        if self._compute_shapley_values == 1:
            # log(INFO, "Calculate Shapley Values")
            # Shapley Value computation.
            self._shapley_values.set_client_index_dictionary(self._id_and_client_number)
            self._shapley_values.shapley_values_calculation(self._model,
                                                            clients_list,
                                                            client_weights,
                                                            server_round)
            self._shapley_values.set_last_round_results(list_of_metrics)
            # Shapley GTG
            # self._shapley_values.shapley_values_calculation(x_test,
            #                                                 y_test,
            #                                                 self._model,
            #                                                 clients_list,
            #                                                 client_weights,
            #                                                 list_of_metrics,
            #                                                 server_round,
            #                                                 self._metric_list)
            self._gradient_rewards.set_client_index_dictionary(self._id_and_client_number)
            self._gradient_rewards.calculate_rewards(server_round, clients_list, self.clients_data_size_dict)
            # self._gradient_rewards.get_rewards()

            # log(INFO, "Shapley Values: {}".format(self._shapley_values.get_shapley_values()))

            # Evaluation now receives the dataset, the model and the weights of the model.
            # The weights by themselves are insufficient, because they are simply a list.

            # self._accuracy_object.set_value(accuracy)
            # log(INFO, "Loss of global model: {}".format(list_of_metrics.get_value()["CrossEntropyLoss"].get_value()))
            # log(INFO, "Accuracy of global model: {}".format(list_of_metrics.get_value()["Accuracy"].get_value()))

        if server_round == self._max_round:
            if self._compute_shapley_values == 1:
                # for client, fit_res in sorted(results, key=lambda x: x[0].cid):
                #     log(INFO, "Training labels of client {}: {}".format(client.cid, fit_res.metrics))
                # log(INFO, "=" * 50)
                shapley_values_total = self._shapley_values.get_shapley_values()
                gradient_rewards_total = self._gradient_rewards.get_rewards()
                # log(INFO, f"Shapley values keys: {list(shapley_values_total.keys())}")
                for training_round, dict_sv in shapley_values_total.items():
                    for evaluated_client_id, sv in dict_sv.items():
                        evaluated_client = self._id_and_client_number[evaluated_client_id]
                        flower_dict_type = sv.return_flower_dict_as_str()
                        for metric, value in flower_dict_type.items():
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
                result_dictionary = {client_id: return_default_dict_of_metrics(self._metric_list, self.y_test.shape[1])
                                     for client_id in clients_list}
                for training_round in shapley_values_total.keys():
                    single_round_dict = shapley_values_total.get(training_round)
                    result_dictionary = {k: result_dictionary[k] + single_round_dict[k] for k in
                                         result_dictionary.keys()}

                # result_dictionary = {k: str(shapley_values_client) for k, shapley_values_client in
                #                      result_dictionary.items()}
                # log(INFO, "Result_dictionary: {}".format(result_dictionary))

                for round, dict_sv in shapley_values_total.items():
                    for evaluated_client_id, sv in dict_sv.items():
                        evaluated_client = self._id_and_client_number[evaluated_client_id]
                        flower_dict_type = sv.return_flower_dict_as_str()
                        for metric in flower_dict_type:
                            # while type(flower_dict_type[metric] is str):
                            #     flower_dict_type[metric] = ast.literal_eval(flower_dict_type[metric])
                            if '[' in flower_dict_type[metric]:
                                flower_dict_type[metric] = [float(value)
                                                            for value
                                                            in ast.literal_eval(flower_dict_type[metric])]
                            else:
                                # log(INFO, "")
                                flower_dict_type[metric] = float(flower_dict_type[metric])
                        for metric, value in flower_dict_type.items():
                            self._dataset_metrics.add_shapley_value(metric,
                                                                    "Centralized",
                                                                    evaluated_client,
                                                                    round,
                                                                    value)

                        # Cosine Similarity
                        gradient_rewards_round = gradient_rewards_total.get(round)
                        gradients_with_clients_to_id = {self._id_and_client_number[k]: gradient_rewards_round[k]
                                                        for k in gradient_rewards_total.get(round)}
                        self._dataset_metrics.add_shapley_value("CosineSimilarity",
                                                                "Centralized",
                                                                evaluated_client,
                                                                round,
                                                                gradients_with_clients_to_id[evaluated_client])
                log(INFO, "=" * 50)
                log(INFO, "\n")
                log(INFO, "\n")
                log(INFO, "\n")
                log(INFO, "=" * 50)
                log(INFO, "Gradient Rewards:")
                log(INFO, "-" * 50)
                gradient_rewards = self._gradient_rewards.get_rewards()
                gradient_result_dictionary = {client_id: 0 for client_id in clients_list}
                for training_round in gradient_rewards.keys():
                    single_round_dict = gradient_rewards.get(training_round)
                    log(INFO, f"Gradient rewards round {training_round}: {single_round_dict}")
                    gradient_result_dictionary = {k: gradient_result_dictionary[k] + single_round_dict[k]
                                                  for k in result_dictionary.keys()}
                log(INFO, "-" * 50)
                gradient_result_dictionary = {self._id_and_client_number[k]: gradient_result_dictionary[k]
                                              for k in gradient_result_dictionary.keys()}
                log(INFO, f"Aggregated gradient rewards: {gradient_result_dictionary}")
                log(INFO, "=" * 50)

            # Saving model in pickle.
            self._model.set_model(new_model)
            model_to_save = self._model
            save_model_with_tensorflow(ROOT_DIR +
                                       os.sep + "data" +
                                       os.sep + "global_model" +
                                       os.sep + self._model_final_name,
                                       model_to_save.get_model())

        new_model_to_parameters = ndarrays_to_parameters(new_model)

        return new_model_to_parameters, {}

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        log(INFO, "Configure Evaluate()")
        """Configure the next local_round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            # config = self.on_evaluate_config_fn(server_round, self._clients_weights)
            config = self.on_evaluate_config_fn(server_round)
            if self._compute_shapley_values:
                config["compute_shapley_values"] = 1
                config["client_cid_number"] = str(self._id_and_client_number)
                # _, total_dataset_y = get_test_data()
                config["total_num_of_classes"] = self.y_test.shape[1]
                # TODO: Change from pickle to data for configurations.
                save_data_on_pickle(ROOT_DIR + os.sep + "data" + os.sep + "pickled_information" + os.sep + "model.pkl",
                                    self._clients_weights)
                # log(INFO, f"Serializing: {str(self._clients_weights)}")
                # config["all_models_from_clients"] = str(self._clients_weights)
            else:
                config["compute_shapley_values"] = 0

            if server_round == self._max_round:
                config["last_round"] = 1
            else:
                config["last_round"] = 0
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
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if os.path.exists(ROOT_DIR + os.sep + "data" + os.sep + "pickled_information" + os.sep + "model.pkl"):
            os.remove(ROOT_DIR + os.sep + "data" + os.sep + "pickled_information" + os.sep + "model.pkl")

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

        eval_metrics = {client.cid: dict(
            (k, ast.literal_eval(v))
            for k, v in res.metrics.items()
            if k in self._metric_list
        ) for client, res in results}
        # log(INFO, f"Eval metrics: {eval_metrics}")
        sv_metrics = {client.cid: dict(
            (evaluating_client.split('_')[1], from_string_to_dict(v))
            for evaluating_client, v in res.metrics.items()
            if "SV" in evaluating_client
        ) for client, res in results}
        # log(INFO, f"SV metrics: {sv_metrics}")
        total_of_samples_per_class = reduce(np.add,
                                            self.clients_data_size_dict.values(),
                                            np.zeros(shape=len(self.target_classes)))

        normalized_metrics = []
        for client, dictionary_of_metrics in eval_metrics.items():
            # log(INFO, f"Client number {client}")
            # log(INFO, f"Metrics: {dictionary_of_metrics}")

            if self._final_training == 1:
                for metric, value in dictionary_of_metrics.items():
                    self._dataset_metrics.add_result(metric, self._id_and_client_number[client], server_round, value)

            aggregated_metrics = {metric: self.normalize_metric(metric, value,
                                                                self.clients_data_size_dict[client],
                                                                total_of_samples_per_class)
                                  for metric, value in dictionary_of_metrics.items()}
            normalized_metrics.append(aggregated_metrics)
            # log(INFO, f"Metrics normalized: {normalized_metrics}")

        metrics_aggregated = reduce(lambda dict1, dict2: {key: self.sum_metrics(dict1[key], dict2[key])
                                                          for key in dict1},
                                    normalized_metrics)
        if "RMSE" in self._metric_list:
            metrics_aggregated["RMSE"] = math.sqrt(metrics_aggregated["RMSE"])

        # log(INFO, f"Metrics Aggregated: {metrics_aggregated}")
        if self._final_training == 1:
            for metric, value in metrics_aggregated.items():
                self._dataset_metrics.add_result(metric, "Aggregated", server_round, value)

        aggregated_metrics_sv = \
            {client_number: return_default_dict_of_metrics(self._metric_list,
                                                           len(self.target_classes)).return_flower_dict()
             for client_number in self._id_and_client_number.values()}

        if self._compute_shapley_values == 1:
            log(INFO, "")
            for evaluating_client_cid, sv_evaluating_client in sv_metrics.items():
                evaluating_client_number = self._id_and_client_number[evaluating_client_cid]
                for client_evaluated_cid, sv_evaluated_client in sv_evaluating_client.items():
                    evaluated_client_number = self._id_and_client_number[client_evaluated_cid]
                    for metric, value in sv_evaluated_client.items():
                        if '[' in value:
                            value = [float(evaluated_value)
                                     for evaluated_value
                                     in ast.literal_eval(value)]
                        else:
                            value = float(value)
                        self._dataset_metrics.add_shapley_value(metric,
                                                                evaluating_client_number,
                                                                evaluated_client_number,
                                                                server_round,
                                                                value)
                        normalized_metrics = {metric: self.normalize_metric(metric, value,
                                                                            self.clients_data_size_dict[
                                                                                evaluating_client_cid],
                                                                            total_of_samples_per_class)}
                        aggregated_metrics_sv[evaluated_client_number][metric] = self.sum_metrics(
                            normalized_metrics[metric], aggregated_metrics_sv[evaluated_client_number][metric]
                        )
            # log(INFO, f"Aggregated_metrics: {aggregated_metrics_sv}")
            for evaluated_client, sv in aggregated_metrics_sv.items():
                for metric, value in sv.items():
                    self._dataset_metrics.add_shapley_value(metric,
                                                            "Aggregated",
                                                            evaluated_client,
                                                            server_round,
                                                            value)

        if self._final_training == 1 and server_round == self._max_round:
            self._dataset_metrics.save_dataframes_as_csv(self._result_path)

        return loss_aggregated, metrics_aggregated
