import math
from abc import ABC
from logging import INFO, DEBUG

import pandas as pd
from flwr.common.logger import log

from experiment_parameters.aggregation_processes.aggregate import aggregate_nn, aggregate_trees, aggregate_xgboost
from experiment_parameters.model_builder.Model import Model
from metrics.Evaluator import evaluator
from metrics.Metrics import DictOfMetrics, return_default_dict_of_metrics


class ShapleyValues:
    _shapley_values: dict
    _x_test: pd.DataFrame
    _y_test: pd.DataFrame
    _metric_list: list
    _last_round_result: DictOfMetrics

    def __init__(self, x_test, y_test, rounds, metric_list):
        self._x_test = x_test
        self._y_test = y_test
        self._shapley_values = {k: dict() for k in range(1, rounds + 1)}
        self._client_index_dictionary = dict()
        self._client_index_dictionary_set = False
        self._metric_list = metric_list

    def set_last_round_results(self, metrics):
        self._last_round_result = metrics

    def get_shapley_values(self):
        return self._shapley_values

    def get_round_shapley_values(self, round: int):
        return self._shapley_values[round]

    def set_shapley_values(self, shapley_values):
        self._shapley_values = shapley_values

    def update_shapley_value(self, local_round, client_cid, dict_of_values):
        if client_cid not in self._shapley_values[local_round].keys():
            self._shapley_values[local_round][client_cid] = return_default_dict_of_metrics(self._metric_list,
                                                                                           self._y_test.shape[1])

        self._shapley_values[local_round][client_cid] += dict_of_values

    def last_division(self, local_round):
        num_participants = len(self._shapley_values[local_round].keys())
        # log(INFO, "Last division")
        # log(INFO, f"Local round: {local_round}")
        # log(INFO, f"Number of participants: {math.factorial(num_participants)}")
        for client in self._shapley_values[local_round].keys():
            # log(INFO, f"Client number {client} with Shapley Value: {self._shapley_values[local_round][client]}")
            # log(INFO, f"Final result: {self._shapley_values[local_round][client] / math.factorial(num_participants)}")
            self._shapley_values[local_round][client] /= math.factorial(num_participants)

    def get_client_index_dictionary(self):
        return self._client_index_dictionary

    def set_client_index_dictionary(self, client_number_dictionary):
        if not self._client_index_dictionary_set:
            # for client_id, iterator in zip(clients_ids, range(len(clients_ids))):
            #     self._client_index_dictionary[client_id] = iterator
            for client_cid, client_number in client_number_dictionary.items():
                self._client_index_dictionary[client_cid] = int(client_number)
            log(INFO, "Setting client-ip dictionary")
            log(INFO, f"{self._client_index_dictionary}")
            self._client_index_dictionary_set = True


class ShapleyValuesNN(ShapleyValues):

    def __init__(self, x_test, y_test, rounds, metric_list):
        super().__init__(x_test, y_test, rounds, metric_list)

    def _rec_shapley_values_calculation(self,
                                        model,
                                        clients_selected,
                                        clients_remaining,
                                        client_weights,
                                        last_result,
                                        local_round):
        for client in clients_remaining:
            new_client_selection = clients_selected.copy()
            new_remaining_clients = clients_remaining.copy()
            new_client_selection.add(client)
            new_remaining_clients.remove(client)
            # log(INFO, 50 * "=")
            # log(INFO, 50 * "=")
            # log(INFO, "New Calculation")
            # log(INFO, 50 * "=")
            # log(INFO, "Calculation for clients {}".format([client.cid for client in new_client_selection]))

            model.set_model(aggregate_nn([client_weights[client] for client in new_client_selection]))
            participant_subset_dict_metrics = evaluator(self._x_test,
                                                        self._y_test,
                                                        model,
                                                        self._metric_list)
            # log(INFO, "Accuracy adding client {}: {}".format(client.cid, accuracy))
            # log(INFO, "Former Accuracy: {}".format(former_accuracy))
            self.update_shapley_value(local_round,
                                      client,
                                      (participant_subset_dict_metrics - last_result) *
                                      math.factorial(len(new_remaining_clients)))

            # log(INFO, "Difference on accuracy for client {}: {}".format(client.cid,
            #                                                             (accuracy - former_accuracy) *
            #                                                             math.factorial(len(new_remaining_clients))))
            if len(clients_remaining) > 1:
                self._rec_shapley_values_calculation(model,
                                                     new_client_selection,
                                                     new_remaining_clients,
                                                     client_weights,
                                                     participant_subset_dict_metrics,
                                                     local_round)

    def shapley_values_calculation(self,
                                   model: Model,
                                   clients_list,
                                   client_weights,
                                   local_round):
        self._rec_shapley_values_calculation(model,
                                             set(),
                                             clients_list,
                                             client_weights,
                                             self._last_round_result,
                                             local_round)
        self.last_division(local_round)
        # result_dictionary = {client: str(shapley_values_client)
        #                      for client, shapley_values_client
        #                      in self._shapley_values[local_round].items()}
        # log(INFO, 50 * "=")
        # log(INFO, f"Shapley_Values in round {local_round}: {result_dictionary}")
        # log(INFO, 50 * "=")


class ShapleyValuesDT(ShapleyValues):
    def __init__(self, x_test, y_test, rounds, metric_list):
        super().__init__(x_test, y_test, rounds, metric_list)

    def _rec_shapley_values_calculation(self,
                                        model,
                                        clients_selected,
                                        clients_remaining,
                                        client_trees,
                                        last_result,
                                        local_round,
                                        global_model):
        for client in clients_remaining:
            new_client_selection = clients_selected.copy()
            new_remaining_clients = clients_remaining.copy()
            new_client_selection.add(client)
            new_remaining_clients.remove(client)
            # log(INFO, 50 * "=")
            # log(INFO, 50 * "=")
            # log(INFO, "New Calculation")
            # log(INFO, 50 * "=")
            # log(INFO, "Calculation for clients {}".format([client.cid for client in new_client_selection]))

            model.set_model(aggregate_xgboost(
                [client_trees[client] for client in new_client_selection], global_model
            ))
            participant_subset_dict_metrics = evaluator(self._x_test,
                                                        self._y_test,
                                                        model,
                                                        self._metric_list)
            # log(INFO, "Accuracy adding client {}: {}".format(client.cid, accuracy))
            # log(INFO, "Former Accuracy: {}".format(former_accuracy))
            self.update_shapley_value(local_round,
                                      client,
                                      (participant_subset_dict_metrics - last_result) *
                                      math.factorial(len(new_remaining_clients)))

            # log(INFO, "Difference on accuracy for client {}: {}".format(client.cid,
            #                                                             (accuracy - former_accuracy) *
            #                                                             math.factorial(len(new_remaining_clients))))
            if len(clients_remaining) > 1:
                self._rec_shapley_values_calculation(model,
                                                     new_client_selection,
                                                     new_remaining_clients,
                                                     client_trees,
                                                     participant_subset_dict_metrics,
                                                     local_round,
                                                     global_model)

    def shapley_values_calculation(self,
                                   model: Model,
                                   clients_list,
                                   client_weights,
                                   local_round,
                                   global_model):
        self._rec_shapley_values_calculation(model,
                                             set(),
                                             clients_list,
                                             client_weights,
                                             self._last_round_result,
                                             local_round,
                                             global_model)
        self.last_division(local_round)
        # result_dictionary = {client: str(shapley_values_client)
        #                      for client, shapley_values_client
        #                      in self._shapley_values[local_round].items()}
        # log(INFO, 50 * "=")
        # log(INFO, f"Shapley_Values in round {local_round}: {result_dictionary}")
        # log(INFO, 50 * "=")
