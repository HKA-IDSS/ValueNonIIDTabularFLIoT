import itertools
import math
from abc import ABC
from functools import reduce
from logging import INFO, DEBUG
from typing import Union

import numpy as np
import pandas as pd
from flwr.common.logger import log
from torch.utils.data import DataLoader
from xgboost import DMatrix

from experiment_parameters.aggregation_processes.aggregate import aggregate_nn, aggregate_trees, aggregate_xgboost
from experiment_parameters.model_builder.Model import Model
from metrics.Evaluator import evaluator, partial_computation
from metrics.Metrics import DictOfMetrics, return_default_dict_of_metrics, AggregatableMeasuresClassification, \
    AggregatableMeasures, SubpopulationDict


class ShapleyValues:
    _shapley_values: dict
    _metric_list: list
    _last_round_result: AggregatableMeasures
    _index_client_id_dictionary: dict[int, str]
    _client_index_dictionary: dict[str, int]
    _num_classes: int

    def __init__(self, rounds, metric_list, num_classes):
        # self._x_test = x_test
        # self._y_test = y_test
        self._shapley_values = {k: dict() for k in range(1, rounds + 1)}
        self._client_index_dictionary = dict()
        self._client_index_dictionary_set = False
        self._metric_list = metric_list
        self._index_client_id_dictionary = dict()
        self._num_classes = num_classes

    # def set_last_round_results(self, aggregatable_measures):
    #     self._last_round_result = aggregatable_measures

    def get_shapley_values(self):
        return self._shapley_values

    def get_round_shapley_values(self, round: int):
        return self._shapley_values[round]

    def set_shapley_values(self, shapley_values):
        self._shapley_values = shapley_values

    def get_index_client_id_dictionary(self):
        return self._index_client_id_dictionary

    def update_shapley_value(self, local_round, client_cid, dict_of_values):
        if client_cid not in self._shapley_values[local_round].keys():
            if isinstance(dict_of_values, SubpopulationDict):
                self._shapley_values[local_round][client_cid] = SubpopulationDict({})
            else:
                self._shapley_values[local_round][client_cid] = return_default_dict_of_metrics(
                    self._metric_list, self._num_classes)
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

            self._index_client_id_dictionary = {number_id: client_id
                                                for client_id, number_id in self._client_index_dictionary.items()}


def powerset(iterable):
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

def get_clients_powerset(client_index_dictionary: dict[str, int]):
    evaluation_powerset = {reduce(lambda client1, client2: client1 + client2,
                                  [pow(2, int(client_index_dictionary[client]))
                                   for client in model_combination_key],
                                  0): None
                           for model_combination_key in powerset(client_index_dictionary.keys())}
    return evaluation_powerset

def get_all_partial_aggregation_evaluation_results(
        evaluation_data: Union[DataLoader, DMatrix],
        y_test: np.ndarray,
        model: Model,
        columns,
        client_weights,
        index_client_id_dictionary,
        last_round_result: AggregatableMeasures,
        evaluation_powerset,
        subpop_values
):
    # As python does not have binary operations, we are using the number of clients as the powers with base 2.
    # Example: Client 4 is pow(2, 4) = 16.
    # This way, we have a unique number for each client combination, as it resembles the powerset of the binary
    # number with equal length as the number of clients. That is: powerset(6) equals binary of 64 (six numbers for
    # representation).
    for group in evaluation_powerset.keys():
        if group == 0:
            evaluation_powerset[0] = last_round_result
        else:
            clients_nums = list()
            combined_number_group = group
            while combined_number_group > 0:
                client_num_binary_position = int(math.trunc(math.log2(combined_number_group)))
                clients_nums.append(client_num_binary_position)
                combined_number_group -= pow(2, int(client_num_binary_position))

            model.set_model(aggregate_nn([client_weights[index_client_id_dictionary[client]]
                                          for client in clients_nums]))
            predictions = model.predict(evaluation_data)
            evaluation_results = partial_computation(predictions, y_test, columns, subpopulation_values=subpop_values)
            # clients_num_combined = reduce(
            #     lambda num1, num2: num1 + num2,
            #     [pow(2, int(self._client_index_dictionary[client_id]))
            #      for client_id in evaluation_powerset[group]],
            #     0
            # )

            evaluation_powerset[group] = evaluation_results

    return evaluation_powerset

def get_all_metrics_from_partial_results(partial_evaluations, metric_list):
    metrics_dict = evaluator(partial_evaluations, metric_list)
    return metrics_dict

class ShapleyValuesNN(ShapleyValues):

    def __init__(self, rounds, metric_list, num_classes):
        super().__init__(rounds, metric_list, num_classes)

    def _rec_shapley_values_calculation(self,
                                        metric_dict_powerset,
                                        clients_selected,
                                        clients_remaining,
                                        former_index_number,
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
            current_index_number = reduce(lambda client_num1, client_num2: client_num1 + client_num2,
                                                        [pow(2, int(self._client_index_dictionary[client_id]))
                                                             for client_id in new_client_selection],
                                                        0)

            # log(INFO, "Current index number: " + str(current_index_number))
            # log(INFO, "Former index number: " + str(former_index_number))
            # log(INFO, "Metric dict powerset: " + str(metric_dict_powerset[current_index_number]))
            # log(INFO, "Former dict powerset: " + str(metric_dict_powerset[former_index_number]))
            self.update_shapley_value(local_round,
                                      client,
                                      (metric_dict_powerset[current_index_number] -
                                       metric_dict_powerset[former_index_number]) *
                                      math.factorial(len(new_remaining_clients)))

            # log(INFO, "Difference on accuracy for client {}: {}".format(client.cid,
            #                                                             (accuracy - former_accuracy) *
            #                                                             math.factorial(len(new_remaining_clients))))
            if len(clients_remaining) > 1:
                self._rec_shapley_values_calculation(metric_dict_powerset,
                                                     new_client_selection,
                                                     new_remaining_clients,
                                                     current_index_number,
                                                     local_round)

    def shapley_values_calculation(self,
                                   metric_dict_powersets,
                                   clients_list,
                                   local_round):
        #
        # evaluation_powerset = get_all_partial_aggregation_evaluation_results(evaluation_powerset,
        #                                                                           model,
        #                                                                           client_weights,
        #                                                                           index_client_id_dictionary)
        # for k, metric_dict in metric_dict_powersets.items():
        #     log(INFO, f"Metric dict with key {k}: {metric_dict}")
        self._rec_shapley_values_calculation(metric_dict_powersets,
                                             set(),
                                             clients_list,
                                             0,
                                             local_round)
        self.last_division(local_round)


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
