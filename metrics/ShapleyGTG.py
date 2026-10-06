# import copy
# import math
# import random
# from logging import INFO
#
# import numpy as np
# from flwr.common.logger import log
# from flwr.server.strategy.aggregate import aggregate
#
# from metrics.Evaluator import evaluator
# from metrics.Metrics import DictOfMetrics, CrossEntropyLoss, Accuracy, SVCompatibleF1Score, \
#     return_default_dict_of_metrics  # , min_out_two_metrics, max_out_two_metrics
# from util.Util import
#
#
# # def get_min_value_metric(shapley_values, default_value, metric):
# #     min_value_metric = None
# #     for shapley_value_list_of_metrics in shapley_values.values():
# #         if min_value_metric is None:
# #             dict_of_min_values = shapley_value_list_of_metrics.get_value()[metric]
# #         dict_of_min_values = min_out_two_metrics(dict_of_min_values, shapley_value_list_of_metrics)
# #
# #     return dict_of_min_values
# #
# #
# # def get_all_max_values(shapley_values, default_value, metric):
# #     dict_of_max_values = None
# #     for shapley_value_list_of_metrics in shapley_values.values():
# #         if dict_of_max_values is None:
# #             dict_of_max_values = DictOfMetrics(shapley_value_list_of_metrics.get_value())
# #         dict_of_max_values = max_out_two_metrics(dict_of_max_values, shapley_value_list_of_metrics)
# #
# #     return dict_of_max_values
#
#
# def return_min_for_positive_max_for_negative(value):
#     return min if value > 0 else max
#
#
# def return_sub_for_positive_add_for_negative(value):
#     return "sub" if value > 0 else "add"
#
#
# def normalize_shapley_values(shapley_values: dict, marginal_gain: DictOfMetrics, default_dict):
#     dict_of_scaling_values = default_dict
#     dict_of_min_max_functions = {}
#     dict_of_add_sub_functions = {}
#
#     for metric in marginal_gain.get_value().keys():
#         if metric == "F1Score":
#             dict_of_min_max_functions[metric] = [return_min_for_positive_max_for_negative(value)
#                                                  for value in marginal_gain.get_value_of_metric(metric)]
#             dict_of_add_sub_functions[metric] = [return_sub_for_positive_add_for_negative(value)
#                                                  for value in marginal_gain.get_value_of_metric(metric)]
#         else:
#             dict_of_min_max_functions[metric] = return_min_for_positive_max_for_negative(marginal_gain.get_value_of_metric(metric))
#             dict_of_add_sub_functions[metric] = return_sub_for_positive_add_for_negative(marginal_gain.get_value_of_metric(metric))
#
#     for shapley_value_list_of_metrics in shapley_values.values():
#         dict_of_scaling_values = dict_of_scaling_values.obtain_min_or_max(shapley_value_list_of_metrics,
#                                                                           dict_of_min_max_functions)
#
#     for client in shapley_values.keys():
#         shapley_values[client] = \
#             shapley_values[client].addition_or_substraction(dict_of_scaling_values,
#                                                             dict_of_add_sub_functions)
#
#     sum_value = None
#     for client, shapley_value_list_of_metrics in shapley_values.items():
#         if sum_value is None:
#             log(INFO, "Shapley Value metrics for client {}: {}".format(str(client), str(shapley_value_list_of_metrics)))
#             sum_value = shapley_value_list_of_metrics
#         else:
#             log(INFO, "Shapley Value metrics for client {}: {}".format(str(client), str(shapley_value_list_of_metrics)))
#             sum_value = sum_value + shapley_value_list_of_metrics
#
#     # for shapley_value_list_of_metrics in shapley_values.values():
#     #     # print("Shapley Values: {}".format(shapley_value_list_of_metrics.get_value()[1].get_value()))
#     #     if sum_value is None:
#     #         sum_value = shapley_value_list_of_metrics
#     #     else:
#     #         sum_value = sum_value + shapley_value_list_of_metrics
#     #         # print(sum_value.get_value()[1].get_value())
#     #
#     #     # log(INFO, "Sum_value: {}".format(str(sum_value)))
#
#     for metric in sum_value.get_value().values():
#         if isinstance(metric.get_value(), list):
#             list_of_f1_scores = metric.get_value()
#             # metric.set_value(
#             #     list(map(lambda x: 1 if math.isclose(x, 0, rel_tol=1e-03, abs_tol=0.0) else x,
#             #              list_of_f1_scores)))
#             for iterator in range(len(list_of_f1_scores)):
#                 if math.isclose(list_of_f1_scores[iterator], 0, rel_tol=1e-9, abs_tol=1e-3):
#                     log(INFO, "Null value on {}".format(iterator))
#                     log(INFO, "Division by zero")
#                     list_of_f1_scores[iterator] = 1e9
#         else:
#             if math.isclose(metric.get_value(), 0, rel_tol=1e-9, abs_tol=1e-3):
#                 log(INFO, "Division by zero")
#                 metric.set_value(1e9)
#
#     log(INFO, "Sum_value: {}".format(str(sum_value)))
#
#     return {k: (marginal_gain * v / sum_value) for k, v in shapley_values.items()}
#
#
# """Old function for Normalize Shapley Values
#
# def normalize_shapley_values(shapley_values: dict, marginal_gain: float):
#     sum_value: float = 0
#     if marginal_gain >= 0:
#         # log(INFO, "Marginal gain: {}".format(marginal_gain))
#         # log(INFO, "Shapley_Values Values marginal gain > 0: {}".format(shapley_values.values()))
#         sum_value = sum([v for v in shapley_values.values() if v) >= 0])
#         # log(INFO, "Sum_value higher than 0: {}".format(sum_value))
#         if math.isclose(sum_value, 0):
#             sum_value = 1e-9
#     else:
#         # log(INFO, "Marginal gain: {}".format(marginal_gain))
#         # log(INFO, "Shapley_Values Values marginal gain < 0: {}".format(shapley_values.values()))
#         sum_value = sum([v for v in shapley_values.values() if v < 0])
#         if math.isclose(sum_value, 0):
#             sum_value = -1e-9
#
#     return {k: float(marginal_gain * v / sum_value) for k, v in shapley_values.items()}
#
# """
#
#
# class ShapleyGTG:
#     _worker_number: int
#     _shapley_values: dict
#     _shapley_values_S: dict
#     _converge_min: float
#     _eps: float
#     _round_trunc_threshold: float
#
#     _last_k: int
#     _converge_criteria: float
#     _last_round_result: DictOfMetrics
#
#     _client_index_dictionary: dict
#     _client_index_dictionary_set: bool
#
#     def __init__(self, worker_number):
#         self._worker_number = worker_number
#         self._shapley_values = dict()
#         self._shapley_values_S = dict()
#         self._eps = 0.001
#         self._round_trunc_threshold = 0.001
#         self._converge_min = max(30, self._worker_number)
#         self._last_k = 10
#
#         self._converge_criteria = 0.05
#
#         self._max_percentage = 0.8
#         self._max_number = min(
#             2 ** self._worker_number,
#             max(
#                 self._converge_min,
#                 self._max_percentage * (2 ** self._worker_number)
#                 + np.random.randint(-5, +5),
#             ),
#         )
#         # self._last_round_result
#         self._client_index_dictionary = dict()
#         self._client_index_dictionary_set = False
#
#     def set_initial_metrics(self, metrics):
#         self._last_round_result = metrics
#
#     def set_client_index_dictionary(self, client_number_dictionary):
#         if not self._client_index_dictionary_set:
#             # for client_id, iterator in zip(clients_ids, range(len(clients_ids))):
#             #     self._client_index_dictionary[client_id] = iterator
#             for client_cid, client_number in client_number_dictionary.items():
#                 self._client_index_dictionary[client_cid] = int(client_number)
#             log(INFO, "Setting client-ip dictionary")
#             log(INFO, f"{self._client_index_dictionary}")
#             self._client_index_dictionary_set = True
#
#     def get_shapley_values(self):
#         return self._shapley_values
#
#     def shapley_values_calculation(self,
#                                    x_test,
#                                    y_test,
#                                    model,
#                                    clients_list,
#                                    client_weights,
#                                    this_round_result,
#                                    local_round,
#                                    metric_list):
#         # log(INFO, "Last Round Result: {}".format(self._last_round_result))
#         # log(INFO, "This Round Result: {}".format(this_round_result))
#         # Truncation based local_round. It exits early the computation if the difference between rounds is not high.
#         # Kind of complicated way to get metrics, but it is to allow for operations over multiple types.
#         accuracy_difference = (this_round_result - self._last_round_result).get_value()["Accuracy"].get_value()
#         if abs(accuracy_difference) < self._round_trunc_threshold:
#             self._shapley_values[local_round] = {
#                 client_id: return_default_dict_of_metrics(metric_list, y_test.shape[1]) for client_id in clients_list
#             }
#             for client in self._shapley_values[local_round].keys():
#                 self._shapley_values[local_round][client].get_value()["CrossEntropyLoss"].set_value(0)
#                 log(INFO, self._shapley_values[local_round][client])
#             self._shapley_values_S[local_round] = {
#                 client_id: return_default_dict_of_metrics(metric_list, y_test.shape[1]) for client_id in clients_list
#             }
#             for client in self._shapley_values_S[local_round].keys():
#                 self._shapley_values_S[local_round][client].get_value()["CrossEntropyLoss"].set_value(0)
#             log(INFO, "Shapley_value Truncation {}".format(self._shapley_values[local_round]))
#             log(INFO, "Shapley_value_S Truncation {}".format(self._shapley_values_S[local_round]))
#             self._last_round_result = this_round_result
#             return
#
#         metrics = dict()
#         perm_records = dict()
#         index = 0
#         contribution_records: list = []
#         while self.not_convergent(index, contribution_records):
#             for client_id in clients_list:
#                 index += 1
#                 # Be careful, you are modifying here because of the cross entropy loss.
#                 v: list = [0 for _ in range(self._worker_number + 1)]
#                 v[0] = self._last_round_result
#                 marginal_contribution = [return_default_dict_of_metrics(metric_list, num_classes=y_test.shape[1])
#                                          for _ in range(self._worker_number)]
#
#                 # Changing the original code to work with labels, instead of indexes.
#                 remaining_clients_list = [remaining_client_id for remaining_client_id
#                                           in clients_list if remaining_client_id != client_id]
#                 random.shuffle(remaining_clients_list)
#                 perturbed_client_ids = [client_id] + remaining_clients_list
#
#                 # Iteration by every client after the first client selected. This list is generated by a random
#                 # permutation.
#                 for j in range(1, len(perturbed_client_ids) + 1):
#                     subset = tuple(sorted(perturbed_client_ids[:j]))
#                     # truncation
#                     if abs((this_round_result - v[j - 1]).get_value()["Accuracy"].get_value()) >= self._eps:
#                         if subset not in metrics:
#                             # I guess that an empty tuple is null, therefore false. Only valid for first.
#                             if not subset:
#                                 metric = self._last_round_result
#                             else:
#                                 dict_of_metrics = evaluator(x_test,
#                                                                     y_test,
#                                                                     model,
#                                                                     aggregate(
#                                                                      [client_weights[client_cid]
#                                                                       for client_cid
#                                                                       in subset]),
#                                                                     metric_list)
#                                 # log(INFO, "List of metrics: {}".format(dict_of_metrics))
#                                 metric = dict_of_metrics
#
#                             # log(INFO, "Round {} Subset {} Metric {}".format(local_round, subset, metric))
#                             metrics[subset] = metric
#                         # log(INFO, "Subset {} in metrics. Value: {}".format(subset, metrics[subset]))
#                         v[j] = metrics[subset]
#                     else:
#                         v[j] = v[j - 1]
#
#                     # update SV
#                     # log(INFO, "With: {} \tWithout {} \tDifference {}".format(v[j], v[j - 1], v[j] - v[j - 1]))
#                     marginal_contribution[self._client_index_dictionary[perturbed_client_ids[j - 1]]] = v[j] - v[j - 1]
#
#                 # This two store an array equivalent to all the marginal contributions of the participants in
#                 # the permutation.
#                 contribution_records.append(marginal_contribution)
#                 perm_records[tuple(perturbed_client_ids)] = marginal_contribution
#
#         # For best_S (subset of participants)
#         # It sorts the subsets of participants. First, it sorts by result with the first element.
#         # Second, it sorts by lower number of participants. The length is negative, as it is sorted by greater first.
#         # The lower the absolute value of the number, the bigger it is.
#         subset_rank = sorted(
#             metrics.items(), key=lambda x: (x[1], -len(x[0])), reverse=True
#         )
#
#         # I assume that the empty tuple means all the clients, which is not a subset.
#         # Therefore, if it is empty, it is null, and the second is selected.
#         best_S: tuple
#         if subset_rank[0][0]:
#             best_S = subset_rank[0][0]
#         else:
#             best_S = subset_rank[1][0]
#
#         # Because the clients are stored as lists, but here the code wants subsets, the function needs to
#         # transform the code into sets. That is what is done here. K represents the permutation of participants.
#         # This returns a matrix, for what it seems.
#         contrib_S = [
#             v for k, v in perm_records.items() if set(k[: len(best_S)]) == set(best_S)
#         ]
#         # SV_calc_temp = np.sum(contrib_S, axis=0) / len(contrib_S)
#         SV_calc_temp: list = contrib_S[0]
#         for iterator in range(1, len(contrib_S)):
#             for client in range(len(SV_calc_temp)):
#                 SV_calc_temp[client] += contrib_S[iterator][client]
#         SV_calc_temp = [SV_of_client / len(contrib_S) for SV_of_client in SV_calc_temp]
#         round_marginal_gain_S = metrics[best_S] - self._last_round_result
#         round_SV_S = dict()
#         for client_id in best_S:
#             round_SV_S[client_id] = SV_calc_temp[self._client_index_dictionary[client_id]]
#
#         self._shapley_values_S[local_round] = \
#             normalize_shapley_values(round_SV_S,
#                                      round_marginal_gain_S,
#                                      return_default_dict_of_metrics(metric_list, y_test.shape[1]))
#
#         # calculating fullset SV
#         # shapley value calculation
#         if set(best_S) == set(range(self._worker_number)):
#             self._shapley_values[local_round] = copy.deepcopy(
#                 self._shapley_values_S[local_round]
#             )
#         else:
#             contributions = contribution_records[0]
#             log(INFO, "Contributions: {}".format(contributions))
#             for round_to_add in range(1, len(contribution_records)):
#                 # If not done like this, it just concatenates the list
#                 for client in range(len(contributions)):
#                     contributions[client] += contribution_records[round_to_add][client]
#             log(INFO, "Contribution records: {}".format(contribution_records))
#             round_shapley_values = [contribution / len(contribution_records) for contribution in contributions]
#
#             log(INFO, "Round shapley values: {}".format(round_shapley_values))
#             assert len(round_shapley_values) == self._worker_number
#
#             round_marginal_gain = this_round_result - self._last_round_result
#             round_shapley_value_dict = dict()
#             for idx, list_of_metrics in zip(clients_list, round_shapley_values):
#                 # Value should represent the values for the different metrics
#                 round_shapley_value_dict[idx] = list_of_metrics
#
#             self._shapley_values[local_round] = \
#                 normalize_shapley_values(round_shapley_value_dict,
#                                          round_marginal_gain,
#                                          return_default_dict_of_metrics(metric_list, y_test.shape[1]))
#
#         log(INFO, "Shapley_value {}".format(self._shapley_values[local_round]))
#         log(INFO, "Shapley_value_S {}".format(self._shapley_values_S[local_round]))
#         self._last_round_result = this_round_result
#
#     def not_convergent(self, index, contribution_records):
#         if index >= self._max_number:
#             return False
#         if index <= self._converge_min:
#             return True
#         contribution_records_accuracy = copy.deepcopy(contribution_records)
#         for iterator in range(0, len(contribution_records_accuracy)):
#             contribution_records_accuracy[iterator] = [record.get_value()["Accuracy"].get_value()
#                                                        for record in contribution_records_accuracy[iterator]]
#         all_vals = (
#                            np.cumsum(contribution_records_accuracy, 0)
#                            / np.reshape(np.arange(1, len(contribution_records_accuracy) + 1), (-1, 1))
#                    )[-self._last_k:]
#         errors = np.mean(
#             np.abs(all_vals[-self._last_k:] - all_vals[-1:])
#             / (np.abs(all_vals[-1:]) + 1e-12),
#             -1,
#         )
#         if np.max(errors) > self._converge_criteria:
#             return True
#         # get_logger().info(
#         #     "convergent for index %s and converge_min %s max error %s converge_criteria %s",
#         #     index,
#         #     self.converge_min,
#         #     np.max(errors),
#         #     self.converge_criteria,
#         # )
#         return False
#
#
# if __name__ == "__main__":
#     original_round = DictOfMetrics({"CrossEntropyLoss": CrossEntropyLoss(0.70),
#                                     "Accuracy": Accuracy(0.60),
#                                     "F1Score": SVCompatibleF1Score([0.60, 0.60])
#                                     })
#     list_of_metrics1 = DictOfMetrics({"CrossEntropyLoss": CrossEntropyLoss(0.56),
#                                       "Accuracy": Accuracy(0.75),
#                                       "F1Score": SVCompatibleF1Score([0.65, 0.70])
#                                       })
#     list_of_metrics2 = DictOfMetrics({"CrossEntropyLoss": CrossEntropyLoss(0.56),
#                                       "Accuracy": Accuracy(0.80),
#                                       "F1Score": SVCompatibleF1Score([0.65, 0.70])
#                                       })
#     list_of_metrics3 = DictOfMetrics({"CrossEntropyLoss": CrossEntropyLoss(0.42),
#                                       "Accuracy": Accuracy(0.82),
#                                       "F1Score": SVCompatibleF1Score([0.70, 0.75])
#                                       })
#     # for metric in list_of_metrics3.get_value():
#     #     print(metric.get_value())
#     #
#     marginal_contribution1 = (list_of_metrics3 - list_of_metrics2) + (list_of_metrics1 - original_round)
#     marginal_contribution2 = (list_of_metrics3 - list_of_metrics1) + (list_of_metrics2 - original_round)
#     difference_on_round = list_of_metrics3 - original_round
#     #
#     print(marginal_contribution1.get_value()["Accuracy"].get_value())
#     print(marginal_contribution2.get_value()["Accuracy"].get_value())
#     print(difference_on_round.get_value()["Accuracy"].get_value())
#     print(return_default_dict_of_metrics(["F1Score", "MCC"], 6))
#
#     normalized_shapley_values = normalize_shapley_values({"1": marginal_contribution1, "2": marginal_contribution2},
#                                                          difference_on_round,
#                                                          return_default_dict_of_metrics(
#                                                              original_round.get_value().keys(), 2)
#                                                          )
#
#     for key, normalized_shapley_value in normalized_shapley_values.items():
#         print("Key: {}".format(key))
#         print("Shapley_values: {}".format(normalized_shapley_value))
