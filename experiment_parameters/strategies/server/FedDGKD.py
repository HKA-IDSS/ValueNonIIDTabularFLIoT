# from logging import WARNING, INFO
# from typing import Optional, Tuple, Dict, List, Callable
#
# from flwr.common import Parameters, Scalar, FitRes, FitIns, NDArray, parameters_to_ndarrays, ndarrays_to_parameters
# from flwr.common.logger import log
# from flwr.server.client_manager import ClientManager
# from flwr.server.client_proxy import ClientProxy
# from flwr.server.strategy import FedAvg
# from flwr.server.strategy.aggregate import aggregate
# from pandas import DataFrame
#
# from experiment_parameters.strategies.server.FedAvgRewritten import WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW
# from metrics.Metrics import Accuracy, CrossEntropyLoss
# from metrics.Shapley_Values import ShapleyValues
# from util.Util import evaluator
#
#
# class FedDGKD(FedAvg):
#     _centralized_X: DataFrame
#     _centralized_y: DataFrame
#     _shapley_values: ShapleyValues
#     _accuracy_object: Accuracy
#     _ce_loss_object: CrossEntropyLoss
#     _max_round: int
#
#     # pylint: disable=too-many-arguments,too-many-instance-attributes
#     def __init__(
#             self,
#             max_round,
#             dataset_factory,
#             model,
#             shapley_values,
#             accuracy_object,
#             ce_loss_object,
#             fraction_fit: float = 0.1,
#             fraction_eval: float = 0.1,
#             min_fit_clients: int = 2,
#             min_eval_clients: int = 2,
#             min_available_clients: int = 2,
#             eval_fn: Optional[
#                 Callable[[NDArray], Optional[Tuple[float, Dict[str, Scalar]]]]
#             ] = None,
#             on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
#             on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
#             accept_failures: bool = True,
#             initial_parameters: Optional[Parameters] = None,
#     ) -> None:
#         super().__init__()
#
#         if (
#                 min_fit_clients > min_available_clients
#                 or min_eval_clients > min_available_clients
#         ):
#             log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)
#
#         self.fraction_fit = fraction_fit
#         self.fraction_eval = fraction_eval
#         self.min_fit_clients = min_fit_clients
#         self.min_eval_clients = min_eval_clients
#         self.min_available_clients = min_available_clients
#         self.eval_fn = eval_fn
#         self.on_fit_config_fn = on_fit_config_fn
#         self.on_evaluate_config_fn = on_evaluate_config_fn
#         self.accept_failures = accept_failures
#
#         self.global_model_parameters = initial_parameters
#         self._shapley_values = shapley_values
#         self._accuracy_object = accuracy_object
#         self._model = model.get_model()
#         self._evaluation_dataset = dataset_factory.get_dataset()
#         self._ce_loss_object = ce_loss_object
#         self._max_round = max_round
#         self.current_round = 1
#
#     def initialize_parameters(
#             self, client_manager: ClientManager
#     ) -> Tuple[Optional[Parameters]]:
#         """Initialize global model parameters."""
#         initial_parameters = self.global_model_parameters  # Keeping initial parameters on memory
#         return initial_parameters
#
#     # Configure the upcoming aggregation round
#     def configure_fit(
#             self,
#             server_round: int,
#             parameters: Parameters,
#             client_manager: ClientManager
#     ) -> List[Tuple[ClientProxy, FitIns]]:
#         self.current_round = server_round
#         client_instructions = super().configure_fit(server_round, parameters, client_manager)
#         client_dict = client_manager.all()
#         # Build client config
#         for index, _ in enumerate(client_dict):
#             current_client_instructions = client_instructions[index]
#             config = current_client_instructions[1].config
#             config["current_round"] = server_round
#             log(INFO, f"Config per Client: \n{current_client_instructions[1].config}")
#         return client_instructions
#
#     def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]) \
#             -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
#         """
#         Aggregate fit results using weighted average.
#
#         Args:
#             rnd (int): _description_
#             results (List[Tuple[ClientProxy, FitRes]]): _description_
#             failures (List[BaseException]): _description_
#
#         Returns:
#             Tuple[Optional[Parameters], Dict[str, Scalar]]: _description_
#         """
#         if not results:
#             return None, {}
#         # Do not aggregate if there are failures and failures are not accepted
#         if not self.accept_failures and failures:
#             return None, {}
#         # Convert results
#         client_weights = dict()
#         for client, fit_res in results:
#             client_weights[client.cid] = (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
#         weights_results = [
#             (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
#             for client, fit_res in results
#         ]
#
#         current_fed_avg_srv_weights = aggregate(weights_results)
#
#         # Evaluate server model
#         x_test, y_test = self._evaluation_dataset.get_test_data()
#         loss, accuracy, ce_loss_result, metrics_dict = evaluator(x_test, y_test,
#                                                                          model=self._model,
#                                                                          weights=current_fed_avg_srv_weights)
#         self._accuracy_object.set_value(accuracy)
#         log(INFO, f"Accuracy of global model in Round -{server_round}- : {self._accuracy_object.get_value()}")
#         log(INFO, f"F1 Scores of global model in Round -{server_round}- : {metrics_dict['f1_scores']}")
#         log(INFO, "Loss of global model: {}".format(loss))
#         for client, fit_res in results:
#             log(INFO, "Labels of client {}: {}".format(client.cid, fit_res.metrics))
#         latest_global_model_params = ndarrays_to_parameters(current_fed_avg_srv_weights)
#         return latest_global_model_params, {}
