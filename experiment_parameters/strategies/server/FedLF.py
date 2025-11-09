"""
Local Fusion is an experimental strategy that is not related to any paper.
"""
from logging import WARNING, INFO
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import FitIns
from flwr.common import Parameters, Scalar, EvaluateRes, FitRes, NDArray, parameters_to_ndarrays, \
    ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from experiment_parameters.strategies.server.FedAvgRewritten import WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW
from metrics.Metrics import Accuracy
from metrics.ShapleyGTG import ShapleyGTG
from util.Util import evaluate_nn_model


class FedLF(FedAvg):
    _shapley_values: ShapleyGTG
    _accuracy_object: Accuracy

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
            self,
            dataset_factory,
            model,
            shapley_values,
            accuracy_object,
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
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self._shapley_values = shapley_values
        self._accuracy_object = accuracy_object
        self._evaluation_dataset = dataset_factory.get_dataset()
        self._model = model.get_model()

    # One-time init at start of strategy
    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Tuple[Optional[Parameters]]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    # Configure the upcoming aggregation round
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:

        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        client_dict = client_manager.all()
        # Build client config
        for index, _ in enumerate(client_dict):
            current_client_instructions = client_instructions[index]
            config = current_client_instructions[1].config
            config["current_round"] = server_round
            log(INFO, f"Config per Client: \n{current_client_instructions[1].config}")
        return client_instructions

    # Receive and aggregate the results
    # Only collects the results that are defined in the configure_fit function
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # -- Stop if error
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # -- Get client model information and convert
        client_weights = dict()
        for client, fit_res in results:
            client_weights[client.cid] = (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]

        # -- Perform strategy
        new_model = aggregate(weights_results)
        x_test, y_test = self._evaluation_dataset.get_test_data()
        loss, accuracy, ce_loss_result, metrics_dict = evaluate_nn_model(x_test, y_test, model=self._model, weights=new_model)
        self._accuracy_object.set_value(accuracy)

        # -- Display model information
        log(INFO, f"Accuracy of global model in Round -{server_round}- : {self._accuracy_object.get_value()}")
        log(INFO, f"F1 Scores of global model in Round -{server_round}- : {metrics_dict['f1_scores']}")
        log(INFO, "Loss of global model: {}".format(loss))
        for client, fit_res in results:
            log(INFO, "Labels of client {}: {}".format(client.cid, fit_res.metrics))

        new_model_to_parameters = ndarrays_to_parameters(new_model)

        return new_model_to_parameters, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        for client in results:
            print(client[1])
        return super().aggregate_evaluate(server_round, results, failures)