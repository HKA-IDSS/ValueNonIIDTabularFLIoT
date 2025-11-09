import os
from logging import WARNING, INFO
from typing import Optional, Tuple, Dict, List, Callable, Union

import tensorflow as tf
from flwr.common import Parameters, Scalar, FitRes, NDArray, parameters_to_ndarrays, \
    ndarrays_to_parameters, EvaluateRes, EvaluateIns
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from pandas import DataFrame

from Definitions import ROOT_DIR
from experiment_parameters.strategies.server.FedAvgRewritten import WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW, \
    save_data_on_pickle
from metrics.Metrics import Accuracy, CrossEntropyLoss
from metrics.ShapleyGTG import ShapleyGTG
from util.Util import evaluate_nn_model, obtain_prediction_from_ensemble, obtain_server_model


class FedKD(FedAvg):
    _centralized_X: DataFrame
    _centralized_y: DataFrame
    _shapley_values: ShapleyGTG
    _accuracy_object: Accuracy
    _ce_loss_object: CrossEntropyLoss
    _max_round: int
    _compute_shapley_values: bool

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
            self,
            max_round,
            dataset_factory,
            model,
            shapley_values,
            accuracy_object,
            ce_loss_object,
            compute_shapley_values=False,
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
        self.global_model_parameters = initial_parameters

        # Eval synth model uses _evaluation_dataset test data
        # Use _evaluation_dataset train for KD
        self._evaluation_dataset = dataset_factory.get_dataset()
        self._model = model.get_model()
        self._shapley_values = shapley_values
        self._accuracy_object = accuracy_object
        self._ce_loss_object = ce_loss_object
        self._max_round = max_round
        self._compute_shapley_values = compute_shapley_values

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Tuple[Optional[Parameters]]:
        """Initialize global model parameters."""
        initial_parameters = self.global_model_parameters  # Keeping initial parameters on memory
        return initial_parameters

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]) \
            -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
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
        for client, fit_res in results:
            client_weights[client.cid] = (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]

        new_model = aggregate(weights_results)

        """ The following subsection handles the knowledge distillation process.

        """
        x_train, _ = self._evaluation_dataset.get_training_data()

        optimizer = tf.keras.optimizers.Adam()

        model_ensemble = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Perform knowledge distillation with CrossEntropy
        server_weights = knowledge_distillation(x_train=x_train, model_type=self._model, server_weights=new_model,
                                                optimizer=optimizer,
                                                loss_func=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                                model_ensemble_weights=model_ensemble, steps=5)

        # Perform knowledge distillation with KLD
        # server_weights = knowledge_distillation(new_model,optimizer=optimizer, \
        #     loss_func=kl_divergance,\
        #     model_ensemble=model_ensemble,dataset=self.dataset, steps=5)

        # Evaluate server model
        x_test, y_test = self._evaluation_dataset.get_test_data()
        loss, accuracy, _, metrics_dict = evaluate_nn_model(x_test, y_test, self._model,
                                                            server_weights)

        self._accuracy_object.set_value(accuracy)
        log(INFO, f"Accuracy of global model in Round -{server_round}- : {self._accuracy_object.get_value()}")
        log(INFO, f"F1 Scores of global model in Round -{server_round}- : {metrics_dict['f1_scores']}")
        log(INFO, "Loss of global model: {}".format(loss))
        for client, fit_res in results:
            log(INFO, "Labels of client {}: {}".format(client.cid, fit_res.metrics))
        new_model_to_parameters = ndarrays_to_parameters(server_weights)

        return new_model_to_parameters, {}

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
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
                save_data_on_pickle(ROOT_DIR + "\\data\\pickled_information\\model.pkl", self._clients_weights)
            else:
                config["compute_shapley_values"] = 0

            if server_round == self._max_round:
                config["last_round"] = 1
            else:
                config["last_round"] = 0
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

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

        if os.path.exists(ROOT_DIR + "\\data\\pickled_information\\model.pkl"):
            os.remove(ROOT_DIR + "\\data\\pickled_information\\model.pkl")

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated


'''Knowledge Distillation for robust model fusion

Distill knowledge from ensemble model (weights of each client as ensemble and average overall prediction)
to server model.
Remark: In most cases the distillation converges against the (FedAvg)-ensemble model.
'''


def knowledge_distillation(x_train: DataFrame, model_type, server_weights: List[NDArray],
                           model_ensemble_weights: List[List[NDArray]], optimizer, loss_func, steps: int = 20) -> List[
    NDArray]:
    server_model = obtain_server_model(model_type, server_weights)
    x_train = tf.convert_to_tensor(x_train)

    for _ in range(0, steps):
        ensembler_predictions = obtain_prediction_from_ensemble(model_type, model_ensemble_weights, x_train)

        with tf.GradientTape() as tape:
            tape.watch(server_model.trainable_variables)
            server_predictions = server_model(x_train)
            loss = loss_func(ensembler_predictions, server_predictions)

        gradient = tape.gradient(loss, server_model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, server_model.trainable_variables))

    return server_model.get_weights()


