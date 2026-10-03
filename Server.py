import gc
import os
import pickle
from logging import INFO
from typing import Dict, Optional, Tuple, Union

import flwr as fl
import numpy as np
import optuna
import pandas as pd
import ray
import torch
from flwr.common import Scalar, Parameters
from flwr.common.logger import log
from torch.utils.data import DataLoader
from xgboost import DMatrix

from experiment_parameters.TrainerFactory import strategies_dictionary, dataset_model_dictionary, factory_return_model
from experiment_parameters.model_builder.Model import XGBoostModel, KerasModel
from experiment_parameters.model_builder.ModelBuilder import get_training_configuration, Director
from metrics.Evaluator import evaluator, partial_computation
from metrics.GradientRewards import GradientRewards
from metrics.Metrics import return_default_dict_of_metrics, AggregatableMeasuresClassification, \
    return_default_partial_computations
from metrics.Shapley_Values import ShapleyValuesDT, ShapleyValuesNN
from util import OptunaConnection
from util.Util import get_test_data

pd.set_option('display.float_format', lambda x: '%.15f' % x)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Try to use the config for sending the parameters.
def get_fit_config_func(parameters_dict):
    def fit_config_func(server_round):
        config = {
            "server_round": server_round,
            "gradient_tracking": 0
        }
        if server_round == 1:
            config.update(parameters_dict)
            log(INFO, "Updating config: {}".format(parameters_dict))
        return config

    return fit_config_func


# Try to use the config for sending the parameters.
def get_evaluate_config_func(compute_shapley_values, num_rounds):
    def evaluate_config_func(server_round):
        config = {
            "server_round": server_round,
            "compute_shapley_values": compute_shapley_values,
            "num_rounds": num_rounds
        }

        return config

    return evaluate_config_func


# The `evaluate` function will be called after every round
# It needs to be positioned here, as it needs to have the model defined before.
def get_evaluate_function(evaluation_data, y_test, columns, model: Union[KerasModel | XGBoostModel], model_selected,
                          metric_list, x_data_subpopulations=None):
    def evaluate(
            server_round: int, parameters: Optional[Parameters | bytes], config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        log(INFO, "Server round in evaluate: {}".format(server_round))
        log(INFO, "Metric list: {}".format(metric_list))
        # x_test, y_test = get_test_data(data_route)
        if model_selected == "xgboost" and server_round == 0:
            partial_computation_result = return_default_partial_computations(columns)
            evaluation_results = return_default_dict_of_metrics(metric_list, y_test.shape[1])
        else:
            # log(INFO, "Set Model")
            # log(INFO, f"parameters: {parameters}")
            # if model_selected == "xgboost":
            #     log(INFO, f"parameter tensors: {parameters.tensors[0]}")
            #     parameters = parameters.tensors[0]
            model.set_model(parameters)
            log(INFO, "Evaluation")
            predictions = model.predict(evaluation_data)
            partial_computation_result = partial_computation(predictions, y_test, columns, x_data_subpopulations)
            evaluation_results = evaluator(partial_computation_result, metric_list)
            # evaluation_results = evaluate_tree_model(x_test, y_test, parameters, metric_list)

        if "MSE" in metric_list:
            if x_data_subpopulations is not None:
                loss = evaluation_results.get_global().get_value_of_metric("MSE")
            else:
                loss = evaluation_results.get_value_of_metric("MSE")
        else:
            if x_data_subpopulations is not None:
                loss = evaluation_results.get_global().get_value_of_metric("CrossEntropyLoss")
            else:
                loss = evaluation_results.get_value_of_metric("CrossEntropyLoss")

        return loss, {"partial_computation_result": pickle.dumps(partial_computation_result)}

    return evaluate


def generate_server_strategy(strategy_selected,
                             directory_of_data,
                             model_selected,
                             number_of_clients,
                             number_of_rounds,
                             metric_list,
                             model_final_name,
                             optuna_search_experiment,
                             load_best_trial,
                             compute_shapley_values,
                             result_path,
                             experiment_name,
                             exclude_maverick=False,
                             maverick_number=None,
                             x_data_subpopulations=None):
    log(INFO, "Generate server strategy")

    study = OptunaConnection.load_study(optuna_search_experiment)

    log(INFO, "After load study")

    if load_best_trial == 0:
        trial = study.ask()
        parameters_dict = get_training_configuration(trial, model_selected)
        log(INFO, "Trying Hyperparameter Optimization")

    else:
        # trial = None  # Quickfix. Not so nicely programmed.
        trial: optuna.Trial = study.best_trial
        parameters_dict = get_training_configuration(trial, model_selected)
        log(INFO, "Loading best combination")
    # Dataset factory. In this case, because the dataset is not used directly in this file,
    # it is not instantiated. This factory is used by the strategy, to pass the data for
    # evaluation.
    # dataset_factory = dataset_model_dictionary[dataset_selected]()
    # X_train, y_train = dataset_factory.get_dataset().get_training_data()

    # This is just to get the model prepared for KD.
    # logits = False
    # if strategy_selected in ["FedKD", "FedDKD"]:
    #     logits = True

    directory_of_data = "data" + os.sep + "partitioned_training_data" + os.sep + directory_of_data

    director = Director()
    X_train, y_train = pd.DataFrame(), pd.DataFrame()

    for file in os.listdir(directory_of_data):
        if "train" in file:
            if "_X_" in file:
                X_train = pd.concat([X_train, pd.read_csv(directory_of_data + os.sep + file, index_col=0)],
                                    ignore_index=True)
            elif "_y_" in file:
                y_train = pd.concat([y_train, pd.read_csv(directory_of_data + os.sep + file, index_col=0)],
                                    ignore_index=True)

    X_test, y_test = get_test_data(directory_of_data, number_of_clients, exclude_maverick, maverick_number)

    if x_data_subpopulations is not None:
        x_data_subpopulations = x_data_subpopulations.loc[X_test.index]

    columns = list(y_train.columns)
    # X_train, y_train = dataset_factory.get_dataset().get_training_data()
    # X_test, y_test = dataset_factory.get_dataset().get_test_data()

    # log(INFO, "Logits: {}".format(logits))
    # This is the model. For retrieving the model class (type of instance TFModel)
    # from the file Model.py.
    shape: int
    try:
        shape = y_test.shape[1]
    except:
        shape = 1
    if model_selected != "xgboost":
        # model = factory_return_model(dataset_factory, model_selected, parameters_dict)
        model = director.create_mlp(X_test.shape[1], shape, parameters_dict)

        # Flower work with NDArrays, which is what you get when you use get_weights op.
        weights = model.get_model().get_weights()

        parameters = fl.common.ndarrays_to_parameters(weights)

        shapley_values = ShapleyValuesNN(number_of_rounds, metric_list, y_test.shape[1])
        gradient_rewards = GradientRewards(number_of_clients)

        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test.to_numpy()), torch.from_numpy(y_test.to_numpy())
        )
        test_data = torch.utils.data.DataLoader(
            test_dataset, batch_size=512, shuffle=False, pin_memory=True
        )

    else:
        model = XGBoostModel()
        parameters = None

        xgboost_training_params = director.create_xgboost(X_test.shape[1], shape, parameters_dict)

        test_data = DMatrix(X_test, label=np.argmax(y_test, axis=1))

        # model.fit(xgboost_training_params,
        #           x_train=X_train,
        #           x_test=X_test,
        #           y_train=y_train,
        #           y_test=y_test,
        #           num_local_rounds=1)

        shapley_values = ShapleyValuesDT(X_test, y_test, number_of_rounds, metric_list)
        gradient_rewards = GradientRewards(number_of_clients)

    # shapley_values = ShapleyGTG(number_of_clients)
    # Strategies are retrieved from the factory, which is retrieved using the string on the dictionary.
    # Then, a type of the class of the strategy is returned. It needs to be instantiated afterwards
    # by passing all the parameters.
    # There are multiple new parameters on strategies:
    #   Dataset_factory, for retrieving data for evaluation.
    #   TFModel
    #   Shapley Values: To enable the computation. (Currently obligatory, but no need to call the function)
    #   Accuracy
    # Instantiate only the class that possesses the interested strategy. If the () does not appear after the
    # selected strategy, the code instantiates all of them.s
    # log(INFO, f"Model in server: {model}")
    # log(INFO, f"Model in server xgboost: {model.get_model()}")
    # log(INFO, f"Model in server bytes: {bytes(model.get_model().save_raw('json'))}")
    strategy_type = strategies_dictionary[strategy_selected]().create_strategy()
    strategy = strategy_type(
        # ... other fedavg arguments
        max_round=number_of_rounds,
        strategy_aggregation=strategy_selected,
        data_loader_evaluation=test_data,
        y_test=y_test,
        x_test_subpopulation_values=x_data_subpopulations,
        model=model,
        final_training=load_best_trial,
        compute_shapley_values=compute_shapley_values,
        shapley_values=shapley_values,
        gradient_rewards=gradient_rewards,
        metric_list=metric_list,
        result_path=result_path,
        experiment_name=experiment_name,
        study=study,
        trial=trial,
        min_fit_clients=number_of_clients,
        min_eval_clients=number_of_clients,
        fraction_eval=0.2,
        min_available_clients=number_of_clients,
        initial_parameters=parameters,
        # eval_fn=get_evaluate_function(directory_of_data, model, model_selected, metric_list,
        #                               study, trial, load_best_trial, number_of_rounds),
        eval_fn=get_evaluate_function(test_data, y_test, columns, model, model_selected, metric_list, x_data_subpopulations),
        on_fit_config_fn=get_fit_config_func(parameters_dict),
        on_evaluate_config_fn=get_evaluate_config_func(compute_shapley_values, number_of_rounds),
        model_final_name=model_final_name,
    )

    return strategy


#@ray.remote
@ray.remote(num_gpus=0)
def start_server(strategy_selected,
                 directory_of_data,
                 model_selected,
                 number_of_clients,
                 number_of_rounds,
                 metric_list,
                 model_final_name,
                 optuna_search_experiment,
                 load_best_trial,
                 compute_shapley_values,
                 result_path,
                 experiment_name,
                 is_maverick=False,
                 maverick_number=None,
                 x_data_subpopulations=None):
    # print(f"Ray GPUS: {ray.get_gpu_ids()}")
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # try:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)
    # except RuntimeError as e:
    #     print(e)
    log(INFO, "Server.py loaded before strategy")
    strategy = generate_server_strategy(strategy_selected,
                                        directory_of_data,
                                        model_selected,
                                        number_of_clients,
                                        number_of_rounds,
                                        metric_list,
                                        model_final_name,
                                        optuna_search_experiment,
                                        load_best_trial,
                                        compute_shapley_values,
                                        result_path,
                                        experiment_name,
                                        is_maverick,
                                        maverick_number,
                                        x_data_subpopulations)

    log(INFO, "Server.py loaded")
    # Start Flower server
    fl.server.start_server(
        server_address="localhost:54080",
        config=fl.server.ServerConfig(num_rounds=number_of_rounds),
        strategy=strategy
    )

    gc.collect()
    return None


if __name__ == "__main__":
    print("Not sure why I am here")
    # strategy_selected = argv[1]
    # dataset_selected = argv[2]
    # model_selected = argv[3]
    # number_of_clients = int(argv[4])
    # number_of_rounds = int(argv[5])
    # metric_list = argv[6].split("-")
    # model_final_name = argv[7]
    # load_best_trial = int(argv[8])
    # compute_shapley_values = int(argv[9])
    #
    # ray.put(strategy_selected)
    # ray.put(dataset_selected)
    # ray.put(model_selected)
    # ray.put(number_of_clients)
    # ray.put(number_of_rounds)
    # ray.put(metric_list)
    # ray.put(model_final_name)
    # ray.put(load_best_trial)
    # ray.put(compute_shapley_values)
    #
    # ray.get(start_server.remote(strategy_selected,
    #                             dataset_selected,
    #                             model_selected,
    #                             number_of_clients,
    #                             number_of_rounds,
    #                             metric_list,
    #                             model_final_name,
    #                             load_best_trial,
    #                             compute_shapley_values))
