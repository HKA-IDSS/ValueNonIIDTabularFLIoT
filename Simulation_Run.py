import gc
import itertools
import logging
import os
import sys
import time

from tqdm import tqdm

from util.manual_partition_scripts.ManualSamplerBuilder import sample_data_manual_partition

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["KERAS_BACKEND"] = "torch"

import keras
import ray
from flwr.common import log

from Client import start_client
from Server import start_server
from Definitions import ROOT_DIR
from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util import OptunaConnection
from util.DataSampler import sample_data_dirichlet, sample_data_feature_skew_clustering


def generate_experiments() -> list[dict]:
    # Valid (strategy, model) pairs — matched by shared type key (nn/dt)

    # strategies = {"nn": ["FedAvg"], "dt": ["FedXGB"]}
    # models = {"nn": ["mlp"], "dt": ["xgboost"]}
    strategies = {"nn": ["FedAvg", "FedProx", "Scaffold"]}
    # strategies = {"nn": ["FedAvg"]}
    # strategies = {"nn": ["FedProx"]}
    models = {"nn": ["mlp"]}
    datasets = {
        "classification": [
            # "wine",
            "har",
        #     # "adult",
        #     # "covertype",
        #     # "heart",
        #     # "binary_heart"
            "edge-iot-coreset"
        ],
        "regression": [
            "wids_dataset"
        ]
    }

    partition_types = [
        "dirichlet",
        # "feature_skew",
        "manual"
    ]

    hyperparameter_search_rounds = 30
    rounds = 50

    metrics = {"classification": ["CrossEntropyLoss", "Accuracy", "F1Score", "MCC", "F1ScoreMacro", "F1ScoreMicro"],
               "regression": ["MSE", "RMSE", "MAE"]}

    DEFAULT_DATA_SPLIT = [20, 20, 20, 20, 20]
    data_split = {
        "har": [16, 16, 16, 16, 16, 16],
    }

    alpha = [
        0.1,
        10,
        1000,
    ]
    n_clients = [5]
    run_number_list = [number for number in range(1, 6)]

    # Manual partitions come in two shapes:
    #  - a plain list of names (no maverick semantics)
    #  - a dict {name: (is_maverick, maverick_client_number, eval_type, special_class)}
    #    for scenarios where one client is a "maverick" holding unique data. I separate between Mavericks of class
    #    and feature by string.
    manual_partition_names = {
        "har": {
            "HAR_1_Maverick_Laying": (True, 0, "Class", 0, None),
            "HAR_1_Maverick_WalkingUpstairs": (True, 5, "Class", 5, None),
            "HAR_1_Maverick_Laying_Balanced": (True, 0, "Class", 0, None),
            "HAR_1_Maverick_WalkingUpstairs_Balanced": (True, 5, "Class", 5, None),
            # "HAR_1_Maverick_1_MissingOneLabel": (True, 5, "Class", 5, None),
            # "HAR_1_Maverick_1_MissingTwoLabels": (True, 5, "Class", 5, None),
        },
        "edge-iot-coreset": {
            "edgeiot_coreset_1_Maverick_Least_Class": (True, 4, "Class", 6, None),
            "edgeiot_coreset_1_Maverick_Only_Normal": (True, 4, "Class", 7, None),
            "edgeiot_coreset_1_Maverick_sql_injection": (True, 4, "Class", 11, None),
            "edgeiot_coreset_1_Maverick_ddos_udp": (True, 4, "Class", 4, None),
        },
        "wids_dataset": {
            "wids_energy_non_iid_by_label": (False, None, None, None, None),
            "wids_energy_iid_sampling": (False, None, None, None, None),
            "wids_energy_feature_skew_building_type": (False, None, None, None, None),
            "wids_energy_feature_skew_state_factor": (False, None, None, None, None),
            "wids_energy_feature_skew_facility_type": (False, None, None, None, None),
            "wids_energy_maverick_facility_type_grocery_store": (
                True, 5, "Feature", ["facility_type"], "Grocery_store_or_food_market"
            ),
            "wids_energy_maverick_facility_type_uncategorized_multifamily": (
                True, 5, "Feature", ["facility_type"], "Multifamily_Uncategorized"
            ),
        }
    }

    strategy_model_pairs = [
        (strategy, model)
        for type_key in set(strategies) & set(models)
        for strategy in strategies[type_key]
        for model in models[type_key]
    ]

    dataset_metrics_pairs = [
        (dataset, metrics[type_key], type_key)
        for type_key in set(datasets) & set(metrics)
        for dataset in datasets[type_key]
    ]

    experiments = []

    for (strategy, model), (dataset, selected_metrics, task_type), number in itertools.product(
            strategy_model_pairs, dataset_metrics_pairs, run_number_list
    ):
        base = {
            "strategy": strategy,
            "model": model,
            "dataset": dataset,
            "metrics": selected_metrics,
            "rounds": rounds,
            "hyperparameter_search": False,
            "hyperparameter_search_rounds": hyperparameter_search_rounds,
            "run_number": number
        }

        # Regression tasks are restricted to manual partitions only.
        allowed_partitions = ["manual"] if task_type == "regression" else partition_types

        for partition in allowed_partitions:
            entry = {**base, "partition": partition}

            if partition == "dirichlet":
                split = data_split.get(dataset, DEFAULT_DATA_SPLIT)
                for a in alpha:
                    experiments.append({**entry, "alpha": a, "data_split": split})

            elif partition == "feature_skew":
                for nc in n_clients:
                    experiments.append({**entry, "n_clients": nc})

            elif partition == "manual":
                partitions = manual_partition_names.get(dataset, [])

                if isinstance(partitions, dict):
                    # Maverick-style entries — generate both a "with maverick"
                    # and a "without maverick" training run per scenario.
                    for name, (is_maverick, maverick_client_number, eval_type,
                               special_class_or_columns_for_subpopulation,
                               maverick_category_value) in partitions.items():
                        if is_maverick:
                            for exclude_maverick in (True, False):
                                experiments.append({
                                    **entry,
                                    "name": name,
                                    "is_maverick_scenario": True,
                                    "maverick_client_number": maverick_client_number,
                                    "maverick_eval_type": eval_type,
                                    "maverick_special_class_or_columns_for_subpopulation": special_class_or_columns_for_subpopulation,
                                    "maverick_category_value": maverick_category_value,  # NEW
                                    "exclude_maverick": exclude_maverick,
                                })
                        else:
                            experiments.append({**entry, "name": name})
                else:
                    for name in partitions:
                        experiments.append({**entry, "name": name})

    return experiments

if __name__ == "__main__":
    experiments = generate_experiments()

    ray.init(num_cpus=12, num_gpus=0)

    for experiment in tqdm(experiments):
        strategy = experiment.get("strategy", None)
        dataset = experiment.get("dataset", None)
        model = experiment.get("model", None)
        partition = experiment.get("partition", None)
        rounds = experiment.get("rounds", None)
        selected_metrics = experiment.get("metrics", None)
        hyperparameter_search = experiment.get("hyperparameter_search", None)
        hyperparameter_search_rounds = experiment.get("hyperparameter_search_rounds", None)
        alpha = experiment.get("alpha", None)
        data_split = experiment.get("data_split", None)
        n_clusters_and_clients = experiment.get("n_clients", None)
        name = experiment.get("name", None)
        run_number = experiment.get("run_number", None)

        # Maverick-related fields (only populated for maverick scenarios)
        is_maverick_scenario = experiment.get("is_maverick_scenario", False)
        maverick_client_number = experiment.get("maverick_client_number", None)
        exclude_maverick = experiment.get("exclude_maverick", False)
        eval_type = experiment.get("maverick_eval_type", None)
        columns_for_subpopulations = experiment.get("maverick_special_class_or_columns_for_subpopulation", None)

        LOG_PATH = "results" + os.sep + "logs"
        RESULTS_PATH = "results" + os.sep + "dataframes"
        HYPERPARAMETER_LOG_PATH = "results" + os.sep + "hyperparameter" + os.sep + "logs"

        if partition == "dirichlet":
            num_clients = len(data_split)

            model_final_name = strategy + "_" + dataset + "_" + partition + "_" + str(alpha) + "_" + model + "_" + str(run_number)

            route = (os.sep + strategy + \
                    os.sep + dataset + \
                    os.sep + "dirichlet" + \
                    os.sep + "alpha_" + str(alpha) + \
                    os.sep + model +
                    os.sep + str(run_number))

            optuna_search_experiment = model_final_name

            sample_data_dirichlet(name_dataset=dataset, percentages_data_clients=data_split, alpha=alpha, seed=run_number)

        elif partition == "feature_skew":
            num_clients = n_clusters_and_clients

            route = (os.sep + strategy + \
                    os.sep + dataset + \
                    os.sep + "fs_clustering" + \
                    os.sep + "n_clients_" + str(num_clients) + \
                    os.sep + model +
                    os.sep + str(run_number))

            model_final_name = strategy + "_" + \
                               dataset + "_" + \
                               "feature_skew_clustering_" + \
                               str(n_clusters_and_clients) + "clients_" + \
                               model + "_" + \
                               str(run_number)

            optuna_search_experiment = model_final_name

            sample_data_feature_skew_clustering(name_dataset=dataset, n_clients_and_clusters=n_clusters_and_clients)

        elif partition == "manual":
            directory_of_manual_partitions = ROOT_DIR + \
                                             os.sep + "data" + \
                                             os.sep + "partitioned_training_data" + \
                                             os.sep + "manual"

            # Distinguish the two maverick variants in the route / model name
            # so they don't collide and each gets its own Optuna study + results dir.
            maverick_suffix = ""
            if is_maverick_scenario:
                maverick_suffix = "_excl_maverick" if exclude_maverick else "_incl_maverick"

            route = (os.sep + strategy + \
                    os.sep + dataset + \
                    os.sep + "manual" + \
                    os.sep + name + maverick_suffix + \
                    os.sep + model + \
                    os.sep + str(run_number))

            optuna_search_experiment = strategy + "_" + \
                               dataset + "_" + \
                               "manual_" + \
                               name + "_" + \
                               model + "_" + \
                               str(run_number)

            model_final_name = strategy + "_" + \
                               dataset + "_" + \
                               "manual_" + \
                               name + maverick_suffix + "_" + \
                               model + "_" + \
                               str(run_number)

            sample_data_manual_partition(name, run_number)

            num_clients = 0
            for files in os.listdir(directory_of_manual_partitions + os.sep + name + os.sep + str(run_number)):
                num_clients += 1

            num_clients = num_clients / 4

            if is_maverick_scenario:
                num_clients = num_clients - 1

        # Work out which client indices actually participate in *training*.
        # For a maverick scenario with exclude_maverick=True, we drop that one
        # client from the training run while the underlying partitioned data
        # on disk (used for later centralized/per-client evaluation) is untouched.
        # if partition == "manual" and is_maverick_scenario and exclude_maverick and maverick_client_number is not None:
        #     client_indices = [c for c in range(int(num_clients)) if c != maverick_client_number]
        # else:
        #     client_indices = list(range(int(num_clients)))
        if columns_for_subpopulations is not None and type(columns_for_subpopulations) is list:
            _, x_subpopulation_combinations = dataset_model_dictionary[dataset]().get_dataset().get_subpopulation_of_interest(columns_for_subpopulations)
        else:
            x_subpopulation_combinations = None

        client_indices = list(range(int(num_clients)))
        num_training_clients = len(client_indices)

        HYPERPARAMETER_LOG_PATH = HYPERPARAMETER_LOG_PATH + route
        LOG_PATH = LOG_PATH + route
        RESULTS_PATH = RESULTS_PATH + route

        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH, exist_ok=True)

        if not os.path.exists(HYPERPARAMETER_LOG_PATH):
            os.makedirs(HYPERPARAMETER_LOG_PATH, exist_ok=True)

        server_log = open(f'{LOG_PATH}/server_log.log', 'a')
        server_log.write(f'{time.ctime()}  - Start logging \n')
        server_log.flush()

        server_log_hs = open(f'{HYPERPARAMETER_LOG_PATH}/server_log.log', 'a')
        server_log_hs.write(f'{time.ctime()}  - Start logging \n')
        server_log_hs.flush()

        if partition == "dirichlet":
            directory_of_data = (os.sep + "dirichlet" +
                                 os.sep + "dataset_" + dataset +
                                 os.sep + "alpha_" + str(alpha) +
                                 os.sep + str(run_number))

        elif partition == "manual":
            directory_of_data = os.sep + "manual" + os.sep + name + os.sep + str(run_number)

        elif partition == "feature_skew":
            directory_of_data = os.sep + "feature_skew_clustering" + \
                                os.sep + "dataset_" + dataset + \
                                os.sep + str(num_clients) + "_clients"

        total_data_count = len(dataset_model_dictionary[dataset]().get_dataset().get_training_data()[0])

        compute_shapley_value = 0
        load_best_trial = 0
        RESULTS_PATH_WITH_SEED = RESULTS_PATH + os.sep + str(run_number)
        experiment_name_route = route
        if os.path.exists(RESULTS_PATH_WITH_SEED):
            print(f"Results already exists for {RESULTS_PATH_WITH_SEED}. Passing to next experiment")
        else:
            try:
                print(f"Model final name: {model_final_name}")
                print(f"Optuna study: {optuna_search_experiment}")
                OptunaConnection.load_study(optuna_search_experiment)
            except Exception:
                print("Creating new study")
                OptunaConnection.optuna_create_study(optuna_search_experiment, "minimize")
                hyperparameter_search = True
            try:
                if hyperparameter_search:
                    for trial in range(hyperparameter_search_rounds):
                        subprocesses = []
                        subprocesses.append(start_server.remote(strategy,
                                                                directory_of_data,
                                                                model,
                                                                int(num_training_clients),
                                                                rounds,
                                                                selected_metrics,
                                                                model_final_name,
                                                                optuna_search_experiment,
                                                                load_best_trial,
                                                                compute_shapley_value,
                                                                RESULTS_PATH_WITH_SEED,
                                                                experiment_name_route,
                                                                exclude_maverick,
                                                                maverick_client_number,
                                                                x_subpopulation_combinations))
                        if model == "mlp":
                            time.sleep(2)
                        elif model == "xgboost":
                            time.sleep(2)

                        if is_maverick_scenario and exclude_maverick:
                            for client_number in client_indices:
                                subprocesses.append(
                                    start_client.remote(strategy, model, client_number, directory_of_data,
                                                        selected_metrics,
                                                        client_number == maverick_client_number,
                                                        exclude_maverick,
                                                        x_subpopulation_combinations))
                            _ = ray.get(subprocesses)
                        else:
                            for client_number in client_indices:
                                subprocesses.append(
                                    start_client.remote(strategy, model, client_number, directory_of_data,
                                                        selected_metrics,
                                                        client_number == maverick_client_number,
                                                        exclude_maverick,
                                                        x_subpopulation_combinations))
                            _ = ray.get(subprocesses)

                        keras.backend.clear_session()
                        gc.collect()

                if exclude_maverick:
                    compute_shapley_value = 0
                else:
                    compute_shapley_value = 1
                load_best_trial = 1
                subprocesses = []
                subprocesses.append(
                    start_server.remote(strategy, directory_of_data, model, int(num_training_clients), rounds, selected_metrics,
                                        model_final_name, optuna_search_experiment,
                                        load_best_trial, compute_shapley_value, RESULTS_PATH_WITH_SEED,
                                        experiment_name_route, exclude_maverick, maverick_client_number,
                                        x_subpopulation_combinations))
                if model == "mlp":
                    time.sleep(2)
                elif model == "xgboost":
                    time.sleep(2)

                if is_maverick_scenario and exclude_maverick:
                    for client_number in client_indices:
                        subprocesses.append(
                            start_client.remote(strategy, model, client_number, directory_of_data, selected_metrics,
                                                client_number == maverick_client_number,
                                                exclude_maverick,
                                                x_subpopulation_combinations))
                    _ = ray.get(subprocesses)
                else:
                    for client_number in client_indices:
                        subprocesses.append(
                            start_client.remote(strategy, model, client_number, directory_of_data, selected_metrics,
                                                client_number == maverick_client_number,
                                                exclude_maverick,
                                                x_subpopulation_combinations))
                    _ = ray.get(subprocesses)
            finally:
                ray.shutdown()
                keras.backend.clear_session()
                gc.collect()