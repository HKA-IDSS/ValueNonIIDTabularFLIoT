import logging
import os
import sys
import time
import warnings
from logging import INFO


os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
import tensorflow as tf
from flwr.common import log

from Client import start_client
from Server import start_server
from Definitions import ROOT_DIR
from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util import OptunaConnection
from util.DataSampler import sample_data_dirichlet, sample_data_feature_skew_clustering
from util.Util import arg_parser

if __name__ == "__main__":
    # Right now, the only parameter is a yaml with all the parameters.
    # All the possible parameters are on experiment_configuration\CompleteExperiment.yaml.
    filename = sys.argv[1]
    experiment_config = arg_parser("experiment_configuration" + os.sep + filename + ".yaml")
    LOG_PATH = "results" + os.sep + "logs"
    RESULTS_PATH = "results" + os.sep + "dataframes"
    HYPERPARAMETER_LOG_PATH = "results" + os.sep + "hyperparameter" + os.sep + "logs"

    # TODO: It would be a good idea if in a file can be set all the combinations for different configs.
    strategy = experiment_config["strategy"]
    dataset = experiment_config["dataset"]
    model = experiment_config["model"]

    partition = experiment_config["partition"]

    rounds = experiment_config["rounds"]

    selected_metrics = experiment_config["metrics"]
    hyperparameter_search = experiment_config["hyperparameter_search"]
    hyperparameter_search_rounds = experiment_config["hyperparameter_search_rounds"]




    # def logging_setup_func():
    #     logger = logging.getLogger("ray")
    #     logger.setLevel(logging.DEBUG)
    #     warnings.simplefilter("always")
    #     os.environ["RAY_DEDUP_LOGS"] = "0"

    ray.init(num_cpus=12, num_gpus=1)
    # ray.init(num_cpus=12, num_gpus=1, runtime_env={"worker_process_setup_hook": logging_setup_func})

    # logging_setup_func()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

    if partition == "dirichlet":
        alpha = experiment_config["alpha"]
        # seed = experiment_config["seed"]
        data_split = experiment_config["data_split"]
        num_clients = len(data_split)

        model_final_name = strategy + "_" + dataset + "_" + partition + "_" + str(alpha) + "_" + model

        route = os.sep + strategy + \
                os.sep + dataset + \
                os.sep + "dirichlet" + \
                os.sep + "alpha_" + str(alpha) + \
                os.sep + model

        # # Create datasets for each client
        # subprocess.run(['python', 'Main.py', dataset, str(num_clients), str(alpha)], capture_output=True,
        #                creationflags=subprocess.CREATE_NEW_CONSOLE)

        sample_data_dirichlet(name_dataset=dataset, percentages_data_clients=data_split, alpha=alpha, seed=1)

    elif partition == "feature_skew":
        n_clusters_and_clients = experiment_config["n_clients"]
        num_clients = n_clusters_and_clients

        route = os.sep + strategy + \
                os.sep + dataset + \
                os.sep + "fs_clustering" + \
                os.sep + "n_clients_" + str(num_clients) + \
                os.sep + model

        model_final_name = strategy + "_" + \
                           dataset + "_" + \
                           "feature_skew_clustering_" + \
                           str(n_clusters_and_clients) + "clients_" + \
                           model

        sample_data_feature_skew_clustering(name_dataset=dataset, n_clients_and_clusters=n_clusters_and_clients)

    elif partition == "manual":
        name = experiment_config["name"]
        directory_of_manual_partitions = ROOT_DIR + \
                                         os.sep + "data" + \
                                         os.sep + "partitioned_training_data" + \
                                         os.sep + "manual"
        num_clients = 0
        for files in os.listdir(directory_of_manual_partitions + os.sep + name):
            num_clients += 1

        num_clients = num_clients / 4

        route = os.sep + strategy + \
                os.sep + dataset + \
                os.sep + "manual" + \
                os.sep + name + \
                os.sep + model

        model_final_name = strategy + "_" + \
                           dataset + "_" + \
                           "manual_" + \
                           name + "_" + \
                           model

    HYPERPARAMETER_LOG_PATH = HYPERPARAMETER_LOG_PATH + route
    LOG_PATH = LOG_PATH + route
    RESULTS_PATH = RESULTS_PATH + route

    # Check if log dir exists and create it if needed
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH, exist_ok=True)

    if not os.path.exists(HYPERPARAMETER_LOG_PATH):
        os.makedirs(HYPERPARAMETER_LOG_PATH, exist_ok=True)

    # selected_metrics += ["CrossEntropyLoss", "Accuracy"]

    # Start Flower server
    server_log = open(f'{LOG_PATH}/server_log.log', 'a')
    server_log.write(f'{time.ctime()}  - Start logging \n')
    server_log.flush()

    # Start Flower server
    server_log_hs = open(f'{HYPERPARAMETER_LOG_PATH}/server_log.log', 'a')
    server_log_hs.write(f'{time.ctime()}  - Start logging \n')
    server_log_hs.flush()
    # TODO: Fabian: Maybe, you need to add here the gpu_mem_limit.
    # isUsingGPU = False
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     isUsingGPU = True

    if partition == "dirichlet":
        directory_of_data = os.sep + "dirichlet" + os.sep + "dataset_" + dataset + os.sep + "alpha_" + str(alpha)

    elif partition == "manual":
        # Create the virtual clients
        directory_of_data = os.sep + "manual" + os.sep + name

    elif partition == "feature_skew":
        directory_of_data = os.sep + "feature_skew_clustering" + \
                            os.sep + "dataset_" + dataset + \
                            os.sep + str(num_clients) + "_clients"

    total_data_count = len(dataset_model_dictionary[dataset]().get_dataset().get_training_data()[0])
    log(INFO, f"Data total count: {total_data_count}")

    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    compute_shapley_value = 0
    load_best_trial = 0
    start_server.options(num_gpus=1/(num_clients + 1))
    start_client.options(num_gpus=1/(num_clients + 1))
    if hyperparameter_search:
        OptunaConnection.optuna_create_study(model_final_name, ["minimize"])
        for trial in range(hyperparameter_search_rounds):
            subprocesses = []
            # server_log_hs.write(f'{time.ctime()}  - Start logging \n')
            # server_log_hs.flush()
            # subprocesses.append(
            #     subprocess.Popen(['python', 'Server.py', strategy, dataset, model, str(int(num_clients)),
            #                       str(rounds), str(isUsingGPU), metrics_string, model_final_name,
            #                       str(load_best_trial), str(compute_shapley_value)],
            #                      creationflags=subprocess.CREATE_NEW_CONSOLE, stdout=server_log_hs,
            #                      stderr=server_log_hs))
            subprocesses.append(start_server.remote(strategy,
                                                    directory_of_data,
                                                    model,
                                                    int(num_clients),
                                                    rounds,
                                                    selected_metrics,
                                                    model_final_name,
                                                    load_best_trial,
                                                    compute_shapley_value,
                                                    RESULTS_PATH))
            # Wait until server is running
            if model == "mlp":
                time.sleep(10)
            elif model == "xgboost":
                time.sleep(10)

            # Create the virtual clients
            for client_number in range(0, int(num_clients)):
                subprocesses.append(
                    start_client.remote(strategy, model,
                                        client_number,
                                        directory_of_data,
                                        selected_metrics,
                                        total_data_count))
                # client_log_hs = open(f'{HYPERPARAMETER_LOG_PATH}/client_{i}_log.log', 'a')
                # client_log_hs.write(f'{time.ctime()}  - Start Client {i} logging \n')
                # client_log_hs.flush()
                # subprocesses.append(subprocess.Popen(['python', 'Client.py', strategy, dataset, model,
                #                                       str(i), directory_of_data, str(isUsingGPU), metrics_string],
                #                                      creationflags=subprocess.CREATE_NEW_CONSOLE, stdout=client_log_hs,
                #                                      stderr=client_log_hs))
                time.sleep(2)

            # [subproc.wait() for subproc in subprocesses]
            # ready_refs, remaining_refs = ray.wait(subprocesses, num_returns=len(subprocesses), timeout=None)
            _ = ray.get(subprocesses)
            # client_strategy = generate_client_strategy(strategy, dataset, model, num_clients)

    compute_shapley_value = 1
    load_best_trial = 1
    subprocesses = []
    subprocesses.append(
        start_server.remote(strategy, directory_of_data, model, int(num_clients), rounds, selected_metrics, model_final_name,
                            load_best_trial, compute_shapley_value, RESULTS_PATH))
    # Wait until server is running
    if model == "mlp":
        time.sleep(10)
    elif model == "xgboost":
        time.sleep(10)

    # Create the virtual clients
    for client_number in range(0, int(num_clients)):
        subprocesses.append(
            start_client.remote(strategy, model, client_number, directory_of_data, selected_metrics, total_data_count))
        time.sleep(2)
    # [subproc.wait() for subproc in subprocesses]
    # ready_refs, remaining_refs = ray.wait(subprocesses, num_returns=len(subprocesses), timeout=None)
    _ = ray.get(subprocesses)


    # subprocesses.append(
    #     subprocess.Popen(['python', 'Server.py', strategy, dataset, model, str(int(num_clients)),
    #                       str(rounds), str(isUsingGPU), metrics_string, model_final_name,
    #                       str(load_best_trial), str(compute_shapley_value)],
    #                      creationflags=subprocess.CREATE_NEW_CONSOLE, stdout=server_log, stderr=server_log))
    # # subprocess.Popen(['python', 'Server.py', strategy, dataset, model, str(int(num_clients)),
    # #                   str(rounds), str(isUsingGPU), metrics_string, model_final_name,
    # #                   str(load_best_trial), str(compute_shapley_value)],
    # #                  creationflags=subprocess.CREATE_NEW_CONSOLE))
    #
    # # Wait until server is running
    # if model == "mlp":
    #     time.sleep(30)
    # elif model == "xgboost":
    #     time.sleep(30)
    #
    # # Create the virtual clients
    # for i in range(0, int(num_clients)):
    #     client_log = open(f'{LOG_PATH}/client_{i}_log.log', 'a')
    #     client_log.write(f'{time.ctime()}  - Start Client {i} logging \n')
    #     client_log.flush()
    #     subprocesses.append(subprocess.Popen(['python', 'Client.py', strategy, dataset, model,
    #                                           str(i), directory_of_data, str(isUsingGPU), metrics_string],
    #                                          creationflags=subprocess.CREATE_NEW_CONSOLE, stdout=client_log,
    #                                          stderr=client_log))
    #     # subprocesses.append(subprocess.Popen(['python', 'Client.py', strategy, dataset, model,
    #     #                                       str(i), directory_of_data, str(isUsingGPU), metrics_string],
    #     #                                      creationflags=subprocess.CREATE_NEW_CONSOLE))
    #     time.sleep(2)
    #
    # [subproc.wait() for subproc in subprocesses]
