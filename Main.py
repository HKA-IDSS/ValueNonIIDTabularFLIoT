import os
import subprocess
import sys
import time

import tensorflow as tf

from Definitions import ROOT_DIR
from util import OptunaConnection
from util.DataSampler import sample_data_dirichlet, sample_data_feature_skew_clustering
from util.Util import arg_parser

if __name__ == "__main__":

    # Right now, the only parameter is a yaml with all the parameters.
    # All the possible parameters are on experiment_configuration\CompleteExperiment.yaml.
    filename = sys.argv[1]
    experiment_config = arg_parser("experiment_configuration" + os.sep + filename + ".yaml")
    LOG_PATH = "results/logs"
    HYPERPARAMETER_LOG_PATH = "results/hyperparameter/logs"

    # TODO: It would be a good idea if in a file can be set all the combinations for different configs.
    strategy = experiment_config["strategy"]
    dataset = experiment_config["dataset"]
    model = experiment_config["model"]

    partition = experiment_config["partition"]

    # batch_size = experiment_config["batch_size"]
    rounds = experiment_config["rounds"]

    selected_metrics = experiment_config["metrics"]
    hyperparameter_search = experiment_config["hyperparameter_search"]
    hyperparameter_search_rounds = experiment_config["hyperparameter_search_rounds"]

    # I would like to propose a new combination for the log_name: See below
    # if log_name == "default":
    #     log_name = f"{dataset_name}_{aggregation_type}_{client_split}_{alpha}"
    # else:
    #     log_name = f"{dataset_name}_{aggregation_type}_{client_split}_{alpha}_{log_name}"

    # Despite it can be repeated, I am not that interested at the moment in researching with quantity skew. So, for the
    # moment, I think we can stick with equally dividing the amount of data among the participants.
    # log_name = f"{num_clients}"

    # TODO: The original point here is good. Maybe Florian can update the current process to remove the split of data
    #   once the training is done. I think this way, the training can be parallelized by models, while maintaining the
    #   the number of clients. So far, I leave this code here.
    # Remind to clean up previous datasets
    # Dont remove them automatically, because csv might be saved for documentation
    # check_path = f"data\\partitioned_training_data\\dirichlet\\dataset_" + dataset + "\\alpha_{alpha}"
    # if os.path.isdir(check_path):
    #     if (len(os.listdir(check_path)) / 2) != num_clients:
    #         print(
    #             f"Previous dataset was for {int(len(os.listdir(check_path)) / 2)} clients,"
    #             f" but current split is for {num_clients} clients.")
    #         print(f"Please remove path: {check_path} and start again")
    #         exit()

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

    # Check if log dir exists and create it if needed
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH, exist_ok=True)

    if not os.path.exists(HYPERPARAMETER_LOG_PATH):
        os.makedirs(HYPERPARAMETER_LOG_PATH, exist_ok=True)

    metrics_string = ""
    if selected_metrics is not None:
        for metric in selected_metrics:
            metrics_string += metric + "-"
    metrics_string = metrics_string[:-1]

    subprocesses = []

    # Start Flower server
    server_log = open(f'{LOG_PATH}/server_log.log', 'a')
    server_log.write(f'{time.ctime()}  - Start logging \n')
    server_log.flush()

    # Start Flower server
    server_log_hs = open(f'{HYPERPARAMETER_LOG_PATH}/server_log.log', 'a')
    server_log_hs.write(f'{time.ctime()}  - Start logging \n')
    server_log_hs.flush()
    # TODO: Fabian: Maybe, you need to add here the gpu_mem_limit.
    isUsingGPU = False
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        isUsingGPU = True

    if partition == "dirichlet":
        directory_of_data = os.sep + "dirichlet" + os.sep + "dataset_" + dataset + os.sep + "alpha_" + str(alpha)

    elif partition == "manual":
        # Create the virtual clients
        directory_of_data = os.sep + "manual" + os.sep + name

    elif partition == "feature_skew":
        directory_of_data = os.sep + "feature_skew_clustering" + \
                            os.sep + "dataset_" + dataset + \
                            os.sep + str(num_clients) + "_clients"

    compute_shapley_value = 0
    load_best_trial = 0
    if hyperparameter_search:
        OptunaConnection.optuna_create_study(model_final_name, ["minimize", "maximize"])
        for trial in range(hyperparameter_search_rounds):
            server_log_hs.write(f'{time.ctime()}  - Start logging \n')
            server_log_hs.flush()
            subprocesses.append(
                subprocess.Popen(['python', 'Server.py', strategy, dataset, model, str(int(num_clients)),
                                  str(rounds), metrics_string, model_final_name, str(load_best_trial),
                                  str(compute_shapley_value), str(isUsingGPU)]))

            # Wait until server is running
            if model == "mlp":
                time.sleep(10)
            elif model == "xgboost":
                time.sleep(30)

            # Create the virtual clients
            for i in range(0, int(num_clients)):
                client_log_hs = open(f'{HYPERPARAMETER_LOG_PATH}/client_{i}_log.log', 'a')
                client_log_hs.write(f'{time.ctime()}  - Start Client {i} logging \n')
                client_log_hs.flush()
                subprocesses.append(subprocess.Popen(['python', 'Client.py', strategy, model,
                                                      str(i), directory_of_data, str(isUsingGPU), metrics_string]))
                time.sleep(2)

            [subproc.wait() for subproc in subprocesses]

    compute_shapley_value = 1
    load_best_trial = 1
    subprocesses.append(
        subprocess.Popen(['python', 'Server.py', strategy, dataset, model, str(int(num_clients)),
                                  str(rounds), metrics_string, model_final_name, str(load_best_trial),
                                  str(compute_shapley_value), str(isUsingGPU)]))
        # subprocess.Popen(['python', 'Server.py', strategy, dataset, model, str(int(num_clients)),
        #                   str(rounds), str(isUsingGPU), metrics_string, model_final_name,
        #                   str(load_best_trial), str(compute_shapley_value)],
        #                  creationflags=subprocess.CREATE_NEW_CONSOLE))

    # Wait until server is running
    if model == "mlp":
        time.sleep(10)
    elif model == "xgboost":
        time.sleep(30)

    # Create the virtual clients
    for i in range(0, int(num_clients)):
        client_log = open(f'{LOG_PATH}/client_{i}_log.log', 'a')
        client_log.write(f'{time.ctime()}  - Start Client {i} logging \n')
        client_log.flush()
        subprocesses.append(subprocess.Popen(['python', 'Client.py', strategy, model,
                                              str(i), directory_of_data, str(isUsingGPU), metrics_string]))
        # subprocesses.append(subprocess.Popen(['python', 'Client.py', strategy, dataset, model,
        #                                       str(i), directory_of_data, str(isUsingGPU), metrics_string],
        #                                      creationflags=subprocess.CREATE_NEW_CONSOLE))
        time.sleep(2)

    [subproc.wait() for subproc in subprocesses]
