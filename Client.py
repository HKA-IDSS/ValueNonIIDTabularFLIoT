import os
from sys import argv

import ray
import flwr
import pandas as pd
import tensorflow as tf

# from SyntheticDataGenerator import sample_synthetic_unlabeled_data
from experiment_parameters.TrainerFactory import strategies_dictionary

"""
Look for similar comments on the Server.py file.
"""

pd.set_option('display.float_format', lambda x: '%.15f' % x)


def generate_client_strategy(strategy_selected,
                             model_selected,
                             client_number,
                             route_to_dataset,
                             metric_list,
                             total_data_count):
    # Here, the dataset factory is only used for retrieving the model.
    # dataset_factory = dataset_model_dictionary[dataset_selected]
    # dataset = dataset_factory.get_dataset()

    # TODO: If logits are true, than crash
    # Function call stack:
    # train_function -> assert_greater_equal_Assert_AssertGuard_false_811
    # -> train_function -> assert_greater_equal_Assert_AssertGuard_false_811

    logits = False
    if strategy_selected in ["FedKD", "FedDKD"]:
        logits = True

    # The training data is retrieved by using the name of the dataset.
    working_directory = os.getcwd()
    directory_for_training_data = working_directory + os.sep + "data" + os.sep + "partitioned_training_data"
    path_to_train_datasets = directory_for_training_data + route_to_dataset

    X_train = pd.read_csv(path_to_train_datasets + os.sep + "client_" + str(client_number) + "_X_training.csv", index_col=0)
    y_train = pd.read_csv(path_to_train_datasets + os.sep + "client_" + str(client_number) + "_y_training.csv", index_col=0)
    X_test = pd.read_csv(path_to_train_datasets + os.sep + "client_" + str(client_number) + "_X_test.csv", index_col=0)
    y_test = pd.read_csv(path_to_train_datasets + os.sep + "client_" + str(client_number) + "_y_test.csv", index_col=0)
    X_train.sort_index(inplace=True)
    y_train.sort_index(inplace=True)
    X_test.sort_index(inplace=True)
    y_test.sort_index(inplace=True)

    # Again, the factory is used to retrieve the code, in this case of the client.
    # An example of the code can be found on strategy\client\FedAvgClient.py

    # Same here, instantiate only the class of the interested strategy.
    client_strategy_type = strategies_dictionary[strategy_selected]().create_client()
    client_strategy = client_strategy_type(model_selected,
                                           X_train,
                                           X_test,
                                           y_train,
                                           y_test,
                                           client_number,
                                           metric_list,
                                           total_data_count)

    if model_selected == "mlp":
        client_strategy = client_strategy.to_client()

    return client_strategy


#@ray.remote
@ray.remote(num_gpus=0.1)
def start_client(strategy_selected,
                 model_selected,
                 client_number,
                 route_to_dataset,
                 metric_list,
                 total_data_count):
    print(f"Ray GPUS: {ray.get_gpu_ids()}")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
    client_strategy = generate_client_strategy(strategy_selected,
                                               model_selected,
                                               client_number,
                                               route_to_dataset,
                                               metric_list,
                                               total_data_count)
    # Start Flower client
    flwr.client.start_client(server_address="localhost:54080", client=client_strategy)

    return None


if __name__ == "__main__":
    strategy_selected = argv[1]
    model_selected = argv[2]
    client_number = argv[3]
    route_to_dataset = argv[4]
    metric_list = argv[6].split("-")

    # if bool(argv[5]):
    #     gpus = tf.config.experimental.list_physical_devices('GPU')
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)

    ray.get(start_client.remote(strategy_selected,
                                model_selected,
                                client_number,
                                route_to_dataset,
                                metric_list))
