import math

import numpy as np

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import return_dataframes_by_label_distribution, store_datasets

if __name__ == "__main__":
    random_state = 5
    clients = ["client_0", "client_1", "client_2", "client_3", "client_4", "client_5"]
    X_train, y_train = dataset_model_dictionary["har"]().get_dataset().get_training_data()
    X_test, y_test = dataset_model_dictionary["har"]().get_dataset().get_test_data()
    partition_name = "HAR_1_Maverick_1_HellingerTrap"

    labels = y_train.columns
    total_label_distribution_train = [len(y_train[y_train[label] == 1.0]) for label in labels]
    label_distribution_client_train = [list(map(lambda x: int(x / 7), total_label_distribution_train[:-1]))
                                       for _ in clients]
    label_distribution_client_train[-1].append(int(total_label_distribution_train[-1] / 7))

    total_label_distribution_test = [len(y_test[y_test[label] == 1.0]) for label in labels]
    label_distribution_client_test = [list(map(lambda x: int(x / 7), total_label_distribution_test[:-1]))
                                      for _ in clients]
    label_distribution_client_test[-1].append(int(total_label_distribution_test[-1] / 7))

    for client_number in range(0, 5):
        label_distribution_client_train[client_number].append(0)
        label_distribution_client_test[client_number].append(0)

    total_samples_train_non_mavericks = sum(label_distribution_client_train[1])
    total_samples_test_non_mavericks = sum(label_distribution_client_test[1])

    label_distribution_client_train[-1] = np.subtract(label_distribution_client_train[-1],
                                                      np.array(math.trunc(total_samples_train_non_mavericks / 42),
                                                               ndmin=len(label_distribution_client_train[0]))).reshape(
        -1)
    label_distribution_client_test[-1] = np.subtract(label_distribution_client_test[-1],
                                                     np.array(math.trunc(total_samples_test_non_mavericks / 42),
                                                              ndmin=len(label_distribution_client_test[0]))).reshape(-1)

    dirichlet = np.random.dirichlet(np.ones(5) * 1000, size=1)
    train_label_distribution = np.multiply(dirichlet, sum(label_distribution_client_train[-1][:-1]))
    test_label_distribution = np.multiply(dirichlet, sum(label_distribution_client_test[-1][:-1]))
    label_distribution_client_train[-1][:-1] = train_label_distribution
    label_distribution_client_test[-1][:-1] = test_label_distribution

    X_dataframes_train, y_dataframes_train = \
        return_dataframes_by_label_distribution(X_train, y_train, labels, label_distribution_client_train, random_state)
    X_dataframes_test, y_dataframes_test = \
        return_dataframes_by_label_distribution(X_test, y_test, labels, label_distribution_client_test, random_state)

    store_datasets(clients,
                   X_dataframes_train,
                   y_dataframes_train,
                   X_dataframes_test,
                   y_dataframes_test,
                   partition_name)
