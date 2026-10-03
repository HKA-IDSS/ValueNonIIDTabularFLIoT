import os

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import return_dataframes_by_label_distribution, store_datasets, \
    indexes_special_class_samples, store_maverick_removed_data


def partition_edgeiiot_coreset_1_Maverick_least_class(random_state):
    clients = ["client_0", "client_1", "client_2", "client_3", "client_4"]
    X_train, y_train = dataset_model_dictionary["edge-iot-coreset"]().get_dataset().get_training_data()
    X_test, y_test = dataset_model_dictionary["edge-iot-coreset"]().get_dataset().get_test_data()
    partition_name = "edgeiot_coreset_1_Maverick_Least_Class" + os.sep + str(random_state)

    labels = y_train.columns
    total_label_distribution_train = [len(y_train[y_train[label] == 1.0]) for label in labels]
    label_distribution_client_train = [list(map(lambda x: int(x / 5), total_label_distribution_train[:6]))
                                       for _ in clients]
    #
    total_label_distribution_test = [len(y_test[y_test[label] == 1.0]) for label in labels]
    label_distribution_client_test = [list(map(lambda x: int(x / 5), total_label_distribution_test[:6]))
                                      for _ in clients]

    for client_number in range(4):
        label_distribution_client_train[client_number].insert(6, 0)
        label_distribution_client_test[client_number].insert(6, 0)
        label_distribution_client_train[client_number] += list(map(lambda x: int(x / 5),
                                                                   total_label_distribution_train[7:]))
        label_distribution_client_test[client_number] += list(map(lambda x: int(x / 5),
                                                                  total_label_distribution_test[7:]))

    label_distribution_client_train[-1].insert(6, total_label_distribution_train[6])
    label_distribution_client_test[-1].insert(6, total_label_distribution_test[6])
    label_distribution_client_train[-1] += list(map(lambda x: int(x / 5), total_label_distribution_train[7:]))
    label_distribution_client_test[-1] += list(map(lambda x: int(x / 5), total_label_distribution_test[7:]))

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

    X_non_mav_train = X_dataframes_train[-1]
    y_non_mav_train = y_dataframes_train[-1]
    X_non_mav_test = X_dataframes_test[-1]
    y_non_mav_test = y_dataframes_test[-1]

    indexes_to_remove_train, indexes_to_remove_test = indexes_special_class_samples(y_non_mav_train,
                                                                                    y_non_mav_test,
                                                                                    "MITM")
    store_maverick_removed_data(X_non_mav_train,
                                y_non_mav_train,
                                X_non_mav_test,
                                y_non_mav_test,
                                partition_name,
                                4,
                                indexes_to_remove_train,
                                indexes_to_remove_test)
