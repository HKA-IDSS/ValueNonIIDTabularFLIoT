import os

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import return_dataframes_by_label_distribution, store_datasets, \
    indexes_special_class_samples, store_maverick_removed_data


def partition_har_1_maverick_walkingupstairs(random_state):
    clients = ["client_0", "client_1", "client_2", "client_3", "client_4", "client_5"]
    X_train, y_train = dataset_model_dictionary["har"]().get_dataset().get_training_data()
    X_test, y_test = dataset_model_dictionary["har"]().get_dataset().get_test_data()
    partition_name = "HAR_1_Maverick_WalkingUpstairs" + os.sep + str(random_state)

    labels = y_train.columns
    total_label_distribution_train = [len(y_train[y_train[label] == 1.0]) for label in labels]
    label_distribution_client_train = [list(map(lambda x: int(x / 6), total_label_distribution_train[:-1]))
                                       for _ in clients]
    label_distribution_client_train[-1].append(total_label_distribution_train[-1])

    total_label_distribution_test = [len(y_test[y_test[label] == 1.0]) for label in labels]
    label_distribution_client_test = [list(map(lambda x: int(x / 6), total_label_distribution_test[:-1]))
                                      for _ in clients]
    label_distribution_client_test[-1].append(total_label_distribution_test[-1])

    X_dataframes_train, y_dataframes_train = \
        return_dataframes_by_label_distribution(X_train, y_train, labels, label_distribution_client_train)
    X_dataframes_test, y_dataframes_test = \
        return_dataframes_by_label_distribution(X_test, y_test, labels, label_distribution_client_test)

    store_datasets(clients,
                   X_dataframes_train,
                   y_dataframes_train,
                   X_dataframes_test,
                   y_dataframes_test,
                   partition_name)

    X_dataframe_removed_maverick_train = X_dataframes_train[5]
    X_dataframe_removed_maverick_test = X_dataframes_test[5]
    y_dataframe_removed_maverick_train = y_dataframes_train[5]
    y_dataframe_removed_maverick_test = y_dataframes_test[5]

    indexes_to_remove_train, indexes_to_remove_test = indexes_special_class_samples(y_dataframe_removed_maverick_train,
                                                                                    y_dataframe_removed_maverick_test,
                                                                                    "WALKING_UPSTAIRS")
    store_maverick_removed_data(X_dataframe_removed_maverick_train,
                                y_dataframe_removed_maverick_train,
                                X_dataframe_removed_maverick_test,
                                y_dataframe_removed_maverick_test,
                                partition_name,
                                5,
                                indexes_to_remove_train,
                                indexes_to_remove_test)
