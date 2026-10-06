import os

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import return_dataframes_by_label_distribution, store_datasets, \
    sample_directory


# def maverick_wine(random_state):
if __name__ == "__main__":
    random_state = 1
    clients = ["client_0", "client_1", "client_2"]
    X_train, y_train = dataset_model_dictionary["wine"]().get_dataset().get_training_data()
    X_test, y_test = dataset_model_dictionary["wine"]().get_dataset().get_test_data()
    partition_name = "Wine_Maverick" + os.sep + str(random_state)

    labels = y_train.columns
    total_label_distribution_train = [len(y_train[y_train[label] == 1.0]) for label in labels]
    label_distribution_client_train = [list(map(lambda x: int(x / 3), total_label_distribution_train[:-1])) for _ in clients]
    label_distribution_client_train[-1].append(total_label_distribution_train[-1])

    total_label_distribution_test = [len(y_test[y_test[label] == 1.0]) for label in labels]
    label_distribution_client_test = [list(map(lambda x: int(x / 3), total_label_distribution_test[:-1])) for _ in clients]
    label_distribution_client_test[-1].append(total_label_distribution_test[-1])

    X_dataframes_train, y_dataframes_train = return_dataframes_by_label_distribution(X_train,
                                                                                     y_train,
                                                                                     labels,
                                                                                     label_distribution_client_train,
                                                                                     random_state)
    X_dataframes_test, y_dataframes_test = return_dataframes_by_label_distribution(X_test,
                                                                                   y_test,
                                                                                   labels,
                                                                                   label_distribution_client_test,
                                                                                   random_state)

    store_datasets(clients,
                   X_dataframes_train,
                   y_dataframes_train,
                   X_dataframes_test,
                   y_dataframes_test,
                   partition_name)

    y_dataframes_train_non_mav = y_dataframes_train[-1]
    X_dataframes_train_non_mav = X_dataframes_train[-1]
    y_dataframes_test_non_mav = y_dataframes_test[-1]
    X_dataframes_test_non_mav = X_dataframes_test[-1]

    train_indexes_to_remove = list(y_dataframes_train_non_mav[y_dataframes_train_non_mav["x0_2"] == 1].index)
    test_indexes_to_remove = list(y_dataframes_test_non_mav[y_dataframes_test_non_mav["x0_2"] == 1].index)
    X_dataframes_train_non_mav.drop(train_indexes_to_remove, axis=0, inplace=True)
    X_dataframes_test_non_mav.drop(test_indexes_to_remove, axis=0, inplace=True)
    y_dataframes_train_non_mav.drop(train_indexes_to_remove, axis=0, inplace=True)
    y_dataframes_test_non_mav.drop(test_indexes_to_remove, axis=0, inplace=True)
    final_directory = sample_directory + os.sep + partition_name
    X_dataframes_train_non_mav.to_csv(final_directory + os.sep + "client_2_mav_data_excluded_X_training.csv")
    X_dataframes_test_non_mav.to_csv(final_directory + os.sep + "client_2_mav_data_excluded_X_test.csv")
    y_dataframes_train_non_mav.to_csv(final_directory + os.sep + "client_2_mav_data_excluded_y_training.csv")
    y_dataframes_test_non_mav.to_csv(final_directory + os.sep + "client_2_mav_data_excluded_y_test.csv")


