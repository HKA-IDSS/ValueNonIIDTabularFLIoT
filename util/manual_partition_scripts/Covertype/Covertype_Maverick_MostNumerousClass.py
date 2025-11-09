from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import return_dataframes_by_label_distribution, store_datasets

if __name__ == "__main__":
    clients = ["client_0", "client_1", "client_2", "client_3"]
    X_train, y_train = dataset_model_dictionary["covertype"]().get_dataset().get_training_data()
    X_test, y_test = dataset_model_dictionary["covertype"]().get_dataset().get_test_data()
    partition_name = "Covertype_Maverick_MostNumerousClass"

    labels = y_train.columns
    total_label_distribution_train = [len(y_train[y_train[label] == 1.0]) for label in labels]
    label_distribution_client_train = [list(map(lambda x: int(x / 4), total_label_distribution_train[:-1]))
                                       for _ in clients]
    label_distribution_client_train[-1].append(total_label_distribution_train[-1])

    total_label_distribution_test = [len(y_test[y_test[label] == 1.0]) for label in labels]
    label_distribution_client_test = [list(map(lambda x: int(x / 4), total_label_distribution_test[:-1]))
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
