from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import store_datasets, divide_by_categorical_feature, \
    return_dataframes_by_label_distribution

if __name__ == "__main__":
    clients = ["client_0", "client_1", "client_2", "client_3"]
    X_train, y_train = dataset_model_dictionary["covertype"]().get_dataset().get_training_data()
    X_test, y_test = dataset_model_dictionary["covertype"]().get_dataset().get_test_data()
    partition_name = "Covertype_FullBalanced"

    labels = y_train.columns
    total_label_distribution_train = [len(y_train[y_train[label] == 1.0]) for label in labels]
    label_distribution_client_train = [list(map(lambda x: int(x / 4), total_label_distribution_train[:])) for _ in
                                       clients]

    total_label_distribution_test = [len(y_test[y_test[label] == 1.0]) for label in labels]
    label_distribution_client_test = [list(map(lambda x: int(x / 6), total_label_distribution_test[:-2])) for _ in
                                      clients]

    X_dataframes_training, y_dataframes_training = \
        return_dataframes_by_label_distribution(X_train, y_train, labels, label_distribution_client_train)
    X_dataframes_test, y_dataframes_test = \
        return_dataframes_by_label_distribution(X_test, y_test, labels, label_distribution_client_test)

    clients = ["client_" + str(number) for number in range(len(X_dataframes_training))]
    store_datasets(clients,
                   X_dataframes_training,
                   y_dataframes_training,
                   X_dataframes_test,
                   y_dataframes_test,
                   partition_name)
