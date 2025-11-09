from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import return_dataframes_by_label_distribution, store_datasets

if __name__ == "__main__":
    clients = ["client_0", "client_1", "client_2", "client_3", "client_4", "client_5"]
    X_training, y_training = dataset_model_dictionary["har"]().get_dataset().get_training_data()
    X_test, y_test = dataset_model_dictionary["har"]().get_dataset().get_test_data()
    partition_name = "HAR_1_Maverick_1_MissingTwoLabels"

    labels = y_training.columns
    total_label_distribution_training = [len(y_training[y_training[label] == 1.0]) for label in labels]
    first_3_label_distribution_client_training = \
        [list(map(lambda x: int(x / 6), total_label_distribution_training[:-3])) for _ in clients]
    two_remaining_label_distribution_training = \
        [list(map(lambda x: int(x / 5), total_label_distribution_training[-3:-1])) for _ in clients[1:]]
    label_distribution_client_training = [client + fifth_label
                                          for client, fifth_label
                                          in zip(first_3_label_distribution_client_training[1:],
                                                 two_remaining_label_distribution_training)]
    label_distribution_client_training = [first_3_label_distribution_client_training[0]] + \
                                         label_distribution_client_training
    label_distribution_client_training[-1].append(total_label_distribution_training[-1])
    print(label_distribution_client_training)

    total_label_distribution_test = [len(y_test[y_test[label] == 1.0]) for label in labels]
    first_3_label_distribution_client_test = [list(map(lambda x: int(x / 6), total_label_distribution_test[:-3]))
                                              for _ in clients]
    two_remaining_label_distribution_test = [list(map(lambda x: int(x / 5), total_label_distribution_test[-3:-1]))
                                             for _ in clients[1:]]
    label_distribution_client_test = [client + fifth_label
                                      for client, fifth_label
                                      in zip(first_3_label_distribution_client_test[1:],
                                             two_remaining_label_distribution_test)]
    label_distribution_client_test = [first_3_label_distribution_client_test[0]] + label_distribution_client_test
    label_distribution_client_test[-1].append(total_label_distribution_test[-1])
    print(label_distribution_client_test)

    X_dataframes_training, y_dataframes_training = \
        return_dataframes_by_label_distribution(X_training,
                                                y_training,
                                                labels,
                                                label_distribution_client_training)
    X_dataframes_test, y_dataframes_test = \
        return_dataframes_by_label_distribution(X_test, y_test, labels, label_distribution_client_test)

    store_datasets(clients,
                   X_dataframes_training,
                   y_dataframes_training,
                   X_dataframes_test,
                   y_dataframes_test,
                   partition_name)
