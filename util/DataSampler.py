import os
from array import array
from logging import INFO

import numpy as np
import pandas as pd
from flwr.common import log
# Get working directory to later on save the datasets.
from sklearn.cluster import MiniBatchKMeans

from Definitions import ROOT_DIR
# working_directory = os.getcwd()
from experiment_parameters.TrainerFactory import dataset_model_dictionary

directory_for_data = ROOT_DIR + os.sep + "data" + os.sep + "partitioned_training_data"


def datasets_division(X_train, y_train, X_test, y_test, percentage_data_participant, alpha, seed, labels):
    """
    This function takes divides the complete dataset in smaller datasets.
    It does that following the former dirichlet distribution.

    Args:
        X_train (dataframe): X part of the dataframe.
        y_train (dataframe): y part of the dataframe. In this case, is the one-hot-encoded version.
        percentage_data_participant (array[integers]): Percentage of data per client.
        alpha (float): Alpha parameter for the dirichlet function. Higher than 0
        seed (int): Seed in the case that it is needed to control the randomness of the dirichlet function.

    Returns:
        list[Tuple[DataFrame, DataFrame]]: The datasets per client, both for the X and y.
    """
    label_distribution_by_client_training, label_distribution_by_client_testing = \
        dirichlet_division(y_train, y_test, percentage_data_participant, alpha, seed)
    log(INFO, f"Label_distribution_by_client_training: {label_distribution_by_client_training}")
    log(INFO, f"label_distribution_by_client_testing: {label_distribution_by_client_testing}")
    training_dataset_per_client = []
    test_dataset_per_client = []

    # Saving the distribution as an image.
    # images_directory = working_directory + "\\images"
    # plot_class_distribution(label_distribution_by_client, images_directory)

    for client in label_distribution_by_client_training:
        X_train_dataframe_to_be_returned = pd.DataFrame()
        y_train_dataframe_to_be_returned = pd.DataFrame()
        for label, position_of_label in zip(labels, range(len(labels))):
            filter_by_label = y_train[y_train[label] == 1.0]
            filter_X_by_label = X_train[X_train.index.isin(filter_by_label.index)]
            X_samples = filter_X_by_label.sample(int(client[position_of_label]), random_state=1)
            y_samples = y_train[y_train.index.isin(X_samples.index)]
            X_train_dataframe_to_be_returned = pd.concat([X_train_dataframe_to_be_returned, X_samples])
            y_train_dataframe_to_be_returned = pd.concat([y_train_dataframe_to_be_returned, y_samples])
            X_train.drop(X_samples.index.values, axis=0, inplace=True)
            y_train.drop(X_samples.index.values, axis=0, inplace=True)

        training_dataset_per_client.append((X_train_dataframe_to_be_returned, y_train_dataframe_to_be_returned))

    for client in label_distribution_by_client_testing:
        X_test_dataframe_to_be_returned = pd.DataFrame()
        y_test_dataframe_to_be_returned = pd.DataFrame()
        for label, position_of_label in zip(labels, range(len(labels))):
            filter_by_label = y_test[y_test[label] == 1.0]
            filter_X_by_label = X_test[X_test.index.isin(filter_by_label.index)]
            log(INFO, f"Filter_X_by_label: {filter_X_by_label}")
            log(INFO, f"Client_position_of_label: {int(client[position_of_label])}")
            X_samples = filter_X_by_label.sample(min(len(filter_X_by_label), int(client[position_of_label])),
                                                 random_state=1)
            y_samples = y_test[y_test.index.isin(X_samples.index)]
            X_test_dataframe_to_be_returned = pd.concat([X_test_dataframe_to_be_returned, X_samples])
            y_test_dataframe_to_be_returned = pd.concat([y_test_dataframe_to_be_returned, y_samples])
            X_test.drop(X_samples.index.values, axis=0, inplace=True)
            y_test.drop(X_samples.index.values, axis=0, inplace=True)

        test_dataset_per_client.append((X_test_dataframe_to_be_returned, y_test_dataframe_to_be_returned))

    return training_dataset_per_client, test_dataset_per_client


"""
I am one-hot-encoding the labels, as I will need them this way for the attack. In the case you want to remove the encoding,
you need to make a couple of changes in the function "obtain_label_division".
"""


def dirichlet_division(y_train, y_test, percentage_data_participant, alpha, seed):
    """
    This is the main function that carries out the division of labels. I will also add some comments throughout the
    function, with the hope that it helps with the understanding of it.

    Basically, the idea is to divide the samples based on two parameters:
        percentage_data_participant: Out of the total data, what is the percentage that one participant will get.
        alpha: Parameter for the dirichlet function. The lower that Alpha is, the higher the non-i.i.d settings get.

    Args:
        y (dataframe): A dataset with the one hot encoded labels.
        percentage_data_participant (array[int]): The percentage of data that each participant gets.
        alpha (float): Alpha parameter for the dirichlet function. It must be higher than 0, but it can be close to it.
        seed (int): Optional argument. In the case that it is needed for controlling the randomness
            of the dirichlet function.

    Returns: list[list[int]]: It contains an array per participant, with the number of labels of each type. Example:
    [[15,13,16], [14,17,15]]
    """
    np.random.seed(seed)  # Remove comment if you want to control the randomness of the dirichlet function.

    # With this function, I extract the amount of labels per class. This is reflected in the variable counts.
    labels_train, counts_train = np.unique(np.argmax(np.asarray(y_train), axis=1), return_counts=True)
    labels_test, counts_test = np.unique(np.argmax(np.asarray(y_test), axis=1), return_counts=True)
    remaining_instances_training = counts_train
    remaining_instances_test = counts_test

    classes_percentages_training = counts_train / np.sum(counts_train)
    classes_percentages_test = counts_test / np.sum(counts_test)

    # I think this part is self explanatory.
    percentage_per_client = np.asarray(percentage_data_participant)
    classes_for_client_training = np.floor((percentage_per_client / 100) * sum(remaining_instances_training))
    classes_for_client_test = np.floor((percentage_per_client / 100) * sum(remaining_instances_test))
    distribution_of_classes_training = []
    distribution_of_classes_test = []

    # The variable alpha is passed into an array. This is done in order to feed it into the dirichlet function.
    # The justification is just above the dirichlet function.
    dirichlet_alpha_array = np.multiply(np.ones(len(labels_train)), alpha)
    # probabilities_class_test = np.multiply(np.ones(len(labels_test)), alpha)

    # For that gives each client the number of samples corresponding to the percentage introduced by the user.
    for classes_client, classes_test_client in zip(classes_for_client_training, classes_for_client_test):
        classes_assigned_training = np.zeros(len(labels_train), dtype=int)
        classes_assigned_test = np.zeros(len(labels_test), dtype=int)

        # The dirichlet function takes two arguments: first an array with the probability distribution and
        # then the size. The array is the variable probabilities_class, just before the for instruction. The
        # size in this case would be the number of participants. Nevertheless, because the function may give
        # the three participants all the labels for only one class, the function is written by only accepting
        # one participant.
        dirichlet_distribution_labels = \
            np.random.dirichlet(np.multiply(classes_percentages_training, dirichlet_alpha_array), 1)[0]
        while (remaining_instances_training > 0).any() and sum(classes_assigned_training) < classes_client:
            classes_to_assign_training = classes_client - sum(classes_assigned_training)
            classes_to_assign_test = classes_test_client - sum(classes_assigned_test)

            if classes_to_assign_training < 0:
                classes_to_assign_training = 0

            if classes_to_assign_test < 0:
                classes_to_assign_test = 0

            # Here, only the integer part is obtained. Also, reshape to put it into a one dimensional array
            instances_assigned_per_class_training = np.ceil(
                np.multiply(dirichlet_distribution_labels, classes_to_assign_training).reshape(-1).astype(int))
            instances_assigned_per_class_test = np.ceil(
                np.multiply(dirichlet_distribution_labels, classes_to_assign_test).reshape(-1).astype(int))

            # print("Instances assigned per class: {}".format(instances_assigned_per_class)) Then, the instances
            # are assigned to the participants. To make sure that none of the values is higher than the available
            # classes, min is used.
            actual_instances_assigned_training = [min(instance_assigned, remaining_instance)
                                                  for instance_assigned, remaining_instance
                                                  in zip(instances_assigned_per_class_training,
                                                         remaining_instances_training)]
            actual_instances_assigned_test = [min(instance_assigned, remaining_instance)
                                              for instance_assigned, remaining_instance
                                              in zip(instances_assigned_per_class_test,
                                                     remaining_instances_test)]

            if all([int(actual_instance) == 0 for actual_instance in actual_instances_assigned_training]):
                dirichlet_distribution_labels = \
                    np.random.dirichlet(np.multiply(classes_percentages_training, dirichlet_alpha_array), 1)[0]
                instances_assigned_per_class_training = [1 if label > 0 else 0 for label in
                                                         dirichlet_distribution_labels]
                actual_instances_assigned_training = [min(instance_assigned, remaining_instance)
                                                      for instance_assigned, remaining_instance
                                                      in zip(instances_assigned_per_class_training,
                                                             remaining_instances_training)]

            if all([int(actual_instance) == 0 for actual_instance in actual_instances_assigned_test]):
                dirichlet_distribution_labels = \
                    np.random.dirichlet(np.multiply(classes_percentages_test, dirichlet_alpha_array), 1)[0]
                instances_assigned_per_class_test = [1 if label > 0 else 0 for label in dirichlet_distribution_labels]
                actual_instances_assigned_test = [min(instance_assigned, remaining_instance)
                                                  for instance_assigned, remaining_instance
                                                  in zip(instances_assigned_per_class_test,
                                                         remaining_instances_test)]

            # print("Actual instances assigned per class: {}".format(actual_instances_assigned))
            # Having to do the cast to int, because of problems with types
            actual_instances_assigned_training = [int(x) for x in
                                                  actual_instances_assigned_training]

            classes_assigned_training += actual_instances_assigned_training

            # print("Total instances assigned per class: {}".format(classes_assigned))
            remaining_instances_training -= actual_instances_assigned_training

            # Moving all this code here, as it depends on the if.
            actual_instances_assigned_test = [int(x) for x in
                                              actual_instances_assigned_test]
            classes_assigned_test += actual_instances_assigned_test
            remaining_instances_test -= actual_instances_assigned_test

            # print("Remaining classes per class: {}".format(remaining_instances))

        distribution_of_classes_training.append(classes_assigned_training)
        distribution_of_classes_test.append(classes_assigned_test)

    return distribution_of_classes_training, distribution_of_classes_test


def prepare_datasets(name_dataset, percentage_data_participant, alpha, seed):
    """
    This function saves the dataframes in a subdirectory following this format:
        working_directory\\data\\dirichlet\\dataset_(name_of_dataset)\\alpha_(value_of_alpha)
            \\client_(number_of_client)_(X or y).csv

    Args:
        name_dataset:
        X (dataframe): X part of the dataframe.
        y (dataframe): y part of the dataframe. In this case, is the one-hot-encoded version.
        percentage_data_participant (array[integers]): Percentage of data per client.
        alpha (float): Alpha parameter for the dirichlet function. Higher than 0
        seed (int): Seed in the case that it is needed to control the randomness of the dirichlet function.
    """
    num_participants = len(percentage_data_participant)
    X_train, y_train = dataset_model_dictionary[name_dataset]().get_dataset().get_training_data()
    X_test, y_test = dataset_model_dictionary[name_dataset]().get_dataset().get_test_data()
    labels = dataset_model_dictionary[name_dataset]().get_dataset().get_labels()
    dataset_per_client_training, dataset_per_client_testing = datasets_division(X_train, y_train, X_test, y_test,
                                                                                percentage_data_participant, alpha,
                                                                                seed, labels)
    # dataset_per_client_testing = datasets_division(X_test, y_test, percentage_data_participant, alpha, seed, labels)

    if alpha >= 1:
        alpha = int(alpha)

    final_directory = (directory_for_data +
                       os.sep + "dirichlet" +
                       os.sep + "dataset_" + name_dataset +
                       os.sep + "alpha_" + str(alpha))

    os.makedirs(final_directory, exist_ok=True)

    for participant_number in range(num_participants):
        X_dataset, y_dataset = dataset_per_client_training[participant_number]
        X_dataset.to_csv(final_directory + os.sep + "client_" + str(participant_number) + "_X_training.csv")
        y_dataset.to_csv(final_directory + os.sep + "client_" + str(participant_number) + "_y_training.csv")
        X_test_dataset, y_test_dataset = dataset_per_client_testing[participant_number]
        X_test_dataset.to_csv(final_directory + os.sep + "client_" + str(participant_number) + "_X_test.csv")
        y_test_dataset.to_csv(final_directory + os.sep + "client_" + str(participant_number) + "_y_test.csv")


def sample_data_dirichlet(name_dataset, percentages_data_clients, alpha, seed):
    if not os.path.exists(directory_for_data +
                          os.sep + "dirichlet" +
                          os.sep + "dataset_" + name_dataset +
                          os.sep + "alpha_" + str(alpha)):
        prepare_datasets(name_dataset, percentages_data_clients, alpha, seed)


""" Feature Skew Functions """


def divide_by_clustering(X_dataset, y_dataset, n_clusters):
    labels = list(y_dataset.columns)
    X_datasets, y_datasets = [pd.DataFrame() for _ in range(n_clusters)], [pd.DataFrame() for _ in range(n_clusters)]
    for i in range(len(labels)):
        y_train_per_label = y_dataset[y_dataset[labels[i]] == 1]
        X_train_per_label = X_dataset.iloc[list(y_train_per_label.index.values)]
        clusters = MiniBatchKMeans(n_clusters=n_clusters).fit_predict(X_train_per_label)
        cluster_labels = np.unique(clusters)
        for cluster_label in cluster_labels:
            list_of_indexes = np.where(clusters == cluster_label)[0].tolist()
            partial_cluster_X_dataframe = X_train_per_label.iloc[list_of_indexes]
            partial_cluster_y_dataframe = y_train_per_label.iloc[list_of_indexes]
            X_datasets[cluster_label] = pd.concat([X_datasets[cluster_label], partial_cluster_X_dataframe])
            y_datasets[cluster_label] = pd.concat([y_datasets[cluster_label], partial_cluster_y_dataframe])

    return X_datasets, y_datasets


def sample_data_feature_skew_clustering(name_dataset, n_clients_and_clusters):
    if not os.path.exists(directory_for_data +
                          os.sep + "feature_skew_clustering" +
                          os.sep + "dataset_" + name_dataset +
                          os.sep + str(n_clients_and_clusters) + "_clients"):
        X_train, y_train = dataset_model_dictionary[name_dataset]().get_dataset().get_training_data()
        X_test, y_test = dataset_model_dictionary[name_dataset]().get_dataset().get_test_data()
        X_train.sort_index(inplace=True)
        y_train.sort_index(inplace=True)
        X_test.sort_index(inplace=True)
        y_test.sort_index(inplace=True)
        X_train.reset_index(inplace=True, drop=True)
        y_train.reset_index(inplace=True, drop=True)
        X_test.reset_index(inplace=True, drop=True)
        y_test.reset_index(inplace=True, drop=True)
        X_train_dataframes, y_train_dataframes = divide_by_clustering(X_train, y_train, n_clients_and_clusters)
        X_test_dataframes, y_test_dataframes = divide_by_clustering(X_test, y_test, n_clients_and_clusters)
        final_directory = directory_for_data + \
                          os.sep + "feature_skew_clustering" + \
                          os.sep + "dataset_" + name_dataset + \
                          os.sep + str(n_clients_and_clusters) + "_clients"

        os.makedirs(final_directory, exist_ok=True)

        for number, X_train_dataframe, y_train_dataframe, X_test_dataframe, y_test_dataframe \
                in zip(range(n_clients_and_clusters),
                       X_train_dataframes,
                       y_train_dataframes,
                       X_test_dataframes,
                       y_test_dataframes):
            X_train_dataframe.sort_index(inplace=True)
            y_train_dataframe.sort_index(inplace=True)
            X_test_dataframe.sort_index(inplace=True)
            y_test_dataframe.sort_index(inplace=True)
            X_train_dataframe.to_csv(final_directory + os.sep + "client_" + str(number) + "_X_training.csv")
            y_train_dataframe.to_csv(final_directory + os.sep + "client_" + str(number) + "_y_training.csv")
            X_test_dataframe.to_csv(final_directory + os.sep + "client_" + str(number) + "_X_test.csv")
            y_test_dataframe.to_csv(final_directory + os.sep + "client_" + str(number) + "_y_test.csv")


# def remove_data(name_dataset, alpha):
#     shutil.rmtree(directory_for_data + "\\dataset_" + name_dataset + "\\alpha_" + str(alpha))


if __name__ == "__main__":
    dataset_factory = dataset_model_dictionary["edge-iot-coreset"]()
    X_train, y_train = dataset_factory.get_dataset().get_training_data()
    X_test, y_test = dataset_factory.get_dataset().get_test_data()

    print(datasets_division(X_train, y_train, X_test, y_test, [20, 20, 20, 20, 20], 100, 1, y_train.columns))
