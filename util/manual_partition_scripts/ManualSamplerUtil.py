import os

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from Definitions import ROOT_DIR

sample_directory = ROOT_DIR + os.sep + "data" + os.sep + "partitioned_training_data" + os.sep + "manual"


def return_dataframes_by_label_distribution(X, y, labels, label_distribution, random_state=1):
    X_dataframe_list = []
    y_dataframe_list = []

    for client_labels in label_distribution:
        X_dataframe_to_be_assigned = pd.DataFrame()
        y_dataframe_to_be_assigned = pd.DataFrame()
        for label_quantity, position_of_label in zip(client_labels, range(len(client_labels))):
            filter_by_label = y[y[labels[position_of_label]] == 1.0]
            filter_X_by_label = X[X.index.isin(filter_by_label.index)]
            X_samples = filter_X_by_label.sample(int(label_quantity), random_state=random_state)
            y_samples = y[y.index.isin(X_samples.index)]
            X_dataframe_to_be_assigned = pd.concat([X_dataframe_to_be_assigned, X_samples])
            y_dataframe_to_be_assigned = pd.concat([y_dataframe_to_be_assigned, y_samples])
            X.drop(X_samples.index.values, axis=0, inplace=True)
            y.drop(X_samples.index.values, axis=0, inplace=True)

        X_dataframe_list.append(X_dataframe_to_be_assigned)
        y_dataframe_list.append(y_dataframe_to_be_assigned)

    return X_dataframe_list, y_dataframe_list


def store_datasets(clients,
                   X_dataframes_train,
                   y_dataframes_train,
                   X_dataframe_test,
                   y_dataframes_test,
                   partition_name):
    if not os.path.exists(sample_directory + os.sep + __file__):
        final_directory = sample_directory + os.sep + partition_name
        os.makedirs(final_directory, exist_ok=True)
        for client, X_dataframe_train, y_dataframe_train, X_dataframe_test, y_dataframe_test \
                in zip(clients, X_dataframes_train, y_dataframes_train, X_dataframe_test, y_dataframes_test):
            X_dataframe_train.sort_index(inplace=True)
            y_dataframe_train.sort_index(inplace=True)
            X_dataframe_test.sort_index(inplace=True)
            y_dataframe_test.sort_index(inplace=True)
            X_dataframe_train.to_csv(final_directory + os.sep + client + "_X_training.csv")
            y_dataframe_train.to_csv(final_directory + os.sep + client + "_y_training.csv")
            X_dataframe_test.to_csv(final_directory + os.sep + client + "_X_test.csv")
            y_dataframe_test.to_csv(final_directory + os.sep + client + "_y_test.csv")


""" Functions for feature skew """


def divide_by_categorical_feature(X_dataset, y_dataset, slice_functions, n_clients_sharing_partition=1):
    X_datasets, y_datasets = [], []
    for slice_function in slice_functions:
        X_sliced_dataset = X_dataset[slice_function]
        if n_clients_sharing_partition > 1:
            # for client in range(n_clients_sharing_partition):
            shuffled = X_sliced_dataset.sample(frac=1)
            partitioned_dataframes = np.array_split(shuffled, n_clients_sharing_partition)
            for dataframe in partitioned_dataframes:
                X_datasets.append(dataframe)
                y_datasets.append(y_dataset.loc[list(dataframe.index.values)])
        else:
            y_sliced_dataset = y_dataset.loc[list(X_sliced_dataset.index.values)]
            X_datasets.append(X_sliced_dataset)
            y_datasets.append(y_sliced_dataset)

    return X_datasets, y_datasets


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
