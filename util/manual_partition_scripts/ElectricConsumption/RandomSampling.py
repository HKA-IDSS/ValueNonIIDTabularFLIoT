import os

import numpy as np
import pandas as pd

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import store_datasets

def wids_energy_iid_sampling(random_seed):
    clients = ["client_0", "client_1", "client_2", "client_3", "client_4", "client_5"]
    dataset = dataset_model_dictionary["wids_dataset"]()
    X_train, y_train = dataset.get_dataset().get_training_data()
    X_test, y_test = dataset.get_dataset().get_test_data()
    groups_train, groups_test = dataset.get_dataset().groups_train, dataset.get_dataset().groups_test
    partition_name = "wids_energy_iid_sampling" + os.sep + str(random_seed)

    rng = np.random.RandomState(random_seed)
    train_chunks = np.array_split(rng.permutation(groups_train.unique()), len(clients))
    test_chunks  = np.array_split(rng.permutation(groups_test.unique()),  len(clients))

    X_dataframes_train, y_dataframes_train = [], []
    X_dataframes_test, y_dataframes_test = [], []

    for train_groups, test_groups in zip(train_chunks, test_chunks):
        train_mask = groups_train.isin(train_groups)
        test_mask  = groups_test.isin(test_groups)

        X_dataframes_train.append(X_train[train_mask])
        y_dataframes_train.append(y_train[train_mask])
        X_dataframes_test.append(X_test[test_mask])
        y_dataframes_test.append(y_test[test_mask])

    store_datasets(clients, X_dataframes_train, y_dataframes_train, X_dataframes_test, y_dataframes_test, partition_name)

# import os
#
# import pandas as pd
#
# from experiment_parameters.TrainerFactory import dataset_model_dictionary
# from util.manual_partition_scripts.ManualSamplerUtil import return_dataframes_by_label_distribution, store_datasets
#
# def electric_consumption_iid_sampling(random_seed):
#     clients = ["client_0", "client_1", "client_2", "client_3", "client_4", "client_5"]
#     X_train, y_train = dataset_model_dictionary["electric-consumption"]().get_dataset().get_training_data()
#     X_test, y_test = dataset_model_dictionary["electric-consumption"]().get_dataset().get_test_data()
#     partition_name = "electric_consumption_iid_sampling" + os.sep + str(random_seed)
#
#     samples_in_training = len(X_train)
#     samples_in_test = len(X_test)
#
#     X_dataframes_train = []
#     y_dataframes_train = []
#
#     X_dataframes_test = []
#     y_dataframes_test = []
#
#     for client in clients:
#         X_train_dataframe_to_be_assigned = pd.DataFrame()
#         y_train_dataframe_to_be_assigned = pd.DataFrame()
#         X_test_dataframe_to_be_assigned = pd.DataFrame()
#         y_test_dataframe_to_be_assigned = pd.DataFrame()
#
#         X_train_samples = X_train.sample(int(samples_in_training / len(clients)), random_state=random_seed)
#         y_train_samples = y_train[y_train.index.isin(X_train_samples.index)]
#
#         X_test_samples = X_test.sample(int(samples_in_test / len(clients)), random_state=random_seed)
#         y_test_samples = y_test[y_test.index.isin(X_test_samples.index)]
#
#         X_train_dataframe_to_be_assigned = pd.concat([X_train_dataframe_to_be_assigned, X_train_samples])
#         y_train_dataframe_to_be_assigned = pd.concat([y_train_dataframe_to_be_assigned, y_train_samples])
#
#         X_test_dataframe_to_be_assigned = pd.concat([X_test_dataframe_to_be_assigned, X_test_samples])
#         y_test_dataframe_to_be_assigned = pd.concat([y_test_dataframe_to_be_assigned, y_test_samples])
#
#         X_train.drop(X_train_samples.index.values, axis=0, inplace=True)
#         y_train.drop(X_train_samples.index.values, axis=0, inplace=True)
#
#         X_test.drop(X_test_samples.index.values, axis=0, inplace=True)
#         y_test.drop(X_test_samples.index.values, axis=0, inplace=True)
#
#         X_dataframes_train.append(X_train_dataframe_to_be_assigned)
#         y_dataframes_train.append(y_train_dataframe_to_be_assigned)
#
#         X_dataframes_test.append(X_test_dataframe_to_be_assigned)
#         y_dataframes_test.append(y_test_dataframe_to_be_assigned)
#
#     store_datasets(clients,
#                    X_dataframes_train,
#                    y_dataframes_train,
#                    X_dataframes_test,
#                    y_dataframes_test,
#                    partition_name)
