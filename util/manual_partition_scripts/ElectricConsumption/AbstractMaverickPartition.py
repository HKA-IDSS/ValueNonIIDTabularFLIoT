import os

import pandas as pd

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import (
    store_datasets, divide_by_categorical_feature,
    indexes_special_category_samples, store_maverick_removed_data
)


def partition_maverick_categorical_feature(column,
                                           category_value,
                                           n_clients=6,
                                           dataset_key="wids_dataset",
                                           partition_name_prefix=None):
    def partition_random_seed(random_seed):
        """
        Builds an n_clients partition where the last client is a Maverick:
        it gets an even share of the "common" data (like every other client)
        PLUS every building_group that carries `column == category_value`,
        a trait no other client has.

        A building_group is routed to the exclusive pool if ANY of its rows
        match the category -- this keeps the split group-safe even when
        `column` isn't one of the columns building_group is built from.

        Also writes a second, "_mav_data_excluded" copy of the maverick's
        files with the exclusive rows stripped out, so you can run the same
        client with vs. without its unique trait.
        """
        dataset = dataset_model_dictionary[dataset_key]().get_dataset()
        x_train, y_train = dataset.get_training_data()
        x_test, y_test = dataset.get_test_data()
        groups_train, groups_test = dataset.groups_train, dataset.groups_test

        # NEW: raw category values, index-aligned with x_train/x_test/groups_train/groups_test
        raw_train, raw_test = dataset.get_subpopulation_of_interest([column])

        partition_name = (partition_name_prefix or f"wids_energy_maverick_{column}") + os.sep + str(random_seed)

        # groups that own at least one row with the target category
        exclusive_groups_train = set(groups_train[raw_train[column] == 1].unique())
        exclusive_groups_test = set(groups_test[raw_test[column] == 1].unique())

        slice_training_common = ~groups_train.isin(exclusive_groups_train)
        slice_test_common = ~groups_test.isin(exclusive_groups_test)

        # split the common pool group-safely across n_clients
        X_training_dataframe, y_training_dataframe = divide_by_categorical_feature(
            x_train, y_train, [slice_training_common], n_clients, random_seed, groups=groups_train)
        X_test_dataframe, y_test_dataframe = divide_by_categorical_feature(
            x_test, y_test, [slice_test_common], n_clients, random_seed, groups=groups_test)

        # fold the exclusive groups entirely into the last client -> the Maverick
        slice_training_exclusive = groups_train.isin(exclusive_groups_train)
        slice_test_exclusive = groups_test.isin(exclusive_groups_test)

        x_training_exclusive_data = x_train[slice_training_exclusive]
        X_training_dataframe[-1] = pd.concat([X_training_dataframe[-1], x_training_exclusive_data])
        y_training_dataframe[-1] = pd.concat([y_training_dataframe[-1],
                                              y_train.loc[x_training_exclusive_data.index.values]])

        x_test_exclusive_data = x_test[slice_test_exclusive]
        X_test_dataframe[-1] = pd.concat([X_test_dataframe[-1], x_test_exclusive_data])
        y_test_dataframe[-1] = pd.concat([y_test_dataframe[-1],
                                          y_test.loc[x_test_exclusive_data.index.values]])

        clients = ["client_" + str(number) for number in range(len(X_training_dataframe))]
        store_datasets(clients, X_training_dataframe, y_training_dataframe,
                       X_test_dataframe, y_test_dataframe, partition_name)

        # -- second file set: the same maverick client, with the exclusive trait removed --
        maverick_index = len(X_training_dataframe) - 1
        X_mav_train = X_training_dataframe[maverick_index].copy()
        y_mav_train = y_training_dataframe[maverick_index].copy()
        X_mav_test = X_test_dataframe[maverick_index].copy()
        y_mav_test = y_test_dataframe[maverick_index].copy()

        # CHANGED: raw category lookup, not a one-hot column read off X_mav_train
        indexes_to_remove_train, indexes_to_remove_test = indexes_special_category_samples(
            raw_train.loc[X_mav_train.index], raw_test.loc[X_mav_test.index], column, category_value)


        store_maverick_removed_data(X_mav_train, y_mav_train, X_mav_test, y_mav_test,
                                    partition_name, maverick_index,
                                    indexes_to_remove_train, indexes_to_remove_test)

        return clients
    return partition_random_seed
