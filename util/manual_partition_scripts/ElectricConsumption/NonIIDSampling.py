import os
import numpy as np
import pandas as pd

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import store_datasets


def wids_energy_non_iid_by_label(random_seed, n_buckets=6, representative='last'):
    """
    Non-iid partition where each client is skewed toward a range of site_eui.
    Buckets are assigned per building_group (not per row), using a single
    representative site_eui value per group, so a building's full history
    always lands on exactly one client. Bucket boundaries are fit on train
    only and reused for test, so no leakage from test into the boundaries.
    """
    dataset = dataset_model_dictionary["wids_dataset"]()
    X_train, y_train = dataset.get_dataset().get_training_data()
    X_test, y_test = dataset.get_dataset().get_test_data()
    groups_train, groups_test = dataset.get_dataset().groups_train, dataset.get_dataset().groups_test
    partition_name = "wids_energy_non_iid_by_label" + os.sep + str(random_seed)

    # 1. one representative site_eui per building_group, computed separately
    #    for train and test's own labels (a test-only building has no train
    #    representative to fall back on, so each side needs its own)
    def representative_per_group(y, groups, how):
        g = y.groupby(groups)
        return g.mean() if how == 'mean' else g.last()

    if representative not in ('mean', 'last'):
        raise ValueError("representative must be 'mean' or 'last'")

    group_repr_train = representative_per_group(y_train, groups_train, representative)
    group_repr_test  = representative_per_group(y_test,  groups_test,  representative)

    # 2. quantile edges computed dynamically from TRAIN only — test never
    #    influences where the bucket boundaries fall
    quantile_points = np.linspace(0, 1, n_buckets + 1)[1:-1]  # e.g. [.25, .5, .75] for 4 buckets
    edges = group_repr_train.quantile(quantile_points).values
    bin_edges = [-np.inf] + list(edges) + [np.inf]

    # 3. bucket each side independently, against the same (train-derived) edges
    group_bucket_train = pd.cut(group_repr_train, bins=bin_edges, labels=False)
    group_bucket_test  = pd.cut(group_repr_test,  bins=bin_edges, labels=False)

    clients = [f"client_{i}" for i in range(n_buckets)]
    X_dataframes_train, y_dataframes_train = [], []
    X_dataframes_test, y_dataframes_test = [], []

    for bucket_idx in range(n_buckets):
        train_groups_in_bucket = group_bucket_train[group_bucket_train == bucket_idx].index
        test_groups_in_bucket  = group_bucket_test[group_bucket_test == bucket_idx].index

        train_mask = groups_train.isin(train_groups_in_bucket)
        test_mask  = groups_test.isin(test_groups_in_bucket)

        X_dataframes_train.append(X_train[train_mask])
        y_dataframes_train.append(y_train[train_mask])
        X_dataframes_test.append(X_test[test_mask])
        y_dataframes_test.append(y_test[test_mask])

    store_datasets(clients, X_dataframes_train, y_dataframes_train, X_dataframes_test, y_dataframes_test, partition_name)

# import os
#
# from experiment_parameters.TrainerFactory import dataset_model_dictionary
# from util.manual_partition_scripts.ManualSamplerUtil import store_datasets, divide_by_categorical_feature
#
# def electric_consumption_non_iid_sampling(random_seed):
#     x_train, y_train = dataset_model_dictionary["electric-consumption"]().get_dataset().get_training_data()
#     x_test, y_test = dataset_model_dictionary["electric-consumption"]().get_dataset().get_test_data()
#     partition_name = "electric_consumption_non_iid_sampling" + os.sep + str(random_seed)
#
#     X_training_dataframe = []
#     y_training_dataframe = []
#     X_test_dataframe = []
#     y_test_dataframe = []
#
#     y_train_filtered = y_train[y_train < 54.52]
#     y_training_dataframe.append(y_train_filtered)
#     X_training_dataframe.append(x_train[x_train.index.isin(y_train_filtered.index.values)])
#
#     y_test_filtered = y_test[y_test < 54.52]
#     y_test_dataframe.append(y_test_filtered)
#     X_test_dataframe.append(x_test[x_test.index.isin(y_test_filtered.index.values)])
#
#     y_train_filtered = y_train[(y_train > 54.52) & (y_train < 75.29)]
#     y_training_dataframe.append(y_train_filtered)
#     X_training_dataframe.append(x_train[x_train.index.isin(y_train_filtered.index.values)])
#
#     y_test_filtered = y_test[(y_test > 54.52) & (y_test < 75.29)]
#     y_test_dataframe.append(y_test_filtered)
#     X_test_dataframe.append(x_test[x_test.index.isin(y_test_filtered.index.values)])
#
#     y_train_filtered = y_train[(y_train > 75.29) & (y_train < 97.28)]
#     y_training_dataframe.append(y_train_filtered)
#     X_training_dataframe.append(x_train[x_train.index.isin(y_train_filtered.index.values)])
#
#     y_test_filtered = y_test[(y_test > 75.29) & (y_test < 97.28)]
#     y_test_dataframe.append(y_test_filtered)
#     X_test_dataframe.append(x_test[x_test.index.isin(y_test_filtered.index.values)])
#
#     y_train_filtered = y_train[y_train > 97.28]
#     y_training_dataframe.append(y_train_filtered)
#     X_training_dataframe.append(x_train[x_train.index.isin(y_train_filtered.index.values)])
#
#     y_test_filtered = y_test[y_test > 97.28]
#     y_test_dataframe.append(y_test_filtered)
#     X_test_dataframe.append(x_test[x_test.index.isin(y_test_filtered.index.values)])


    # y_train_client_1 = y_train[y_train]
    #
    # slice_train_functions = [slice_training_function_1, slice_training_function_2, slice_training_function_3]
    # slice_test_functions = [slice_test_function_1, slice_test_function_2, slice_test_function_3]
    #
    # X_training_dataframe, y_training_dataframe = divide_by_categorical_feature(x_train,
    #                                                                            y_train,
    #                                                                            slice_train_functions,
    #                                                                            2)
    # X_test_dataframe, y_test_dataframe = divide_by_categorical_feature(x_test,
    #                                                                    y_test,
    #                                                                    slice_test_functions,
    #                                                                    2)
    #
    # clients = ["client_" + str(number) for number in range(len(X_training_dataframe))]
    # # for y_training in y_training_dataframe:
    # #     print(np.unique(np.argmax(y_training, axis=1), return_counts=True))
    # store_datasets(clients,
    #                X_training_dataframe,
    #                y_training_dataframe,
    #                X_test_dataframe,
    #                y_test_dataframe,
    #                partition_name)
