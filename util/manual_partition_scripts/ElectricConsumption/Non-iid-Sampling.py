# print(len(dataset[dataset["site_eui"] < 54.52]))
# print(len(dataset[(dataset["site_eui"] > 54.52) & (dataset["site_eui"] < 75.29)]))
# print(len(dataset[(dataset["site_eui"] > 75.29) & (dataset["site_eui"] < 97.28)]))
# print(len(dataset[dataset["site_eui"] > 97.28]))

import numpy as np

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import store_datasets, divide_by_categorical_feature

if __name__ == "__main__":
    x_train, y_train = dataset_model_dictionary["electric-consumption"]().get_dataset().get_training_data()
    x_test, y_test = dataset_model_dictionary["electric-consumption"]().get_dataset().get_test_data()
    partition_name = "ElectricConsumptionNoniid"

    X_training_dataframe = []
    y_training_dataframe = []
    X_test_dataframe = []
    y_test_dataframe = []

    y_train_filtered = y_train[y_train < 54.52]
    y_training_dataframe.append(y_train_filtered)
    X_training_dataframe.append(x_train[x_train.index.isin(y_train_filtered.index.values)])

    y_test_filtered = y_test[y_test < 54.52]
    y_test_dataframe.append(y_test_filtered)
    X_test_dataframe.append(x_test[x_test.index.isin(y_test_filtered.index.values)])

    y_train_filtered = y_train[(y_train > 54.52) & (y_train < 75.29)]
    y_training_dataframe.append(y_train_filtered)
    X_training_dataframe.append(x_train[x_train.index.isin(y_train_filtered.index.values)])

    y_test_filtered = y_test[(y_test > 54.52) & (y_test < 75.29)]
    y_test_dataframe.append(y_test_filtered)
    X_test_dataframe.append(x_test[x_test.index.isin(y_test_filtered.index.values)])

    y_train_filtered = y_train[(y_train > 75.29) & (y_train < 97.28)]
    y_training_dataframe.append(y_train_filtered)
    X_training_dataframe.append(x_train[x_train.index.isin(y_train_filtered.index.values)])

    y_test_filtered = y_test[(y_test > 75.29) & (y_test < 97.28)]
    y_test_dataframe.append(y_test_filtered)
    X_test_dataframe.append(x_test[x_test.index.isin(y_test_filtered.index.values)])

    y_train_filtered = y_train[y_train > 97.28]
    y_training_dataframe.append(y_train_filtered)
    X_training_dataframe.append(x_train[x_train.index.isin(y_train_filtered.index.values)])

    y_test_filtered = y_test[y_test > 97.28]
    y_test_dataframe.append(y_test_filtered)
    X_test_dataframe.append(x_test[x_test.index.isin(y_test_filtered.index.values)])


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
    clients = ["client_" + str(number) for number in range(len(X_training_dataframe))]
    # for y_training in y_training_dataframe:
    #     print(np.unique(np.argmax(y_training, axis=1), return_counts=True))
    store_datasets(clients,
                   X_training_dataframe,
                   y_training_dataframe,
                   X_test_dataframe,
                   y_test_dataframe,
                   partition_name)
