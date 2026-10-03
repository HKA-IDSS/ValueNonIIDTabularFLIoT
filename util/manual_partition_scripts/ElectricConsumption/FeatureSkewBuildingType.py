import os

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import store_datasets, divide_by_categorical_feature

def feature_skew_building_type(random_seed):
    dataset = dataset_model_dictionary["wids_dataset"]()
    x_train, y_train = dataset.get_dataset().get_training_data()
    x_test, y_test = dataset.get_dataset().get_test_data()
    groups_train, groups_test = dataset.get_dataset().groups_train, dataset.get_dataset().groups_test
    partition_name = "wids_energy_feature_skew_building_type" + os.sep + str(random_seed)

    slice_training_function_1 = (x_train["building_class_Commercial"] == 1)
    slice_training_function_2 = (x_train["building_class_Residential"] == 1)

    slice_test_function_1 = (x_test["building_class_Commercial"] == 1)
    slice_test_function_2 = (x_test["building_class_Residential"] == 1)

    slice_train_functions = [slice_training_function_1, slice_training_function_2]
    slice_test_functions = [slice_test_function_1, slice_test_function_2]

    X_training_dataframe, y_training_dataframe = divide_by_categorical_feature(x_train,
                                                                               y_train,
                                                                               slice_train_functions,
                                                                               3,
                                                                               random_seed,
                                                                               groups=groups_train)
    X_test_dataframe, y_test_dataframe = divide_by_categorical_feature(x_test,
                                                                       y_test,
                                                                       slice_test_functions,
                                                                       3,
                                                                       random_seed,
                                                                       groups=groups_test)

    clients = ["client_" + str(number) for number in range(len(X_training_dataframe))]
    store_datasets(clients,
                   X_training_dataframe,
                   y_training_dataframe,
                   X_test_dataframe,
                   y_test_dataframe,
                   partition_name)