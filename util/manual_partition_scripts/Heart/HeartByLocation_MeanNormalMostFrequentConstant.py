import functools

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import store_datasets, divide_by_categorical_feature

from itertools import product

if __name__ == "__main__":

    imputers = ["mean", "none"]
    numerical_preprocesser = ["standardize", "none"]
    imputer_string_categorical = ["most_frequent", "none"]
    imputer_numerical_categorical = ["constant", "none"]

    preprocessing_pipeline = [imputers, numerical_preprocesser, imputer_string_categorical, imputer_numerical_categorical]

    for pipeline in product(*preprocessing_pipeline):
        print(pipeline)

        x_train, y_train = dataset_model_dictionary["heart"](pipeline).get_training_data()
        x_test, y_test = dataset_model_dictionary["heart"](pipeline).get_test_data()
        pipeline_name_reducer = functools.reduce(lambda x, y: x + "_" + y, pipeline)
        partition_name = "Heart_FS_Location_" + pipeline_name_reducer

        slice_train_function_1 = (x_train["location_cleveland"] == 1)
        slice_train_function_2 = (x_train["location_hungarian"] == 1)
        slice_train_function_3 = (x_train["location_switzerland"] == 1)
        slice_train_function_4 = (x_train["location_va"] == 1)

        slice_test_function_1 = (x_test["location_cleveland"] == 1)
        slice_test_function_2 = (x_test["location_hungarian"] == 1)
        slice_test_function_3 = (x_test["location_switzerland"] == 1)
        slice_test_function_4 = (x_test["location_va"] == 1)

        slice_train_functions = [slice_train_function_1,
                                 slice_train_function_2,
                                 slice_train_function_3,
                                 slice_train_function_4]

        slice_test_functions = [slice_test_function_1,
                                 slice_test_function_2,
                                 slice_test_function_3,
                                 slice_test_function_4]

        X_train_dataframes, y_train_dataframes = divide_by_categorical_feature(x_train, y_train, slice_train_functions)
        X_test_dataframes, y_test_dataframes = divide_by_categorical_feature(x_test, y_test, slice_test_functions)

        clients = ["client_" + str(number) for number in range(len(X_train_dataframes))]
        store_datasets(clients,
                       X_train_dataframes,
                       y_train_dataframes,
                       X_test_dataframes,
                       y_test_dataframes,
                       partition_name)
