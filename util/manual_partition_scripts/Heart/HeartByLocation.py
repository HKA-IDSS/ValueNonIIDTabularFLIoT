from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import store_datasets, divide_by_categorical_feature

if __name__ == "__main__":
    x_train, y_train = dataset_model_dictionary["heart"]().get_dataset().get_training_data()
    x_test, y_test = dataset_model_dictionary["heart"]().get_dataset().get_test_data()
    partition_name = "Heart_FS_Location"

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
