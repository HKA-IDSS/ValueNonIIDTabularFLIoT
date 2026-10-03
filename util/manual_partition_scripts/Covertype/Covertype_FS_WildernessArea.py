from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import store_datasets, divide_by_categorical_feature

if __name__ == "__main__":
    x_train, y_train = dataset_model_dictionary["covertype"]().get_dataset().get_training_data()
    x_test, y_test = dataset_model_dictionary["covertype"]().get_dataset().get_test_data()
    partition_name = "Covertype_FS_WildernessArea"

    slice_train_function_1 = (x_train["Wilderness_Area1"] == 1)
    slice_train_function_2 = (x_train["Wilderness_Area2"] == 1)
    slice_train_function_3 = (x_train["Wilderness_Area3"] == 1)
    slice_train_function_4 = (x_train["Wilderness_Area4"] == 1)

    slice_test_function_1 = (x_test["Wilderness_Area1"] == 1)
    slice_test_function_2 = (x_test["Wilderness_Area2"] == 1)
    slice_test_function_3 = (x_test["Wilderness_Area3"] == 1)
    slice_test_function_4 = (x_test["Wilderness_Area4"] == 1)

    slice_train_functions = [slice_train_function_1,
                             slice_train_function_2,
                             slice_train_function_3,
                             slice_train_function_4]

    slice_test_functions = [slice_test_function_1,
                            slice_test_function_2,
                            slice_test_function_3,
                            slice_test_function_4]

    X_training_dataframe, y_training_dataframe = divide_by_categorical_feature(x_train, y_train, slice_train_functions)
    X_test_dataframe, y_test_dataframe = divide_by_categorical_feature(x_test, y_test, slice_test_functions)

    clients = ["client_" + str(number) for number in range(len(X_training_dataframe))]
    store_datasets(clients,
                   X_training_dataframe,
                   y_training_dataframe,
                   X_test_dataframe,
                   y_test_dataframe,
                   partition_name)
