from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import store_datasets, divide_by_categorical_feature

if __name__ == "__main__":
    x_train, y_train = dataset_model_dictionary["adult"]().get_dataset().get_training_data()
    x_test, y_test = dataset_model_dictionary["adult"]().get_dataset().get_test_data()
    partition_name = "Adult_FeatureSkew_Occupation"

    slice_training_function_1 = ((x_train["Other-service"] == 1)
                                 | (x_train["Priv-house-serv"] == 1)
                                 | (x_train["Protective-serv"] == 1)
                                 | (x_train["Sales"] == 1)
                                 | (x_train["Tech-support"] == 1))

    slice_training_function_2 = ((x_train["Adm-clerical"] == 1)
                                 | (x_train["Exec-managerial"] == 1))

    slice_training_function_3 = ((x_train["Handlers-cleaners"] == 1)
                                 | (x_train["Craft-repair"] == 1)
                                 | (x_train["Farming-fishing"] == 1)
                                 | (x_train["Armed-Forces"] == 1)
                                 | (x_train["Machine-op-inspct"] == 1)
                                 | (x_train["Transport-moving"] == 1))

    slice_test_function_1 = ((x_test["Other-service"] == 1)
                             | (x_test["Priv-house-serv"] == 1)
                             | (x_test["Protective-serv"] == 1)
                             | (x_test["Sales"] == 1)
                             | (x_test["Tech-support"] == 1))

    slice_test_function_2 = ((x_test["Adm-clerical"] == 1)
                             | (x_test["Exec-managerial"] == 1))

    slice_test_function_3 = ((x_test["Handlers-cleaners"] == 1)
                             | (x_test["Craft-repair"] == 1)
                             | (x_test["Farming-fishing"] == 1)
                             | (x_test["Armed-Forces"] == 1)
                             | (x_test["Machine-op-inspct"] == 1)
                             | (x_test["Transport-moving"] == 1))

    slice_train_functions = [slice_training_function_1, slice_training_function_2, slice_training_function_3]
    slice_test_functions = [slice_test_function_1, slice_test_function_2, slice_test_function_3]

    X_training_dataframe, y_training_dataframe = divide_by_categorical_feature(x_train, y_train, slice_train_functions, 2)
    X_test_dataframe, y_test_dataframe = divide_by_categorical_feature(x_test, y_test, slice_test_functions, 2)

    clients = ["client_" + str(number) for number in range(len(X_training_dataframe))]
    store_datasets(clients,
                   X_training_dataframe,
                   y_training_dataframe,
                   X_test_dataframe,
                   y_test_dataframe,
                   partition_name)
