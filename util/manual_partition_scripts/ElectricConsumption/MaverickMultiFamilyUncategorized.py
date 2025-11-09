import pandas as pd

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import store_datasets, divide_by_categorical_feature

if __name__ == "__main__":
    x_train, y_train = dataset_model_dictionary["electric-consumption"]().get_dataset().get_training_data()
    x_test, y_test = dataset_model_dictionary["electric-consumption"]().get_dataset().get_test_data()
    partition_name = "ElectricConsumption_FeatureSkew_MaverickMultifamilyUncategorized"

    slice_training_function_1 = (x_train["facility_type_Multifamily_Uncategorized"] == 0)
    slice_test_function_1 = (x_test["facility_type_Multifamily_Uncategorized"] == 0)

    slice_training_function_2 = (x_train["facility_type_Multifamily_Uncategorized"] == 1)
    slice_test_function_2 = (x_test["facility_type_Multifamily_Uncategorized"] == 1)

    slice_train_functions = [slice_training_function_1]
    slice_test_functions = [slice_test_function_1]

    X_training_dataframe, y_training_dataframe = divide_by_categorical_feature(x_train, y_train, slice_train_functions,
                                                                               5)
    X_test_dataframe, y_test_dataframe = divide_by_categorical_feature(x_test, y_test, slice_test_functions, 5)

    x_training_category_dataset = x_train[slice_training_function_2]
    X_training_dataframe[-1] = pd.concat([X_training_dataframe[-1], x_training_category_dataset])
    y_training_dataframe[-1] = pd.concat([y_training_dataframe[-1], y_train.loc[x_training_category_dataset.index.values]])

    x_test_category_dataset = x_test[slice_test_function_2]
    X_test_dataframe[-1] = pd.concat([X_test_dataframe[-1], x_test_category_dataset])
    y_test_dataframe[-1] = pd.concat([y_test_dataframe[-1], y_test.loc[x_test_category_dataset.index.values]])

    clients = ["client_" + str(number) for number in range(len(X_training_dataframe))]
    store_datasets(clients,
                   X_training_dataframe,
                   y_training_dataframe,
                   X_test_dataframe,
                   y_test_dataframe,
                   partition_name)
