import numpy as np

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import store_datasets, divide_by_categorical_feature

if __name__ == "__main__":
    x_train, y_train = dataset_model_dictionary["adult"]().get_dataset().get_training_data()
    x_test, y_test = dataset_model_dictionary["adult"]().get_dataset().get_test_data()
    partition_name = "Adult_FeatureSkew_Studies"

    slice_training_function_1 = ((x_train["1st-4th"] == 1)
                                 | (x_train["5th-6th"] == 1)
                                 | (x_train["7th-8th"] == 1)
                                 | (x_train["9th"] == 1)
                                 | (x_train["10th"] == 1)
                                 | (x_train["11th"] == 1)
                                 | (x_train["12th"] == 1)
                                 | (x_train["HS-grad"] == 1))

    slice_training_function_2 = ((x_train["Assoc-acdm"] == 1)
                                 | (x_train["Assoc-voc"] == 1)
                                 | (x_train["Prof-school"] == 1)
                                 | (x_train["Some-college"] == 1))

    slice_training_function_3 = ((x_train["Bachelors"] == 1)
                                 | (x_train["Masters"] == 1)
                                 | (x_train["Doctorate"] == 1))

    slice_test_function_1 = ((x_test["1st-4th"] == 1)
                             | (x_test["5th-6th"] == 1)
                             | (x_test["7th-8th"] == 1)
                             | (x_test["9th"] == 1)
                             | (x_test["10th"] == 1)
                             | (x_test["11th"] == 1)
                             | (x_test["12th"] == 1)
                             | (x_test["HS-grad"] == 1))

    slice_test_function_2 = ((x_test["Assoc-acdm"] == 1)
                             | (x_test["Assoc-voc"] == 1)
                             | (x_test["Prof-school"] == 1)
                             | (x_test["Some-college"] == 1))

    slice_test_function_3 = ((x_test["Bachelors"] == 1)
                             | (x_test["Masters"] == 1)
                             | (x_test["Doctorate"] == 1))

    slice_train_functions = [slice_training_function_1, slice_training_function_2, slice_training_function_3]
    slice_test_functions = [slice_test_function_1, slice_test_function_2, slice_test_function_3]

    X_training_dataframe, y_training_dataframe = divide_by_categorical_feature(x_train,
                                                                               y_train,
                                                                               slice_train_functions,
                                                                               2)
    X_test_dataframe, y_test_dataframe = divide_by_categorical_feature(x_test,
                                                                       y_test,
                                                                       slice_test_functions,
                                                                       2)

    clients = ["client_" + str(number) for number in range(len(X_training_dataframe))]
    for y_training in y_training_dataframe:
        print(np.unique(np.argmax(y_training, axis=1), return_counts=True))
    store_datasets(clients,
                   X_training_dataframe,
                   y_training_dataframe,
                   X_test_dataframe,
                   y_test_dataframe,
                   partition_name)
