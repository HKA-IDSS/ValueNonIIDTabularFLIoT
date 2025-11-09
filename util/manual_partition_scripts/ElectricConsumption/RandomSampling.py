import pandas as pd

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import return_dataframes_by_label_distribution, store_datasets

if __name__ == "__main__":
    clients = ["client_0", "client_1", "client_2", "client_3", "client_4"]
    X_train, y_train = dataset_model_dictionary["electric-consumption"]().get_dataset().get_training_data()
    X_test, y_test = dataset_model_dictionary["electric-consumption"]().get_dataset().get_test_data()
    partition_name = "Electric_Consumption_Random_Sampling"

    samples_in_training = len(X_train)
    samples_in_test = len(X_test)

    X_dataframes_train = []
    y_dataframes_train = []

    X_dataframes_test = []
    y_dataframes_test = []

    for client in clients:
        X_train_dataframe_to_be_assigned = pd.DataFrame()
        y_train_dataframe_to_be_assigned = pd.DataFrame()
        X_test_dataframe_to_be_assigned = pd.DataFrame()
        y_test_dataframe_to_be_assigned = pd.DataFrame()

        X_train_samples = X_train.sample(int(samples_in_training / len(clients)), random_state=1)
        y_train_samples = y_train[y_train.index.isin(X_train_samples.index)]

        X_test_samples = X_test.sample(int(samples_in_test / len(clients)), random_state=1)
        y_test_samples = y_test[y_test.index.isin(X_test_samples.index)]

        X_train_dataframe_to_be_assigned = pd.concat([X_train_dataframe_to_be_assigned, X_train_samples])
        y_train_dataframe_to_be_assigned = pd.concat([y_train_dataframe_to_be_assigned, y_train_samples])

        X_test_dataframe_to_be_assigned = pd.concat([X_test_dataframe_to_be_assigned, X_test_samples])
        y_test_dataframe_to_be_assigned = pd.concat([y_test_dataframe_to_be_assigned, y_test_samples])

        X_train.drop(X_train_samples.index.values, axis=0, inplace=True)
        y_train.drop(X_train_samples.index.values, axis=0, inplace=True)

        X_test.drop(X_test_samples.index.values, axis=0, inplace=True)
        y_test.drop(X_test_samples.index.values, axis=0, inplace=True)

        X_dataframes_train.append(X_train_dataframe_to_be_assigned)
        y_dataframes_train.append(y_train_dataframe_to_be_assigned)

        X_dataframes_test.append(X_test_dataframe_to_be_assigned)
        y_dataframes_test.append(y_test_dataframe_to_be_assigned)

    store_datasets(clients,
                   X_dataframes_train,
                   y_dataframes_train,
                   X_dataframes_test,
                   y_dataframes_test,
                   partition_name)
