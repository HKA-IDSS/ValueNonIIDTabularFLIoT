import os

from sklearn.model_selection import train_test_split

from experiment_parameters.data_preparation.Dataset import encode_training_and_test_y_data
from util.manual_partition_scripts.NewAdult.ACSDataLoader import acs_get_full_data

if __name__ == "__main__":
    name_partition = "New_Adult_AL_CO_CT_MA"
    # name_dataset = "New_Adult"
    countries_selected = {
        'AL': "ST_Alabama/AL",
        'CO': "ST_Colorado/CO",
        'CT': "ST_Connecticut/CT",
        'MA': "ST_Massachusetts/MA"
    }

    X, y = acs_get_full_data(countries_selected)
    centralized_X = X.copy()
    centralized_y = y.copy()

    centralized_X.drop(countries_selected.values(), axis=1)
    # centralized_y.drop(countries_selected.values(), inplace=True)

    centralized_X.to_csv("data" + os.sep + "datasets" + os.sep + "New_Adult" + os.sep + "X.csv")
    centralized_y.to_csv("data" + os.sep + "datasets" + os.sep + "New_Adult" + os.sep + "y.csv")

    for client_number, country in zip(range(len(countries_selected)), countries_selected.keys()):
        sliced_X = X[X[countries_selected[country]] == 1]
        sliced_y = y.loc[sliced_X.index]
        sliced_X.drop(countries_selected.values(), axis=1)
        X_train, X_test, y_train, y_test = train_test_split(sliced_X, sliced_y, test_size=0.3, random_state=1)
        y_train, y_test, labels = encode_training_and_test_y_data(y_train, y_test)

        route_for_data = "data" + os.sep + "partitioned_training_data" + os.sep + "manual" + os.sep + name_partition
        os.makedirs(route_for_data, exist_ok=True)

        X_train.to_csv(route_for_data + os.sep + "client_" + str(client_number) + "_X_training.csv")
        X_test.to_csv(route_for_data + os.sep + "client_" + str(client_number) + "_X_test.csv")
        y_train.to_csv(route_for_data + os.sep + "client_" + str(client_number) + "_y_training.csv")
        y_test.to_csv(route_for_data + os.sep + "client_" + str(client_number) + "_y_test.csv")
