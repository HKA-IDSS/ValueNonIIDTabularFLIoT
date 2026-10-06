from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import store_datasets, divide_by_categorical_feature

if __name__ == "__main__":
    x_train, y_train = dataset_model_dictionary["adult"]().get_dataset().get_training_data()
    partition_name = "Adult_FeatureSkew_WorldRegion"

    slice_function_1 = ((x_train["categorical_preprocess__native-country_England"] == 1)
                        | (x_train["categorical_preprocess__native-country_France"] == 1)
                        | (x_train["categorical_preprocess__native-country_Germany"] == 1)
                        | (x_train["categorical_preprocess__native-country_Greece"] == 1)
                        | (x_train["categorical_preprocess__native-country_Holand-Netherlands"] == 1)
                        | (x_train["categorical_preprocess__native-country_Hungary"] == 1)
                        | (x_train["categorical_preprocess__native-country_Ireland"] == 1)
                        | (x_train["categorical_preprocess__native-country_Italy"] == 1)
                        | (x_train["categorical_preprocess__native-country_Poland"] == 1)
                        | (x_train["categorical_preprocess__native-country_Portugal"] == 1)
                        | (x_train["categorical_preprocess__native-country_Scotland"] == 1)
                        | (x_train["categorical_preprocess__native-country_Yugoslavia"] == 1))

    slice_function_2 = ((x_train["categorical_preprocess__native-country_Columbia"] == 1)
                        | (x_train["categorical_preprocess__native-country_Cuba"] == 1)
                        | (x_train["categorical_preprocess__native-country_Dominican-Republic"] == 1)
                        | (x_train["categorical_preprocess__native-country_Ecuador"] == 1)
                        | (x_train["categorical_preprocess__native-country_El-Salvador"] == 1)
                        | (x_train["categorical_preprocess__native-country_Guatemala"] == 1)
                        | (x_train["categorical_preprocess__native-country_Haiti"] == 1)
                        | (x_train["categorical_preprocess__native-country_Honduras"] == 1)
                        | (x_train["categorical_preprocess__native-country_Jamaica"] == 1)
                        | (x_train["categorical_preprocess__native-country_Mexico"] == 1)
                        | (x_train["categorical_preprocess__native-country_Nicaragua"] == 1)
                        | (x_train["categorical_preprocess__native-country_Peru"] == 1)
                        | (x_train["categorical_preprocess__native-country_Puerto-Rico"] == 1)
                        | (x_train["categorical_preprocess__native-country_Trinadad&Tobago"] == 1))

    slice_function_3 = ((x_train["categorical_preprocess__native-country_Cambodia"] == 1)
                        | (x_train["categorical_preprocess__native-country_China"] == 1)
                        | (x_train["categorical_preprocess__native-country_Hong"] == 1)
                        | (x_train["categorical_preprocess__native-country_India"] == 1)
                        | (x_train["categorical_preprocess__native-country_Iran"] == 1)
                        | (x_train["categorical_preprocess__native-country_Japan"] == 1)
                        | (x_train["categorical_preprocess__native-country_Laos"] == 1)
                        | (x_train["categorical_preprocess__native-country_Philippines"] == 1)
                        | (x_train["categorical_preprocess__native-country_South"] == 1)
                        | (x_train["categorical_preprocess__native-country_Taiwan"] == 1)
                        | (x_train["categorical_preprocess__native-country_Thailand"] == 1)
                        | (x_train["categorical_preprocess__native-country_Vietnam"] == 1))

    slice_function_4 = ((x_train["categorical_preprocess__native-country_Canada"] == 1)
                        | (x_train["categorical_preprocess__native-country_United-States"] == 1)
                        | (x_train["categorical_preprocess__native-country_Outlying-US(Guam-USVI-etc)"] == 1))

    slice_functions = [slice_function_1, slice_function_2, slice_function_3, slice_function_4]

    X_dataframes, y_dataframes = divide_by_categorical_feature(x_train, y_train, slice_functions)

    clients = ["client_" + str(number) for number in range(len(X_dataframes))]
    store_datasets(clients, X_dataframes, y_dataframes, partition_name)
