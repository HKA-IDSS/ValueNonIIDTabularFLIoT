import numpy as np

from experiment_parameters.TrainerFactory import dataset_model_dictionary
from util.manual_partition_scripts.ManualSamplerUtil import store_datasets, divide_by_categorical_feature

if __name__ == "__main__":
    x_train, y_train = dataset_model_dictionary["electric-consumption"]().get_dataset().get_training_data()
    x_test, y_test = dataset_model_dictionary["electric-consumption"]().get_dataset().get_test_data()
    partition_name = "ElectricConsumptionFacilityType"

    slice_training_function_1 = ((x_train["facility_type_Retail_Enclosed_mall"] == 1)
                                 | (x_train["facility_type_Retail_Strip_shopping_mall"] == 1)
                                 | (x_train["facility_type_Retail_Uncategorized"] == 1)
                                 | (x_train["facility_type_Retail_Vehicle_dealership_showroom"] == 1)
                                 | (x_train["facility_type_Service_Drycleaning_or_Laundry"] == 1)
                                 | (x_train["facility_type_Service_Uncategorized"] == 1)
                                 | (x_train["facility_type_Service_Vehicle_service_repair_shop"] == 1)
                                 | (x_train["facility_type_Religious_worship"] == 1)
                                 | (x_train["facility_type_Lodging_Dormitory_or_fraternity_sorority"] == 1)
                                 | (x_train["facility_type_Lodging_Hotel"] == 1)
                                 | (x_train["facility_type_Lodging_Other"] == 1)
                                 | (x_train["facility_type_Lodging_Uncategorized"] == 1)
                                 | (x_train["facility_type_Industrial"] == 1)
                                 | (x_train["facility_type_Data_Center"] == 1)
                                 | (x_train["facility_type_Food_Sales"] == 1)
                                 | (x_train["facility_type_Food_Service_Other"] == 1)
                                 | (x_train["facility_type_Food_Service_Restaurant_or_cafeteria"] == 1)
                                 | (x_train["facility_type_Food_Service_Uncategorized"] == 1)
                                 | (x_train["facility_type_Grocery_store_or_food_market"] == 1)
                                 | (x_train["facility_type_Office_Bank_or_other_financial"] == 1)
                                 | (x_train["facility_type_Office_Medical_non_diagnostic"] == 1)
                                 | (x_train["facility_type_Office_Mixed_use"] == 1)
                                 | (x_train["facility_type_Office_Uncategorized"] == 1)
                                 | (x_train["facility_type_Parking_Garage"] == 1)
                                 | (x_train["facility_type_Mixed_Use_Commercial_and_Residential"] == 1)
                                 | (x_train["facility_type_Mixed_Use_Predominantly_Commercial"] == 1))

    slice_training_function_2 = ((x_train["facility_type_2to4_Unit_Building"] == 1)
                                 | (x_train["facility_type_5plus_Unit_Building"] == 1)
                                 | (x_train["facility_type_Mixed_Use_Predominantly_Residential"] == 1)
                                 | (x_train["facility_type_Multifamily_Uncategorized"] == 1))

    slice_training_function_3 = ((x_train["facility_type_Public_Assembly_Drama_theater"] == 1)
                                 | (x_train["facility_type_Public_Assembly_Entertainment_culture"] == 1)
                                 | (x_train["facility_type_Public_Assembly_Library"] == 1)
                                 | (x_train["facility_type_Public_Assembly_Movie_Theater"] == 1)
                                 | (x_train["facility_type_Public_Assembly_Other"] == 1)
                                 | (x_train["facility_type_Public_Assembly_Recreation"] == 1)
                                 | (x_train["facility_type_Public_Assembly_Social_meeting"] == 1)
                                 | (x_train["facility_type_Public_Assembly_Stadium"] == 1)
                                 | (x_train["facility_type_Public_Assembly_Uncategorized"] == 1)
                                 | (x_train["facility_type_Public_Safety_Courthouse"] == 1)
                                 | (x_train["facility_type_Public_Safety_Fire_or_police_station"] == 1)
                                 | (x_train["facility_type_Public_Safety_Penitentiary"] == 1)
                                 | (x_train["facility_type_Public_Safety_Uncategorized"] == 1)
                                 | (x_train["facility_type_Education_College_or_university"] == 1)
                                 | (x_train["facility_type_Education_Other_classroom"] == 1)
                                 | (x_train["facility_type_Education_Preschool_or_daycare"] == 1)
                                 | (x_train["facility_type_Education_Uncategorized"] == 1)
                                 | (x_train["facility_type_Health_Care_Inpatient"] == 1)
                                 | (x_train["facility_type_Health_Care_Outpatient_Clinic"] == 1)
                                 | (x_train["facility_type_Health_Care_Outpatient_Uncategorized"] == 1)
                                 | (x_train["facility_type_Health_Care_Uncategorized"] == 1)
                                 | (x_train["facility_type_Laboratory"] == 1)
                                 | (x_train["facility_type_Nursing_Home"] == 1))

    slice_test_function_1 = ((x_test["facility_type_Retail_Enclosed_mall"] == 1)
                             | (x_test["facility_type_Retail_Strip_shopping_mall"] == 1)
                             | (x_test["facility_type_Retail_Uncategorized"] == 1)
                             | (x_test["facility_type_Retail_Vehicle_dealership_showroom"] == 1)
                             | (x_test["facility_type_Service_Drycleaning_or_Laundry"] == 1)
                             | (x_test["facility_type_Service_Uncategorized"] == 1)
                             | (x_test["facility_type_Service_Vehicle_service_repair_shop"] == 1)
                             | (x_test["facility_type_Religious_worship"] == 1)
                             | (x_test["facility_type_Lodging_Dormitory_or_fraternity_sorority"] == 1)
                             | (x_test["facility_type_Lodging_Hotel"] == 1)
                             | (x_test["facility_type_Lodging_Other"] == 1)
                             | (x_test["facility_type_Lodging_Uncategorized"] == 1)
                             | (x_test["facility_type_Industrial"] == 1)
                             | (x_test["facility_type_Data_Center"] == 1)
                             | (x_test["facility_type_Food_Sales"] == 1)
                             | (x_test["facility_type_Food_Service_Other"] == 1)
                             | (x_test["facility_type_Food_Service_Restaurant_or_cafeteria"] == 1)
                             | (x_test["facility_type_Food_Service_Uncategorized"] == 1)
                             | (x_test["facility_type_Grocery_store_or_food_market"] == 1)
                             | (x_test["facility_type_Office_Bank_or_other_financial"] == 1)
                             | (x_test["facility_type_Office_Medical_non_diagnostic"] == 1)
                             | (x_test["facility_type_Office_Mixed_use"] == 1)
                             | (x_test["facility_type_Office_Uncategorized"] == 1)
                             | (x_test["facility_type_Parking_Garage"] == 1)
                             | (x_test["facility_type_Mixed_Use_Commercial_and_Residential"] == 1)
                             | (x_test["facility_type_Mixed_Use_Predominantly_Commercial"] == 1))

    slice_test_function_2 = ((x_test["facility_type_2to4_Unit_Building"] == 1)
                             | (x_test["facility_type_5plus_Unit_Building"] == 1)
                             | (x_test["facility_type_Mixed_Use_Predominantly_Residential"] == 1)
                             | (x_test["facility_type_Multifamily_Uncategorized"] == 1))

    slice_test_function_3 = ((x_test["facility_type_Public_Assembly_Drama_theater"] == 1)
                             | (x_test["facility_type_Public_Assembly_Entertainment_culture"] == 1)
                             | (x_test["facility_type_Public_Assembly_Library"] == 1)
                             | (x_test["facility_type_Public_Assembly_Movie_Theater"] == 1)
                             | (x_test["facility_type_Public_Assembly_Other"] == 1)
                             | (x_test["facility_type_Public_Assembly_Recreation"] == 1)
                             | (x_test["facility_type_Public_Assembly_Social_meeting"] == 1)
                             | (x_test["facility_type_Public_Assembly_Stadium"] == 1)
                             | (x_test["facility_type_Public_Assembly_Uncategorized"] == 1)
                             | (x_test["facility_type_Public_Safety_Courthouse"] == 1)
                             | (x_test["facility_type_Public_Safety_Fire_or_police_station"] == 1)
                             | (x_test["facility_type_Public_Safety_Penitentiary"] == 1)
                             | (x_test["facility_type_Public_Safety_Uncategorized"] == 1)
                             | (x_test["facility_type_Education_College_or_university"] == 1)
                             | (x_test["facility_type_Education_Other_classroom"] == 1)
                             | (x_test["facility_type_Education_Preschool_or_daycare"] == 1)
                             | (x_test["facility_type_Education_Uncategorized"] == 1)
                             | (x_test["facility_type_Health_Care_Inpatient"] == 1)
                             | (x_test["facility_type_Health_Care_Outpatient_Clinic"] == 1)
                             | (x_test["facility_type_Health_Care_Outpatient_Uncategorized"] == 1)
                             | (x_test["facility_type_Health_Care_Uncategorized"] == 1)
                             | (x_test["facility_type_Laboratory"] == 1)
                             | (x_test["facility_type_Nursing_Home"] == 1))

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
    store_datasets(clients,
                   X_training_dataframe,
                   y_training_dataframe,
                   X_test_dataframe,
                   y_test_dataframe,
                   partition_name)
