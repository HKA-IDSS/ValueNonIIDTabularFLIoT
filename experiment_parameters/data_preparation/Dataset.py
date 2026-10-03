import functools
import itertools
import os
from typing import Tuple, List, Union, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from rich.jupyter import display
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_wine, load_iris
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector

from Definitions import ROOT_DIR
from util.Util import one_hot_encode_labels

directory_of_data = ROOT_DIR + os.sep + "data" + os.sep + "datasets"


def encode_training_and_test_y_data(y, y_test):
    y_reshaped = np.reshape(y.values, (-1, 1))
    y_test_reshaped = np.reshape(y_test.values, (-1, 1))

    encoder = OneHotEncoder().fit(y_reshaped)
    y_encoded = encoder.transform(y_reshaped)
    y_test_encoded = encoder.transform(y_test_reshaped)

    label_names = list(itertools.chain.from_iterable(list(map(lambda name: name.split("_", 1)[1:],
                                                              encoder.get_feature_names_out()))))
    y = pd.DataFrame(y_encoded.toarray(), index=y.index, columns=label_names)
    y_test = pd.DataFrame(y_test_encoded.toarray(), index=y_test.index, columns=label_names)

    return y, y_test, label_names


def sorting_columns(columns, original_order):
    new_list_of_columns = []
    for column_position in range(0, len(original_order)):
        filtered_columns = list(filter(lambda column: "__" + original_order[column_position] in column, columns))
        if any(["categorical_preprocess" in column for column in filtered_columns]):
            filtered_columns = list(
                filter(lambda column: "__" + original_order[column_position] + "_" in column, columns))

        new_list_of_columns.append(filtered_columns)

    new_list_of_columns = functools.reduce(lambda list1, list2: list1 + list2, new_list_of_columns, [])

    return new_list_of_columns


def remove_categorical_preprocess_or_remainder_string(column_name, continuous_features_names,
                                                      categorical_features_names):
    # if "categorical_preprocess__" in column_name:
    #     for categorical_features_name in categorical_features_names:
    #         if categorical_features_name in column_name:
    #             column_name = column_name.removeprefix("categorical_preprocess__" + categorical_features_name + "_")
    #             return column_name
    # elif "inputer_numerical__" in column_name:
    #     for categorical_features_name in categorical_features_names:
    #         if categorical_features_name in column_name:
    #             column_name = column_name.removeprefix("categorical_preprocess__" + categorical_features_name + "_")
    #             return column_name
    # elif "remainder__" in column_name:
    #     for continuous_feature_name in continuous_features_names:
    #         if continuous_feature_name in column_name:
    #             column_name = column_name.removeprefix("remainder__")
    #             return column_name
    column_name = column_name.split("__")[1]
    return column_name


class Dataset:
    X: DataFrame
    y: Union[Series, DataFrame]
    x_test: DataFrame
    y_test: Union[Series, DataFrame]
    x_train: DataFrame
    y_train: Union[Series, DataFrame]
    labels: List[str]
    _original_values_of_categories_of_interest_train: Optional[pd.DataFrame] = None
    _original_values_of_categories_of_interest_test: Optional[pd.DataFrame] = None

    def get_dataset(self) -> Tuple[DataFrame, Union[Series, DataFrame]]:
        return self.X, self.y

    def get_min_max_X_train(self):
        min_X = self.x_train.min(axis=0)
        max_X = self.x_train.max(axis=0)
        return min_X, max_X

    def get_min_max_X_test(self):
        min_X = self.x_test.min(axis=0)
        max_X = self.x_test.max(axis=0)
        return min_X, max_X

    def get_test_data(self):
        return self.x_test, self.y_test

    def get_training_data(self):
        return self.x_train, self.y_train

    def get_labels(self):
        return self.labels

    def get_subpopulation_of_interest(self, columns_of_interest):
        assert self._original_values_of_categories_of_interest_test is not None
        return (self._original_values_of_categories_of_interest_train[columns_of_interest],
                self._original_values_of_categories_of_interest_test[columns_of_interest])

    # def get_number_of_classes(self):
    #     return self.y.shape[1]


class WineDataset(Dataset):
    def __init__(self):
        self.X, self.y = load_wine(return_X_y=True, as_frame=True)
        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)
        self.y, self.labels = one_hot_encode_labels(self.y)
        self.X.rename(columns={"od280/od315_of_diluted_wines": "od_of_diluted_wines"}, inplace=True)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=2)

        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(self.x_train)
        scaled_x_train = min_max_scaler.transform(self.x_train)
        scaled_x_test = min_max_scaler.transform(self.x_test)
        self.x_train = pd.DataFrame(scaled_x_train, columns=self.x_train.columns, index=self.x_train.index)
        self.x_test = pd.DataFrame(scaled_x_test, columns=self.x_test.columns, index=self.x_test.index)

        # self.x_train.reset_index(drop=True, inplace=True)
        # self.x_test.reset_index(drop=True, inplace=True)
        # self.y_train.reset_index(drop=True, inplace=True)
        # self.y_test.reset_index(drop=True, inplace=True)


class IrisDataset(Dataset):
    def __init__(self):
        self.X, self.y = load_iris(return_X_y=True, as_frame=True)
        self.y, self.labels = one_hot_encode_labels(self.y)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=1)


class HumanActivityRecognitionDataset(Dataset):
    def __init__(self):
        training_data = pd.read_csv(directory_of_data + os.sep + "Human_Activity_Recognition" + os.sep + "train.csv")
        test_data = pd.read_csv(directory_of_data + os.sep + "Human_Activity_Recognition" + os.sep + "test.csv")
        training_data.drop('subject', axis=1)
        test_data.drop('subject', axis=1)
        training_data.rename({"angle(tBodyAccJerkMean),gravityMean)": "angle(tBodyAccJerkMean,gravityMean)"},
                             axis=1,
                             inplace=True)
        test_data.rename({"angle(tBodyAccJerkMean),gravityMean)": "angle(tBodyAccJerkMean,gravityMean)"},
                         axis=1,
                         inplace=True)
        merged_index = [number for number in range(len(training_data) + len(test_data))]
        training_data.index = merged_index[:len(training_data)]
        test_data.index = merged_index[len(training_data):]
        # training_data.reset_index(drop=True, inplace=True)
        # test_data.reset_index(drop=True, inplace=True)

        y_train = training_data.pop('Activity')
        y_test = test_data.pop('Activity')
        self.x_train = training_data
        self.x_test = test_data
        self.y_train, self.y_test, self.labels = encode_training_and_test_y_data(y_train, y_test)

        self.X = self.x_train
        self.y = self.y_train

        # TODO: Should be used instead of X = x_train
        # self.X = pd.concat([self.x_train,self.x_test])
        # self.y = pd.concat([self.y_train,self.y_test])


class AdultDataset(Dataset):
    def __init__(self):
        training_data = pd.read_csv(directory_of_data + os.sep + "Adult" + os.sep + "train.csv", skipinitialspace=True)
        test_data = pd.read_csv(directory_of_data + os.sep + "Adult" + os.sep + "test.csv", skipinitialspace=True)
        training_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)
        y_train = training_data.pop('label')
        y_test = test_data.pop('label')
        training_data.drop("education-num", axis=1, inplace=True)
        test_data.drop("education-num", axis=1, inplace=True)
        x_train = training_data
        x_test = test_data

        feature_names_original_order = list(training_data.columns)

        y_train, y_test, self.labels = encode_training_and_test_y_data(y_train, y_test)

        continuous_features = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
        categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                                'relationship', 'race', 'sex', 'native-country']

        x_train[x_train == '?'] = np.nan
        x_test[x_test == '?'] = np.nan

        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(x_train[continuous_features])
        x_train[continuous_features] = min_max_scaler.transform(x_train[continuous_features])
        x_test[continuous_features] = min_max_scaler.transform(x_test[continuous_features])

        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        ct = ColumnTransformer([("categorical_preprocess", categorical_transformer, categorical_features)],
                               remainder="passthrough")

        encoder_ct = ct.fit(x_train)
        x_train = pd.DataFrame(encoder_ct.transform(x_train).toarray(), index=x_train.index,
                               columns=encoder_ct.get_feature_names_out())
        x_test = pd.DataFrame(encoder_ct.transform(x_test).toarray(), index=x_test.index,
                              columns=encoder_ct.get_feature_names_out())

        sorted_columns = list(sorting_columns(list(x_train.columns), feature_names_original_order))

        x_train = x_train.reindex(sorted_columns, axis=1)
        x_test = x_test.reindex(sorted_columns, axis=1)

        x_train_new_columns = []
        for column_name in sorted_columns:
            x_train_new_columns.append(remove_categorical_preprocess_or_remainder_string(
                column_name=column_name,
                continuous_features_names=continuous_features,
                categorical_features_names=categorical_features))

        x_train.columns = x_train_new_columns
        x_test.columns = x_train_new_columns

        # x_train.drop("nan", axis=1, inplace=True)
        # x_test.drop("nan", axis=1, inplace=True)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


class CovertypeDataset(Dataset):
    def __init__(self):
        training_data = pd.read_csv(directory_of_data + os.sep + "Covertype_Forest" + os.sep + "train.csv",
                                    skipinitialspace=True,
                                    index_col=0)
        test_data = pd.read_csv(directory_of_data + os.sep + "Covertype_Forest" + os.sep + "test.csv",
                                skipinitialspace=True,
                                index_col=0)

        training_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)

        y_train = training_data.pop('Cover_Type')
        y_test = test_data.pop('Cover_Type')
        x_train = training_data
        x_test = test_data

        y_train, y_test, self.labels = encode_training_and_test_y_data(y_train, y_test)

        continuous_features = ["Elevation", "Aspect", "Slope", "Distance_To_Hydrology",
                               "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
                               "Horizontal_Distance_To_Fire_Points"]

        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(x_train[continuous_features])
        x_train[continuous_features] = min_max_scaler.transform(x_train[continuous_features])
        x_test[continuous_features] = min_max_scaler.transform(x_test[continuous_features])

        x_train[np.isnan(x_train)] = 0
        x_test[np.isnan(x_test)] = 0

        x_train_size = len(x_train)
        x_test_size = len(x_test)
        index_x_train = [i for i in range(x_train_size)]
        # index_y_train = [i for i in range(x_train_size)]
        index_x_test = [i for i in range(x_train_size, x_train_size + x_test_size)]
        # index_y_test = [i for i in range(x_train_size, x_train_size + x_test_size)]

        x_train.index = index_x_train
        y_train.index = index_x_train
        x_test.index = index_x_test
        y_test.index = index_x_test

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


class HeartDataset(Dataset):
    def __init__(self):
        train_full_dataset = pd.read_csv(directory_of_data + os.sep + "Heart" + os.sep + "Full_Dataset_Train.csv",
                                         skipinitialspace=True,
                                         index_col=0)
        test_full_dataset = pd.read_csv(directory_of_data + os.sep + "Heart" + os.sep + "Full_Dataset_Test.csv",
                                        skipinitialspace=True,
                                        index_col=0)

        train_full_dataset.reset_index(drop=True, inplace=True)
        test_full_dataset.reset_index(drop=True, inplace=True)

        y_train = train_full_dataset.pop("num")
        y_test = test_full_dataset.pop("num")
        x_train = train_full_dataset
        x_test = test_full_dataset

        y_train, y_test, self.labels = encode_training_and_test_y_data(y_train, y_test)

        # CA could be considered categorical feature or numerical. It is a count of distinct values.
        continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

        categorical_string =['sex', 'location', 'cp', 'restecg', 'exang', 'slope', 'thal']
        categorical_numerical = ['fbs', 'ca']

        x_train.drop(categorical_numerical, axis=1, inplace=True)
        x_test.drop(categorical_numerical, axis=1, inplace=True)

        numerical_pipeline = Pipeline(steps=[('imputer_numerical',
                                              SimpleImputer(strategy='mean')),
                                             ('scaler', StandardScaler())])
        # categorical_numerical_pipeline = Pipeline(steps=[('imputer_numerical_categorical',
        #                                                   SimpleImputer(strategy='constant', fill_value=0.0)),
        #                                      ('scaler', MinMaxScaler())])
        categorical_string_pipeline = Pipeline(steps=[('imputer_string_categorical',
                                                       SimpleImputer(strategy='most_frequent')),
                                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        ct = ColumnTransformer(
            [('numerical_pipeline', numerical_pipeline, continuous_features),
             # ('categorical_numerical_pipeline', categorical_numerical_pipeline, categorical_numerical),
             ('categorical_string_pipeline', categorical_string_pipeline, categorical_string)],
            remainder="passthrough", verbose_feature_names_out=False)

        # ct = ColumnTransformer([("categorical_preprocess", OneHotEncoder(), categorical_features)],
        #                        remainder="passthrough")

        # min_max_scaler = MinMaxScaler()
        # min_max_scaler.fit(x_train[continuous_features])
        # x_train[continuous_features] = min_max_scaler.transform(x_train[continuous_features])
        # x_test[continuous_features] = min_max_scaler.transform(x_test[continuous_features])

        encoder_ct = ct.fit(x_train)
        x_train = pd.DataFrame(encoder_ct.transform(x_train), index=x_train.index,
                               columns=encoder_ct.get_feature_names_out())
        x_test = pd.DataFrame(encoder_ct.transform(x_test), index=x_test.index,
                              columns=encoder_ct.get_feature_names_out())

        # x_train[np.isnan(x_train)] = 0
        # x_test[np.isnan(x_test)] = 0

        x_train_size = len(x_train)
        x_test_size = len(x_test)
        index_x_train = [i for i in range(x_train_size)]
        # index_y_train = [i for i in range(x_train_size)]
        index_x_test = [i for i in range(x_train_size, x_train_size + x_test_size)]
        # index_y_test = [i for i in range(x_train_size, x_train_size + x_test_size)]

        x_train.index = index_x_train
        y_train.index = index_x_train
        x_test.index = index_x_test
        y_test.index = index_x_test

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


class HeartDatasetBinary(HeartDataset):
    def __init__(self):
        super().__init__()
        self.y_train.loc[(self.y_train["2"] == 1) | (self.y_train["3"] == 1) | (self.y_train["4"] == 1), "1"] = 1
        self.y_train.drop(["2", "3", "4"], axis=1, inplace=True)
        self.y_test.loc[(self.y_test["2"] == 1) | (self.y_test["3"] == 1) | (self.y_test["4"] == 1), "1"] = 1
        self.y_test.drop(["2", "3", "4"], axis=1, inplace=True)
        # self.y_train["1", self.y_train[(self.y_train["2"] == 1) | (self.y_train["3"] == 1) | (self.y_train["4"] == 1)]] = 1
        # self.y_train.drop(["2", "3", "4"], axis=1, inplace=True)
        # self.y_test["1", (self.y_test["2"] == 1) | (self.y_test["3"] == 1) | (self.y_test["4"] == 1)] = 1
        # self.y_test.drop(["2", "3", "4"], axis=1, inplace=True)


class NewAdultDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        X = pd.read_csv(ROOT_DIR + os.sep + "data" + os.sep + "datasets" + os.sep + "New_Adult" + os.sep + "X.csv",
                        index_col=0)
        y = pd.read_csv(ROOT_DIR + os.sep + "data" + os.sep + "datasets" + os.sep + "New_Adult" + os.sep + "y.csv",
                        index_col=0)
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        y_train, y_test, self.labels = encode_training_and_test_y_data(y_train, y_test)

        # x_train.reset_index(drop=True, inplace=True)
        # x_test.reset_index(drop=True, inplace=True)
        # y_train.reset_index(drop=True, inplace=True)
        # y_test.reset_index(drop=True, inplace=True)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


class HELOCDataset(Dataset):
    def __init__(self):
        super().__init__()
        full_dataset = pd.read_csv(
            ROOT_DIR + os.sep + "data" + os.sep + "datasets" + os.sep + "HELOC" + os.sep + "X.csv")
        y = full_dataset.pop("RiskPerformance")
        X = full_dataset
        y.reset_index(drop=True, inplace=True)
        X.reset_index(drop=True, inplace=True)
        # y = pd.read_csv(ROOT_DIR + os.sep + "data" + os.sep + "datasets" + os.sep + "HELOC" + os.sep + "y.csv",
        #                 index_col=0)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        y_train, y_test, self.labels = encode_training_and_test_y_data(y_train, y_test)

        # x_train.reset_index(drop=True, inplace=True)
        # x_test.reset_index(drop=True, inplace=True)
        # y_train.reset_index(drop=True, inplace=True)
        # y_test.reset_index(drop=True, inplace=True)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


class EdgeIIoTDataset(Dataset):
    def __init__(self):
        super().__init__()

        df = pd.read_csv(
            ROOT_DIR + os.sep + "data" + os.sep + "datasets" + os.sep + 'EdgeIIoT/MLDL/DNN-EdgeIIoT-dataset.csv',
            low_memory=False)

        drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
                        "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp", "http.request.uri.query",
                        "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
                        "http.request.version", "http.referer", "dns.qry.name.len", "mqtt.conack.flags", "icmp.unused",
                        "http.tls_port", "dns.qry.type", 'mqtt.msg_decoded_as']

        df.drop(drop_columns, axis=1, inplace=True)
        df.dropna(axis=0, how='any', inplace=True)
        df.drop_duplicates(subset=None, keep="first", inplace=True)

        y_type = df.pop('Attack_type')
        df.drop("Attack_Label", axis=1, inplace=True)

        categorical_columns = list(df.select_dtypes(include=["object"]).columns)
        numerical_columns = list(df.select_dtypes(include=['int64', "float64"]).columns)
        feature_names_original_order = list(df.columns)

        df["http.request.method"].replace("0", "HTTPMethodNotAvailable", inplace=True)
        df["http.request.method"].replace("0.0", "HTTPMethodNotAvailable", inplace=True)
        df["mqtt.protoname"].replace("0", "ProtonameNotAvailable", inplace=True)
        df["mqtt.protoname"].replace("0.0", "ProtonameNotAvailable", inplace=True)
        df["mqtt.topic"].replace("0", "TopicNotAvailable", inplace=True)
        df["mqtt.topic"].replace("0.0", "TopicNotAvailable", inplace=True)

        x_train, x_test, y_train, y_test = train_test_split(df, y_type, test_size=0.3, random_state=1)

        y_train, y_test, self.labels = encode_training_and_test_y_data(y_train, y_test)

        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(x_train[numerical_columns])
        x_train[numerical_columns] = min_max_scaler.transform(x_train[numerical_columns])
        x_test[numerical_columns] = min_max_scaler.transform(x_test[numerical_columns])

        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        ct = ColumnTransformer([("categorical_preprocess", categorical_transformer, categorical_columns)],
                               remainder="passthrough")

        encoder_ct = ct.fit(x_train)
        x_train = pd.DataFrame(encoder_ct.transform(x_train), index=x_train.index,
                               columns=encoder_ct.get_feature_names_out())
        x_test = pd.DataFrame(encoder_ct.transform(x_test), index=x_test.index,
                              columns=encoder_ct.get_feature_names_out())

        sorted_columns = list(sorting_columns(list(x_train.columns), feature_names_original_order))

        x_train = x_train.reindex(sorted_columns, axis=1)
        x_test = x_test.reindex(sorted_columns, axis=1)

        x_train_new_columns = []
        for column_name in sorted_columns:
            x_train_new_columns.append(remove_categorical_preprocess_or_remainder_string(
                column_name=column_name,
                continuous_features_names=numerical_columns,
                categorical_features_names=categorical_columns))

        x_train.columns = x_train_new_columns
        x_test.columns = x_train_new_columns

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


class EdgeIIoTCoresetDataset(Dataset):
    def __init__(self):
        super().__init__()

        df = pd.read_csv(
            ROOT_DIR + os.sep + "data" + os.sep + "datasets" + os.sep + 'EdgeIIoT/MLDL/IIOT_DNN_Coreset_Reduced.csv',
            index_col=0, low_memory=False)

        df.reset_index(drop=True, inplace=True)

        y_type = df.pop('Attack_type')

        x_train, x_test, y_train, y_test = train_test_split(df, y_type, test_size=0.3, random_state=1)

        y_train, y_test, self.labels = encode_training_and_test_y_data(y_train, y_test)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


class ElectricConsumptionDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(directory_of_data + os.sep + "ElectricConsumption" + os.sep + "train.csv")
        # test_full_dataset = pd.read_csv(directory_of_data + os.sep + "ElectricConsumption" + os.sep + "test.csv")

        df.drop("id", axis=1, inplace=True)
        y_type = df.pop("site_eui")

        x_train, x_test, y_train, y_test = train_test_split(df, y_type, test_size=0.3, random_state=1)

        numerical_pipeline = Pipeline(steps=[('inputer_numerical', SimpleImputer(strategy='mean')),
                                             ('scaler', MinMaxScaler())])
        categorical_pipeline = Pipeline(steps=[('imputer_categorical', SimpleImputer(strategy='most_frequent')),
                                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        ct = ColumnTransformer(
            [('numerical_pipeline', numerical_pipeline, make_column_selector(dtype_include=["int", "float"])),
             ('categorical_pipeline', categorical_pipeline, make_column_selector(dtype_include=["object", "category"]))],
            remainder="passthrough", verbose_feature_names_out=False)

        encoder = ct.fit(x_train)
        x_train = pd.DataFrame(encoder.transform(x_train), index=x_train.index,
                               columns=encoder.get_feature_names_out())
        x_test = pd.DataFrame(encoder.transform(x_test), index=x_test.index,
                              columns=encoder.get_feature_names_out())

        x_train.sort_index(inplace=True)
        x_test.sort_index(inplace=True)
        y_train.sort_index(inplace=True)
        y_test.sort_index(inplace=True)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

from sklearn.model_selection import GroupShuffleSplit

class WidsEnergyDataset(Dataset):
    GROUP_COLS = ['State_Factor', 'building_class', 'facility_type', 'floor_area', 'year_built']

    def __init__(self, random_seed=1, test_size=0.3):
        df = pd.read_csv(directory_of_data + os.sep + "ElectricConsumption" + os.sep + "train.csv")
        df = df.reset_index(drop=True)

        # --- 1. row-local weather compression: pure per-row function, safe pre-split ---
        df = self._add_weather_row_stats(df)

        # --- 2. synthetic building id, used both for the split and later for FL partitioning ---
        df['building_group'] = df[self.GROUP_COLS].astype(str).agg('_'.join, axis=1)

        # --- 3. GROUP-safe split: a building's full row history goes entirely to one side ---
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_idx, test_idx = next(splitter.split(df, groups=df['building_group']))

        df_train = df.iloc[train_idx].sort_values(self.GROUP_COLS + ['Year_Factor']).copy()
        df_test  = df.iloc[test_idx].sort_values(self.GROUP_COLS + ['Year_Factor']).copy()

        self._original_values_of_categories_of_interest_train = df_train[['State_Factor', 'building_class', 'facility_type']].copy()
        self._original_values_of_categories_of_interest_test = df_test[['State_Factor', 'building_class', 'facility_type']].copy()

        # --- 4. lag/delta features, computed independently per split (groups never cross the split) ---
        df_train = self._add_lag_delta_features(df_train)
        df_test  = self._add_lag_delta_features(df_test)

        # --- 5. cohort (weather-station) stats: FIT on train, TRANSFORM test ---
        df_train, df_test = self._fit_transform_cohort_stats(df_train, df_test)

        # keep raw (pre-encoding) group keys for manual FL partitioning later
        self.groups_train = df_train['building_group'].copy()
        self.groups_test  = df_test['building_group'].copy()
        self.state_train  = df_train['State_Factor'].copy()
        self.state_test   = df_test['State_Factor'].copy()

        df_train.drop(columns=['id', 'building_group'], inplace=True, errors='ignore')
        df_test.drop(columns=['id', 'building_group'], inplace=True, errors='ignore')

        y_train = df_train.pop('site_eui')
        y_test  = df_test.pop('site_eui')
        x_train, x_test = df_train, df_test

        # --- 6. standard impute/scale/one-hot, fit on train only ---
        numerical_pipeline = Pipeline(steps=[('inputer_numerical', SimpleImputer(strategy='mean')),
                                              ('scaler', MinMaxScaler())])
        categorical_pipeline = Pipeline(steps=[('imputer_categorical', SimpleImputer(strategy='most_frequent')),
                                                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        ct = ColumnTransformer(
            [('numerical_pipeline', numerical_pipeline, make_column_selector(dtype_include=["int", "float"])),
             ('categorical_pipeline', categorical_pipeline, make_column_selector(dtype_include=["object", "category"]))],
            remainder="passthrough", verbose_feature_names_out=False)

        encoder = ct.fit(x_train)
        x_train = pd.DataFrame(encoder.transform(x_train), index=x_train.index, columns=encoder.get_feature_names_out())
        x_test  = pd.DataFrame(encoder.transform(x_test),  index=x_test.index,  columns=encoder.get_feature_names_out())

        x_train.sort_index(inplace=True); x_test.sort_index(inplace=True)
        y_train.sort_index(inplace=True); y_test.sort_index(inplace=True)

        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

    @staticmethod
    def _add_weather_row_stats(df):
        min_temp = [c for c in df.columns if 'min_temp' in c]
        max_temp = [c for c in df.columns if 'max_temp' in c]
        avg_temp = [c for c in df.columns if 'avg_temp' in c]
        for prefix, cols in [('min_temp', min_temp), ('max_temp', max_temp), ('avg_temp', avg_temp)]:
            df[f'{prefix}1'] = df[cols].min(axis=1)
            df[f'{prefix}2'] = df[cols].max(axis=1)
            df[f'{prefix}3'] = df[cols].mean(axis=1)
            df[f'{prefix}4'] = df[cols].std(axis=1)
            df[f'{prefix}5'] = df[cols].skew(axis=1)
        df.attrs['min_temp_cols'] = min_temp
        return df

    @classmethod
    def _add_lag_delta_features(cls, df_split):
        df_split = df_split.copy()
        for i in [1, 2, 3]:
            df_split[f'site_eui_lag{i}']           = df_split.groupby(cls.GROUP_COLS)['site_eui'].shift(i)
            df_split[f'energy_star_rating_lag{i}'] = df_split.groupby(cls.GROUP_COLS)['energy_star_rating'].shift(i)
            df_split[f'delta_energy{i}'] = np.where(
                df_split['energy_star_rating'].notnull() & df_split[f'energy_star_rating_lag{i}'].notnull(),
                (df_split['energy_star_rating'] - df_split[f'energy_star_rating_lag{i}']) / df_split['energy_star_rating'],
                np.nan)
        return df_split

    @staticmethod
    def _fit_transform_cohort_stats(df_train, df_test):
        min_temp = df_train.attrs['min_temp_cols']
        grp1 = df_train.groupby(min_temp, as_index=True).agg({
            "floor_area": ['mean', 'max', 'min', 'sum'],
            "year_built": ['mean', 'median', 'max', 'min'],
            "energy_star_rating": ['min', 'max', 'mean', 'median'],
            "id": "count"
        })
        grp1.columns = ["_".join(x) for x in grp1.columns.ravel()]
        # fit on train, apply to both — test rows whose station-cohort wasn't seen in train get NaN,
        # which the numerical imputer downstream will handle
        df_train_index, df_test_index = df_train.index, df_test.index  # NEW
        df_train = df_train.merge(grp1, on=min_temp, how='left')
        df_train.index = df_train_index  # NEW — restore pre-merge row labels
        df_test = df_test.merge(grp1, on=min_temp, how='left')
        df_test.index = df_test_index  # NEW
        return df_train, df_test


class TimeSeriesDataset:
    df: DataFrame

    def get_full_dataset(self) -> DataFrame:
        return self.df


class AirQualityDataset(TimeSeriesDataset):
    def __init__(self):
        super().__init__()

        df = pd.read_csv(
            ROOT_DIR + os.sep + "data" + os.sep + "datasets" + os.sep + 'DatosHistoricosQAir/dataframe_for_ML.csv',
            index_col=0, low_memory=False)

        columns_to_scale = ["Temperature", "Precipitations", "Racha", "Max_speed", "CO(µg/m3)"]

        min_max_scaler = MinMaxScaler()
        df[columns_to_scale] = min_max_scaler.fit_transform(df[columns_to_scale])

        self.df = df

# if __name__ == "__main__":
#     X_train, y_train = ElectricConsumptionDataset().get_training_data()
