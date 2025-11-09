import ast
import os
from logging import INFO

import pandas as pd
from flwr.common import log

type_of_metric = {
    "Accuracy": "Single",
    "CrossEntropyLoss": "Single",
    "F1Score": "Multiple",
    "F1ScoreMacro": "Single",
    "F1ScoreMicro": "Single",
    "AUC": "Single",
    "MCC": "Single",
    "CosineSimilarity": "Single", # Only in SV
    "RMSE": "Single",
    "RSE": "Single",
    "MAE": "Single",
    "R2": "Single"
}


def get_aggregated_sv_clients(dataframe_sv):
    if isinstance(dataframe_sv.index, pd.MultiIndex):
        for label, new_df in dataframe_sv.groupby(level=1):
            dataframe_sv.loc[("aggregated", label), :] = (
                    new_df.iloc[:-1].sum(axis="rows") / new_df.iloc[:-1].shape[0]
            )
    else:
        dataframe_sv.loc["aggregated"] = (
                dataframe_sv.iloc[:-1].sum(axis="rows") / dataframe_sv.iloc[:-1].shape[0]
        )
    return dataframe_sv


class FlowerMetricManager:
    metric_list: list[str]
    client_list: list[str]
    number_of_rounds: int
    list_classes: list[str]
    evaluation_results = {}

    def __init__(self, metric_list, client_list, number_of_rounds, classes):
        self.metric_list = metric_list
        # if metric_list is None:
        #     self.metric_list = ["CrossEntropyLoss", "Accuracy", "AUC"]
        # else:
        #     self.metric_list += ["CrossEntropyLoss", "Accuracy"]  # Quickfix, to work with Flower.
        self.client_list = client_list
        self.number_of_rounds = number_of_rounds
        self.list_classes = classes
        log(INFO, f"Classes: {self.list_classes}")

        for metric in self.metric_list:  # FML is used to separate evaluation results from the SV results.
            if type_of_metric[metric] == "Single":
                self.evaluation_results["Evaluation_" + metric] = (
                    pd.DataFrame(index=[i for i in range(1, number_of_rounds + 1)],
                                 columns=self.client_list + ["Global", "Aggregated"]))
                self.evaluation_results["Evaluation_" + metric].sort_index(inplace=True)
            elif type_of_metric[metric] == "Multiple":
                evaluating_clients = self.client_list + ["Global", "Aggregated"]
                multiIndex_Columns = pd.MultiIndex.from_product([evaluating_clients,
                                                                 self.list_classes])
                self.evaluation_results["Evaluation_" + metric] = (
                    pd.DataFrame(index=[i for i in range(1, number_of_rounds + 1)],
                                 columns=multiIndex_Columns))
                self.evaluation_results["Evaluation_" + metric].sort_index(inplace=True)
            else:
                raise Exception(f"Metric {metric} is not supported. Please, add it into the dictionary"
                                f"of file ResultManager.")

    def add_result(self, metric, client, round, value):
        # log(INFO, f"Metric for round {round}.  Metric: {metric} \t Value: {value}")
        # if type(value) is str:
        #     evaluted_metric = ast.literal_eval(metric)
        #     log(INFO, f"Type of metric after eval: {type(evaluted_metric)}")
        if type(value) is list:
            for label in range(len(value)):
                self.evaluation_results["Evaluation_" + metric].loc[round, (client, self.list_classes[label])] = value[
                    label]
        else:
            self.evaluation_results["Evaluation_" + metric].loc[round, client] = value

    def get_global_dataframes(self):
        return self.evaluation_results

    def save_dataframes_as_csv(self, path):
        evaluation_dir_path = path + os.sep + "Evaluation"
        os.makedirs(evaluation_dir_path, exist_ok=True)
        for metric, dataframe in self.evaluation_results.items():
            dataframe.to_csv(evaluation_dir_path + os.sep + metric, float_format='%.15f')


class SVCompatibleFlowerMetricManager(FlowerMetricManager):
    sv_results = {}

    def __init__(self, metric_list, client_list, number_of_rounds, classes):
        super().__init__(metric_list, client_list, number_of_rounds, classes)
        metrics_and_sv_methods = self.metric_list + ["CosineSimilarity"]
        for metric in metrics_and_sv_methods:  # FML is used to separate evaluation results from the SV results.
            if type_of_metric[metric] == "Single":
                multiple_index_one_class = pd.MultiIndex.from_product(
                    [[i for i in range(number_of_rounds + 1)], client_list + ["Centralized", "Aggregated"]],
                    names=["Round", "Evaluator"],
                )
                self.sv_results["SV_" + metric] = (
                    pd.DataFrame(index=multiple_index_one_class,
                                 columns=client_list))
                self.sv_results["SV_" + metric].sort_index(inplace=True)
            elif type_of_metric[metric] == "Multiple":
                multiple_index_multiple_classes = pd.MultiIndex.from_product(
                    [[i for i in range(number_of_rounds + 1)],
                     client_list + ["Centralized", "Aggregated"],
                     self.list_classes],
                    names=["Round", "Evaluator", "Classes"],
                )
                self.sv_results["SV_" + metric] = (
                    pd.DataFrame(index=multiple_index_multiple_classes,
                                 columns=client_list))
                self.sv_results["SV_" + metric].sort_index(inplace=True)
            else:
                raise Exception(f"Metric {metric} is not supported. Please, add it into the dictionary"
                                f"of file ResultManager.")
            # self.evaluation_results["SV_Centralized_" + metric] = (
            #     pd.DataFrame(index=[i for i in range(number_of_rounds + 1)],
            #                  columns=["Global"]))
            # self.evaluation_results["SV_Decentralized_" + metric] = (
            #     pd.DataFrame(index=[i for i in range(number_of_rounds + 1)],
            #                  columns=["Global"]))

    def add_shapley_value(self, metric, evaluating_client, evaluated_client, round, value):
        # log(INFO, f"Shapley Values for round {round},"
        #           f" from evaluating client {evaluating_client}"
        #           f" to evaluated client {evaluated_client}.  Metric: {metric} \t Value: {value}")
        # log(INFO, f"Type of metric: {type(metric)}")
        # log(INFO, f"Type of metric: {type(ast.literal_eval(metric))}")
        if type(value) is list:
            for label in range(len(value)):
                self.sv_results["SV_" + metric].loc[
                    (round, evaluating_client, self.list_classes[label]), evaluated_client
                ] = value[label]
        else:
            self.sv_results["SV_" + metric].loc[
                (round, evaluating_client), evaluated_client
            ] = value

    # def process_shapley_value_dict(self, dictionary, evaluating_client, round):
    #     for client, sv in dictionary.items():
    #         metric_dict = sv.return_flower_dict()
    #         for metric, value in metric_dict.items():
    #             self.add_shapley_value(metric, evaluating_client, client, round, value)

    def get_sv_dataframes(self):
        return self.sv_results

    def save_dataframes_as_csv(self, path):
        super().save_dataframes_as_csv(path)
        sv_dir_path = path + os.sep + "Shapley_Value"
        os.makedirs(sv_dir_path, exist_ok=True)
        for metric, dataframe in self.get_sv_dataframes().items():
            dataframe.to_csv(sv_dir_path + os.sep + metric, float_format='%.15f')

# class MetricManager(ABC):
#     metric_list: List[str]
#     client_list: List[str]
#     number_of_rounds: int
#
#     def __init__(self, metric_list, client_list, number_of_rounds):
#         self.metric_list = metric_list
#         self.client_list = client_list
#         self.number_of_rounds = number_of_rounds
#
#     def add_metric_value(self, evaluator, metric, value):
#         pass
#
#
# class FlowerMetricManager(MetricManager):
#     evaluation_results = {}
#
#     def __init__(self, metric_list, client_list, number_of_rounds):
#         super().__init__(metric_list, client_list, number_of_rounds)
#
#         for metric in metric_list:
#             self.evaluation_results["Centralized_" + metric] = (
#                 pd.DataFrame(index=[i for i in range(number_of_rounds + 1)],
#                              columns=["Global"]))
#             self.evaluation_results["Decentralized_" + metric] = (
#                 pd.DataFrame(index=[i for i in range(number_of_rounds + 1)],
#                              columns=["Global"]))
#
#
# class SVMetricManager(MetricManager):
#     shapley_values_evaluation_results = {}


