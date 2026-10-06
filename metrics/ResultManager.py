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
    "CosineSimilarity": "Single",  # Only in SV
    "RMSE": "Single",
    "MSE": "Single",
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

    def __init__(self, metric_list, client_list, number_of_rounds, classes):
        self.metric_list = metric_list
        self.client_list = client_list
        self.number_of_rounds = number_of_rounds
        self.list_classes = classes
        log(INFO, f"Classes: {self.list_classes}")

        self.evaluation_results = self._create_evaluation_dataframes()
        self._subpopulation_rows = []

    def _create_evaluation_dataframes(self):
        dataframes = {}
        for metric in self.metric_list:
            if type_of_metric[metric] == "Single":
                dataframes["Evaluation_" + metric] = pd.DataFrame(
                    index=[i for i in range(1, self.number_of_rounds + 1)],
                    columns=self.client_list + ["Global", "Aggregated"])
                dataframes["Evaluation_" + metric].sort_index(inplace=True)
            elif type_of_metric[metric] == "Multiple":
                evaluating_clients = self.client_list + ["Global", "Aggregated"]
                multiIndex_Columns = pd.MultiIndex.from_product([evaluating_clients, self.list_classes])
                dataframes["Evaluation_" + metric] = pd.DataFrame(
                    index=[i for i in range(1, self.number_of_rounds + 1)],
                    columns=multiIndex_Columns)
                dataframes["Evaluation_" + metric].sort_index(inplace=True)
            else:
                raise Exception(f"Metric {metric} is not supported. Please, add it into the dictionary"
                                f"of file ResultManager.")
        return dataframes

    def _write_result(self, dataframes, metric, client, round, value):
        if type(value) is list:
            for label in range(len(value)):
                dataframes["Evaluation_" + metric].loc[round, (client, self.list_classes[label])] = value[label]
        else:
            dataframes["Evaluation_" + metric].loc[round, client] = value

    def add_result(self, metric, client, round, value):
        self._write_result(self.evaluation_results, metric, client, round, value)

    def add_subpopulation_result(self, subpopulation_key, metric, client, round, value):
        if type(value) is list:
            for label in range(len(value)):
                self._subpopulation_rows.append({
                    "Round": round, "Evaluator": client, "Subpopulation": subpopulation_key,
                    "Metric": metric, "Class": self.list_classes[label], "Value": value[label]
                })
        else:
            self._subpopulation_rows.append({
                "Round": round, "Evaluator": client, "Subpopulation": subpopulation_key,
                "Metric": metric, "Class": None, "Value": value
            })

    def get_global_dataframes(self):
        return self.evaluation_results

    def get_subpopulation_dataframe(self):
        return pd.DataFrame(self._subpopulation_rows)

    def save_dataframes_as_csv(self, path, last_valid_round: int):
        evaluation_dir_path = path + os.sep + "Evaluation"
        os.makedirs(evaluation_dir_path, exist_ok=True)
        for metric, dataframe in self.evaluation_results.items():
            dataframe.loc[:last_valid_round].to_csv(evaluation_dir_path + os.sep + metric, float_format='%.15f')

        subpop_df = self.get_subpopulation_dataframe()
        if not subpop_df.empty:
            subpop_df = subpop_df[subpop_df["Round"] <= last_valid_round]
            subpop_df.to_csv(evaluation_dir_path + os.sep + "Evaluation_Subpopulations.csv",
                             index=False, float_format='%.15f')


class SVCompatibleFlowerMetricManager(FlowerMetricManager):

    def __init__(self, metric_list, client_list, number_of_rounds, classes):
        super().__init__(metric_list, client_list, number_of_rounds, classes)
        # CHANGED: instance attributes, not class-level -- the old `sv_results = {}` /
        # `subpopulation_sv_results = {}` at class scope would be one shared dict across
        # every instance of this class (a classic mutable-default-style bug); harmless while
        # only one experiment ran at a time, but worth fixing now that this file is being
        # touched anyway, before it causes cross-experiment bleed under Ray's process reuse.
        self.sv_results = self._create_sv_dataframes()
        self.subpopulation_sv_results = {}  # NEW: {subpop_key: {"SV_<metric>": DataFrame}}, lazy per key

    def _create_sv_dataframes(self):
        """Same SV DataFrame layout used at dataset level, factored out so a
        subpopulation gets an identical set of DataFrames on first use."""
        dataframes = {}
        metrics_and_sv_methods = self.metric_list + ["CosineSimilarity"]
        for metric in metrics_and_sv_methods:
            if type_of_metric[metric] == "Single":
                multiple_index_one_class = pd.MultiIndex.from_product(
                    [[i for i in range(self.number_of_rounds + 1)], self.client_list + ["Centralized", "Aggregated"]],
                    names=["Round", "Evaluator"],
                )
                dataframes["SV_" + metric] = pd.DataFrame(index=multiple_index_one_class, columns=self.client_list)
                dataframes["SV_" + metric].sort_index(inplace=True)
            elif type_of_metric[metric] == "Multiple":
                multiple_index_multiple_classes = pd.MultiIndex.from_product(
                    [[i for i in range(self.number_of_rounds + 1)],
                     self.client_list + ["Centralized", "Aggregated"],
                     self.list_classes],
                    names=["Round", "Evaluator", "Classes"],
                )
                dataframes["SV_" + metric] = pd.DataFrame(index=multiple_index_multiple_classes, columns=self.client_list)
                dataframes["SV_" + metric].sort_index(inplace=True)
            else:
                raise Exception(f"Metric {metric} is not supported. Please, add it into the dictionary"
                                f"of file ResultManager.")
        return dataframes

    def _parse_subpopulation_metric(self, metric):
        """
        A SubpopulationDict's return_flower_dict() flattens keys as
        "<subpop_key>__<metric_name>". Metric names ("MSE", "F1Score", ...)
        never contain "__" themselves, so splitting on the LAST occurrence
        reliably recovers (subpop_key, real_metric_name) even when the
        subpopulation key itself contains underscores (e.g. "__global__" or
        a raw category value like "Health_Care_Outpatient_Clinic").
        Returns (None, metric) unchanged when metric isn't a flattened form.
        """
        if "__" in metric:
            prefix, _, suffix = metric.rpartition("__")
            if suffix in self.metric_list or suffix == "CosineSimilarity":
                return prefix, suffix
        return None, metric

    def add_shapley_value(self, metric, evaluating_client, evaluated_client, round, value):
        subpopulation_key, real_metric = self._parse_subpopulation_metric(metric)

        if subpopulation_key is not None:
            if subpopulation_key not in self.subpopulation_sv_results:
                self.subpopulation_sv_results[subpopulation_key] = self._create_sv_dataframes()
            dataframes = self.subpopulation_sv_results[subpopulation_key]
        else:
            dataframes = self.sv_results

        if type(value) is list:
            for label in range(len(value)):
                dataframes["SV_" + real_metric].loc[
                    (round, evaluating_client, self.list_classes[label]), evaluated_client
                ] = value[label]
        else:
            dataframes["SV_" + real_metric].loc[
                (round, evaluating_client), evaluated_client
            ] = value

    def get_sv_dataframes(self):
        return self.sv_results

    def save_dataframes_as_csv(self, path, last_valid_round: int):
        super().save_dataframes_as_csv(path, last_valid_round)
        sv_dir_path = path + os.sep + "Shapley_Value"
        os.makedirs(sv_dir_path, exist_ok=True)
        for metric, dataframe in self.get_sv_dataframes().items():
            dataframe.loc[:last_valid_round].to_csv(sv_dir_path + os.sep + metric, float_format='%.15f')

        # NEW: one combined file per SV metric, mirroring the Evaluation_<metric>_Subpopulations.csv
        # shape -- (Round, Evaluator[, Classes]) index plus a Subpopulation column, one row per
        # subpopulation key that actually occurred.
        metrics_and_sv_methods = self.metric_list + ["CosineSimilarity"]
        for metric in metrics_and_sv_methods:
            key = "SV_" + metric
            pieces = []
            for subpopulation_key, dataframes in self.subpopulation_sv_results.items():
                df = dataframes[key]
                df = df.loc[df.index.get_level_values("Round") <= last_valid_round].copy()
                df = df.reset_index()
                df["Subpopulation"] = str(subpopulation_key)
                pieces.append(df)
            if pieces:
                index_cols = ["Round", "Evaluator", "Subpopulation"] if "Classes" not in pieces[0].columns \
                    else ["Round", "Evaluator", "Classes", "Subpopulation"]
                combined = pd.concat(pieces, ignore_index=True).set_index(index_cols)
                combined.to_csv(sv_dir_path + os.sep + key + "_Subpopulations.csv", float_format='%.15f')