import math
from logging import INFO
from typing import List, Union, Optional, Tuple

import numpy
import numpy as np
import pandas as pd
import xgboost
from flwr.common import log
from pandas import DataFrame
from sklearn.metrics import log_loss, accuracy_score, matthews_corrcoef, f1_score, root_mean_squared_error, \
    mean_absolute_error, r2_score, mean_squared_error
from torch.utils.data import DataLoader
from xgboost import DMatrix

from experiment_parameters.model_builder.Model import Model, XGBoostModel
from metrics.Metrics import DictOfMetrics, Accuracy, CrossEntropyLoss, SVCompatibleF1Score, \
    SVCompatibleMatthewsCorrelationCoefficient, F1ScoreMacro, F1ScoreMicro, RMSE, MAE, ConfusionMatrix, \
    AggregatableMeasuresClassification, CrossEntropyLossPartialMeasure, AggregatableMeasures, MSEPartialMeasure, \
    MAEPartialMeasure, MSE, AggregatableMeasuresRegression, SubpopulationDict, GLOBAL_KEY, \
    group_indices_by_subpopulation


def return_labels(y_true_argmaxed, labels):
    # y_true_argmaxed = np.reshape(y_true_argmaxed, newshape=(-1, 1))
    # print(np.apply_along_axis(lambda x: print(x), 0, y_true_argmaxed))
    return [labels[selected_class] for selected_class in y_true_argmaxed]

def cross_entropy_loss(y_test, y_pred_proba, labels) -> CrossEntropyLoss:
    # log(INFO, "CE Loss")
    ground_truth_np = return_labels(np.argmax(y_test, axis=1), labels)
    return CrossEntropyLoss(log_loss(ground_truth_np, y_pred_proba, labels=labels))

computation_type_to_function = {
    "CrossEntropyLoss": cross_entropy_loss,
    "ConfusionMatrix": ConfusionMatrix,
    "MSE": MSEPartialMeasure,
    "MAE": MAEPartialMeasure
    # "R2": r2
}

metric_name_to_computation_type_dict = {
    "CrossEntropyLoss": "CrossEntropyLoss",
    "Accuracy": "ConfusionMatrix",
    "F1Score": "ConfusionMatrix",
    "F1ScoreMacro": "ConfusionMatrix",
    "F1ScoreMicro": "ConfusionMatrix",
    "MCC": "ConfusionMatrix",
    "RMSE": "MSE",
    "MSE": "MSE",
    "MAE": "MAE",
    # "R2": "R2"
}

metric_name_to_metric = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "Accuracy": Accuracy,
    "F1Score": SVCompatibleF1Score,
    "F1ScoreMacro": F1ScoreMacro,
    "F1ScoreMicro": F1ScoreMicro,
    "MCC": SVCompatibleMatthewsCorrelationCoefficient,
    "RMSE": RMSE,
    "MSE": MSE,
    "MAE": MAE,
    # "R2": "R2"
}


def _single_group_partial_measure(y_pred, y_test_values, output_columns):
    # unchanged body of your original partial_computation
    if len(output_columns) > 1:
        partial_ce_loss = CrossEntropyLossPartialMeasure(y_test_values, y_pred)
        partial_confusion_matrix = ConfusionMatrix(output_columns, y_test_values, y_pred)
        return AggregatableMeasuresClassification(output_columns, partial_ce_loss, partial_confusion_matrix)
    else:
        partial_mse = MSEPartialMeasure(y_test_values, y_pred)
        partial_mae = MAEPartialMeasure(y_test_values, y_pred)
        return AggregatableMeasuresRegression(partial_mse, partial_mae)


def partial_computation(y_pred: np.ndarray,
                        y_test_values: np.ndarray,
                        output_columns: list[str],
                        subpopulation_values: Optional[Union[pd.Series, pd.DataFrame]] = None):
    y_pred = np.asarray(y_pred)              # NEW — normalizes DataFrame/Series/ndarray to plain ndarray
    y_test_values = np.asarray(y_test_values)  # NEW

    global_measure = _single_group_partial_measure(y_pred, y_test_values, output_columns)

    if subpopulation_values is None:
        return global_measure  # exactly the old behavior, unchanged type

    result = {GLOBAL_KEY: global_measure}
    for key, idx in group_indices_by_subpopulation(subpopulation_values).items():
        result[key] = _single_group_partial_measure(y_pred[idx], y_test_values[idx], output_columns)

    return SubpopulationDict(result)


def evaluator(partial_measurements: Union[AggregatableMeasures, SubpopulationDict], metric_list):
    if isinstance(partial_measurements, SubpopulationDict):
        return SubpopulationDict({key: evaluator(value, metric_list)
                                  for key, value in partial_measurements.items()})

    metric_dict = {}
    for metric_computation in metric_list:
        metric_dict[metric_computation] = metric_name_to_metric[metric_computation](partial_measurements)
    return DictOfMetrics(metric_dict)