from logging import INFO
from typing import List, Union, Optional
import numpy as np
import xgboost
from flwr.common import log
from pandas import DataFrame
from sklearn.metrics import log_loss, accuracy_score, matthews_corrcoef, f1_score, root_mean_squared_error, \
    mean_absolute_error, r2_score, mean_squared_error
from xgboost import DMatrix

from experiment_parameters.model_builder.Model import Model, XGBoostModel
from metrics.Metrics import DictOfMetrics, Accuracy, CrossEntropyLoss, SVCompatibleF1Score, \
    SVCompatibleMatthewsCorrelationCoefficient, F1ScoreMacro, F1ScoreMicro, RMSE, R2, MAE


def return_labels(y_true_argmaxed, labels):
    # y_true_argmaxed = np.reshape(y_true_argmaxed, newshape=(-1, 1))
    # print(np.apply_along_axis(lambda x: print(x), 0, y_true_argmaxed))
    return [labels[selected_class] for selected_class in y_true_argmaxed]


def cross_entropy_loss(y_test, y_pred_proba, labels) -> CrossEntropyLoss:
    # log(INFO, "CE Loss")
    ground_truth_np = return_labels(np.argmax(y_test, axis=1), labels)
    return CrossEntropyLoss(log_loss(ground_truth_np, y_pred_proba, labels=labels))


def accuracy(y_test, y_pred, labels) -> Accuracy:
    # log(INFO, "Accuracy")
    y_pred = np.argmax(y_pred, axis=1)
    ground_truth_np = np.argmax(y_test, axis=1)
    return Accuracy(accuracy_score(ground_truth_np, y_pred))


def f1_score_local(y_test, y_pred_proba, labels) -> SVCompatibleF1Score:
    # log(INFO, "F1 Score")
    ground_truth_np = return_labels(np.argmax(y_test, axis=1), labels)
    predictions_np = return_labels(np.argmax(y_pred_proba, axis=1), labels)
    # print(f1_score(ground_truth_np, predictions_np, labels=labels, average=None, zero_division=0))
    return SVCompatibleF1Score(f1_score(ground_truth_np, predictions_np, labels=labels, average=None, zero_division=0))


def f1_score_micro(y_test, y_pred_proba, labels) -> F1ScoreMicro:
    # log(INFO, "F1 Score")
    ground_truth_np = return_labels(np.argmax(y_test, axis=1), labels)
    predictions_np = return_labels(np.argmax(y_pred_proba, axis=1), labels)
    # print(f1_score(ground_truth_np, predictions_np, labels=labels, average=None, zero_division=0))
    return F1ScoreMicro(f1_score(ground_truth_np, predictions_np, labels=labels, average="micro", zero_division=0))


def f1_score_macro(y_test, y_pred_proba, labels) -> F1ScoreMacro:
    # log(INFO, "F1 Score")
    ground_truth_np = return_labels(np.argmax(y_test, axis=1), labels)
    predictions_np = return_labels(np.argmax(y_pred_proba, axis=1), labels)
    # print(f1_score(ground_truth_np, predictions_np, labels=labels, average=None, zero_division=0))
    return F1ScoreMacro(f1_score(ground_truth_np, predictions_np, labels=labels, average="macro", zero_division=0))


def mcc(y_test, y_pred, labels) -> SVCompatibleMatthewsCorrelationCoefficient:
    # log(INFO, "MCC")
    y_pred = np.argmax(y_pred, axis=1)
    ground_truth_np = np.argmax(y_test, axis=1)
    return SVCompatibleMatthewsCorrelationCoefficient(matthews_corrcoef(ground_truth_np, y_pred))


def rmse(y_test, y_pred, labels=None):
    return RMSE(root_mean_squared_error(y_test, y_pred))


def rse(y_test, y_pred, labels=None):
    return RMSE(mean_squared_error(y_test, y_pred))


def mae(y_test, y_pred, labels=None):
    return MAE(mean_absolute_error(y_test, y_pred))


def r2(y_test, y_pred, labels=None):
    return R2(r2_score(y_test, y_pred))


metric_function_dict = {
    "CrossEntropyLoss": cross_entropy_loss,
    "Accuracy": accuracy,
    "F1Score": f1_score_local,
    "F1ScoreMacro": f1_score_macro,
    "F1ScoreMicro": f1_score_micro,
    "MCC": mcc,
    "RMSE": rmse,
    "RSE": rse,
    "MAE": mae,
    "R2": r2
}


def evaluator(x_test: Union[DataFrame, DMatrix], y_test, model: Model, metric_list: List[str]):
    # log(INFO, "Evaluating")
    metric_dict = DictOfMetrics()
    columns: Optional[list] = None
    try:
        columns = list(y_test.columns)
    except:
        columns = None

    if type(model) is XGBoostModel:
        if columns is None or len(columns) == 1:
            y_pred = model.predict(x_test)
        else:
            y_pred = model.predict_proba(x_test)
    else:
        if columns is None or len(columns) == 1:
            y_pred = model.predict(x_test)
        else:
            y_pred = model.predict_proba(x_test)

    for metric in metric_list:
        # TODO: Replace in the dockerize version with the input parameters
        metric_dict.add_metric(metric_function_dict[metric](y_test, y_pred, labels=columns))

    return metric_dict
