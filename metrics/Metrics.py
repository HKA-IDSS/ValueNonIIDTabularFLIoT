import math
from abc import ABC
from logging import INFO
from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
from flwr.common import log
from sklearn.metrics import log_loss, mean_squared_error, mean_absolute_error


class PartialMeasure(ABC):

    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    # def __mul__(self, constant):
    #     raise NotImplementedError
    #
    # def __truediv__(self, constant):
    #     raise NotImplementedError


class CrossEntropyLossPartialMeasure(PartialMeasure):
    _cross_entropy_loss_value: float
    _number_of_samples: int

    def __init__(self, y_test=None, y_pred=None, cross_entropy_loss_partial=None, num_samples=None):
        if y_test is None and y_pred is None and cross_entropy_loss_partial is None and num_samples is None:
            self._cross_entropy_loss_value = 0
            self._number_of_samples = 0
        elif y_test is None and y_pred is None and cross_entropy_loss_partial is not None and num_samples is not None:
            self._cross_entropy_loss_value = cross_entropy_loss_partial
            self._number_of_samples = num_samples
        else:
            self._cross_entropy_loss_value = log_loss(y_test, y_pred)
            self._number_of_samples = len(y_test)

    def __add__(self, other):
        total_num_samples = self._number_of_samples + other._number_of_samples
        cross_entropy_loss_value = ((self._number_of_samples / total_num_samples) * self._cross_entropy_loss_value
                                    + (other._number_of_samples / total_num_samples) * other._cross_entropy_loss_value)
        return CrossEntropyLossPartialMeasure(cross_entropy_loss_partial=cross_entropy_loss_value, num_samples=total_num_samples)

    def __sub__(self, other):
        total_num_samples = self._number_of_samples + other._number_of_samples
        cross_entropy_loss_value = ((self._number_of_samples / total_num_samples) * self._cross_entropy_loss_value
                                    - (other._number_of_samples / total_num_samples) * other._cross_entropy_loss_value)
        return CrossEntropyLossPartialMeasure(cross_entropy_loss_partial=cross_entropy_loss_value, num_samples=total_num_samples)

    def set_value(self, cross_entropy_loss):
        self._cross_entropy_loss_value = cross_entropy_loss._cross_entropy_loss_value
        self._number_of_samples = cross_entropy_loss._number_of_samples
        return self


class ConfusionMatrix(PartialMeasure):
    confusion_matrix: np.ndarray
    labels: Optional[list[str]]

    def __init__(self, labels: Optional[list[str]]=None, y_test=None, y_pred=None, confusion_matrix=None):
        if labels is not None:
            self.labels = labels
            self.confusion_matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)

            if y_test is not None:
                y_test_argmaxed = np.argmax(y_test, axis=1)
                y_pred_argmaxed = np.argmax(y_pred, axis=1)

                np.add.at(self.confusion_matrix, (y_test_argmaxed, y_pred_argmaxed), 1)

                # for y_test_value, y_pred_value in zip(y_test_argmaxed, y_pred_argmaxed):
                #     self.confusion_matrix[y_test_value, y_pred_value] += 1
        else:
            self.confusion_matrix = confusion_matrix

    def __add__(self, other):
        confusion_matrix = self.confusion_matrix + other.confusion_matrix
        return ConfusionMatrix(confusion_matrix=confusion_matrix)

    def __sub__(self, other):
        confusion_matrix = self.confusion_matrix - other.confusion_matrix
        return ConfusionMatrix(confusion_matrix=confusion_matrix)

    def set_value(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix.confusion_matrix
        self.labels = confusion_matrix.labels


class MSEPartialMeasure(PartialMeasure):
    mse_value: float
    number_of_samples: int

    def __init__(self, y_pred: Optional[np.ndarray]=None,
                 y_test: Optional[np.ndarray]=None,
                 mse_value: Optional[float]=None,
                 number_of_samples: Optional[int]=None):
        if y_pred is None and y_test is None and mse_value is None and number_of_samples is None:
            self.mse_value = 0
            self.number_of_samples = 0
        elif y_pred is None and y_test is None:
            self.mse_value = mse_value
            self.number_of_samples = number_of_samples
        else:
            self.mse_value = mean_squared_error(y_pred, y_test)
            self.number_of_samples = len(y_pred)

    def __add__(self, other):
        total_number_of_samples = self.number_of_samples + other.number_of_samples
        aux_mse_value = (self.mse_value * (self.number_of_samples / total_number_of_samples)
                         + other.mse_value * (other.number_of_samples / total_number_of_samples))
        return MSEPartialMeasure(mse_value=aux_mse_value, number_of_samples=total_number_of_samples)

    def __sub__(self, other):
        total_number_of_samples = self.number_of_samples + other.number_of_samples
        aux_mse_value = (self.mse_value * (self.number_of_samples / total_number_of_samples)
                         - other.mse_value * (other.number_of_samples / total_number_of_samples))
        return MSEPartialMeasure(mse_value=aux_mse_value, number_of_samples=total_number_of_samples)


class MAEPartialMeasure(PartialMeasure):
    mae_value: float
    number_of_samples: int

    def __init__(self, y_pred: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None,
                 mae_value: Optional[float] = None,
                 number_of_samples: Optional[int] = None):
        if y_pred is None and y_test is None and mae_value is None and number_of_samples is None:
            self.mae_value = 0
            self.number_of_samples = 0
        elif y_pred is None and y_test is None:
            self.mae_value = mae_value
            self.number_of_samples = number_of_samples
            if self.mae_value is None:
                log(INFO, f"MAE: {self.mae_value}")
                log(INFO, f"This should not be happening")
        else:
            self.mae_value = mean_absolute_error(y_pred, y_test)
            self.number_of_samples = len(y_pred)

    def __add__(self, other):
        total_number_of_samples = self.number_of_samples + other.number_of_samples
        aux_mae_value = (self.mae_value * (self.number_of_samples / total_number_of_samples)
                         + other.mae_value * (other.number_of_samples / total_number_of_samples))
        return MAEPartialMeasure(mae_value=aux_mae_value, number_of_samples=total_number_of_samples)

    def __sub__(self, other):
        total_number_of_samples = self.number_of_samples + other.number_of_samples
        aux_mae_value = (self.mae_value * (self.number_of_samples / total_number_of_samples)
                         - other.mae_value * (other.number_of_samples / total_number_of_samples))
        return MAEPartialMeasure(mae_value=aux_mae_value, number_of_samples=total_number_of_samples)



class AggregatableMeasures(ABC):

    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError


class AggregatableMeasuresClassification(AggregatableMeasures):
    labels: Optional[list[str]]
    _cross_entropy_partial_measure: Optional[CrossEntropyLossPartialMeasure]
    _confusion_matrix: Optional[ConfusionMatrix]

    def __init__(self,
                 labels: Optional[list[str]],
                 cross_entropy_partial_measure: Optional[CrossEntropyLossPartialMeasure] = None,
                 confusion_matrix: Optional[ConfusionMatrix] = None):
        self.labels = labels
        if cross_entropy_partial_measure is None:
            self._cross_entropy_partial_measure = CrossEntropyLossPartialMeasure()
        else:
            self._cross_entropy_partial_measure = cross_entropy_partial_measure

        if confusion_matrix is None:
            self._confusion_matrix = ConfusionMatrix(labels)
        else:
            self._confusion_matrix = confusion_matrix

    # @classmethod
    # def initialize_default_aggregatable_measures(cls, labels):
    #     cross_entropy_partial_measure = CrossEntropyLossPartialMeasure()
    #     confusion_matrix = ConfusionMatrix(labels)
    #     return cls(cross_entropy_partial_measure, confusion_matrix)


    def __add__(self, other):
        cross_entropy_loss_value = self._cross_entropy_partial_measure + other._cross_entropy_partial_measure
        confusion_matrix = self._confusion_matrix + other._confusion_matrix
        return AggregatableMeasuresClassification(self.labels, cross_entropy_loss_value, confusion_matrix)

    def __sub__(self, other):
        cross_entropy_loss_value = self._cross_entropy_partial_measure - other._cross_entropy_partial_measure
        confusion_matrix = self._confusion_matrix - other._confusion_matrix
        return AggregatableMeasuresClassification(self.labels, cross_entropy_loss_value, confusion_matrix)

    def __str__(self):
        return str(self._cross_entropy_partial_measure._cross_entropy_loss_value)

    def get_measure(self, measure_name: str):
        if "CrossEntropyLoss" == measure_name:
            return self._cross_entropy_partial_measure
        elif "ConfusionMatrix" == measure_name:
            return self._confusion_matrix
        else:
            raise NotImplementedError

    def set_aggregatable_measures(self, aggregatable_measures):
        self._cross_entropy_partial_measure.set_value(aggregatable_measures._cross_entropy_partial_measure)
        self._confusion_matrix.set_value(aggregatable_measures._confusion_matrix)

    def set_value_cross_entropy(self, cross_entropy_partial_measure: CrossEntropyLossPartialMeasure):
        self._cross_entropy_partial_measure.set_value(cross_entropy_partial_measure)

    def set_value_confusion_matrix(self, confusion_matrix: ConfusionMatrix):
        self._confusion_matrix.set_value(confusion_matrix)


class AggregatableMeasuresRegression(AggregatableMeasures):
    _mse_partial_measure: Optional[MSEPartialMeasure]
    _mae_partial_measure: Optional[MAEPartialMeasure]

    def __init__(self,
                 msePartialMeasure: Optional[MSEPartialMeasure] = None,
                 maePartialMeasure: Optional[MAEPartialMeasure] = None):
        if msePartialMeasure is None:
            self._mse_partial_measure = MSEPartialMeasure()
        else:
            self._mse_partial_measure = msePartialMeasure

        if maePartialMeasure is None:
            self._mae_partial_measure = MAEPartialMeasure()
        else:
            self._mae_partial_measure = maePartialMeasure

    def __add__(self, other):
        mse_partial_measure = self._mse_partial_measure + other._mse_partial_measure
        mae_partial_measure = self._mae_partial_measure + other._mae_partial_measure
        return AggregatableMeasuresRegression(mse_partial_measure, mae_partial_measure)

    def __sub__(self, other):
        mse_partial_measure = self._mse_partial_measure - other._mse_partial_measure
        mae_partial_measure = self._mae_partial_measure - other._mae_partial_measure
        return AggregatableMeasuresRegression(mse_partial_measure, mae_partial_measure)

    def __str__(self):
        return str(self._mse_partial_measure.mse_value)

    def get_measure(self, measure_name: str):
        if "MSE" == measure_name:
            return self._mse_partial_measure
        elif "MAE" == measure_name:
            return self._mae_partial_measure
        else:
            raise NotImplementedError

    def set_aggregatable_measures(self, aggregatable_measures):
        self._mse_partial_measure.set_value(aggregatable_measures._mse_partial_measure)
        self._mae_partial_measure.set_value(aggregatable_measures._mae_partial_measure)

    # def set_value_mse(self, mse_partial: CrossEntropyLossPartialMeasure):
    #     self._cross_entropy_partial_measure.set_value(cross_entropy_partial_measure)
    #
    # def set_value_confusion_matrix(self, confusion_matrix: ConfusionMatrix):
    #     self._confusion_matrix.set_value(confusion_matrix)


class Metric(ABC):

    def get_name(self) -> str:
        pass

    def get_value(self):
        pass

    def set_value(self, value):
        pass

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __mul__(self, constant):
        pass
    #
    # def __truediv__(self, other):
    #     pass
    #
    # def __abs__(self):
    #     pass
    #
    # def obtain_min_or_max(self, other, func):
    #     pass


class Accuracy(Metric):
    _name: str
    _value: float

    def __init__(self, partial_measurement: Optional[Union[AggregatableMeasuresClassification, float]] = None):
        super().__init__()
        self._name = "Accuracy"
        if partial_measurement is None:
            self._value = 0
        elif type(partial_measurement) in [float, np.float64]:
            self._value = partial_measurement
        else:
            self._value = partial_measurement._confusion_matrix.confusion_matrix.diagonal().sum() / partial_measurement._confusion_matrix.confusion_matrix.sum()

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._value

    def set_value(self, confusion_matrix):
        self._value = confusion_matrix.diagonal().sum() / confusion_matrix.sum()

    def __add__(self, other):
        result = self._value + other.get_value()
        return Accuracy(result)

    def __sub__(self, other):
        result = self._value - other.get_value()
        return Accuracy(result)

    def __mul__(self, constant):
        result = self._value * constant
        return Accuracy(result)

    def __truediv__(self, constant):
        result = self._value / constant
        return Accuracy(result)

    # def __mul__(self, other):
    #     if isinstance(other, Accuracy):
    #         return Accuracy(self.get_value() * other.get_value())
    #     elif isinstance(other, int):
    #         return Accuracy(self.get_value() * other)
    #
    # def __truediv__(self, other):
    #     if isinstance(other, Accuracy):
    #         return Accuracy(self.get_value() / other.get_value())
    #     elif isinstance(other, int):
    #         return Accuracy(self.get_value() / other)

    # def __abs__(self):
    #     return abs(self._accuracy())
    #
    # def obtain_min_or_max(self, other, func):
    #     return Accuracy(func(self._accuracy_value, other.get_value()))

    # def addition_or_substraction(self, other, func):
    #     if func == "add":
    #         return self + other
    #     elif func == "sub":
    #         return self - other


class CrossEntropyLoss(Metric):
    _name: str
    _cross_entropy_loss_value: float

    def __init__(self, partial_measurement: Optional[Union[AggregatableMeasuresClassification, float]] = None):
        self._name = "CrossEntropyLoss"
        if partial_measurement is None:
            self._cross_entropy_loss_value = 0
        elif type(partial_measurement) == float:
            self._cross_entropy_loss_value = partial_measurement
        else:
            self._cross_entropy_loss_value = partial_measurement._cross_entropy_partial_measure._cross_entropy_loss_value

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._cross_entropy_loss_value

    def set_value(self, value):
        self._cross_entropy_loss_value = value

    def __add__(self, other):
        result = self._cross_entropy_loss_value + other._cross_entropy_loss_value
        return CrossEntropyLoss(result)

    def __sub__(self, other):
        result = other._cross_entropy_loss_value - self._cross_entropy_loss_value
        return CrossEntropyLoss(result)  # Turned around for the Shapley Values

    def __mul__(self, constant):
        result = self._cross_entropy_loss_value * constant
        return CrossEntropyLoss(result)

    def __truediv__(self, constant):
        result = self._cross_entropy_loss_value / constant
        return CrossEntropyLoss(result)
    #
    # def __mul__(self, other):
    #     if isinstance(other, CrossEntropyLoss):
    #         return CrossEntropyLoss(self.get_value() * other.get_value())
    #     elif isinstance(other, int):
    #         return CrossEntropyLoss(self.get_value() * other)
    #
    # def __truediv__(self, other):
    #     if isinstance(other, CrossEntropyLoss):
    #         return CrossEntropyLoss(self.get_value() / other.get_value())
    #     elif isinstance(other, int):
    #         return CrossEntropyLoss(self.get_value() / other)
    #
    # def obtain_min_or_max(self, other, func):
    #     return CrossEntropyLoss(func(self._cross_entropy_loss_value, other.get_value()))
    #
    # def __abs__(self):
    #     return CrossEntropyLoss(abs(self._cross_entropy_loss_value))
    #
    # def addition_or_substraction(self, other, func):
    #     if func == "add":
    #         return self + other
    #     elif func == "sub":
    #         return self - other


# class AggregatedF1Score(Metric):
#     _name: str
#     _aggregated_f1score_value: float
#
#     def __init__(self, initial_aggregated_f1score=None):
#         self._name = "AggregatedF1Score"
#         if initial_aggregated_f1score is None:
#             self._aggregated_f1score_value = 0
#         else:
#             self._aggregated_f1score_value = initial_aggregated_f1score
#
#     def get_name(self) -> str:
#         return self._name
#
#     def get_value(self):
#         return self._aggregated_f1score_value
#
#     def set_value(self, value):
#         self._aggregated_f1score_value = value
#
#     def __add__(self, other):
#         return AggregatedF1Score(self.get_value() + other.get_value())
#
#     def __sub__(self, other):
#         return AggregatedF1Score(self.get_value() - other.get_value())
#
#     def __mul__(self, other):
#         if isinstance(other, AggregatedF1Score):
#             return AggregatedF1Score(self.get_value() * other.get_value())
#         elif isinstance(other, int):
#             return AggregatedF1Score(self.get_value() * other)
#
#     def __truediv__(self, other):
#         if isinstance(other, AggregatedF1Score):
#             return AggregatedF1Score(self.get_value() / other.get_value())
#         elif isinstance(other, int):
#             return AggregatedF1Score(self.get_value() / other)
#
#     def __abs__(self):
#         return AggregatedF1Score(abs(self._aggregated_f1score_value))
#
#     def obtain_min_or_max(self, other, func):
#         return AggregatedF1Score(func(self._aggregated_f1score_value, other.get_value()))
#
#     def addition_or_substraction(self, other, func):
#         if func == "add":
#             return self + other
#         elif func == "sub":
#             return self - other

# Auxiliar function for F1Score computations
def _f1_score_arrays(confusion_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    true_positives = confusion_matrix.diagonal()
    false_negatives = confusion_matrix.sum(axis=1) - true_positives
    false_positives = confusion_matrix.sum(axis=0) - true_positives
    a = (2 * true_positives)
    b = (2 * true_positives + false_positives + false_negatives)
    f1_score_calculated = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    f1_score_weights = np.sum(confusion_matrix, axis=0) / np.sum(confusion_matrix)
    return f1_score_calculated, f1_score_weights


class SVCompatibleF1Score(Metric):
    _name: str
    _f1_score: list[float]

    def __init__(self, partial_measurement: Optional[Union[AggregatableMeasuresClassification, list[float]]] = None):
        self._name = "F1Score"
        if partial_measurement is None:
            self._f1_score = []
        elif type(partial_measurement) == list:
            self._f1_score = partial_measurement
        else:
            self._f1_score, _ = _f1_score_arrays(partial_measurement._confusion_matrix.confusion_matrix)

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._f1_score

    def set_value(self, value):
        self._f1_score = value

    def __add__(self, other):
        aux_list = []
        for i in range(len(self._f1_score)):
            aux_list.append(self._f1_score[i] + other.get_value()[i])
        return SVCompatibleF1Score(aux_list)

    def __sub__(self, other):
        aux_list = []
        for i in range(len(self._f1_score)):
            aux_list.append(self._f1_score[i] - other.get_value()[i])
        return SVCompatibleF1Score(aux_list)

    def __mul__(self, constant):
        aux_list = [value * constant for value in self._f1_score]
        return SVCompatibleF1Score(aux_list)

    def __truediv__(self, constant):
        aux_list = [value / constant for value in self._f1_score]
        return SVCompatibleF1Score(aux_list)

    # def __mul__(self, other):
    #     aux_list = []
    #     if isinstance(other, SVCompatibleF1Score):
    #         for element_list_self, element_list_other in zip(self.get_value(), other.get_value()):
    #             aux_list.append(element_list_self * element_list_other)
    #         return SVCompatibleF1Score(aux_list)
    #     elif isinstance(other, int):
    #         for element_list_self in self.get_value():
    #             aux_list.append(element_list_self * other)
    #         return SVCompatibleF1Score(aux_list)
    #
    # def __truediv__(self, other):
    #     aux_list = []
    #     if isinstance(other, SVCompatibleF1Score):
    #         for element_list_self, element_list_other in zip(self.get_value(), other.get_value()):
    #             aux_list.append(element_list_self / element_list_other)
    #         return SVCompatibleF1Score(aux_list)
    #     elif isinstance(other, int):
    #         for element_list_self in self.get_value():
    #             aux_list.append(element_list_self / other)
    #         return SVCompatibleF1Score(aux_list)
    #
    # def __abs__(self):
    #     return SVCompatibleF1Score([abs(f1_score_value) for f1_score_value in self._f1_score])
    #
    # def obtain_min_or_max(self, other, func):
    #     list_of_min = [function(v1, v2) for v1, v2, function in zip(self._f1_score, other.get_value(), func)]
    #     return SVCompatibleF1Score(list_of_min)
    #
    # def addition_or_substraction(self, other, func):
    #     aux_list = []
    #     for f1_score_value, other_f1_score_value, function in zip(self.get_value(), other.get_value(), func):
    #         if function == "add":
    #             aux_list.append(f1_score_value + other_f1_score_value)
    #         elif function == "sub":
    #             aux_list.append(f1_score_value - other_f1_score_value)
    #     return SVCompatibleF1Score(aux_list)

    # def max_out_two(self, other):
    #     list_of_max = [max(v1, v2) for v1, v2 in zip(self._f1_score, other.get_value())]
    #     return SVCompatibleF1Score(list_of_max)



class F1ScoreMacro(Metric):
    _name: str
    _f1score_macro_value: float

    def __init__(self, partial_measurement: Optional[Union[AggregatableMeasuresClassification, float]] = None):
        self._name = "F1ScoreMacro"
        if partial_measurement is None:
            self._f1score_macro_value = 0
        elif type(partial_measurement) == float:
            self._f1score_macro_value = partial_measurement
        else:
            f1_score_values, _ = _f1_score_arrays(partial_measurement._confusion_matrix.confusion_matrix)
            self._f1score_macro_value = float(np.mean(f1_score_values))

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._f1score_macro_value

    def set_value(self, confusion_matrix: ConfusionMatrix):
        f1_score_values, _ = _f1_score_arrays(confusion_matrix.confusion_matrix)
        self._f1score_macro_value = float(np.mean(f1_score_values))

    def __add__(self, other):
        result = self._f1score_macro_value + other._f1score_macro_value
        return F1ScoreMacro(result)

    def __sub__(self, other):
        result = self._f1score_macro_value - other._f1score_macro_value
        return F1ScoreMacro(result)

    def __mul__(self, constant):
        result = self._f1score_macro_value * constant
        return F1ScoreMacro(result)

    def __truediv__(self, constant):
        result = self._f1score_macro_value / constant
        return F1ScoreMacro(result)

    # def __mul__(self, other):
    #     if isinstance(other, F1ScoreMacro):
    #         return F1ScoreMacro(self.get_value() * other.get_value())
    #     elif isinstance(other, int):
    #         return F1ScoreMacro(self.get_value() * other)
    #
    # def __truediv__(self, other):
    #     if isinstance(other, F1ScoreMacro):
    #         return F1ScoreMacro(self.get_value() / other.get_value())
    #     elif isinstance(other, int):
    #         return F1ScoreMacro(self.get_value() / other)
    #
    # def __abs__(self):
    #     return F1ScoreMacro(abs(self._f1score_macro_value))
    #
    # def obtain_min_or_max(self, other, func):
    #     return F1ScoreMacro(func(self._f1score_macro_value, other.get_value()))
    #
    # def addition_or_substraction(self, other, func):
    #     if func == "add":
    #         return self + other
    #     elif func == "sub":
    #         return self - other


class F1ScoreMicro(Metric):
    _name: str
    _f1score_micro_value: float

    def __init__(self, partial_measurement: Optional[Union[AggregatableMeasuresClassification, float]] = None):
        self._name = "F1ScoreMicro"
        if partial_measurement is None:
            self._f1score_micro_value = 0
        elif type(partial_measurement) == float:
            self._f1score_micro_value = partial_measurement
        else:
            f1_score_values, f1_score_weights = _f1_score_arrays(partial_measurement._confusion_matrix.confusion_matrix)
            self._f1score_micro_value = float(np.average(f1_score_values, weights=f1_score_weights))

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._f1score_micro_value

    def set_value(self, confusion_matrix: ConfusionMatrix):
        f1_score_values, f1_score_weights = _f1_score_arrays(confusion_matrix.confusion_matrix)
        self._f1score_micro_value = float(np.average(f1_score_values, weights=f1_score_weights))

    def __add__(self, other):
        result = self._f1score_micro_value + other._f1score_micro_value
        return F1ScoreMicro(result)

    def __sub__(self, other):
        result = self._f1score_micro_value - other._f1score_micro_value
        return F1ScoreMicro(result)
    #
    def __mul__(self, constant):
        result = self._f1score_micro_value * constant
        return F1ScoreMicro(result)

    def __truediv__(self, constant):
        result = self._f1score_micro_value / constant
        return F1ScoreMicro(result)
    #
    # def __truediv__(self, other):
    #     if isinstance(other, F1ScoreMicro):
    #         return F1ScoreMicro(self.get_value() / other.get_value())
    #     elif isinstance(other, int):
    #         return F1ScoreMicro(self.get_value() / other)
    #
    # def __abs__(self):
    #     return F1ScoreMicro(abs(self._f1score_micro_value))
    #
    # def obtain_min_or_max(self, other, func):
    #     return F1ScoreMicro(func(self._f1score_micro_value, other.get_value()))
    #
    # def addition_or_substraction(self, other, func):
    #     if func == "add":
    #         return self + other
    #     elif func == "sub":
    #         return self - other


# class SVCompatibleF1Score(Metric):
#     _name: str
#     _f1_score: list
#
#     def __init__(self, initial_f1_score=None, num_classes=None):
#         self._name = "F1Score"
#         if initial_f1_score is None:
#             if num_classes is None:
#                 raise Exception("Need number of classes if initial value is null")
#             else:
#                 self._f1_score = [0 for _ in range(num_classes)]
#         else:
#             self._f1_score = initial_f1_score
#
#     def get_name(self) -> str:
#         return self._name
#
#     def get_value(self):
#         return self._f1_score
#
#     def set_value(self, value):
#         self._f1_score = value
#
#     def __add__(self, other):
#         aux_list = []
#         for list_self_element, list_other_element in zip(self.get_value(), other.get_value()):
#             aux_list.append(list_self_element + list_other_element)
#         return SVCompatibleF1Score(aux_list)
#
#     def __sub__(self, other):
#         aux_list = []
#         for list_self_element, list_other_element in zip(self.get_value(), other.get_value()):
#             aux_list.append(list_self_element - list_other_element)
#         return SVCompatibleF1Score(aux_list)
#
#     def __mul__(self, other):
#         aux_list = []
#         if isinstance(other, SVCompatibleF1Score):
#             for element_list_self, element_list_other in zip(self.get_value(), other.get_value()):
#                 aux_list.append(element_list_self * element_list_other)
#             return SVCompatibleF1Score(aux_list)
#         elif isinstance(other, int):
#             for element_list_self in self.get_value():
#                 aux_list.append(element_list_self * other)
#             return SVCompatibleF1Score(aux_list)
#
#     def __truediv__(self, other):
#         aux_list = []
#         if isinstance(other, SVCompatibleF1Score):
#             for element_list_self, element_list_other in zip(self.get_value(), other.get_value()):
#                 aux_list.append(element_list_self / element_list_other)
#             return SVCompatibleF1Score(aux_list)
#         elif isinstance(other, int):
#             for element_list_self in self.get_value():
#                 aux_list.append(element_list_self / other)
#             return SVCompatibleF1Score(aux_list)
#
#     def __abs__(self):
#         return SVCompatibleF1Score([abs(f1_score_value) for f1_score_value in self._f1_score])
#
#     def obtain_min_or_max(self, other, func):
#         list_of_min = [function(v1, v2) for v1, v2, function in zip(self._f1_score, other.get_value(), func)]
#         return SVCompatibleF1Score(list_of_min)
#
#     def addition_or_substraction(self, other, func):
#         aux_list = []
#         for f1_score_value, other_f1_score_value, function in zip(self.get_value(), other.get_value(), func):
#             if function == "add":
#                 aux_list.append(f1_score_value + other_f1_score_value)
#             elif function == "sub":
#                 aux_list.append(f1_score_value - other_f1_score_value)
#         return SVCompatibleF1Score(aux_list)
#
#     # def max_out_two(self, other):
#     #     list_of_max = [max(v1, v2) for v1, v2 in zip(self._f1_score, other.get_value())]
#     #     return SVCompatibleF1Score(list_of_max)
#
#
# class SVCompatibleWeightedF1Score(SVCompatibleF1Score):
#
#     def __init__(self, initial_f1_score=None, num_classes=None):
#         super().__init__(initial_f1_score, num_classes)
#
#     def __truediv__(self, other):
#         aux_list = []
#         if isinstance(other, SVCompatibleWeightedF1Score):
#             for element_list_self, element_list_other in zip(self.get_value(), other.get_value()):
#                 aux_list.append(element_list_self / element_list_other)
#             return SVCompatibleWeightedF1Score(aux_list)
#         elif isinstance(other, int):
#             for element_list_self in self.get_value():
#                 aux_list.append(element_list_self / other)
#             return SVCompatibleWeightedF1Score(aux_list)


class SVCompatibleMatthewsCorrelationCoefficient(Metric):
    _name: str
    _mcc_value: float

    def __init__(self, partial_measurement_or_value: Optional[Union[AggregatableMeasuresClassification, float]] = None):
        self._name = "MCC"
        if partial_measurement_or_value is None:
            self._mcc_value = 0
        elif type(partial_measurement_or_value) in [int, float, np.float64]:
            self._mcc_value = partial_measurement_or_value
        else:
            true_positives = partial_measurement_or_value._confusion_matrix.confusion_matrix.diagonal().sum()
            total_number_of_instances = partial_measurement_or_value._confusion_matrix.confusion_matrix.sum()
            total_actual_classes = (partial_measurement_or_value._confusion_matrix.confusion_matrix.sum(axis=1))
            total_predictions = (partial_measurement_or_value._confusion_matrix.confusion_matrix.sum(axis=0))
            false_per_label = int(np.sum(np.multiply(total_predictions, total_actual_classes)))
            numerator = true_positives * total_number_of_instances - false_per_label
            denominator = (
                    math.sqrt(pow(total_number_of_instances, 2) - np.sum(np.power(total_predictions, 2))) *
                    math.sqrt(pow(total_number_of_instances, 2) - np.sum(np.power(total_actual_classes, 2)))
            )
            if denominator == 0:
                self._mcc_value = 0.0
            else:
                self._mcc_value = numerator / denominator


    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._mcc_value

    def set_value(self, confusion_matrix: ConfusionMatrix):
        true_positives = confusion_matrix.confusion_matrix.diagonal().sum()
        total_number_of_instances = confusion_matrix.confusion_matrix.sum()
        total_false_negatives = (confusion_matrix.confusion_matrix.sum(axis=1) - true_positives).sum()
        total_false_positives = (confusion_matrix.confusion_matrix.sum(axis=0) - true_positives).sum()
        false_per_label = int(np.sum(total_false_positives * total_false_negatives))
        numerator = true_positives * total_number_of_instances - false_per_label
        denominator = (math.sqrt(total_number_of_instances - np.power(total_false_positives, 2)) *
                       math.sqrt(total_number_of_instances - np.power(total_false_negatives, 2)))
        self._mcc_value = numerator / denominator

    def __add__(self, other):
        result = self._mcc_value + other._mcc_value
        return SVCompatibleMatthewsCorrelationCoefficient(result)

    def __sub__(self, other):
        result = self._mcc_value - other._mcc_value
        return SVCompatibleMatthewsCorrelationCoefficient(result)

    def __mul__(self, constant):
        result = self._mcc_value * constant
        return SVCompatibleMatthewsCorrelationCoefficient(result)
    #

    def __truediv__(self, constant):
        result = self._mcc_value / constant
        return SVCompatibleMatthewsCorrelationCoefficient(result)
    #
    # def __abs__(self):
    #     return SVCompatibleMatthewsCorrelationCoefficient(abs(self._mcc_value))
    #
    # def obtain_min_or_max(self, other, func):
    #     return SVCompatibleMatthewsCorrelationCoefficient(func(self._mcc_value, other.get_value()))
    #
    # def addition_or_substraction(self, other, func):
    #     if func == "add":
    #         return self + other
    #     elif func == "sub":
    #         return self - other


class RMSE(Metric):
    _name: str
    _rmse_value: float

    def __init__(self, mse_partial_measure: Optional[Union[AggregatableMeasuresRegression, float]]=None):
        self._name = "RMSE"
        if mse_partial_measure is None:
            self._rmse_value = 0
        elif type(mse_partial_measure) in [int, float, np.float64]:
            self._rmse_value = mse_partial_measure
        else:
            self._rmse_value = math.sqrt(mse_partial_measure._mse_partial_measure.mse_value)

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._rmse_value

    def set_value(self, value):
        self._rmse_value = value

    def __add__(self, other):
        return RMSE(self.get_value() + other.get_value())

    def __sub__(self, other):
        return RMSE(other.get_value() - self.get_value())  # Turned around for the Shapley Values

    def __mul__(self, other):
        if isinstance(other, RMSE):
            return RMSE(self.get_value() * other.get_value())
        elif isinstance(other, int):
            return RMSE(self.get_value() * other)

    def __truediv__(self, other):
        if isinstance(other, RMSE):
            return RMSE(self.get_value() / other.get_value())
        elif isinstance(other, int):
            return RMSE(self.get_value() / other)

    def __abs__(self):
        return RMSE(abs(self._rmse_value))

    def obtain_min_or_max(self, other, func):
        return RMSE(func(self._rmse_value, other.get_value()))

    def addition_or_substraction(self, other, func):
        if func == "add":
            return self + other
        elif func == "sub":
            return self - other
        

class MSE(Metric):
    _name: str
    _mse_value: float

    def __init__(self, mse_partial_measure: Optional[Union[AggregatableMeasuresRegression, float]] = None):
        self._name = "MSE"
        if mse_partial_measure is None:
            self._mse_value = 0
        elif type(mse_partial_measure) in [int, float, np.float64]:
            self._mse_value = mse_partial_measure
        else:
            self._mse_value = mse_partial_measure._mse_partial_measure.mse_value

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._mse_value

    def set_value(self, value):
        self._mse_value = value

    def __add__(self, other):
        return MSE(self.get_value() + other.get_value())

    def __sub__(self, other):
        return MSE(other.get_value() - self.get_value())  # Turned around for the Shapley Values

    def __mul__(self, other):
        if isinstance(other, MSE):
            return MSE(self.get_value() * other.get_value())
        elif isinstance(other, int):
            return MSE(self.get_value() * other)

    def __truediv__(self, other):
        if isinstance(other, MSE):
            return MSE(self.get_value() / other.get_value())
        elif isinstance(other, int):
            return MSE(self.get_value() / other)

    def __abs__(self):
        return MSE(abs(self._mse_value))

    def obtain_min_or_max(self, other, func):
        return MSE(func(self._mse_value, other.get_value()))

    def addition_or_substraction(self, other, func):
        if func == "add":
            return self + other
        elif func == "sub":
            return self - other


class MAE(Metric):
    _name: str
    _mae_value: float

    def __init__(self, mae_partial_measure: Optional[Union[AggregatableMeasuresRegression, float]] = None):
        self._name = "MAE"
        if mae_partial_measure is None:
            self._mae_value = 0
        elif type(mae_partial_measure) in [int, float, np.float64]:
            self._mae_value = mae_partial_measure
        else:
            self._mae_value = mae_partial_measure._mae_partial_measure.mae_value

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._mae_value

    def set_value(self, value):
        self._mae_value = value

    def __add__(self, other):
        return MAE(self.get_value() + other.get_value())

    def __sub__(self, other):
        return MAE(other.get_value() - self.get_value())  # Turned around for the Shapley Values

    def __mul__(self, other):
        if isinstance(other, MAE):
            return MAE(self.get_value() * other.get_value())
        elif isinstance(other, int):
            return MAE(self.get_value() * other)

    def __truediv__(self, other):
        if isinstance(other, MAE):
            return MAE(self.get_value() / other.get_value())
        elif isinstance(other, int):
            return MAE(self.get_value() / other)

    def __abs__(self):
        return MAE(abs(self._mae_value))

    def obtain_min_or_max(self, other, func):
        return MAE(func(self._mae_value, other.get_value()))

    def addition_or_substraction(self, other, func):
        if func == "add":
            return self + other
        elif func == "sub":
            return self - other


class R2(Metric):
    _name: str
    _r2_value: float

    def __init__(self, initial_r2_value=None):
        self._name = "R2"
        if initial_r2_value is None:
            self._r2_value = 0
        else:
            self._r2_value = initial_r2_value

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._r2_value

    def set_value(self, value):
        self._r2_value = value

    def __add__(self, other):
        return R2(self.get_value() + other.get_value())

    def __sub__(self, other):
        return R2(self.get_value() - other.get_value())

    def __mul__(self, other):
        if isinstance(other, R2):
            return R2(self.get_value() * other.get_value())
        elif isinstance(other, int):
            return R2(self.get_value() * other)

    def __truediv__(self, other):
        if isinstance(other, R2):
            return R2(self.get_value() / other.get_value())
        elif isinstance(other, int):
            return R2(self.get_value() / other)

    def __abs__(self):
        return R2(abs(self._r2_value))

    def obtain_min_or_max(self, other, func):
        return R2(func(self._r2_value, other.get_value()))

    def addition_or_substraction(self, other, func):
        if func == "add":
            return self + other
        elif func == "sub":
            return self - other


def string_cast(param):
    if type(param) is np.ndarray:
        param = param.tolist()
        return str(param)
    else:
        return str(param)


class DictOfMetrics:
    _dictionary_of_metrics: Dict

    def __init__(self, first_dict_of_metrics=None):
        if first_dict_of_metrics is None:
            first_dict_of_metrics = {}
        self._dictionary_of_metrics = first_dict_of_metrics

    def add_metric(self, metric: Metric):
        self._dictionary_of_metrics[metric.get_name()] = metric

    def get_value(self):
        return self._dictionary_of_metrics

    def set_value(self, value):
        self._dictionary_of_metrics = value

    def get_value_of_metric(self, metric_name):
        return self.get_value()[metric_name].get_value()

    def set_value_of_metric(self, metric_name, value):
        self.get_value()[metric_name].set_value(value)

    def __add__(self, other):
        aux_dict = {}
        # dict_of_metrics_1 = self.get_value()
        dict_of_metrics_2 = other.get_value()
        for key in self._dictionary_of_metrics.keys():
            aux_dict[key] = self._dictionary_of_metrics[key] + dict_of_metrics_2[key]
        return DictOfMetrics(aux_dict)

    def __sub__(self, other):
        aux_dict = {}
        # dict_of_metrics_1 = self.get_value()
        dict_of_metrics_2 = other.get_value()
        for key in self._dictionary_of_metrics.keys():
            aux_dict[key] = self._dictionary_of_metrics[key] - dict_of_metrics_2[key]
        return DictOfMetrics(aux_dict)

    def __mul__(self, other):
        aux_dict = {}
        for key in self._dictionary_of_metrics.keys():
            aux_dict[key] = self._dictionary_of_metrics[key] * other
        return DictOfMetrics(aux_dict)

    def __truediv__(self, other):
        aux_dict = {}
        for key in self._dictionary_of_metrics.keys():
            aux_dict[key] = self._dictionary_of_metrics[key] / other
        return DictOfMetrics(aux_dict)

    def __lt__(self, other):
        list_of_metrics_1 = self.get_value()
        list_of_metrics_2 = other.get_value()
        # The second metric is always accuracy.
        return list_of_metrics_1["Accuracy"].get_value() < list_of_metrics_2["Accuracy"].get_value()

    def __str__(self):
        full_string = ""
        for metric in self.get_value().values():
            full_string += str(metric.get_name()) + ":" + str(metric.get_value()) + ","
        return full_string[:-1]

    def __abs__(self):
        dict_of_metrics_1 = self.get_value()
        aux_dict = {}
        for key in dict_of_metrics_1.keys():
            aux_dict[key] = abs(dict_of_metrics_1[key])
        return DictOfMetrics(aux_dict)

    def obtain_min_or_max(self, other, functions):
        dict_of_metrics_1 = self.get_value()
        aux_dict = {}
        for key in dict_of_metrics_1.keys():
            aux_dict[key] = self.get_value()[key].obtain_min_or_max(other.get_value()[key], functions[key])
        return DictOfMetrics(aux_dict)

    def addition_or_substraction(self, other, functions):
        dict_of_metrics_1 = self.get_value()
        aux_dict = {}
        for key in dict_of_metrics_1.keys():
            aux_dict[key] = self.get_value()[key].addition_or_substraction(other.get_value()[key], functions[key])
        return DictOfMetrics(aux_dict)

    def return_flower_dict(self):
        metrics_to_return = {metric: self._dictionary_of_metrics[metric].get_value()
                             for metric in self._dictionary_of_metrics.keys()}
        if "F1Score" in metrics_to_return:
            if type(metrics_to_return["F1Score"]) is np.ndarray:
                metrics_to_return["F1Score"]: List = metrics_to_return["F1Score"].tolist()
        return metrics_to_return

    def return_flower_dict_as_str(self):
        return {metric: string_cast(self._dictionary_of_metrics[metric].get_value())
                for metric in self._dictionary_of_metrics.keys()}

    def eval_flower_dict_from_str(self):
        return

def return_default_partial_computations(labels):
    partial_computations_default: AggregatableMeasures
    if len(labels) > 1:
        partial_computations_default = AggregatableMeasuresClassification(labels)
    else:
        partial_computations_default = AggregatableMeasuresRegression()

    return partial_computations_default


def return_default_dict_of_metrics(metrics, num_classes):
    metric_dict = DictOfMetrics()
    if "CrossEntropyLoss" in metrics:
        metric_dict.add_metric(CrossEntropyLoss())
    if "Accuracy" in metrics:
        metric_dict.add_metric(Accuracy())
    if "F1Score" in metrics:
        metric_dict.add_metric(SVCompatibleF1Score([0.0] * num_classes))
    # if "WeightedF1Score" in metrics:
    #     metric_dict.add_metric(SVCompatibleWeightedF1Score(num_classes=num_classes))
    if "MCC" in metrics:
        metric_dict.add_metric(SVCompatibleMatthewsCorrelationCoefficient())
    if "F1ScoreMacro" in metrics:
        metric_dict.add_metric(F1ScoreMacro())
    if "F1ScoreMicro" in metrics:
        metric_dict.add_metric(F1ScoreMicro())
    if "RMSE" in metrics:
        metric_dict.add_metric(RMSE())
    if "MSE" in metrics:
        metric_dict.add_metric(MSE())
    if "MAE" in metrics:
        metric_dict.add_metric(MAE())
    if "R2" in metrics:
        metric_dict.add_metric(R2())
    return metric_dict


GLOBAL_KEY = "__global__"

class SubpopulationDict:
    def __init__(self, mapping: dict = None):
        self._mapping = mapping if mapping is not None else {}

    def keys(self):
        return self._mapping.keys()

    def items(self):
        return self._mapping.items()

    def get_value(self):
        return self._mapping

    def __getitem__(self, key):
        return self._mapping[key]

    def __contains__(self, key):
        return key in self._mapping

    def get_global(self):
        return self._mapping.get(GLOBAL_KEY)

    def get_global_flower_dict(self):
        global_value = self.get_global()
        if global_value is None:
            return {}
        return global_value.return_flower_dict() if hasattr(global_value, "return_flower_dict") else {}

    def subpopulation_items(self):
        """All entries except the global one -- what the ResultManager needs to fan out."""
        return [(key, value) for key, value in self._mapping.items() if key != GLOBAL_KEY]

    def _combine(self, other, op):
        keys = set(self._mapping.keys()) | set(other._mapping.keys())
        combined = {}
        for key in keys:
            if key in self._mapping and key in other._mapping:
                combined[key] = op(self._mapping[key], other._mapping[key])
            elif key in self._mapping:
                combined[key] = self._mapping[key]
            else:
                combined[key] = other._mapping[key]
        return SubpopulationDict(combined)

    def __add__(self, other):
        return self._combine(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._combine(other, lambda a, b: a - b)

    def __mul__(self, constant):
        return SubpopulationDict({k: v * constant for k, v in self._mapping.items()})

    def __truediv__(self, constant):
        return SubpopulationDict({k: v / constant for k, v in self._mapping.items()})

    def __str__(self):
        return " | ".join(f"{key}: {value}" for key, value in self._mapping.items())

    def return_flower_dict(self):
        flat = {}
        for key, value in self._mapping.items():
            nested = value.return_flower_dict() if hasattr(value, "return_flower_dict") else {"value": value}
            for metric_name, metric_value in nested.items():
                if key == GLOBAL_KEY:
                    flat[metric_name] = metric_value  # NEW — no prefix for the global entry
                else:
                    flat[f"{key}__{metric_name}"] = metric_value
        return flat


def group_indices_by_subpopulation(subpopulation_values: Union[pd.Series, pd.DataFrame]):
    if isinstance(subpopulation_values, pd.Series):
        subpopulation_values = subpopulation_values.to_frame()
    columns = list(subpopulation_values.columns)
    group_by = columns if len(columns) > 1 else columns[0]
    return subpopulation_values.reset_index(drop=True).groupby(group_by).indices