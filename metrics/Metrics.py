from abc import ABC
from typing import Dict, List
import numpy as np


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

    def __mul__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __abs__(self):
        pass

    def obtain_min_or_max(self, other, func):
        pass


class Accuracy(Metric):
    _name: str
    _accuracy_value: float

    def __init__(self, initial_accuracy=None):
        self._name = "Accuracy"
        if initial_accuracy is None:
            self._accuracy_value = 0
        else:
            self._accuracy_value = initial_accuracy

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._accuracy_value

    def set_value(self, value):
        self._accuracy_value = value

    def __add__(self, other):
        return Accuracy(self.get_value() + other.get_value())

    def __sub__(self, other):
        return Accuracy(self.get_value() - other.get_value())

    def __mul__(self, other):
        if isinstance(other, Accuracy):
            return Accuracy(self.get_value() * other.get_value())
        elif isinstance(other, int):
            return Accuracy(self.get_value() * other)

    def __truediv__(self, other):
        if isinstance(other, Accuracy):
            return Accuracy(self.get_value() / other.get_value())
        elif isinstance(other, int):
            return Accuracy(self.get_value() / other)

    def __abs__(self):
        return Accuracy(abs(self._accuracy_value))

    def obtain_min_or_max(self, other, func):
        return Accuracy(func(self._accuracy_value, other.get_value()))

    def addition_or_substraction(self, other, func):
        if func == "add":
            return self + other
        elif func == "sub":
            return self - other


class CrossEntropyLoss(Metric):
    _name: str
    _cross_entropy_loss_value: float

    def __init__(self, initial_loss=None):
        self._name = "CrossEntropyLoss"
        if initial_loss is None:
            self._cross_entropy_loss_value = 0
        else:
            self._cross_entropy_loss_value = initial_loss

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._cross_entropy_loss_value

    def set_value(self, value):
        self._cross_entropy_loss_value = value

    def __add__(self, other):
        return CrossEntropyLoss(self.get_value() + other.get_value())

    def __sub__(self, other):
        return CrossEntropyLoss(other.get_value() - self.get_value())  # Turned around for the Shapley Values

    def __mul__(self, other):
        if isinstance(other, CrossEntropyLoss):
            return CrossEntropyLoss(self.get_value() * other.get_value())
        elif isinstance(other, int):
            return CrossEntropyLoss(self.get_value() * other)

    def __truediv__(self, other):
        if isinstance(other, CrossEntropyLoss):
            return CrossEntropyLoss(self.get_value() / other.get_value())
        elif isinstance(other, int):
            return CrossEntropyLoss(self.get_value() / other)

    def obtain_min_or_max(self, other, func):
        return CrossEntropyLoss(func(self._cross_entropy_loss_value, other.get_value()))

    def __abs__(self):
        return CrossEntropyLoss(abs(self._cross_entropy_loss_value))

    def addition_or_substraction(self, other, func):
        if func == "add":
            return self + other
        elif func == "sub":
            return self - other


class AggregatedF1Score(Metric):
    _name: str
    _aggregated_f1score_value: float

    def __init__(self, initial_aggregated_f1score=None):
        self._name = "AggregatedF1Score"
        if initial_aggregated_f1score is None:
            self._aggregated_f1score_value = 0
        else:
            self._aggregated_f1score_value = initial_aggregated_f1score

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._aggregated_f1score_value

    def set_value(self, value):
        self._aggregated_f1score_value = value

    def __add__(self, other):
        return AggregatedF1Score(self.get_value() + other.get_value())

    def __sub__(self, other):
        return AggregatedF1Score(self.get_value() - other.get_value())

    def __mul__(self, other):
        if isinstance(other, AggregatedF1Score):
            return AggregatedF1Score(self.get_value() * other.get_value())
        elif isinstance(other, int):
            return AggregatedF1Score(self.get_value() * other)

    def __truediv__(self, other):
        if isinstance(other, AggregatedF1Score):
            return AggregatedF1Score(self.get_value() / other.get_value())
        elif isinstance(other, int):
            return AggregatedF1Score(self.get_value() / other)

    def __abs__(self):
        return AggregatedF1Score(abs(self._aggregated_f1score_value))

    def obtain_min_or_max(self, other, func):
        return AggregatedF1Score(func(self._aggregated_f1score_value, other.get_value()))

    def addition_or_substraction(self, other, func):
        if func == "add":
            return self + other
        elif func == "sub":
            return self - other


class F1ScoreMacro(Metric):
    _name: str
    _f1score_macro_value: float

    def __init__(self, initial_f1score_macro=None):
        self._name = "F1ScoreMacro"
        if initial_f1score_macro is None:
            self._f1score_macro_value = 0
        else:
            self._f1score_macro_value = initial_f1score_macro

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._f1score_macro_value

    def set_value(self, value):
        self._f1score_macro_value = value

    def __add__(self, other):
        return F1ScoreMacro(self.get_value() + other.get_value())

    def __sub__(self, other):
        return F1ScoreMacro(self.get_value() - other.get_value())

    def __mul__(self, other):
        if isinstance(other, F1ScoreMacro):
            return F1ScoreMacro(self.get_value() * other.get_value())
        elif isinstance(other, int):
            return F1ScoreMacro(self.get_value() * other)

    def __truediv__(self, other):
        if isinstance(other, F1ScoreMacro):
            return F1ScoreMacro(self.get_value() / other.get_value())
        elif isinstance(other, int):
            return F1ScoreMacro(self.get_value() / other)

    def __abs__(self):
        return F1ScoreMacro(abs(self._f1score_macro_value))

    def obtain_min_or_max(self, other, func):
        return F1ScoreMacro(func(self._f1score_macro_value, other.get_value()))

    def addition_or_substraction(self, other, func):
        if func == "add":
            return self + other
        elif func == "sub":
            return self - other


class F1ScoreMicro(Metric):
    _name: str
    _f1score_micro_value: float

    def __init__(self, initial_f1score_micro=None):
        self._name = "F1ScoreMicro"
        if initial_f1score_micro is None:
            self._f1score_micro_value = 0
        else:
            self._f1score_micro_value = initial_f1score_micro

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._f1score_micro_value

    def set_value(self, value):
        self._f1score_micro_value = value

    def __add__(self, other):
        return F1ScoreMicro(self.get_value() + other.get_value())

    def __sub__(self, other):
        return F1ScoreMicro(self.get_value() - other.get_value())

    def __mul__(self, other):
        if isinstance(other, F1ScoreMicro):
            return F1ScoreMicro(self.get_value() * other.get_value())
        elif isinstance(other, int):
            return F1ScoreMicro(self.get_value() * other)

    def __truediv__(self, other):
        if isinstance(other, F1ScoreMicro):
            return F1ScoreMicro(self.get_value() / other.get_value())
        elif isinstance(other, int):
            return F1ScoreMicro(self.get_value() / other)

    def __abs__(self):
        return F1ScoreMicro(abs(self._f1score_micro_value))

    def obtain_min_or_max(self, other, func):
        return F1ScoreMicro(func(self._f1score_micro_value, other.get_value()))

    def addition_or_substraction(self, other, func):
        if func == "add":
            return self + other
        elif func == "sub":
            return self - other


class SVCompatibleF1Score(Metric):
    _name: str
    _f1_score: list

    def __init__(self, initial_f1_score=None, num_classes=None):
        self._name = "F1Score"
        if initial_f1_score is None:
            if num_classes is None:
                raise Exception("Need number of classes if initial value is null")
            else:
                self._f1_score = [0 for _ in range(num_classes)]
        else:
            self._f1_score = initial_f1_score

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._f1_score

    def set_value(self, value):
        self._f1_score = value

    def __add__(self, other):
        aux_list = []
        for list_self_element, list_other_element in zip(self.get_value(), other.get_value()):
            aux_list.append(list_self_element + list_other_element)
        return SVCompatibleF1Score(aux_list)

    def __sub__(self, other):
        aux_list = []
        for list_self_element, list_other_element in zip(self.get_value(), other.get_value()):
            aux_list.append(list_self_element - list_other_element)
        return SVCompatibleF1Score(aux_list)

    def __mul__(self, other):
        aux_list = []
        if isinstance(other, SVCompatibleF1Score):
            for element_list_self, element_list_other in zip(self.get_value(), other.get_value()):
                aux_list.append(element_list_self * element_list_other)
            return SVCompatibleF1Score(aux_list)
        elif isinstance(other, int):
            for element_list_self in self.get_value():
                aux_list.append(element_list_self * other)
            return SVCompatibleF1Score(aux_list)

    def __truediv__(self, other):
        aux_list = []
        if isinstance(other, SVCompatibleF1Score):
            for element_list_self, element_list_other in zip(self.get_value(), other.get_value()):
                aux_list.append(element_list_self / element_list_other)
            return SVCompatibleF1Score(aux_list)
        elif isinstance(other, int):
            for element_list_self in self.get_value():
                aux_list.append(element_list_self / other)
            return SVCompatibleF1Score(aux_list)

    def __abs__(self):
        return SVCompatibleF1Score([abs(f1_score_value) for f1_score_value in self._f1_score])

    def obtain_min_or_max(self, other, func):
        list_of_min = [function(v1, v2) for v1, v2, function in zip(self._f1_score, other.get_value(), func)]
        return SVCompatibleF1Score(list_of_min)

    def addition_or_substraction(self, other, func):
        aux_list = []
        for f1_score_value, other_f1_score_value, function in zip(self.get_value(), other.get_value(), func):
            if function == "add":
                aux_list.append(f1_score_value + other_f1_score_value)
            elif function == "sub":
                aux_list.append(f1_score_value - other_f1_score_value)
        return SVCompatibleF1Score(aux_list)

    # def max_out_two(self, other):
    #     list_of_max = [max(v1, v2) for v1, v2 in zip(self._f1_score, other.get_value())]
    #     return SVCompatibleF1Score(list_of_max)


class SVCompatibleWeightedF1Score(SVCompatibleF1Score):

    def __init__(self, initial_f1_score=None, num_classes=None):
        super().__init__(initial_f1_score, num_classes)

    def __truediv__(self, other):
        aux_list = []
        if isinstance(other, SVCompatibleWeightedF1Score):
            for element_list_self, element_list_other in zip(self.get_value(), other.get_value()):
                aux_list.append(element_list_self / element_list_other)
            return SVCompatibleWeightedF1Score(aux_list)
        elif isinstance(other, int):
            for element_list_self in self.get_value():
                aux_list.append(element_list_self / other)
            return SVCompatibleWeightedF1Score(aux_list)


class SVCompatibleMatthewsCorrelationCoefficient(Metric):
    _name: str
    _mcc_value: float

    def __init__(self, initial_mcc_value=None):
        self._name = "MCC"
        if initial_mcc_value is None:
            self._mcc_value = 0
        else:
            self._mcc_value = initial_mcc_value

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._mcc_value

    def set_value(self, value):
        self._mcc_value = value

    def __add__(self, other):
        return SVCompatibleMatthewsCorrelationCoefficient(self.get_value() + other.get_value())

    def __sub__(self, other):
        return SVCompatibleMatthewsCorrelationCoefficient(self.get_value() - other.get_value())

    def __mul__(self, other):
        if isinstance(other, SVCompatibleMatthewsCorrelationCoefficient):
            return SVCompatibleMatthewsCorrelationCoefficient(self.get_value() * other.get_value())
        elif isinstance(other, int):
            return SVCompatibleMatthewsCorrelationCoefficient(self.get_value() * other)

    def __truediv__(self, other):
        if isinstance(other, SVCompatibleMatthewsCorrelationCoefficient):
            return SVCompatibleMatthewsCorrelationCoefficient(self.get_value() / other.get_value())
        elif isinstance(other, int):
            return SVCompatibleMatthewsCorrelationCoefficient(self.get_value() / other)

    def __abs__(self):
        return SVCompatibleMatthewsCorrelationCoefficient(abs(self._mcc_value))

    def obtain_min_or_max(self, other, func):
        return SVCompatibleMatthewsCorrelationCoefficient(func(self._mcc_value, other.get_value()))

    def addition_or_substraction(self, other, func):
        if func == "add":
            return self + other
        elif func == "sub":
            return self - other


class RMSE(Metric):
    _name: str
    _rmse_value: float

    def __init__(self, initial_rmse_value=None):
        self._name = "RMSE"
        if initial_rmse_value is None:
            self._rmse_value = 0
        else:
            self._rmse_value = initial_rmse_value

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
        

class RSE(Metric):
    _name: str
    _rse_value: float

    def __init__(self, initial_rse_value=None):
        self._name = "RSE"
        if initial_rse_value is None:
            self._rse_value = 0
        else:
            self._rse_value = initial_rse_value

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._rse_value

    def set_value(self, value):
        self._rse_value = value

    def __add__(self, other):
        return RSE(self.get_value() + other.get_value())

    def __sub__(self, other):
        return RSE(other.get_value() - self.get_value())  # Turned around for the Shapley Values

    def __mul__(self, other):
        if isinstance(other, RSE):
            return RSE(self.get_value() * other.get_value())
        elif isinstance(other, int):
            return RSE(self.get_value() * other)

    def __truediv__(self, other):
        if isinstance(other, RSE):
            return RSE(self.get_value() / other.get_value())
        elif isinstance(other, int):
            return RSE(self.get_value() / other)

    def __abs__(self):
        return RSE(abs(self._rse_value))

    def obtain_min_or_max(self, other, func):
        return RSE(func(self._rse_value, other.get_value()))

    def addition_or_substraction(self, other, func):
        if func == "add":
            return self + other
        elif func == "sub":
            return self - other


class MAE(Metric):
    _name: str
    _mae_value: float

    def __init__(self, initial_mae_value=None):
        self._name = "MAE"
        if initial_mae_value is None:
            self._mae_value = 0
        else:
            self._mae_value = initial_mae_value

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

    def __init__(self, first_list_of_metrics=None):
        if first_list_of_metrics is None:
            first_list_of_metrics = {}
        self._dictionary_of_metrics = first_list_of_metrics

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
        dict_of_metrics_1 = self.get_value()
        dict_of_metrics_2 = other.get_value()
        aux_dict = {}
        for key in dict_of_metrics_1.keys():
            aux_dict[key] = dict_of_metrics_1[key] + dict_of_metrics_2[key]
        return DictOfMetrics(aux_dict)

    def __sub__(self, other):
        dict_of_metrics_1 = self.get_value()
        dict_of_metrics_2 = other.get_value()
        aux_dict = {}
        for key in dict_of_metrics_1.keys():
            aux_dict[key] = dict_of_metrics_1[key] - dict_of_metrics_2[key]
        return DictOfMetrics(aux_dict)

    def __mul__(self, other):
        dict_of_metrics_1 = self.get_value()
        aux_dict = {}
        if isinstance(other, DictOfMetrics):
            dict_of_metrics_2 = other.get_value()
            for key in dict_of_metrics_1.keys():
                aux_dict[key] = dict_of_metrics_1[key] * dict_of_metrics_2[key]
        elif isinstance(other, (int, float)):
            for key in dict_of_metrics_1.keys():
                aux_dict[key] = dict_of_metrics_1[key] * other
        return DictOfMetrics(aux_dict)

    def __truediv__(self, other):
        aux_dict = {}
        dict_of_metrics_1 = self.get_value()
        if isinstance(other, DictOfMetrics):
            dict_of_metrics_2 = other.get_value()
            for key in dict_of_metrics_1.keys():
                aux_dict[key] = dict_of_metrics_1[key] / dict_of_metrics_2[key]
        elif isinstance(other, (int, float)):
            for key in dict_of_metrics_1.keys():
                aux_dict[key] = dict_of_metrics_1[key] / other
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


def return_default_dict_of_metrics(metrics, num_classes):
    metric_dict = DictOfMetrics()
    if "CrossEntropyLoss" in metrics:
        metric_dict.add_metric(CrossEntropyLoss())
    if "Accuracy" in metrics:
        metric_dict.add_metric(Accuracy())
    if "F1Score" in metrics:
        metric_dict.add_metric(SVCompatibleF1Score(num_classes=num_classes))
    if "WeightedF1Score" in metrics:
        metric_dict.add_metric(SVCompatibleWeightedF1Score(num_classes=num_classes))
    if "MCC" in metrics:
        metric_dict.add_metric(SVCompatibleMatthewsCorrelationCoefficient())
    if "F1ScoreMacro" in metrics:
        metric_dict.add_metric(F1ScoreMacro())
    if "F1ScoreMicro" in metrics:
        metric_dict.add_metric(F1ScoreMicro())
    if "RMSE" in metrics:
        metric_dict.add_metric(RMSE())
    if "RSE" in metrics:
        metric_dict.add_metric(RSE())
    if "MAE" in metrics:
        metric_dict.add_metric(MAE())
    if "R2" in metrics:
        metric_dict.add_metric(R2())
    return metric_dict


if __name__ == "__main__":
    default_dict = return_default_dict_of_metrics(["F1Score", "F1ScoreMacro", "F1ScoreMicro"], 6)

    print(default_dict.return_flower_dict_as_str())
