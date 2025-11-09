from abc import ABC, abstractmethod
from typing import Type, List, Tuple

from flwr.client import NumPyClient
from flwr.server.strategy import Strategy
from pandas import DataFrame

from experiment_parameters.data_preparation.Dataset import IrisDataset, WineDataset, Dataset, \
    HumanActivityRecognitionDataset, \
    AdultDataset, CovertypeDataset, HeartDataset, HeartDatasetBinary, NewAdultDataset, EdgeIIoTDataset, \
    EdgeIIoTCoresetDataset, AirQualityDataset, ElectricConsumptionDataset
from experiment_parameters.model_builder.Model import Model
from experiment_parameters.model_builder.ModelBuilder import Director
from experiment_parameters.strategies.client.FedAvgClient import FedAvgClient
from experiment_parameters.strategies.client.FedDGKDClient import FedDGKDClient
from experiment_parameters.strategies.client.FedGKDClient import FedGKDClient
from experiment_parameters.strategies.client.FedLFClient import FedLFClient
from experiment_parameters.strategies.client.FedXGBoostClient import FedXGBoostClient
from experiment_parameters.strategies.server.FedAvgRewritten import FedAvgRewritten
# from experiment_parameters.strategies.server.FedDGKD import FedDGKD
# from experiment_parameters.strategies.server.FedDKD import FedDKD
# from experiment_parameters.strategies.server.FedGKD import FedGKD
# from experiment_parameters.strategies.server.FedKD import FedKD
# from experiment_parameters.strategies.server.FedLF import FedLF
from experiment_parameters.strategies.server.FedXgbBagging import FedXgbBagging

"""
Two Abstract Factories in this file:
    1. Factory for Strategy-Client
    2. Factory for Dataset-Model

First strategies and their corresponding code on client.

Second, factories for retrieving the dataset and the models with
parametrized for such datasets.
"""


class AbstractFactoryStrategyClient(ABC):
    @abstractmethod
    def create_strategy(self) -> Type[Strategy]:
        pass

    @abstractmethod
    def create_client(self) -> Type[NumPyClient]:
        pass


class ConcreteFactoryFedAvg(AbstractFactoryStrategyClient):

    def create_strategy(self) -> Type[Strategy]:
        return FedAvgRewritten

    def create_client(self) -> Type[NumPyClient]:
        return FedAvgClient


# class ConcreteFactoryCentralizedKD(AbstractFactoryStrategyClient):
#
#     def create_strategy(self) -> Type[Strategy]:
#         return FedKD
#
#     def create_client(self) -> Type[NumPyClient]:
#         return FedAvgClient


class ConcreteFactoryXGBoostBagging(AbstractFactoryStrategyClient):

    def create_strategy(self) -> Type[Strategy]:
        return FedXgbBagging

    def create_client(self) -> Type[NumPyClient]:
        return FedXGBoostClient


# class ConcreteFactoryLocalKnowledgeFusion(AbstractFactoryStrategyClient):
#
#     def create_strategy(self) -> Type[Strategy]:
#         return FedLF
#
#     def create_client(self) -> Type[NumPyClient]:
#         return FedLFClient
#
#
# class ConcreteFactoryGlobalKD(AbstractFactoryStrategyClient):
#
#     def create_strategy(self) -> Type[Strategy]:
#         return FedGKD
#
#     def create_client(self) -> Type[NumPyClient]:
#         return FedGKDClient
#
#
# class ConcreteFactoryDecoupledGKD(AbstractFactoryStrategyClient):
#
#     def create_strategy(self) -> Type[Strategy]:
#         return FedDGKD
#
#     def create_client(self) -> Type[NumPyClient]:
#         return FedDGKDClient
#
#
# class ConcreteFactoryDecoupledKD(AbstractFactoryStrategyClient):
#
#     def create_strategy(self) -> Type[Strategy]:
#         return FedDKD
#
#     def create_client(self) -> Type[NumPyClient]:
#         return FedAvgClient


class AbstractFactoryModelPerDataset(ABC):
    """
    In this abstract factory, you need to update whenever you add a new type of model.
    """
    _dataset: Dataset

    def get_dataset(self) -> Dataset:
        return self._dataset

    def get_input_dimensions(self):
        return self._dataset.x_test.shape[1]

    def get_output_dimensions(self):
        return self._dataset.y_test.shape[1]


class ConcreteFactoryDatasetWine(AbstractFactoryModelPerDataset):

    def __init__(self):
        self._dataset = WineDataset()


class ConcreteFactoryDatasetIris(AbstractFactoryModelPerDataset):

    def __init__(self):
        self._dataset = IrisDataset()


class ConcreteFactoryDatasetHumanActivityRecognition(AbstractFactoryModelPerDataset):
    def __init__(self):
        self._dataset = HumanActivityRecognitionDataset()

    def get_synthetic_dataset(self) -> Tuple[DataFrame, DataFrame]:
        pass


class ConcreteFactoryDatasetAdult(AbstractFactoryModelPerDataset):
    def __init__(self):
        self._dataset = AdultDataset()


class ConcreteFactoryDatasetCovertype(AbstractFactoryModelPerDataset):
    def __init__(self):
        self._dataset = CovertypeDataset()


class ConcreteFactoryDatasetHeart(AbstractFactoryModelPerDataset):

    def __init__(self):
        self._dataset = HeartDataset()


class ConcreteFactoryDatasetBinaryHeart(AbstractFactoryModelPerDataset):

    def __init__(self):
        self._dataset = HeartDatasetBinary()


class ConcreteFactoryDatasetNewAdult(AbstractFactoryModelPerDataset):

    def __init__(self):
        self._dataset = NewAdultDataset()


class ConcreteFactoryEdgeIoT(AbstractFactoryModelPerDataset):

    def __init__(self):
        self._dataset = EdgeIIoTDataset()


class ConcreteFactoryEdgeIoTCoreset(AbstractFactoryModelPerDataset):
    def __init__(self):
        self._dataset = EdgeIIoTCoresetDataset()


class ConcreteFactoryAirQuality(AbstractFactoryModelPerDataset):
    def __init__(self):
        self._dataset = AirQualityDataset()


class ConcreteFactoryElectricConsumption(AbstractFactoryModelPerDataset):
    def __init__(self):
        self._dataset = ElectricConsumptionDataset()


class ModelBuilder(ABC):

    @abstractmethod
    def set_input(self):
        pass

    @abstractmethod
    def create_mlp(self, input_dim: int, neurons_and_activation_function_layer: List[Tuple[int, str]], output_dim: int):
        pass

    @abstractmethod
    def create_tabnet(self, input_dim: int, output_dim: int):
        pass

    @abstractmethod
    def define_compiler(self):
        pass


class ModelBuilderSoftmax(ModelBuilder):

    @abstractmethod
    def set_input(self):
        pass

    @abstractmethod
    def create_mlp(self, input_dim: int, neurons_and_activation_function_layer: List[Tuple[int, str]], output_dim: int):
        pass

    @abstractmethod
    def create_tabnet(self, input_dim, output_dim):
        pass

    @abstractmethod
    def define_compiler(self):
        pass


class ModelBuilderLogits(ModelBuilder):
    """
    Unused class, because at the moment, I am not thinking on adding parametrization for models
    from the arguments parsing.
    """

    @abstractmethod
    def set_input(self):
        pass

    @abstractmethod
    def create_mlp(self, input_dim: int, neurons_and_activation_function_layer: List[Tuple[int, str]], output_dim: int):
        pass

    @abstractmethod
    def create_tabnet(self, input_dim, output_dim):
        pass

    @abstractmethod
    def define_compiler(self):
        pass


# class Director:
#     builder: ModelBuilder
#
#     def __init__(self, builder):
#         self.builder = builder
#
#     def builder(self) -> ModelBuilder:
#         return self.builder
#
#     def build_mpl(self):
#         self.builder.initialize_data()
#         self.builder.initialize_model()
#         self.builder.initialize_strategy()
#         self.builder.initialize_client()


########## Export Dictionaries ##########

"""Add here instances of new concrete Factories."""

strategies_dictionary = {
    "FedAvg": ConcreteFactoryFedAvg,
    # "FedKD": ConcreteFactoryCentralizedKD,
    # "FedLF": ConcreteFactoryLocalKnowledgeFusion,
    # "FedGKD": ConcreteFactoryGlobalKD,
    # "FedDKD": ConcreteFactoryDecoupledKD,
    # "FedDGKD": ConcreteFactoryDecoupledGKD,
    "FedXGB": ConcreteFactoryXGBoostBagging
}


dataset_model_dictionary = {
    "wine": ConcreteFactoryDatasetWine,
    # "iris": ConcreteFactoryDatasetIris,
    "har": ConcreteFactoryDatasetHumanActivityRecognition,
    "adult": ConcreteFactoryDatasetAdult,
    "covertype": ConcreteFactoryDatasetCovertype,
    "heart": ConcreteFactoryDatasetHeart,
    "binary_heart": ConcreteFactoryDatasetBinaryHeart,
    "new_adult": ConcreteFactoryDatasetNewAdult,
    "edge-iot": ConcreteFactoryEdgeIoT,
    "edge-iot-coreset": ConcreteFactoryEdgeIoTCoreset,
    "air-quality": ConcreteFactoryAirQuality,
    "electric-consumption": ConcreteFactoryElectricConsumption
}


def factory_return_model(factory, model: str, parameters) -> Model:
    """
    Function to add the models corresponding to the different datasets.
    """
    director = Director()
    if model == "mlp":
        input_parameters = factory.get_input_dimensions()
        num_classes = factory.get_output_dimensions()
        return director.create_mlp(input_parameters, num_classes, parameters)

    # elif model == "tabnet":
    #     input_parameters = factory.get_input_dimensions()
    #     num_classes = factory.get_output_dimensions()
    #     model = director.create_tabnet(input_parameters, num_classes, parameters)
    #
    #     X, y = factory.get_dataset().get_training_data()
    #     first_X, first_y = X.iloc[0:2], y.iloc[0:2]
    #     model_to_initialize = model.get_model()
    #     model_to_initialize.fit(x=first_X, y=first_y, epochs=1)
    #     model.set_model(model_to_initialize)
    #     return model

    elif model == "xgboost":
        input_parameters = factory.get_input_dimensions()
        num_classes = factory.get_output_dimensions()
        return director.create_xgboost(input_parameters, num_classes, parameters)
