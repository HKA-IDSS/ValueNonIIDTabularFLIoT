from abc import ABC

import keras

from experiment_parameters.data_preparation.Dataset import WineDataset
from experiment_parameters.model_builder.Model import KerasModel, XGBoostModel, MLPModel, LSTMModel

from tensorflow.keras import layers, models, applications, Input, Model


def get_training_configuration(trial, model_type):
    dict_parameters = {}

    if model_type == "mlp" or model_type == "tabnet":
        dict_parameters["batch_size"] = trial.suggest_int("batch_size", 64, 512, log=True)
        dict_parameters["learning_rate_init"] = trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True)
        dict_parameters["decay_steps"] = trial.suggest_int('decay_steps', 500, 2000, step=500)
        dict_parameters["decay_rate"] = trial.suggest_float('decay_rate', 0.8, 0.95, step=0.05)

        if model_type == "mlp":
            num_layers = trial.suggest_int("num_layers", 1, 4)
            dict_parameters["num_layers"] = num_layers
            for layer in range(1, num_layers + 1):
                dict_parameters[f'n_units_l{layer}'] = trial.suggest_int(f'n_units_l{layer}', 4, 128)
                dict_parameters[f'dropout_l{layer}'] = trial.suggest_float(f'dropout_l{layer}', 0, 0.4, step=0.1)
                dict_parameters[f'activation_l{layer}'] = trial.suggest_categorical(f'activation_l{layer}',
                                                                                    ["relu", "tanh"])
        elif model_type == "lstm":
            num_layers = trial.suggest_int("num_layers", 1, 2)
            dict_parameters["num_layers"] = num_layers
            for layer in range(1, num_layers + 1):
                dict_parameters[f'n_units_l{layer}'] = trial.suggest_int(f'n_units_l{layer}', 4, 128)
                # dict_parameters[f'dropout_l{layer}'] = trial.suggest_float(f'dropout_l{layer}', 0.1, 0.4, stepx=0.1)
                dict_parameters[f'activation_l{layer}'] = trial.suggest_categorical(f'activation_l{layer}',
                                                                                    ["relu", "tanh"])

    elif model_type == "xgboost":
        # dict_parameters["objective"]
        dict_parameters["tree_method"] = "hist"
        # dict_parameters["booster"] = trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"])
        dict_parameters["booster"] = trial.suggest_categorical("booster", ["gbtree"])
        # dict_parameters["device"] = "cuda"
        dict_parameters["eta"] = trial.suggest_categorical("eta", [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3])
        dict_parameters["max_depth"] = trial.suggest_int("max_depth", 2, 10)
        dict_parameters["gamma"] = trial.suggest_float("gamma", 0, 0.5)
        dict_parameters["subsample"] = 1
        dict_parameters["max_delta_step"] = trial.suggest_int("max_delta_step", 0, 10)
        dict_parameters["lambda"] = trial.suggest_float("lambda", 0, 1)
        dict_parameters["alpha"] = trial.suggest_float("alpha", 0, 1)
        dict_parameters["min_child_weight"] = trial.suggest_int('min_child_weight', 0, 3)
        dict_parameters["seed"] = 1
        dict_parameters["num_local_rounds"] = trial.suggest_int('num_local_rounds', 1, 2)

    return dict_parameters


class ModelBuilder(ABC):
    ml_model: Model

    def return_model(self):
        pass

    def build_model(self):
        pass

    def define_model_hyperparameters(self, num_classes, parameters):
        pass

    def define_model_architecture(self, input_parameters, num_classes, parameters):
        pass


class KerasModelBuilder(ModelBuilder):
    ml_model: KerasModel

    def return_model(self):
        return self.ml_model

    def build_model(self):
        pass


class NNBuilder(KerasModelBuilder):
    def define_model_hyperparameters(self, num_classes, parameters):
        lr = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=parameters["learning_rate_init"],
            decay_steps=parameters["decay_steps"],
            decay_rate=parameters["decay_rate"],
            staircase=False
        )

        # TODO: Adapt the compiler for different problems.
        adam = keras.optimizers.Adam(lr)
        if num_classes == 1:
            self.ml_model.get_model().compile(adam,
                                              loss=keras.losses.MeanSquaredError(),
                                              metrics=[keras.metrics.MeanSquaredError()])
        else:
            self.ml_model.get_model().compile(adam,
                                              loss=keras.losses.CategoricalCrossentropy(),
                                              metrics=[keras.metrics.CategoricalAccuracy()])


class MLPBuilder(NNBuilder):
    def define_model_architecture(self, input_parameters, num_classes, parameters):
        ml_model = keras.Sequential()
        ml_model.add(keras.Input(shape=(input_parameters,)))  # Not really sure why, but Keras request a tuple
        for layer in range(1, parameters["num_layers"] + 1):
            ml_model.add(keras.layers.Dense(parameters["n_units_l" + str(layer)],
                                            activation=parameters["activation_l" + str(layer)],
                                            kernel_initializer=keras.initializers.HeNormal(seed=1)))
            ml_model.add(keras.layers.Dropout(parameters["dropout_l" + str(layer)]))

        if num_classes == 1:
            ml_model.add(keras.layers.Dense(1,
                                            activation="linear",
                                            kernel_initializer=keras.initializers.Zeros()))
        else:
            ml_model.add(keras.layers.Dense(num_classes,
                                            activation="softmax",
                                            kernel_initializer=keras.initializers.HeNormal(seed=1)))

        self.ml_model = MLPModel(ml_model)


class LSTMBuilder(NNBuilder):
    def define_model_architecture(self, input_parameters, num_classes, parameters):
        ml_model = keras.Sequential()
        ml_model.add(keras.Input(shape=(parameters["window_size"], input_parameters)))
        for layer in range(1, parameters["num_layers"] + 1):
            ml_model.add(keras.layers.LSTM(parameters["n_units_l" + str(layer)],
                                           activation=parameters["activation_l" + str(layer)],
                                           kernel_initializer=keras.initializers.Zeros()))
            # ml_model.add(keras.layers.Dropout(parameters["dropout_l" + str(layer)]))

        ml_model.add(keras.layers.Dense(num_classes,
                                        kernel_initializer=keras.initializers.Zeros()))

        self.ml_model = LSTMModel(ml_model)


class DecisionTreeBuilder(ModelBuilder):
    pass


class XGBoostModelBuilder(DecisionTreeBuilder):
    ml_model: XGBoostModel
    _params: dict

    def __init__(self):
        self._params = {}

    def return_model(self):
        return self._params

    def define_model_hyperparameters(self, num_classes, parameters):
        self._params["tree_method"] = parameters["tree_method"]
        self._params["booster"] = parameters["booster"]
        # At the moment, having a problem with memory allocation. Still, it works well with CPU.
        # self._params["device"] = "cuda"
        self._params["eta"] = parameters["eta"]
        self._params["max_depth"] = parameters["max_depth"]
        self._params["gamma"] = parameters["gamma"]
        self._params["subsample"] = parameters["subsample"]
        self._params["max_delta_step"] = parameters["max_delta_step"]
        self._params["lambda"] = parameters["lambda"]
        self._params["alpha"] = parameters["alpha"]
        self._params["min_child_weight"] = parameters["min_child_weight"]
        self._params["num_boost_rounds"] = parameters["num_local_rounds"]
        self._params["seed"] = 1
        self._params["seed_per_iteration"] = True

    def define_model_architecture(self, input_parameters, num_classes, parameters):
        if num_classes == 1:
            self._params["objective"] = "reg:squarederror"
            self._params["eval_metric"] = "rmse"
        elif num_classes == 2:
            self._params["objective"] = "binary:logistic"
            self._params["eval_metric"] = "logloss"
        elif num_classes > 2:
            self._params["objective"] = "multi:softprob"
            self._params['num_class'] = num_classes
            self._params["disable_default_eval_metric"] = 1
            self._params["eval_metric"] = "mlogloss"


# class TabnetBuilder(NNBuilder):
#     # def define_model_architecture(self, input_parameters, num_classes, parameters):
#     #     # ml_model = TabNetClassifier(feature_columns=None,
#     #     #                             num_features=input_parameters,
#     #     #                             num_classes=num_classes,
#     #     #                             feature_dim=parameters["output_dim"] + 1,
#     #     #                             output_dim=parameters["output_dim"],
#     #     #                             num_decision_steps=parameters["num_decision_steps"],
#     #     #                             relaxation_factor=parameters["relaxation_factor"],
#     #     #                             sparsity_coefficient=parameters["sparsity"],
#     #     #                             batch_momentum=parameters["batch_momentum"],
#     #     #                             virtual_batch_size=None, norm_type='group',
#     #     #                             num_groups=1)
#     #     # self.ml_model = KerasModel(ml_model)

class Director:
    model_builder: ModelBuilder

    def create_mlp(self, input_parameters, num_classes, parameters):
        self.model_builder = MLPBuilder()
        self.model_builder.build_model()
        self.model_builder.define_model_architecture(input_parameters, num_classes, parameters)
        self.model_builder.define_model_hyperparameters(num_classes, parameters)
        return self.model_builder.return_model()

    def create_lstm(self, input_parameters, num_classes, parameters):
        self.model_builder = LSTMBuilder()
        self.model_builder.build_model()
        self.model_builder.define_model_architecture(input_parameters, num_classes, parameters)
        self.model_builder.define_model_hyperparameters(num_classes, parameters)
        return self.model_builder.return_model()

    #  Generating a model with XGBoost is not trivial. What this function should do is return parameters
    #  for direct training. This is because XGBoost does not provide a function to create a model,
    #  then train it. This is why the model is not built here.
    def create_xgboost(self, input_parameters, num_classes, parameters):
        self.model_builder = XGBoostModelBuilder()
        self.model_builder.define_model_architecture(input_parameters, num_classes, parameters)
        self.model_builder.define_model_hyperparameters(num_classes, parameters)
        # self.model_builder.build_model()
        return self.model_builder.return_model()


def wine_optimization(trial):
    x_train, y_train = WineDataset().get_training_data()
    x_test, y_test = WineDataset().get_test_data()
    parameters = get_training_configuration(trial=trial, model_type="mlp")
    director = Director()
    ml_model = director.create_mlp(input_parameters=x_train.shape[1], num_classes=y_train.shape[1],
                                   parameters=parameters).get_model()
    ml_model.fit(x_train, y_train, epochs=20, batch_size=parameters["batch_size"])
    loss, accuracy = ml_model.evaluate(x_test, y_test)

    return loss  # , accuracy, mcc_result


if __name__ == "__main__":
    pass
    # model = NeuralNetworkWine().get_model()
    # optuna_study_test = OptunaConnection.optuna_create_study("wine_optimization", direction=['minimize'])
    # optuna_study_test.optimize(wine_optimization, n_trials=5)
