import ast
import os
from logging import INFO
from typing import Any, List

import flwr as fl
# Define Flower client
import numpy as np
import xgboost as xgb
from flwr.common import Code, EvaluateIns, EvaluateRes, FitRes, FitIns, GetParametersIns, \
    GetParametersRes, Status, Parameters
from flwr.common.logger import log
from numpy import ndarray
from pandas import DataFrame

from Definitions import ROOT_DIR
from experiment_parameters.model_builder.Model import XGBoostModel
from experiment_parameters.model_builder.ModelBuilder import Director
from metrics.Evaluator import evaluator
from metrics.Metrics import DictOfMetrics, return_default_dict_of_metrics
from metrics.Shapley_Values import ShapleyValuesDT
from util.Util import load_data_from_pickle_file


# TODO:
# - Dynamic XGBoost Configuration (Maybe through experimentation config)

class FedXGBoostClient(fl.client.Client):
    _model: Any
    _model_name: str
    _x_train: DataFrame
    _x_test: DataFrame
    _y_train: DataFrame
    _y_test: DataFrame
    _batch_size: int
    _shapley_values: ShapleyValuesDT
    _client_number: int
    _metric_list: list
    _early_stopping_rounds: int
    _last_round_result = DictOfMetrics
    num_local_round = 2
    # Keep initial parameters to initialize SV.
    initial_parameters: List[ndarray] = None

    def __init__(self, model, x_train, x_test, y_train, y_test, client_number, metrics, total_data_size):
        self.cid = client_number
        self._model_name = model
        self._client_number = client_number
        self._metric_list = metrics
        self.bst: XGBoostModel = XGBoostModel()
        self.parameters_config = None
        self.config = None
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        # self.train_dmatrix = xgb.DMatrix(x_train, label=np.argmax(y_train, axis=1))
        self.train_len = len(x_train)
        self.train_label_len = len(y_train)
        # self.valid_dmatrix = xgb.DMatrix(x_test, label=np.argmax(y_test, axis=1))
        self.test_len = len(x_test)
        self.test_label_len = len(y_test)
        self._early_stopping_rounds = 5
        self._total_data_size = total_data_size

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:

        log(INFO, f"get_parameters()")
        self.config = ins.config
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def fit(self, ins: FitIns):
        log(INFO, "fit()")
        labels, counts = np.unique(np.argmax(np.asarray(self._y_train), axis=1), return_counts=True)
        label_names = self._y_train.columns
        metrics = {"client_number": self._client_number}
        for label, count in zip(labels, counts):
            metrics["Label " + str(label_names[label])] = int(count)

        if ins.config["server_round"] == 1:
            director = Director()
            self.parameters_config = director.create_xgboost(self._x_train.shape[1], self._y_train.shape[1], ins.config)
            self.parameters_config["eta"] = self.parameters_config["eta"] * (len(self._x_train) / self._total_data_size)
            log(INFO, f"Parameters: {self.parameters_config}")
            self.num_local_round = self.parameters_config.pop("num_boost_rounds")
            # First round local training
            log(INFO, "Start training at round 1")
            self.bst.fit(self.parameters_config,
                         self._x_train,
                         self._y_train,
                         self._x_test,
                         self._y_test,
                         self.num_local_round)
            # self.config = bst.save_config()
            log(INFO, "Finished training")
            bst = self.bst.get_model()
            local_model = bst.save_raw("json")
            local_model_bytes = bytes(local_model)
            self.bst.set_model(local_model_bytes)

            self._last_round_result = evaluator(self._x_test,
                                                self._y_test,
                                                self.bst,
                                                self._metric_list)
        else:
            log(INFO, "Local Boost Round")
            # self.parameters_config = director.create_xgboost(self._x_train.shape[1], self._y_train.shape[1], ins.config)
            # bst = xgb.Booster(self.parameters_config)
            # for item in ins.parameters.tensors:
            # global_model = bytearray(ins.parameters.tensors[0])

            # Load global model into booster
            # self.bst.load_model(global_model)
            # self.bst.load_config(self.config)
            self.bst.set_model(ins.parameters.tensors[0])
            # bst = self._local_boost(self.bst.get_model())
            previous_model = self.bst.get_model()
            self.bst.fit(self.parameters_config,
                         self._x_train,
                         self._y_train,
                         self._x_test,
                         self._y_test,
                         self.num_local_round,
                         early_stopping_rounds=None,
                         previous_xgb_model=previous_model)
            local_model = self.bst.get_model().save_raw("json")
            local_model_bytes = bytes(local_model)
            self.bst.set_model(local_model_bytes)

        # log(INFO, "Saving models")
        # local_model = bst.save_raw("json")
        # local_model_bytes = bytes(local_model)
        # self.bst.set_model(local_model_bytes)
        # Parameters Object cannot be transformed into ndarray
        # Send the information in the dict for metrics
        # Should be changed in a newer version of flower
        log(INFO, "Send local model to server")
        # return [], self.train_len, {"local_model": local_model_bytes}
        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.train_len,
            metrics=metrics,
        )

    # def _local_boost(self, bst_input):
    # Update trees based on local training data.
    # for i in range(self.num_local_round):
    #     # self.bst.update(self.train_dmatrix, self.bst.num_boosted_rounds())
    #     bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())
    # # Extract the last N=num_local_round trees for sever aggregation
    # bst = (bst_input[
    #        bst_input.num_boosted_rounds()
    #        - self.num_local_round: bst_input.num_boosted_rounds()
    #        ])
    # bst = xgb.train(
    #     self.parameters_config,
    #     self.train_dmatrix,
    #     evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
    #     num_boost_round=self.num_local_round,
    #     # early_stopping_rounds=self._early_stopping_rounds,
    #     xgb_model=bst_input
    # )

    def evaluate(self, ins: EvaluateIns):
        log(INFO, f"evaluate client()")
        # bst = xgb.Booster(params=self.parameters_config)
        # para_b = bytearray(ins.parameters.tensors[0])
        self.bst.set_model(ins.parameters.tensors[0])

        round_result = evaluator(x_test=self._x_test,
                                 y_test=self._y_test,
                                 model=self.bst,
                                 metric_list=self._metric_list)
        metric_results = round_result.return_flower_dict_as_str()

        if ins.config["compute_shapley_values"] == 1:
            client_weights = load_data_from_pickle_file(ROOT_DIR +
                                                        os.sep + "data" +
                                                        os.sep + "pickled_information" +
                                                        os.sep + "model.pkl")

            if ins.config["server_round"] == 1:
                number_of_clients = len(client_weights)
                # self._shapley_values = ShapleyGTG(number_of_clients)
                self._shapley_values = ShapleyValuesDT(self._x_test,
                                                       self._y_test,
                                                       ins.config["num_rounds"],
                                                       self._metric_list)
                list_of_initial_metrics = self._last_round_result
                self._shapley_values.set_last_round_results(list_of_initial_metrics)
                # for client_name
                client_number_dict = ast.literal_eval(ins.config["client_cid_number"])
                # self._shapley_values.set_client_index_dictionary(client_weights.keys())
                self._shapley_values.set_client_index_dictionary(client_number_dict)

            self._shapley_values.shapley_values_calculation(self.bst,
                                                            list(client_weights.keys()),
                                                            client_weights,
                                                            ins.config["server_round"],
                                                            bytes(self.bst.get_model().save_raw("json")))
            self._shapley_values.set_last_round_results(round_result)
            round_sv = self._shapley_values.get_round_shapley_values(ins.config["server_round"])
            sv_round_result = {"SV_" + client_id: str(round_sv[client_id])
                               for client_id, _ in self._shapley_values.get_client_index_dictionary().items()}
            metric_results = metric_results | sv_round_result

            if ins.config["last_round"] == 1:
                log(INFO, "Columns sorted: {}".format(list(self._y_test.columns)))
                labels, counts = np.unique(np.argmax(np.asarray(self._y_test), axis=1), return_counts=True)
                testing_labels = {label: count for label, count in zip(labels, counts)}
                log(INFO, "Testing labels: {}".format(testing_labels))
                log(INFO, "=" * 50)
                shapley_values_total = self._shapley_values.get_shapley_values()
                for training_round in shapley_values_total.keys():
                    log(INFO, "Shapley Values in local_round {}: {}"
                        .format(training_round,
                                {client: str(sv) for client, sv in shapley_values_total[training_round].items()}))
                log(INFO, "=" * 50)
                log(INFO, "=" * 50)
                shapley_values_total = self._shapley_values.get_shapley_values()
                result_dictionary = {client_id: return_default_dict_of_metrics(self._metric_list, self._y_test.shape[1])
                                     for client_id in client_weights.keys()}
                for training_round in shapley_values_total.keys():
                    single_round_dict = shapley_values_total.get(training_round)
                    result_dictionary = {k: result_dictionary[k] + single_round_dict[k] for k in
                                         result_dictionary.keys()}
                # result_dictionary = {k: str(shapley_values_client) for k, shapley_values_client in
                #                      result_dictionary.items()}
                log(INFO, "Result_dictionary: {}".format(result_dictionary))
                # os.makedirs(ROOT_DIR + os.sep + "metrics" + os.sep + "pickled_information", exist_ok=True)
                # save_data_on_pickle(ROOT_DIR +
                #                     os.sep + "metrics" +
                #                     os.sep + "pickled_information" +
                #                     os.sep + "sv" + str(self._client_number) + ".pkl", shapley_values_total)
                log(INFO, "=" * 50)

        # eval_flower_dict = round_result.return_flower_dict_as_str()
        log(INFO, f"Eval_flower_dict: {metric_results}")
        cross_entropy_loss = float(metric_results["CrossEntropyLoss"])
        log(INFO, "Ending evaluate client")
        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=cross_entropy_loss,
            num_examples=self.test_len,
            metrics=metric_results,
        )
