import math
import os

from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from metrics import FedBary

print(os.getcwd())
# os.chdir("..")
# print(os.getcwd())

import numpy

os.environ["KERAS_BACKEND"] = "torch" # This needs to be before importing torch

import torch
import pandas as pd
import numpy as np
import math
import xgboost as xgb
import prince
from experiment_parameters.TrainerFactory import dataset_model_dictionary

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
print(use_cuda)

from geomloss import SamplesLoss


# CONST_ROUTE_MAIN_DIR = ".."


def get_data_from_route(dataset_name, type_of_partition, additional_parameter, random_seed):
    if type_of_partition == "manual":
        path_to_train_datasets = "data" + os.sep + "partitioned_training_data" + os.sep + type_of_partition + os.sep + additional_parameter + os.sep + str(random_seed)
        # path_to_train_datasets = CONST_ROUTE_MAIN_DIR + os.sep + "data" + os.sep + "partitioned_training_data" + os.sep + type_of_partition + os.sep + additional_parameter + os.sep + str(random_seed)
    else:
        path_to_train_datasets = "data" + os.sep + "partitioned_training_data" + os.sep + type_of_partition + os.sep + "dataset_" + dataset_name + os.sep + "alpha_" + additional_parameter + os.sep + str(random_seed)
        # path_to_train_datasets = CONST_ROUTE_MAIN_DIR + os.sep + "data" + os.sep + "partitioned_training_data" + os.sep + type_of_partition + os.sep + "dataset_" + dataset_name + os.sep + "alpha_" + additional_parameter + os.sep + str(random_seed)

    return path_to_train_datasets

def get_results_from_route(aggregation_method, dataset_name, type_of_partition, additional_parameter, random_seed):
    # path_to_result_dataframes = CONST_ROUTE_MAIN_DIR + "results" + os.sep + "FedAvg" + os.sep + dataset_name + os.sep + type_of_partition + os.sep + additional_parameter + os.sep + "mlp"
    if type_of_partition == "dirichlet":
        path_to_result_dataframes = "results" + os.sep + "dataframes" + os.sep + aggregation_method + os.sep + dataset_name + os.sep + type_of_partition + os.sep + "alpha_" + additional_parameter + os.sep + "mlp" + os.sep + str(random_seed)
        # path_to_result_dataframes = CONST_ROUTE_MAIN_DIR + os.sep + "results" + os.sep + "dataframes" + os.sep + aggregation_method + os.sep + dataset_name + os.sep + type_of_partition + os.sep + "alpha_" + additional_parameter + os.sep + "mlp" + os.sep + str(random_seed)
    else:
        path_to_result_dataframes = "results" + os.sep + "dataframes" + os.sep + aggregation_method + os.sep + dataset_name + os.sep + type_of_partition + os.sep + additional_parameter + os.sep + "mlp" + os.sep + str(random_seed)
        # path_to_result_dataframes = CONST_ROUTE_MAIN_DIR + os.sep + "results" + os.sep + "dataframes" + os.sep + aggregation_method + os.sep + dataset_name + os.sep + type_of_partition + os.sep + additional_parameter + os.sep + "mlp" + os.sep + str(random_seed)
    return path_to_result_dataframes


def get_distances_from_route(dataset_name, type_of_partition, additional_parameter, random_seed):
    path_to_result_dataframes = "results" + os.sep + "distances_values" + os.sep + dataset_name + os.sep + type_of_partition + os.sep + additional_parameter + os.sep + str(random_seed)
    # path_to_result_dataframes = CONST_ROUTE_MAIN_DIR + os.sep + "results" + os.sep + "distances_values" + os.sep + dataset_name + os.sep + type_of_partition + os.sep + additional_parameter + os.sep + str(random_seed)
    return path_to_result_dataframes


from itertools import combinations_with_replacement

def number_of_clients_and_all_combinations(path_to_train_datasets, maverick=False):
    num_clients = int(len(os.listdir(path_to_train_datasets)) / 4)
    if maverick:
        num_clients -= 1

    client_numbers_original_order = list(range(num_clients))
    client_numbers_reverse_order = list(range(num_clients - 1, -1, -1))

    all_combinations = list(combinations_with_replacement(client_numbers_original_order, 2)) + list(combinations_with_replacement(client_numbers_reverse_order, 2))
    all_combinations = sorted(list(set(all_combinations)))
    return num_clients, all_combinations


def downcast_types(dataframe):
    for column in dataframe.select_dtypes("int"):
        dataframe[column] = dataframe[column].astype("int16")

    for column in dataframe.select_dtypes("float"):
        dataframe[column] = dataframe[column].astype("float32")

    return dataframe

#
# def compute_coupling(X_src, X_tar, Y_src, Y_tar):
#     cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)
#
#     C = cost_function(X_src, X_tar)
#     P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(X_tar.shape[0]), C, numItermax=100000)
#     W = np.sum(P * np.array(C.numpy()))
#
#     return P, W
#
#
# def compute_CE(P, Y_src, Y_tar):
#     src_label_set = set(sorted(list(Y_src.flatten())))
#     tar_label_set = set(sorted(list(Y_tar.flatten())))
#
#     # joint distribution of source and target label
#     P_src_tar = np.zeros((np.max(Y_src) + 1, np.max(Y_tar) + 1))
#
#     for y1 in src_label_set:
#         y1_idx = np.where(Y_src == y1)
#         for y2 in tar_label_set:
#             y2_idx = np.where(Y_tar == y2)
#
#             RR = y1_idx[0].repeat(y2_idx[0].shape[0])
#             CC = np.tile(y2_idx[0], y1_idx[0].shape[0])
#
#             P_src_tar[y1, y2] = np.sum(P[RR, CC])
#
#     # marginal distribution of source label
#     P_src = np.sum(P_src_tar, axis=1)
#
#     ce = 0.0
#     for y1 in src_label_set:
#         P_y1 = P_src[y1]
#         for y2 in tar_label_set:
#
#             if P_src_tar[y1, y2] != 0:
#                 ce += -(P_src_tar[y1, y2] * math.log(P_src_tar[y1, y2] / P_y1))
#     return ce


# def wasserstein_distance_and_conditional_entropy(src_x, tar_x, src_y, tar_y):
#     print("Wasserstein Distance()")
#     # Define a Sinkhorn (~Wasserstein) loss between sampled measures
#     loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.5, verbose=True) # Although the euclidean distance usually square's root the results, it is not done here. No clue.
#
#     L = loss(torch.from_numpy(src_x).contiguous(), torch.from_numpy(tar_x).contiguous())  # By default, use constant weights = 1/number of samples
#     if use_cuda:
#         torch.cuda.synchronize()
#     return 2 * L.item()

def gaussian_mmd_distance(src_x, tar_x):
    print("Gaussian MMD distance()")
    # Define a Gaussian MMD loss between sampled measures
    loss = SamplesLoss(loss="gaussian", blur=0.05, scaling=0.5, verbose=True) # Although the euclidean distance usually square's root the results, it is not done here. No clue.

    L = loss(torch.from_numpy(src_x).contiguous(), torch.from_numpy(tar_x).contiguous())  # By default, use constant weights = 1/number of samples
    if use_cuda:
        torch.cuda.synchronize()
    return L.item()

def negative_conditional_entropy(source_labels: np.ndarray, target_labels: np.ndarray):
    r"""
    Negative Conditional Entropy in `Transferability and Hardness of Supervised
    Classification Tasks (ICCV 2019) <https://arxiv.org/pdf/1908.08142v1.pdf>`_.

    The NCE :math:`\mathcal{H}` can be described as:

    .. math::
        \mathcal{H}=-\sum_{y \in \mathcal{C}_t} \sum_{z \in \mathcal{C}_s} \hat{P}(y, z) \log \frac{\hat{P}(y, z)}{\hat{P}(z)}

    where :math:`\hat{P}(z)` is the empirical distribution and :math:`\hat{P}\left(y \mid z\right)` is the empirical
    conditional distribution estimated by source and target label.

    Args:
        source_labels (np.ndarray): predicted source labels.
        target_labels (np.ndarray): groud-truth target labels.

    Shape:
        - source_labels: (N, ) elements in [0, :math:`C_s`), with source class number :math:`C_s`.
        - target_labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
    """
    print("Negative conditional entropy()")
    C_t = int(np.max(target_labels) + 1)
    C_s = int(np.max(source_labels) + 1)
    N = len(source_labels)

    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for the joint distribution, shape [C_t, C_s]
    for s, t in zip(source_labels, target_labels):
        s = int(s)
        t = int(t)
        joint[t, s] += 1.0 / N
    p_z = joint.sum(axis=0, keepdims=True)

    # if p_z == 0 or p_z is None:
    #     print(f"p_z: {p_z}")

    p_target_given_source = (joint / p_z).T  # P(y | z), shape [C_s, C_t] # some problem here. Probably why values get so big
    mask = p_z.reshape(-1) != 0  # valid Z, shape [C_s]
    p_target_given_source = p_target_given_source[mask] + 1e-20  # remove NaN where p(z) = 0, add 1e-20 to avoid log (0)
    entropy_y_given_z = np.sum(- p_target_given_source * np.log(p_target_given_source), axis=1, keepdims=True)
    conditional_entropy = np.sum(entropy_y_given_z * p_z.reshape((-1, 1))[mask])

    return -conditional_entropy


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, DMatrix


def accuracy(y_test, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    ground_truth_np = np.argmax(y_test, axis=1)
    return accuracy_score(np.asarray(ground_truth_np), np.asarray(y_pred))

def degradation_decomp(source_X, source_y, other_X_raw, other_y_raw, best_method, column_names, data_sum=20000, K=8, domain_classifier=None, draw_calibration=False, save_calibration_png='calibration.png'):
    print("Degradation Decomposition()")
    perm1 = np.random.permutation(other_X_raw.shape[0])
    other_X = other_X_raw[perm1[:data_sum],:]
    other_y = other_y_raw[perm1[:data_sum]]

    piA = np.zeros(source_X.shape[0])
    piB = np.zeros(other_X.shape[0])
    permA = np.random.permutation(piA.shape[0])
    permB = np.random.permutation(piB.shape[0])

    kf = KFold(n_splits=K, shuffle=False)
    A_train_index_list = []
    A_test_index_list = []
    B_train_index_list = []
    B_test_index_list = []
    for i, (train_index, test_index) in enumerate(kf.split(source_X)):
        A_train_index_list.append(train_index)
        A_test_index_list.append(test_index)
    for i, (train_index, test_index) in enumerate(kf.split(other_X)):
        B_train_index_list.append(train_index)
        B_test_index_list.append(test_index)

    for i in range(K):
        trainX = np.concatenate([source_X[permA[A_train_index_list[i]]],other_X[permB[B_train_index_list[i]]]], axis=0)
        trainT = np.zeros(trainX.shape[0])
        trainT[len(A_train_index_list[i]):] = 1.0

        if domain_classifier is None:
            model = XGBClassifier(random_state=0).fit(trainX, trainT)
        else:
            model = domain_classifier.fit(trainX, trainT)

        piA[permA[A_test_index_list[i]]] = model.predict_proba(source_X[permA[A_test_index_list[i]]])[:,1]
        piB[permB[B_test_index_list[i]]] = model.predict_proba(other_X[permB[B_test_index_list[i]]])[:,1]

    # if draw_calibration:
    #     plot_calibration(piA, piB, save_dir=save_calibration_png)

    alpha = (other_X.shape[0])/ (source_X.shape[0]+other_X.shape[0])
    wA = piA / ((1-alpha)*piA + alpha * (1-piA))
    wB = (1-piB) / ((1-alpha)*piB + alpha * (1-piB))
    # Changing to support the model type of XGBoost.
    # accuracyA = best_method.score(source_X, source_y)
    # accuracyB = best_method.score(other_X, other_y)
    pd_source_X = pd.DataFrame(source_X, columns=column_names)
    pd_other_X = pd.DataFrame(other_X, columns=column_names)
    d_matrix_source = DMatrix(pd_source_X)
    d_matrix_other = DMatrix(pd_other_X)
    accuracyA = accuracy(source_y, best_method.predict(d_matrix_source))
    accuracyB = accuracy(other_y, best_method.predict(d_matrix_other))
    wA = wA / np.sum(wA)
    wB = wB / np.sum(wB)
    # predA = (best_method.predict(source_X) == source_y)
    # predB = (best_method.predict(other_X) == other_y)
    predA = (np.argmax(best_method.predict(d_matrix_source), axis=1) == np.argmax(source_y, axis=1))
    predB = (np.argmax(best_method.predict(d_matrix_other), axis=1) == np.argmax(other_y, axis=1))
    sx_A = np.dot(wA, predA)
    sx_B = np.dot(wB, predB)
    return accuracyA, accuracyB, sx_A, sx_B

def y_shift(src_x, src_y, tar_x, tar_y, tree_model, column_names):
    print("Y Shift()")
    p2p, q2q, p2s, s2q = degradation_decomp(src_x, src_y, tar_x, tar_y, tree_model, column_names, data_sum=20000, K=8, draw_calibration=False, save_calibration_png='calibration.png')
    # print(f"Total Performance Degradation is {p2p-q2q}")
    # print(f"Proportion of Y|X-shift is {(p2s-s2q)/(p2p-q2q)}")
    perf_degradation = p2p-q2q
    if perf_degradation == 0:
        proportion_yshift = 0
    else:
        proportion_yshift = (p2s-s2q)/(p2p-q2q)
    return perf_degradation, proportion_yshift


def task_agnostic_data_valuation(src_x, tar_x):
    print("Task Agnostic Data Valuation()")
    cov_mat_src = src_x.cov()
    # src_eig_vals, src_eig_vecs = np.linalg.eig((cov_mat_src.T @ cov_mat_src) * (1 / len(src_x)))
    src_eig_vals, src_eig_vecs = np.linalg.eig(cov_mat_src)
    # cov_mat_tar = ((tar_x.cov().T @ tar_x.cov()) * (1 / len(tar_x)))
    cov_mat_tar = tar_x.cov()
    tar_eig_vals = [np.sqrt(np.sum(np.square(cov_mat_tar.dot(eigen_vec)))) for eigen_vec in src_eig_vecs]
    src_eig_vals = np.array(src_eig_vals)
    tar_eig_vals = np.array(tar_eig_vals)
    diversity, relevance = 1, 1
    for src_eig, tar_eig in zip(src_eig_vals, tar_eig_vals):
        diversity *= np.power((abs(src_eig - tar_eig) / max(src_eig, tar_eig)), 1 / len(src_eig_vals))
        relevance *= np.power((min(src_eig, tar_eig) / max(src_eig, tar_eig)), 1 / len(src_eig_vals))
    return relevance, diversity


from math import ceil, floor
from collections import defaultdict, Counter

import torch
import numpy as np
from torch import stack

def compute_volumes(datasets, d=1):
    d = datasets[0].shape[1]
    for i in range(len(datasets)):
        datasets[i] = datasets[i].reshape(-1 ,d)

    X = np.concatenate(datasets, axis=0).reshape(-1, d)
    volumes = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        volumes[i] = np.sqrt(np.linalg.det( dataset.T @ dataset ) + 1e-8)

    volume_all = np.sqrt(np.linalg.det(X.T @ X) + 1e-8).round(3)
    return volumes, volume_all

def compute_X_tilde_and_counts(X, omega):
    """
    Compresses the original feature matrix X to  X_tilde with the specified omega.

    Returns:
       X_tilde: compressed np.ndarray
       cubes: a dictionary of cubes with the respective counts in each dcube
    """
    D = X.shape[1]

    # assert 0 < omega <= 1, "omega must be within range [0,1]."

    m = ceil(1.0 / omega) # number of intervals for each dimension

    cubes = Counter() # a dictionary to store the freqs
    # key: (1,1,..)  a d-dimensional tuple, each entry between [0, m-1]
    # value: counts

    Omega = defaultdict(list)
    # Omega = {}

    min_ds = torch.min(X, axis=0).values

    # a dictionary to store cubes of not full size
    for x in X:
        cube = []
        for d, xd in enumerate(x - min_ds):
            d_index = floor(xd / omega)
            cube.append(d_index)

        cube_key = tuple(cube)
        cubes[cube_key] += 1

        Omega[cube_key].append(x)

        '''
        if cube_key in Omega:

            # Implementing mean() to compute the average of all rows which fall in the cube

            Omega[cube_key] = Omega[cube_key] * (1 - 1.0 / cubes[cube_key]) + 1.0 / cubes[cube_key] * x
            # Omega[cube_key].append(x)
        else:
             Omega[cube_key] = x
        '''
    X_tilde = stack([stack(list(value)).mean(axis=0) for key, value in Omega.items()])

    # X_tilde = stack(list(Omega.values()))

    return X_tilde, cubes

def compute_robust_volumes(X_tildes, dcube_collections):

    N = sum([len(X_tilde) for X_tilde in X_tildes])
    alpha = 1.0 / (10 * N) # it means we set beta = 10
    # print("alpha is :{}, and (1 + alpha) is :{}".format(alpha, 1 + alpha))

    volumes, volume_all = compute_volumes(X_tildes, d=X_tildes[0].shape[1])
    robust_volumes = np.zeros_like(volumes)
    for i, (volume, hypercubes) in enumerate(zip(volumes, dcube_collections)):
        rho_omega_prod = 1.0
        for cube_index, freq_count in hypercubes.items():

            # if freq_count == 1: continue # volume does not monotonically increase with omega
            # commenting this if will result in volume monotonically increasing with omega
            rho_omega = (1 - alpha**(freq_count + 1)) / (1 - alpha)

            rho_omega_prod *= rho_omega

        robust_volumes[i] = (volume * rho_omega_prod).round(3)
    return robust_volumes


def robust_volume(Xs, omega=0.1):
    print("RobustVolume()")
    # M = len(Xs)
    D = Xs.shape[1]
    curr_train_X = torch.tensor(Xs.values).reshape(-1, D)

    X_tilde, cubes = compute_X_tilde_and_counts(curr_train_X, omega)

    robust_vol = compute_robust_volumes([X_tilde], [cubes])[0]

    return robust_vol


def hellinger_distance(src_y, tar_y, type_continuous=False):
    print("Hellinger distance()")
    # if type_continuous:
    #
    # else:
    total_instances_per_label_src = np.sum(src_y, axis=0)
    total_instances_per_label_tar = np.sum(tar_y, axis=0)
    p = np.divide(total_instances_per_label_src, np.sum(total_instances_per_label_src))
    q = np.divide(total_instances_per_label_tar, np.sum(total_instances_per_label_tar))

    return (1 / math.sqrt(2)) * np.sqrt(np.sum(np.square(np.sqrt(p) - np.sqrt(q))))


from metrics.Evaluator import partial_computation, evaluator
from util import OptunaConnection
from experiment_parameters.model_builder.ModelBuilder import Director, get_training_configuration
from experiment_parameters.model_builder.Model import XGBoostModel, KerasModel
import gc

director = Director()

def get_parameters(trial, model_type):
    parameters = get_training_configuration(trial=trial, model_type=model_type)
    return parameters

def get_mlp(input_dim, num_classes, parameters):
    return director.create_mlp(input_parameters=input_dim, num_classes=num_classes, parameters=parameters)

def performance_degradation(train_src_x, train_src_y, train_tar_x, train_tar_y, test_tar_x, test_tar_y, dataset_name):
    print("Performance degradation()")
    if dataset_name == "har":
        study = OptunaConnection.load_study("mlp_har")
    elif dataset_name == "edge-iot-coreset":
        study = OptunaConnection.load_study("mlp_edge_iiot_coreset")
    elif dataset_name == "electric-consumption":
        study = OptunaConnection.load_study("mlp_electric_consumption")
    best_trial = study.best_trial
    parameters_dict = get_training_configuration(best_trial, "mlp")

    model_src: KerasModel = get_mlp(train_src_x.shape[1], train_src_y.shape[1], parameters_dict)
    model_tar: KerasModel = get_mlp(train_src_x.shape[1], train_src_y.shape[1], parameters_dict)

    train_dataset_src = torch.utils.data.TensorDataset(
        torch.from_numpy(train_src_x.to_numpy()), torch.from_numpy(train_src_y.to_numpy())
    )
    train_dataloader_src = torch.utils.data.DataLoader(
        train_dataset_src, batch_size=512, shuffle=True, pin_memory=False
    )

    train_dataset_tar = torch.utils.data.TensorDataset(
        torch.from_numpy(train_tar_x.to_numpy()), torch.from_numpy(train_tar_y.to_numpy())
    )
    train_dataloader_tar = torch.utils.data.DataLoader(
        train_dataset_tar, batch_size=512, shuffle=True, pin_memory=False
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_tar_x.to_numpy())
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=512, shuffle=False, pin_memory=False
    )

    model_src.fit(train_dataloader_src, "FedAvg", epochs=50, batch_size=512)
    model_tar.fit(train_dataloader_tar, "FedAvg", epochs=50, batch_size=512)

    if train_src_y.shape[1] == 1:
        metric_list = ["MSE"]
        partial_computations_src = partial_computation(test_dataloader,
                                                   model_src,
                                                   list(train_src_y.columns),
                                                   test_tar_y)
        partial_computations_tar = partial_computation(test_dataloader,
                                                   model_src,
                                                   list(train_src_y.columns),
                                                   test_tar_y)
        evaluation_result_src = evaluator(partial_computations_src, metric_list=metric_list).get_value_of_metric("MSE")
        evaluation_result_tar = evaluator(partial_computations_tar, metric_list=metric_list).get_value_of_metric("MSE")
    else:
        metric_list = ["CrossEntropyLoss"]
        partial_computations_src = partial_computation(test_dataloader,
                                                   model_src,
                                                   list(train_src_y.columns),
                                                   test_tar_y)
        partial_computations_tar = partial_computation(test_dataloader,
                                                   model_src,
                                                   list(train_src_y.columns),
                                                   test_tar_y)
        evaluation_result_src = evaluator(partial_computations_src, metric_list=metric_list).get_value_of_metric("CrossEntropyLoss")
        evaluation_result_tar = evaluator(partial_computations_tar, metric_list=metric_list).get_value_of_metric("CrossEntropyLoss")

    return evaluation_result_src - evaluation_result_tar


def get_xgb_tree(train_x, train_y, test_x, test_y):
    parameters_dict = {"batch_size": 64}
    d_matrix = xgb.DMatrix(train_x, label=np.argmax(train_y, axis=1))
    d_test_matrix = xgb.DMatrix(test_x, label=np.argmax(test_y, axis=1))
    if train_y.shape[1] == 2:
        parameters_dict["objective"] = "binary:logistic"
        parameters_dict["eval_metric"] = "logloss"
    elif train_y.shape[1] > 2:
        parameters_dict["objective"] = "multi:softprob"
        parameters_dict['num_class'] = train_y.shape[1]
        parameters_dict["disable_default_eval_metric"] = 1
        parameters_dict["eval_metric"] = "mlogloss"
    tree_model = xgb.train(parameters_dict, d_matrix, evals=[(d_matrix, "train"), (d_test_matrix, "validate")], num_boost_round=500, early_stopping_rounds=10)
    tree_model = XGBoostModel(tree_model)
    return tree_model

FILENAME_TO_FLAG = {
    "gaussian_mmd.csv": "compute_gaussian_mmd",
    "relevance.csv": "compute_relevance_diversity",
    "diversity.csv": "compute_relevance_diversity",
    "volume.csv": "compute_volume",
    "yShiftDataframe.csv": "compute_yshift",
    "negativeConditionalEntropy.csv": "compute_negative_conditional_entropy",
    "hellinger.csv": "compute_hellinger",
    "wasserstein.csv": "compute_wasserstein",
}


def one_hot_encode_labels_np(array):
    array_reshaped = np.reshape(array, (-1, 1))
    encoder = OneHotEncoder().fit(array_reshaped)
    array_encoded = encoder.transform(array_reshaped)
    return array_encoded.toarray()


def compute_all_distances_and_volumes(dataset_name, path_to_train_datasets, maverick,
                                       compute_gaussian_mmd=True,
                                       compute_relevance_diversity=True,
                                       compute_volume=True,
                                       compute_yshift=True,
                                       compute_negative_conditional_entropy=True,
                                       compute_hellinger=True,
                                       compute_wasserstein=True):
    wassersteinDataframe = pd.DataFrame()
    gaussianMMDDataframe = pd.DataFrame()
    relevanceDataframe = pd.DataFrame()
    diversityDataframe = pd.DataFrame()
    volumeDataframe = pd.Series()

    performanceDegradation = pd.DataFrame()
    negativeConditionalEntropyDataframe = pd.DataFrame()
    yShiftDataframe = pd.DataFrame()
    hellingerDistanceDataframe = pd.DataFrame()

    # classification = False

    global_X_training, global_y_training = dataset_model_dictionary[dataset_name]().get_dataset().get_training_data()
    global_X_test, global_y_test = dataset_model_dictionary[dataset_name]().get_dataset().get_test_data()
    column_names = global_X_training.columns.values.tolist()

    # if len(global_y_training.shape) > 1:  # Classification problem
    #     classification = True

    num_clients, all_combinations = number_of_clients_and_all_combinations(path_to_train_datasets, maverick)

    # y-shift needs classification AND the flag on; negative conditional entropy just needs classification.
    # need_yshift = classification and compute_yshift
    need_yshift = compute_yshift
    # need_negative_conditional_entropy = classification and compute_negative_conditional_entropy
    need_negative_conditional_entropy = compute_negative_conditional_entropy

    for source_client in range(num_clients):
        src_train_x = pd.read_csv(path_to_train_datasets + os.sep + "client_" + str(source_client) + "_X_training.csv",
                                  index_col=0)
        src_train_y = pd.read_csv(path_to_train_datasets + os.sep + "client_" + str(source_client) + "_y_training.csv",
                                  index_col=0)
        src_test_x = pd.read_csv(path_to_train_datasets + os.sep + "client_" + str(source_client) + "_X_test.csv",
                                 index_col=0)
        src_test_y = pd.read_csv(path_to_train_datasets + os.sep + "client_" + str(source_client) + "_y_test.csv",
                                 index_col=0)

        src_train_x = downcast_types(src_train_x)
        src_test_x = downcast_types(src_test_x)

        # Only build the tree model if something that needs it is actually being computed.
        tree_model = get_xgb_tree(src_train_x, src_train_y, src_test_x, src_test_y) if need_yshift else None

        if compute_wasserstein:
            if dataset_name == "electric-consumption" or dataset_name == "wids_dataset":
                wassersteinDataframe.loc[source_client, "Global"] = (FedBary.fed_bary_compute(src_train_x,
                                                                                              np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(pd.DataFrame(src_train_y)).todense(), axis=1).A1,
                                                                                              global_X_test,
                                                                                              np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(pd.DataFrame(global_y_test)).todense(), axis=1).A1,
                                                                                              20)).item()
            else:
                wassersteinDataframe.loc[source_client, "Global"] = (FedBary.fed_bary_compute(src_train_x,
                                                                                              np.argmax(src_train_y,
                                                                                                        axis=1),
                                                                                              global_X_test,
                                                                                              np.argmax(global_y_test,
                                                                                                        axis=1),
                                                                                              src_train_y.shape[1])).item()

        if compute_gaussian_mmd:
            gaussianMMDDataframe.loc[source_client, "Global"] = gaussian_mmd_distance(
                pd.concat([src_train_x, src_test_x]).to_numpy(dtype=numpy.float32),
                pd.concat([global_X_training, global_X_test]).to_numpy(dtype=numpy.float32))

        if compute_relevance_diversity:
            relevanceDataframe.loc[source_client, "Global"], diversityDataframe.loc[
                source_client, "Global"] = task_agnostic_data_valuation(
                prince.PCA(n_components=10).fit_transform(pd.concat([src_train_x, src_test_x])),
                prince.PCA(n_components=10).fit_transform(pd.concat([global_X_training, global_X_test]))
            )

        if compute_volume:
            volumeDataframe.loc[source_client] = robust_volume(
                prince.PCA(n_components=10).fit_transform(pd.concat([src_train_x, src_test_x])))

        if need_yshift:
                # and (dataset_name != "electric-consumption" and dataset_name != "wids_dataset")):
            if dataset_name == "electric-consumption" or dataset_name == "wids_dataset":
                _, yShiftDataframe.loc[source_client, "Global"] = y_shift(
                    pd.concat([src_train_x, src_test_x]).to_numpy(dtype=numpy.float32),
                    one_hot_encode_labels_np(np.concatenate([
                        np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(src_train_y).todense(), axis=1).A1,
                        np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(src_test_y).todense(), axis=1).A1
                    ], axis=0)),
                    pd.concat([global_X_training, global_X_test]).to_numpy(dtype=numpy.float32),
                    one_hot_encode_labels_np(np.concatenate([
                        np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(pd.DataFrame(global_y_training)).todense(), axis=1).A1,
                        np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(pd.DataFrame(global_y_test)).todense(), axis=1).A1,
                    ])),
                    tree_model,
                    column_names)
            else:
                _, yShiftDataframe.loc[source_client, "Global"] = y_shift(
                    pd.concat([src_train_x, src_test_x]).to_numpy(dtype=numpy.float32),
                    pd.concat([src_train_y, src_test_y]).to_numpy(dtype=numpy.float32),
                    pd.concat([global_X_training, global_X_test]).to_numpy(dtype=numpy.float32),
                    pd.concat([global_y_training, global_y_test]).to_numpy(dtype=numpy.float32),
                    tree_model,
                    column_names)

        if need_negative_conditional_entropy and (dataset_name != "electric-consumption" and dataset_name != "wids_dataset"):
            negativeConditionalEntropyDataframe.loc[source_client, "Global"] = negative_conditional_entropy(
                np.argmax(pd.concat([src_train_y, src_test_y]).astype(int).to_numpy(), axis=1),
                np.argmax(pd.concat([global_y_training, global_y_test]).astype(int).to_numpy(), axis=1))

        if compute_hellinger:
            if dataset_name == "electric-consumption" or dataset_name == "wids_dataset":
                src_discretized = np.concatenate([
                    np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(src_train_y).todense(), axis=1).A1,
                    np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(src_test_y).todense(), axis=1).A1
                ])
                tar_discretized = np.concatenate([
                    np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(pd.DataFrame(global_y_training)).todense(), axis=1).A1,
                    np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(pd.DataFrame(global_y_test)).todense(), axis=1).A1
                ])
                hellingerDistanceDataframe.loc[source_client, "Global"] = hellinger_distance(src_discretized, tar_discretized)
            else:
                hellingerDistanceDataframe.loc[source_client, "Global"] = hellinger_distance(
                    pd.concat([src_train_y, src_test_y]).to_numpy(dtype=numpy.float32),
                    pd.concat([global_y_training, global_y_test]).to_numpy(dtype=numpy.float32))

        gc.collect()

        for target_client in range(num_clients):
            tar_train_x = pd.read_csv(
                path_to_train_datasets + os.sep + "client_" + str(target_client) + "_X_training.csv", index_col=0)
            tar_train_y = pd.read_csv(
                path_to_train_datasets + os.sep + "client_" + str(target_client) + "_y_training.csv", index_col=0)
            tar_test_x = pd.read_csv(path_to_train_datasets + os.sep + "client_" + str(target_client) + "_X_test.csv",
                                     index_col=0)
            tar_test_y = pd.read_csv(path_to_train_datasets + os.sep + "client_" + str(target_client) + "_y_test.csv",
                                     index_col=0)

            tar_train_x = downcast_types(tar_train_x)
            tar_test_x = downcast_types(tar_test_x)

            if compute_wasserstein:
                if dataset_name == "electric-consumption" or dataset_name == "wids_dataset":
                    wassersteinDataframe.loc[source_client, target_client] = (
                        FedBary.fed_bary_compute(src_train_x,
                                              np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(
                                                  pd.DataFrame(src_train_y)).todense(), axis=1).A1,
                                              tar_test_x,
                                              np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(
                                                  pd.DataFrame(tar_test_y)).todense(), axis=1).A1,
                                              20)
                    ).item()
                else:
                    wassersteinDataframe.loc[source_client, target_client] = (FedBary.fed_bary_compute(src_train_x,
                                                                                                       np.argmax(
                                                                                                           src_train_y,
                                                                                                           axis=1),
                                                                                                       tar_test_x,
                                                                                                       np.argmax(tar_test_y,
                                                                                                                 axis=1),
                                                                                                       src_train_y.shape[
                                                                                                           1])).item()
                # if dataset_name == "wids_dataset":
                #     print("wids")

            if compute_gaussian_mmd:
                gaussianMMDDataframe.loc[source_client, target_client] = gaussian_mmd_distance(
                    pd.concat([src_train_x, src_test_x]).to_numpy(dtype=numpy.float32),
                    pd.concat([tar_train_x, tar_test_x]).to_numpy(dtype=numpy.float32))

            if compute_relevance_diversity:
                relevanceDataframe.loc[source_client, target_client], diversityDataframe.loc[
                    source_client, target_client] = task_agnostic_data_valuation(
                    prince.PCA(n_components=10).fit_transform(pd.concat([src_train_x, src_test_x])),
                    prince.PCA(n_components=10).fit_transform(pd.concat([tar_train_x, tar_test_x]))
                )

            if need_yshift:
                    # and (dataset_name != "electric-consumption" and dataset_name != "wids_dataset")):
                if dataset_name == "electric-consumption" or dataset_name == "wids_dataset":
                    _, yShiftDataframe.loc[source_client, target_client] = y_shift(
                        pd.concat([src_train_x, src_test_x]).to_numpy(dtype=numpy.float32),
                        one_hot_encode_labels_np(np.concatenate([
                            np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(src_train_y).todense(), axis=1).A1,
                            np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(src_test_y).todense(), axis=1).A1
                        ], axis=0)),
                        pd.concat([tar_train_x, tar_test_x]).to_numpy(dtype=numpy.float32),
                        one_hot_encode_labels_np(np.concatenate([
                            np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(tar_train_y).todense(), axis=1).A1,
                            np.argmax(KBinsDiscretizer(n_bins=20).fit_transform(tar_test_y).todense(), axis=1).A1
                        ], axis=0)),
                        tree_model,
                        column_names)
                else:
                    _, yShiftDataframe.loc[source_client, target_client] = y_shift(
                        pd.concat([src_train_x, src_test_x]).to_numpy(dtype=numpy.float32),
                        pd.concat([src_train_y, src_test_y]).to_numpy(dtype=numpy.float32),
                        pd.concat([tar_train_x, tar_test_x]).to_numpy(dtype=numpy.float32),
                        pd.concat([tar_train_y, tar_test_y]).to_numpy(dtype=numpy.float32),
                        tree_model,
                        column_names)

            if need_negative_conditional_entropy and (dataset_name != "electric-consumption" and dataset_name != "wids_dataset"):
                negativeConditionalEntropyDataframe.loc[source_client, target_client] = negative_conditional_entropy(
                    np.argmax(pd.concat([src_train_y, src_test_y]).astype(int).to_numpy(), axis=1),
                    np.argmax(pd.concat([tar_train_y, tar_test_y]).astype(int).to_numpy(), axis=1))

            if compute_hellinger:
                if dataset_name == "electric-consumption" or dataset_name == "wids_dataset":
                    hellingerDistanceDataframe.loc[source_client, target_client] = hellinger_distance(
                        np.concatenate([KBinsDiscretizer(n_bins=20).fit_transform(src_train_y).todense(),
                                   KBinsDiscretizer(n_bins=20).fit_transform(src_test_y).todense()], axis=0),
                        np.concatenate([KBinsDiscretizer(n_bins=20).fit_transform(tar_train_y).todense(),
                                   KBinsDiscretizer(n_bins=20).fit_transform(tar_test_y).todense()], axis=0))
                else:
                    hellingerDistanceDataframe.loc[source_client, target_client] = hellinger_distance(
                        pd.concat([src_train_y, src_test_y]).to_numpy(dtype=numpy.float32),
                        pd.concat([tar_train_y, tar_test_y]).to_numpy(dtype=numpy.float32))

            gc.collect()

    return wassersteinDataframe, gaussianMMDDataframe, performanceDegradation, yShiftDataframe, negativeConditionalEntropyDataframe, relevanceDataframe, diversityDataframe, volumeDataframe, hellingerDistanceDataframe


def compute_distances_and_values(dataset_name, type_of_partition, additional_parameter, random_seed, maverick):
    path_to_result_dataframes = get_distances_from_route(dataset_name, type_of_partition, additional_parameter, random_seed)

    # wassersteinDataframe / performanceDegradation are never actually populated (their
    # computation is commented out upstream), so they're intentionally excluded here —
    # tracking them would make every run look "incomplete" forever.
    missing_files = [
        filename for filename in FILENAME_TO_FLAG
        if not os.path.exists(os.path.join(path_to_result_dataframes, filename))
    ]

    if not missing_files:
        print(f"Skipping distance computation for {path_to_result_dataframes}, as it was already performed.")
        return

    missing_flags = {FILENAME_TO_FLAG[filename] for filename in missing_files}
    print(f"Computing missing distances for {path_to_result_dataframes}: {sorted(missing_flags)}")

    path_to_train_datasets = get_data_from_route(dataset_name, type_of_partition, additional_parameter, random_seed)
    wassersteinDataframe, gaussianMMDDataframe, performanceDegradation, yShiftDataframe, negativeConditionalEntropy, relevance, diversity, volume, hellinger = compute_all_distances_and_volumes(
        dataset_name, path_to_train_datasets, maverick=maverick,
        compute_gaussian_mmd="compute_gaussian_mmd" in missing_flags,
        compute_relevance_diversity="compute_relevance_diversity" in missing_flags,
        compute_volume="compute_volume" in missing_flags,
        compute_yshift="compute_yshift" in missing_flags,
        compute_negative_conditional_entropy="compute_negative_conditional_entropy" in missing_flags,
        compute_hellinger="compute_hellinger" in missing_flags,
        compute_wasserstein="compute_wasserstein" in missing_flags
    )
    os.makedirs(path_to_result_dataframes, exist_ok=True)

    print("Missing distances computed.")

    dataframes_to_save = {
        "wasserstein.csv": wassersteinDataframe,
        "yShiftDataframe.csv": yShiftDataframe,
        "negativeConditionalEntropy.csv": negativeConditionalEntropy,
        "hellinger.csv": hellinger,
        "gaussian_mmd.csv": gaussianMMDDataframe,
        "relevance.csv": relevance,
        "diversity.csv": diversity,
        "volume.csv": volume,
    }

    for filename in missing_files:
        dataframe = dataframes_to_save[filename]
        if dataframe.shape[0] > 0:
            dataframe.to_csv(os.path.join(path_to_result_dataframes, filename))

import itertools

def experiment_combinations() -> list[dict]:
    datasets = {
        "classification": [
            "har",
            "edge-iot-coreset"
        ],
        "regression": [
            # "electric-consumption",
            "wids_dataset"
        ]
    }

    partition_types = [
        "dirichlet",
        # "feature_skew",
        "manual"
    ]

    metrics = {"classification": ["CrossEntropyLoss", "Accuracy", "F1Score", "MCC", "F1ScoreMacro", "F1ScoreMicro"],
               "regression": ["MSE", "RMSE", "MAE"]}

    DEFAULT_DATA_SPLIT = [20, 20, 20, 20, 20]
    data_split = {
        "har": [16, 16, 16, 16, 16, 16],
    }

    alpha = [
        0.1,
        10,
        1000,
    ]
    n_clients = [5]
    run_number_list = [number for number in range(1, 6)]
    manual_partition_names = {
        "har": {
            "HAR_1_Maverick_Laying": True,
            "HAR_1_Maverick_WalkingUpstairs": True,
            "HAR_1_Maverick_Laying_Balanced": True,
            "HAR_1_Maverick_WalkingUpstairs_Balanced": True,
            "HAR_1_Maverick_1_MissingOneLabel": True,
            "HAR_1_Maverick_1_MissingTwoLabels": True
        },
        "edge-iot-coreset": {
            "edgeiot_coreset_1_Maverick_Least_Class": True,
            "edgeiot_coreset_1_Maverick_Only_Normal": True,
            "edgeiot_coreset_1_Maverick_ddos_udp": True,
            "edgeiot_coreset_1_Maverick_sql_injection": True,
        },
        # "electric-consumption": {
        #     "wids_energy_non_iid_by_label": False,
        #     "wids_energy_iid_sampling": False,
        #     "wids_energy_feature_skew_building_type": False,
        #     "wids_energy_feature_skew_state_factor": False,
        #     "wids_energy_feature_skew_facility_type": False
        # },
        "wids_dataset": {
            "wids_energy_non_iid_by_label": False,
            "wids_energy_iid_sampling": False,
            "wids_energy_feature_skew_building_type": False,
            "wids_energy_feature_skew_state_factor": False,
            "wids_energy_feature_skew_facility_type": False,
            "wids_energy_maverick_facility_type_grocery_store": True,
            "wids_energy_maverick_facility_type_uncategorized_multifamily": True
        }
    }

    # CHANGED: keep type_key alongside (dataset, metrics) so the partition
    # loop below knows whether a given dataset is regression or classification.
    dataset_metrics_pairs = [
        (dataset, metrics[type_key], type_key)
        for type_key in set(datasets) & set(metrics)
        for dataset in datasets[type_key]
    ]

    experiments = []

    # CHANGED: unpack type_key from the pair; partition_types no longer
    # iterated directly over all datasets uniformly.
    for (dataset, selected_metrics, type_key), number in itertools.product(
            dataset_metrics_pairs, run_number_list
    ):
        base = {
            "dataset": dataset,
            "metrics": selected_metrics,
            "run_number": number,
        }

        # NEW: dirichlet is classification-only. Regression datasets never
        # generate a dirichlet combination, regardless of what's toggled
        # on in partition_types above.
        if type_key == "regression":
            allowed_partitions = [p for p in partition_types if p != "dirichlet"]
        else:
            allowed_partitions = partition_types

        for partition in allowed_partitions:
            entry = {**base, "partition": partition}

            if partition == "dirichlet":
                split = data_split.get(dataset, DEFAULT_DATA_SPLIT)
                for a in alpha:
                    experiments.append({**entry, "alpha": a, "data_split": split, "maverick": False})

            elif partition == "feature_skew":
                for nc in n_clients:
                    experiments.append({**entry, "n_clients": nc, "maverick": False})

            elif partition == "manual":
                dataset_partitions = manual_partition_names.get(dataset, {})
                for name, maverick in dataset_partitions.items():
                    experiments.append({**entry, "name": name, "maverick": maverick})

    return experiments

def main():
    training_configurations_for_distances = experiment_combinations()

    for training_config in training_configurations_for_distances:
        dataset_name, type_of_partition, random_seed = training_config["dataset"], training_config["partition"], \
        training_config["run_number"]
        if type_of_partition == "dirichlet":
            additional_parameter = str(training_config["alpha"])
            maverick = False
        elif type_of_partition == "manual":
            additional_parameter = training_config["name"]
            maverick = training_config["maverick"]

        else:
            raise NotImplementedError

        compute_distances_and_values(dataset_name, type_of_partition, additional_parameter, random_seed, maverick)

if __name__ == "__main__":
    main()


# print(training_configurations_for_distances)

# def compute_coupling(X_src, X_tar, Y_src, Y_tar):
#     cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)

#     C = cost_function(X_src, X_tar)
#     P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(X_tar.shape[0]), C.numpy(), numItermax=1000000)
#     W = np.sum(P * np.array(C.numpy()))

#     return P, W

# # def compute_coupling(X_src, X_tar, Y_src, Y_tar):
# #     loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=.05)
# #     # cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)

# #     C = cost_function(X_src, X_tar)
# #     P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(X_tar.shape[0]), C.numpy(), numItermax=1000000)
# #     W = np.sum(P * np.array(C.numpy()))

# #     return P, W


# def compute_CE(P, Y_src, Y_tar):
#     src_label_set = set(sorted(list(Y_src.flatten())))
#     tar_label_set = set(sorted(list(Y_tar.flatten())))

#     # joint distribution of source and target label
#     P_src_tar = np.zeros((np.max(Y_src) + 1, np.max(Y_tar) + 1))

#     for y1 in src_label_set:
#         y1_idx = np.where(Y_src == y1)
#         for y2 in tar_label_set:
#             y2_idx = np.where(Y_tar == y2)

#             RR = y1_idx[0].repeat(y2_idx[0].shape[0])
#             CC = np.tile(y2_idx[0], y1_idx[0].shape[0])

#             P_src_tar[y1, y2] = np.sum(P[RR, CC])

#     # marginal distribution of source label
#     P_src = np.sum(P_src_tar, axis=1)

#     ce = 0.0
#     for y1 in src_label_set:
#         P_y1 = P_src[y1]
#         for y2 in tar_label_set:

#             if P_src_tar[y1, y2] != 0:
#                 ce += -(P_src_tar[y1, y2] * math.log(P_src_tar[y1, y2] / P_y1))
#     return ce


# def test():
#     # -----------start: randomly generate the testing data-----------
#     src_x_list = []
#     src_y_list = []
#     tar_x_list = []
#     tar_y_list = []

#     NUM_SAMPLE = 100

#     # suppose the feature dimension is 512, and the label is in range [0,10].
#     for i in range(NUM_SAMPLE):
#         src_x_list.append(np.random.randn(512))
#         tar_x_list.append(np.random.randn(512))

#         src_y_list.append(np.random.randint(0, 10))
#         tar_y_list.append(np.random.randint(0, 10))

#     # the shape of x is n*512, and the shape of y is n*1
#     src_x = torch.tensor(np.array(src_x_list), dtype=torch.float)
#     tar_x = torch.tensor(np.array(tar_x_list), dtype=torch.float)
#     src_y = np.array(src_y_list)[:, np.newaxis]
#     tar_y = np.array(tar_y_list)[:, np.newaxis]
#     # -----------end: randomly generate the testing data------------

#     # obtain the optimal coupling matrix P and the wasserstein distance W
#     P, W = compute_coupling(src_x, tar_x, src_y, tar_y)

#     # compute the conditonal entropy (ce)
#     ce = compute_CE(P, src_y, tar_y)

#     print('Wasserstein distance:%.4f, Conditonal Entropy: %.4f' % (W, ce))



