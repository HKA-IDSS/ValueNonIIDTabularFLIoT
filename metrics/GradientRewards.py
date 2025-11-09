import os
from logging import INFO
from typing import Dict

import keras
import tensorflow as tf
from flwr.common import log

from Definitions import ROOT_DIR
from util.Util import load_data_from_pickle_file


class GradientRewards:
    _n_clients: int
    _rewards: Dict[int, Dict[str, float]]

    _client_index_dictionary: dict
    _client_index_dictionary_set: bool

    def __init__(self, n_clients):
        self._n_clients = n_clients
        self._rewards = dict()

        self._client_index_dictionary = dict()
        self._client_index_dictionary_set = False

    def get_rewards(self) -> Dict:
        return self._rewards

    def set_client_index_dictionary(self, client_number_dictionary):
        if not self._client_index_dictionary_set:
            # for client_id, iterator in zip(clients_ids, range(len(clients_ids))):
            #     self._client_index_dictionary[client_id] = iterator
            # self._client_index_dictionary = client_number_dictionary
            for client_cid, client_number in client_number_dictionary.items():
                self._client_index_dictionary[client_cid] = int(client_number)
            self._client_index_dictionary_set = True

    def calculate_rewards(self, server_round: int, client_list, clients_data_sizes_dict: dict):
        client_gradients = dict()
        for client in client_list:
            gradients_to_be_flattened = \
                load_data_from_pickle_file(ROOT_DIR +
                                           os.sep + "data" +
                                           os.sep + "pickled_information" +
                                           os.sep + f"gradients_{self._client_index_dictionary[client]}.pkl")
            gradients = list(map(lambda layer: tf.reshape(layer, -1), gradients_to_be_flattened))
            gradient_flattened = tf.concat(gradients, 0)
            gradient_normalized = tf.math.divide_no_nan(gradient_flattened, tf.norm(gradient_flattened))
            client_gradients[client] = gradient_normalized

            if "aggregated" not in client_gradients:
                # log(INFO, f"Client size data: {clients_data_sizes_dict[client]}")
                client_gradients["aggregated"] = gradient_normalized * \
                                                 sum(clients_data_sizes_dict[client])
            else:
                client_gradients["aggregated"] += gradient_normalized * \
                                                  sum(clients_data_sizes_dict[client])

        cosine_sim = keras.metrics.CosineSimilarity(axis=0)
        cosine_rewards = {client: float(cosine_sim(client_gradients["aggregated"], client_gradients[client]))
                          for client in client_list}
        self._rewards[server_round] = cosine_rewards

