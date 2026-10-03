import os
import traceback
from logging import INFO
from typing import Dict

import torch.nn.functional as F
import torch
from flwr.common import log

from Definitions import ROOT_DIR
from util.Util import load_data_from_pickle_file, flatten, add_gradient_updates


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
        self._rs = 0

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

    def calculate_rewards(self, server_model, server_round: int, client_list,
                          clients_data_sizes_dict: dict, alpha: float = 0.9):
        """
        Compute gradient-driven rewards following the Gradient rewards paper.

        alpha: smoothing factor for the exponential moving average of phis.
               Higher = more weight on past reputation. Paper uses 0.9.
        """
        # --- Step 1: Load and normalise each client's gradient update ---
        client_gradients: dict[str, list[torch.Tensor]] = {}
        aggregated_gradient = [torch.zeros(param.shape) for param in server_model.get_model().trainable_weights]

        for client in client_list:
            client_number = self._client_index_dictionary[client]
            log(INFO, f"Loading gradient for client {client} (number {client_number})")

            gradient: list[torch.Tensor] = load_data_from_pickle_file(
                ROOT_DIR + os.sep + "data" + os.sep + "pickled_information" +
                os.sep + f"gradients_{client_number}.pkl"
            )

            # Gradient is a list[list[Tensor]] (steps × layers) from fit().
            # Reduce across steps by summing, matching compute_grad_update semantics.
            if isinstance(gradient[0], list):
                num_layers = len(gradient[0])
                gradient = [
                    torch.stack([gradient[step][layer] for step in range(len(gradient))]).sum(dim=0)
                    for layer in range(num_layers)
                ]

            # Normalise: gradient = Gamma * (gradient / ||gradient||)
            # Your fit() already does this before pickling, so just verify shape is correct.

            # Assert here — gradient is now in its final list[Tensor] form
            assert len(gradient) == len(list(server_model.get_model().trainable_weights)), \
                f"Client {client_number}: gradient layers {len(gradient)} != model layers {len(list(server_model.get_model().trainable_weights))}"

            flattened = flatten(gradient)
            log(INFO, f"Client {client_number} flattened gradient shape: {flattened.shape}")

            client_gradients[client] = gradient

            # --- Step 2: Build weighted aggregated gradient (FedAvg-style) ---
            weight = sum(clients_data_sizes_dict[client])
            add_gradient_updates(aggregated_gradient, gradient, weight=weight)

        # Normalise aggregated gradient by total data size
        total_samples = sum(sum(clients_data_sizes_dict[c]) for c in client_list)
        aggregated_gradient = [layer / total_samples for layer in aggregated_gradient]
        flat_aggre_grad = flatten(aggregated_gradient)
        log(INFO, f"Aggregated gradient shape: {flat_aggre_grad.shape}")

        # --- Step 3: Compute cosine similarity (phis) for each client ---
        phis = torch.tensor([
            F.cosine_similarity(flatten(client_gradients[client]), flat_aggre_grad, dim=0, eps=1e-10)
            for client in client_list
        ])
        log(INFO, f"Phis (cosine similarities): {phis}")

        # --- Step 4: Exponential moving average reputation update ---
        if not hasattr(self, '_rs') or self._rs is None or server_round == 1:
            # First round: initialise rs from phis directly
            rs = phis.clone()
        else:
            rs = alpha * self._rs + (1 - alpha) * phis

        rs = torch.clamp(rs, min=1e-3)  # prevent negative/zero weights
        rs = torch.div(rs, rs.sum())  # normalise to sum to 1
        self._rs = rs  # persist for next round
        log(INFO, f"Reputation scores (rs): {rs}")

        # --- Step 5: Store per-client rewards keyed by client number ---
        cosine_rewards = {
            client: float(rs[i])
            for i, client in enumerate(client_list)
        }
        self._rewards[server_round] = cosine_rewards
        log(INFO, f"Rewards round {server_round}: {cosine_rewards}")

        return cosine_rewards

