import json
from typing import Optional, Tuple, List, Union, Dict

from flwr.common import NDArrays
from flwr.server.strategy.aggregate import aggregate as aggregate_nn_flwr


def get_tree_nums(xgb_model_org: bytes) -> Tuple[int, int]:
    xgb_model = json.loads(bytearray(xgb_model_org))
    # Get the number of trees
    tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ]
    )
    # Get the number of parallel trees
    paral_tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ]
    )
    return tree_num, paral_tree_num


def aggregate_trees(bst_prev_org: Optional[bytes], bst_curr_org: bytes) -> bytes:
    """Conduct bagging aggregation for given trees."""
    if not bst_prev_org:
        return bst_curr_org

    tree_num_prev, _ = get_tree_nums(bst_prev_org)
    tree_num_curr, paral_tree_num_curr = get_tree_nums(bst_curr_org)

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    # Total trees after aggregation
    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(tree_num_prev + tree_num_curr)  # was: tree_num_prev + paral_tree_num_curr

    # iteration_indptr tracks tree boundaries per boosting round.
    # Each client model has its own indptr; we need to offset and append
    # all its entries (skipping the leading 0) onto the global indptr.
    global_indptr = bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"]
    client_indptr = bst_curr["learner"]["gradient_booster"]["model"]["iteration_indptr"]
    offset = global_indptr[-1]
    for entry in client_indptr[1:]:  # skip the leading 0
        global_indptr.append(offset + entry)

    # Append ALL trees from client model, with corrected IDs
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(tree_num_curr):  # was: range(paral_tree_num_curr)
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count  # now covers all trees
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
            trees_curr[tree_count]
        )
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    return bytes(json.dumps(bst_prev), "utf-8")


def aggregate_xgboost(local_models: list[bytes], global_model: Optional[bytes]) -> bytes:
    for client_model in local_models:
        global_model = aggregate_trees(global_model, client_model)
    return global_model


def aggregate_nn(results: List[Tuple[NDArrays, int]]):
    return aggregate_nn_flwr(results)
