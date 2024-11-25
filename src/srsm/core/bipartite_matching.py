from enum import Enum

import numpy as np
import torch
from lapjv import lapjv
from ortools.graph.python import linear_sum_assignment as google_lsa
from scipy.optimize import linear_sum_assignment as scipy_lsa

# Time Optimized version of the greedy matching


def google_linear_sum_assignment(
    cost_matrix: np.ndarray | None, v_i: np.ndarray = None, v_j: np.ndarray = None
) -> np.ndarray:
    """
    Returns the best possible matches.
    """
    # ------------------------------- prepare_data ------------------------------- #
    shape_a = cost_matrix.shape[0]
    shape_b = cost_matrix.shape[1]
    int_scaling = 1000
    flipped_cost_matrix = (-cost_matrix * int_scaling).astype(np.int64)
    assert shape_a == shape_b, "Cost matrix has to be square."
    end_nodes_unraveled, start_nodes_unraveled = np.meshgrid(np.arange(shape_a), np.arange(shape_a))
    start_nodes = start_nodes_unraveled.ravel()
    end_nodes = end_nodes_unraveled.ravel()
    flipped_arc_costs = flipped_cost_matrix.ravel()
    assigner = google_lsa.SimpleLinearSumAssignment()
    assigner.add_arcs_with_cost(start_nodes, end_nodes, flipped_arc_costs)
    status = assigner.solve()
    if status != 0:
        raise ValueError("Could not find a solution.")

    assigned_columns = np.array([assigner.right_mate(i) for i in range(shape_a)])

    # --------------- JUST FOR TESTING THE RETURNED COLS ARE RIGHT --------------- #
    # assigned_rows = np.arange(shape_a)
    # inv_cost = cost_matrix[assigned_rows, assigned_columns] * 1000
    # optimal_cost = assigner.optimal_cost()
    # cost_diff = np.sum(cost_matrix[assigned_rows, assigned_columns]) - np.abs(optimal_cost)

    return assigned_columns


def _auction_lap(X, eps, compute_score=True):
    """
    X: n-by-n matrix w/ integer entries
    eps: "bid size" -- smaller values means higher accuracy w/ longer runtime
    """
    eps = 1 / X.shape[0] if eps is None else eps
    X = torch.from_numpy(X).float()
    # --
    # Init

    cost = torch.zeros((1, X.shape[1]))
    curr_ass = torch.zeros(X.shape[0]).long() - 1
    bids = torch.zeros(X.shape)

    # if X.is_cuda:
    #     cost, curr_ass, bids = cost.cuda(), curr_ass.cuda(), bids.cuda()

    counter = 0
    while (curr_ass == -1).any():
        counter += 1

        # --
        # Bidding

        unassigned = (curr_ass == -1).nonzero().squeeze()

        value = X[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)

        first_idx = top_idx[:, 0]
        first_value, second_value = top_value[:, 0], top_value[:, 1]

        bid_increments = first_value - second_value + eps

        bids_ = bids[unassigned]
        bids_.zero_()
        if unassigned.dim() == 0:
            high_bids, high_bidders = bid_increments, unassigned
            cost[:, first_idx] += high_bids
            curr_ass[(curr_ass == first_idx).nonzero()] = -1
            curr_ass[high_bidders] = first_idx
        else:
            bids_.scatter_(dim=1, index=first_idx.contiguous().view(-1, 1), src=bid_increments.view(-1, 1))

            have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()

            high_bids, high_bidders = bids_[:, have_bidder].max(dim=0)

            high_bidders = unassigned[high_bidders.squeeze()]

            cost[:, have_bidder] += high_bids

            # curr_ass[(curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
            ind = (curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1).nonzero()
            curr_ass[ind] = -1

            curr_ass[high_bidders] = have_bidder.squeeze()

    score = None
    if compute_score:
        score = int(X.gather(dim=1, index=curr_ass.view(-1, 1)).sum())

    return score, curr_ass.numpy(), counter


# Slow as fuck
def auction_match(cost_matrix: np.ndarray | None, v_i: np.ndarray = None, v_j: np.ndarray = None) -> np.ndarray:
    """Calculates the auction algorithm to find assignments (that are likely not globally optimal, but at least locally!)"""
    integer_cost_matrix = (cost_matrix * 1000).astype(np.int64)
    _, assigned_cols, counter = _auction_lap(integer_cost_matrix, eps=None, compute_score=False)
    return assigned_cols


def diagonal_match(cost_matrix, v_i: np.ndarray = None, v_j: np.ndarray = None) -> np.ndarray:
    """This is what is done for Normal Linear CKA"""
    return np.arange(cost_matrix.shape[0])


def hungarian_match(cost_matrix: np.ndarray | None, v_i: np.ndarray = None, v_j: np.ndarray = None) -> np.ndarray:
    """
    Returns the indices of the permutation of the column vector.
    """
    ids = scipy_lsa(-cost_matrix)
    return ids[1]


def lapjv_match(cost_matrix: np.ndarray | None, v_i: np.ndarray = None, v_j: np.ndarray = None) -> np.ndarray:
    """
    Optimal assignment
    Better implementation of Jonker-Volgenant algorithm
    10x faster than linear_sum_assignment (actually 2x after testing)

    """
    # This returns x,y with the first array being the coumns assigned to the rows and
    # if the array would be N x M the second can be used to see assinged rows to columns
    col_ids, _, _ = lapjv(-cost_matrix)  # Has other formatting than linear_sum_assignment!
    return col_ids


def calculate_norms_and_sort(v_i: np.ndarray, v_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the norms and sort them by the norms"""
    norms_i = np.linalg.norm(v_i, axis=-1)
    norms_j = np.linalg.norm(v_j, axis=-1)
    sorted_indices_i = np.argsort(norms_i)
    sorted_indices_j = np.argsort(norms_j)
    return sorted_indices_i, sorted_indices_j


def create_batches_and_cost_matrix(
    v_i: np.ndarray, v_j: np.ndarray, cost_matrix: np.ndarray, k: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Instead of calculating the perfect matching globally we sort the
    ∂values by norm and calculate an optimal matching within them.

    Returns the sorted indices of the values and the cost matrix.
    """
    sorted_indices_i, sorted_indices_j = calculate_norms_and_sort(v_i, v_j)
    num_batches = int(np.ceil(len(v_i) / k))
    batched_sorted_indices_i = []
    batched_sorted_indices_j = []
    cost_matrices = []
    for i in range(num_batches):
        start_i = i * k
        end_i = min((i + 1) * k, len(v_i))
        cost_matrices.append(cost_matrix[sorted_indices_i[start_i:end_i], :][:, sorted_indices_j[start_i:end_i]])
        batched_sorted_indices_i.append(sorted_indices_i[start_i:end_i])
        batched_sorted_indices_j.append(sorted_indices_j[start_i:end_i])
    return batched_sorted_indices_i, batched_sorted_indices_j, cost_matrices


def shifted_window_glsm_match(cost_matrix: np.ndarray | None, v_i: np.ndarray, v_j: np.ndarray, k: int) -> np.ndarray:
    """Calculates the optimal batching not globally but in the local neighbourhood of high norm values."""
    batched_ids_i, batched_ids_j, batched_cost_matrix = create_batches_and_cost_matrix(v_i, v_j, cost_matrix, k)
    permutations = [google_linear_sum_assignment(batch_cost_matrix) for batch_cost_matrix in batched_cost_matrix]
    global_j_perms = [batch_j[p] for p, batch_j in zip(permutations, batched_ids_j)]
    global_i = np.concatenate(batched_ids_i)
    global_j_permed = np.concatenate(global_j_perms)
    reordered_i = np.argsort(global_i)
    final_permutations = global_j_permed[reordered_i]  # Make sure it aligns not with the norm sorted but the original

    return final_permutations


def shifted_window_128_glsm_match(cost_matrix: np.ndarray | None, v_i: np.ndarray, v_j: np.ndarray):
    return shifted_window_glsm_match(cost_matrix, v_i, v_j, 128)


def shifted_window_256_glsm_match(cost_matrix: np.ndarray | None, v_i: np.ndarray, v_j: np.ndarray):
    return shifted_window_glsm_match(cost_matrix, v_i, v_j, 256)


def shifted_window_512_glsm_match(cost_matrix: np.ndarray | None, v_i: np.ndarray, v_j: np.ndarray):
    return shifted_window_glsm_match(cost_matrix, v_i, v_j, 512)


def shifted_window_1024_glsm_match(cost_matrix: np.ndarray | None, v_i: np.ndarray, v_j: np.ndarray):
    return shifted_window_glsm_match(cost_matrix, v_i, v_j, 1024)


def breadth_first_match(cost_matrix, aggregation: str = "sum") -> np.ndarray:
    max_col_ids = np.argsort(cost_matrix, axis=0)[::-1]
    # Get maximum column value for each row
    max_similarity = cost_matrix[np.arange(cost_matrix.shape[0]), max_col_ids[0]]
    # Get the order of the rows based off the maximum similarity
    row_choice_order = np.argsort(max_similarity)[::-1]

    taken_cols = set()
    matched_indices = []
    # For each row index, we try to take the best column index, that has not been taken.
    for row_index in row_choice_order:
        # Get the maximum column index for the current row
        for x in max_col_ids[row_index]:
            if x not in taken_cols:
                max_col_id = x
                taken_cols.add(max_col_id)
                matched_indices.append((row_index, max_col_id))
                break
    sorted_inds = sorted(matched_indices, key=lambda x: x[0])
    col_ids = [x[1] for x in sorted_inds]
    return col_ids


def _measure_cost(cost_matrix: np.ndarray, col_ids: np.ndarray) -> float:
    """
    Measures the cost of the given column ids.
    """
    return np.sum(cost_matrix[np.arange(cost_matrix.shape[0]), col_ids])


class MatchingFunc(Enum):
    lapjv = "lapjv"
    hungarian = "hungarian"
    breadth_first = "breadth_first"
    diagonal = "diagonal"
    google_lsa = "google_lsa"
    sw_128 = "sw_128"
    sw_256 = "sw_256"
    sw_512 = "sw_512"
    sw_1024 = "sw_1024"


def find_matching(matching: MatchingFunc) -> callable:
    """
    Returns the matching function based off the string
    """
    if matching == MatchingFunc.hungarian:
        return hungarian_match
    elif matching == MatchingFunc.breadth_first:
        return breadth_first_match
    elif matching == MatchingFunc.diagonal:
        return diagonal_match
    elif matching == MatchingFunc.lapjv:
        return lapjv_match
    elif matching == MatchingFunc.google_lsa:
        return google_linear_sum_assignment
    elif matching == MatchingFunc.sw_128:
        return shifted_window_128_glsm_match
    elif matching == MatchingFunc.sw_256:
        return shifted_window_256_glsm_match
    elif matching == MatchingFunc.sw_512:
        return shifted_window_512_glsm_match
    elif matching == MatchingFunc.sw_1024:
        return shifted_window_1024_glsm_match
    else:
        raise ValueError("Matching {} not supported".format(matching))


def main():
    values_a = np.array(
        [
            [9.5, 8, 8, 4],
            [9, 1, 4, 2],
            [9, 3, 2, 1],
            [6, 7, 8, 9],
        ]
    )
    result_hungarian = hungarian_match(values_a)
    # result_simple = simple_match(values_a)
    result_lapjv = lapjv_match(values_a)
    results_breadth = breadth_first_match(values_a)
    results_google_lsa = google_linear_sum_assignment(values_a)
    results_diagonal = diagonal_match(values_a)

    print(f"Hungarian: {result_hungarian}, {_measure_cost(values_a, result_hungarian)}")
    print(f"Breadth: {results_breadth}, {_measure_cost(values_a, results_breadth)}")
    print(f"Lapjv: {result_lapjv}, {_measure_cost(values_a, result_lapjv)}")
    print(f"Google LSA: {results_google_lsa}, {_measure_cost(values_a, results_google_lsa)}")
    print(f"Diagonal: {results_diagonal}, {_measure_cost(values_a, results_diagonal)}")
    # print("Faster Greedy: {}".format(result_faster_greedy))
    # print("Faster Greedy v2: {}".format(result_faster_greedy_v2))
    pass


if __name__ == "__main__":
    main()
