from __future__ import annotations

import os
from functools import partial
from itertools import repeat
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from srsm.core.bipartite_matching import breadth_first_match
from srsm.core.bipartite_matching import shifted_window_1024_glsm_match
from srsm.core.bipartite_matching import shifted_window_128_glsm_match
from srsm.core.bipartite_matching import shifted_window_256_glsm_match
from srsm.core.bipartite_matching import shifted_window_512_glsm_match
from srsm.core.sim_func import inner_product
from srsm.core.sim_func import own_rbf
from srsm.core.sim_func import own_rbf_part_1
from srsm.core.sim_func import own_rbf_part_2
from srsm.core.sim_func import rbf
from srsm.util import file_io as io
from loguru import logger
from matplotlib import pyplot as plt
from torch.multiprocessing import Pool
from tqdm import tqdm


def chunker(seq, size):
    """
    From https://stackoverflow.com/questions/434287/how-to-iterate-over-a-list-in-chunks
    since I am lazy
    """
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def prepare_reps(representations: np.ndarray | torch.Tensor) -> np.array:
    """Prepare the representations for the semantic similarity calculation.
    Representations are expected to be in the shape (n, c, h, w) or (n, t, d)"""
    if isinstance(representations, torch.Tensor):
        representations = representations.cpu().numpy()
    rep_shape = representations.shape
    if len(rep_shape) == 4:
        representations = np.reshape(representations, [rep_shape[0], rep_shape[1], -1])
    if len(rep_shape) == 3:
        representations = np.transpose(
            representations, axes=(0, 2, 1)
        )  # out: (n, c, token)  # spatial last as wanted
    else:
        raise "Representation in unexpected shape."
    return representations


def similarity_calculation(v_i: torch.Tensor, v_j: torch.Tensor, sim_func: list[callable]) -> list[np.ndarray]:
    """BASELINE Original translation sensitive way of calculating the similarity matrix!
    Expects the representation vectors to be in the shape (n, whatever). Will be rehapsed to (n, -1), so as long
    as v_i and v_j have the same shape, it works.
    """
    # Faster when on cuda!
    if isinstance(v_i, torch.Tensor):
        v_i = v_i.cpu().numpy()
    if isinstance(v_j, torch.Tensor):
        v_j = v_j.cpu().numpy()
    sim = [sf(v_i, v_j) for sf in sim_func]  # Expected (n, whc), (n, whc)  --> h*w^2 possible combis
    return sim


def _get_cost_matrix(
    v_i: np.ndarray | torch.Tensor,
    v_j: np.ndarray | torch.Tensor,
    cost_matrix: np.ndarray | None | torch.Tensor,
):
    """
    We can either calculate the cost matrix from v_i and v_j (which will be done when we only have CPUs available)
    If we have GPU we can pre-computed the matrix multiplication in the generator function, which is faster.
    So we just have to move that gpu vector back to numpy for the matching.
    """
    # We can either preprocess the cost matrix (on gpu) and pass it along with vectors.
    if cost_matrix is None:
        # Always use the inner product for the inner matching (optimizes alignment & magnitude)
        cost_matrix = inner_product(v_i, v_j)  # Expected (h*w, c), (h*w, c)  --> h*w^2 possible combis
    if isinstance(cost_matrix, torch.Tensor):  # Matching always happens on cpu.
        cost_matrix = cost_matrix.cpu().numpy()
    return cost_matrix


def _get_sim_from_aligned_vectors(v_i: np.ndarray, v_j_permuted: np.ndarray, sim_func: list[callable]):
    """
    Calculates the similarity of the two vectors (a scalar).
    If its the rbf we have to wait to collect the whole batch, to calculate the median distance.
    """
    all_sims = []
    for s in sim_func:
        if s == own_rbf:
            sim = own_rbf_part_1(v_i, v_j_permuted)  # returns the distance
        else:
            # Need to reshape so we can apply the similarity functions
            sim = s(np.reshape(v_i, (1, -1)), np.reshape(v_j_permuted, (1, -1)))
        all_sims.append(sim)
    return all_sims


def perm_inv_semantic_sim(
    v_i: np.ndarray | torch.Tensor,
    v_j: np.ndarray | torch.Tensor,
    cost_matrix: np.ndarray | None | torch.Tensor,
    sim_func: list[callable],
    matching_func: callable,
    pre_sim_func: callable = None,
    ret_permutation: bool = False,
) -> list[float]:
    """
    This is the main function for image-to-image permutation invariant similarity calculation.
    It finds the optimal permutation between the semantic vectors to maximize the similarity.

    :param v_i: np.ndarray [H*W, C] or torch.Tensor [H*W, C]. H*W == S represents the Spatial dimension from the paper.
    :param v_j: np.ndarray [H*W, C] or torch.Tensor [H*W, C]. H*W == S represents the Spatial dimension from the paper.
    :param cost_matrix: (Optional) np.ndarray [H*W, H*W] or torch.Tensor [H*W, H*W]. The cost matrix used for matching.
    Can be pre-computed on GPU or calculated differently. If given, overrides internal `inner product` calculation.
    :param sim_func: callable or list of callables. The similarity function(s) to use for calculating the RSMs.
    CAREFUL: If you use `own_rbf` function distances are returned to calculate the median distance of the entire batch. See `semantic_sim_parallel` which finishes RSM calc.
    :param matching_func: callable. The matching function to use for finding the optimal permutation -- e.g. Hungarian algorithm or an approximation.
    :param pre_sim_func: callable. (Optional) A function to preprocess the vectors before calculating the similarity (e.g. TopK filtering).
    :param ret_permutation: bool. If True, returns the permutation along with the similarity value.
    """
    # In case we want to do approximat matching with topk values (was a bad approximation so not used)
    if pre_sim_func is not None:
        # The v_i are the to be matched through the optimal algorithm.
        # v_i_low and v_j_low are the values that are used with greedy matching
        v_i, v_i_low = pre_sim_func(v_i)
        v_j, v_j_low = pre_sim_func(v_j)
        if v_i_low is not None:
            low_prio_cost_matrix = _get_cost_matrix(v_i_low, v_j_low, None)
            low_prio_permutation = breadth_first_match(low_prio_cost_matrix)
            # this is the permutation of the low prio values
            v_j_low_permuted = v_j_low[low_prio_permutation]
        cost_matrix = None
        # We still need to calculate the cost matrix for the high prio values

    # Cost matrix is the inner product between the two vectors (SxS)
    cost_matrix = _get_cost_matrix(v_i, v_j, cost_matrix)
    if matching_func in [
        shifted_window_128_glsm_match,
        shifted_window_256_glsm_match,
        shifted_window_512_glsm_match,
        shifted_window_1024_glsm_match,
    ]:
        # If we do approximative matching, we need to pass the vectors to calculate norms.
        permutation = matching_func(cost_matrix, v_i, v_j)
    else:
        # If we do exact matching, we only need the cost matrix.
        permutation = matching_func(cost_matrix)
    # We permute the values according to the permutation for sim calculation
    v_j_permuted = v_j[permutation]

    if pre_sim_func is not None:
        # If we did topk similarity we need to append the low prio values back to the high prio values
        v_j_permuted = np.concatenate([v_j_permuted, v_j_low_permuted], axis=0)  # Append the values
        v_i = np.concatenate([v_i, v_i_low], axis=0)  # Append the values

    # Calculate the similarity of the permuted vectors with the sim functions.
    #   IMPORTANT: If you use the RBF you have to wait until all values are collected to calculate the median distance.
    #   This is done in the `semantic_sim_parallel` function. at the end. It is recommended to use that function instead.
    sim = _get_sim_from_aligned_vectors(v_i, v_j_permuted, sim_func)
    if ret_permutation:
        return sim, permutation
    else:
        return sim


def _get_to_be_compared_vectors(
    all_x, all_y, is_symmetric: bool
) -> tuple[list[tuple[int, int]], list[tuple[np.ndarray, np.ndarray]]]:
    """Creates the list of tuples of vectors to be compared,
    and the indices of the matrix to be filled.
    Only upper diagonal is filled, since it's symmetric.
    """
    cnts = []
    for cntx, _ in enumerate(all_x):
        for cnty, _ in enumerate(all_y):
            if is_symmetric:
                if cntx >= cnty:
                    cnts.append((cntx, cnty))
            else:
                cnts.append((cntx, cnty))  # if not symmetric we have to calculate all values
    return cnts


def _fill_sim_matrix(
    cnts: list[tuple[int, int]],
    all_sim_values: list[float],
    mat_shape_x: int,
    mat_shape_y: int,
    is_symmetric: bool = True,
) -> np.ndarray:
    sem_sim = np.zeros((mat_shape_x, mat_shape_y))
    # Fill diagonal and upper diagonal, if lower diagonal copy over.
    for (cntx, cnty), sim_value in zip(cnts, all_sim_values):
        sem_sim[cntx, cnty] = sim_value
        if is_symmetric:
            sem_sim[cnty, cntx] = sim_value
    return sem_sim


def semantic_sim_parallel(
    x: np.ndarray,
    y: np.ndarray,
    sim_func: list[callable] | callable,
    matching_func: callable,
    pre_sim_func: callable = None,
    processes: int = 24,
    is_symmetric: bool = True,
) -> list[np.ndarray] | np.ndarray:
    """
    Measures permutation invariant similarity between a batch of activation maps.

    The aligned vectors are always computed on the inner product, so we can calculate multiple downstream metrics immediately.
    This saves compute time for the optimal measurement of the metric.

    :param x: np.ndarray [N, C, H*W]
    :param y: np.ndarray [N, C, H*W].
    :param sim_func: callable or list of callables
    :param matching_func: callable
    :param pre_sim_func: callable
    :param processes: int
    :return: np.ndarray [N, N] or list[np.ndarray [N, N]] if multiple similarity functions

    """
    is_sim_list: bool = isinstance(sim_func, list)
    if not is_sim_list:
        sim_func = [sim_func]

    assert len(x.shape) == 3, "Expecting (n, c, h*w) shape"
    assert len(y.shape) == 3, "Expecting (n, c, h*w) shape"
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    all_x = np.transpose(x, axes=(0, 2, 1))  # out: (n, h*w, c)
    all_y = np.transpose(y, axes=(0, 2, 1))  # out: (n, h*w, c)

    # Each value corresponds to one pair of activation map responses
    # E.g. 1st to 3rd sample activation response. This will fill the matrix (0, 2) index.
    cnts = _get_to_be_compared_vectors(all_x, all_y, is_symmetric)
    # Drops the lower diagonal since it's symmetric
    n_samples = len(cnts)

    # However this metric is symmetric, so we don't have to calculate n**2 values,
    # But we can only calculate the upper triangle and then mirror it.
    # Keeping this outside and doing it via cuda should speed up the process.
    # Originally kept outside to see if CUDA speeds up, but did not.
    value_gen = [(all_x[idx_x], all_y[idx_y], None) for idx_x, idx_y in cnts]
    # We calculate the cost matrix later multithreaded

    mp = True if processes > 1 else False
    if mp:
        with Pool(processes=processes) as p:
            [r for r in zip(value_gen, repeat(sim_func), repeat(matching_func), repeat(pre_sim_func))]
            all_sim_values = p.starmap(
                perm_inv_semantic_sim,
                list(
                    [
                        [r[0][0], r[0][1], r[0][2], r[1], r[2], r[3]]
                        for r in zip(value_gen, repeat(sim_func), repeat(matching_func), repeat(pre_sim_func))
                    ]
                ),
            )
            p.close()
            p.join()
            # all_sim_values = [sem_sim_calc(*v) for v in tqdm(value_gen, total=n_samples, leave=False)]
        all_sim_values = all_sim_values
    else:
        all_sim_values = [
            perm_inv_semantic_sim(
                v_i,
                v_j,
                cm,
                sim_func,
                matching_func,
                pre_sim_func,
            )
            for v_i, v_j, cm in tqdm(value_gen, total=n_samples, leave=False, disable="LSB_JOBID" in os.environ)
        ]

    all_sim_values_of_funcs = [[] for _ in sim_func]
    for cnt, sf in enumerate(sim_func):
        sim_vals_of_func = [v[cnt] for v in all_sim_values]
        if sf == own_rbf:
            # Finish the similarity calculation!
            sim_vals_of_func = own_rbf_part_2(sim_vals_of_func)
        all_sim_values_of_funcs[cnt] = sim_vals_of_func

    sem_sim = [
        _fill_sim_matrix(cnts, sim_val, all_x.shape[0], all_y.shape[0], is_symmetric)
        for sim_val in all_sim_values_of_funcs
    ]
    if not is_sim_list:
        sem_sim = sem_sim[0]
    return sem_sim


def sim(x: np.ndarray, y: np.ndarray, sim_func: callable):
    x = x.reshape([x.shape[0], -1])
    y = y.reshape([y.shape[0], -1])
    return sim_func(x, y)
