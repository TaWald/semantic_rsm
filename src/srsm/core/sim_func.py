from enum import Enum

import numpy as np
import torch


def rbf(x, y, sigma=None):
    x = np.reshape(x, (x.shape[0], -1))
    y = np.reshape(y, (y.shape[0], -1))
    GX = np.matmul(x, y.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = np.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)

    KX_e = np.exp(KX)
    if np.any(np.isnan(KX_e)):
        print("NaNs in KX_e")
    return KX_e


def own_rbf_part_1(x, y):
    """
    This is for the semantic CKA.

    This is needed since due to permutation we would otherwise have to keep all the permuted vectors in memory.
    By precomputing the distance, we can ignore that, and then do the rbf later (which enables the sigma calculation as median.)
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    # Calc euclidian distance between all dimensions and positions (since they are aligned)
    dist = torch.sum((x - y) ** 2).cpu()  # Only calculated for each pair.
    return dist


def own_rbf_part_2(dist: list[torch.Tensor], sigma=None):
    """
    Continues where part1 left off.

    This is needed since due to permutation we would otherwise have to keep all the permuted vectors in memory.
    By precomputing the distance, we can ignore that, and then do the rbf later (which enables the sigma calculation as median.)

    """
    dist = torch.stack(dist)
    # Median distance
    if sigma is None:
        mdist = torch.median(dist[dist != 0])
        # Sigma
        offset = 1e-3  # To avoid overflows by zero
        sigma = torch.sqrt(mdist) + offset
    # Calc kernel
    K = torch.exp(-0.5 * (dist / (sigma) ** 2))
    return K.cpu().numpy().tolist()


def own_rbf(x, y, sigma=None):
    """This is for the semantic CKA.
    Since we want to normalize based off the median distance, we receive all"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    x = torch.reshape(x, (x.shape[0], -1))  # n x (whatever)
    y = torch.reshape(y, (y.shape[0], -1))  # n x (whatever)
    # Calc euclidian distance between all n x n pairs
    dist = torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=-1).cpu()
    # Median distance
    if sigma is None:
        mdist = torch.median(dist[dist != 0])
        # Sigma
        offset = 1e-3  # To avoid overflows by zero
        sigma = torch.sqrt(mdist) + offset
    # Calc kernel
    K = torch.exp(-0.5 * (dist / (sigma) ** 2))
    return K


def inner_product(x, y):
    """
    Values in n x (whatever) shape.
    """
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    K = x @ y.T
    return K


def cosine_sim(x, y):
    """
    Values in n x (whatever) shape.
    """
    div_offset = 1e-13
    x = x.reshape(x.shape[0], -1)  # n x (whatever)
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + div_offset)
    y = y.reshape(y.shape[0], -1)  # n x (whatever)
    y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + div_offset)
    K = x_norm @ y_norm.T
    return K


class SimFunc(Enum):
    innerproduct = "innerproduct"
    rbf = "rbf"
    cosine = "cosine"
    ownrbf = "ownrbf"


def find_sim_func(sim_func_name: SimFunc) -> callable:
    if sim_func_name == SimFunc.innerproduct:
        return inner_product
    elif sim_func_name == SimFunc.cosine:
        return cosine_sim
    elif sim_func_name == SimFunc.rbf:
        return rbf
    elif sim_func_name == SimFunc.ownrbf:
        return own_rbf
    else:
        raise ValueError(f"Unknown sim_func_name {sim_func_name}")
