from __future__ import annotations

from enum import Enum

import numpy as np


def topk_filtering(values: np.ndarray, topk: int):
    """Receives the values in [n x c x (h*w)] shape and returns
    the topk values for each n according to their channel norm."""
    spatial_size = values.shape[0]
    if spatial_size > topk:
        value_norms = np.linalg.norm(values, axis=(1))
        ids = np.argpartition(value_norms, axis=-1, kth=-topk)
        high_prio = ids[-topk:]
        low_prio = ids[:-topk]

        v_i_high_prio = values[high_prio, :]
        v_i_low_prio = values[low_prio, :]
        return v_i_high_prio, v_i_low_prio
    else:  # No filtering
        return values, None


def top2048_filtering(values: np.ndarray):
    """Receives the values in [n x c x (h*w)] shape and returns
    the topk values for each n according to their channel norm."""
    return topk_filtering(values, 2048)


def top1024_filtering(values: np.ndarray):
    """Receives the values in [n x c x (h*w)] shape and returns
    the topk values for each n according to their channel norm."""
    return topk_filtering(values, 1024)


def top512_filtering(values: np.ndarray):
    """Receives the values in [n x c x (h*w)] shape and returns
    the topk values for each n according to their channel norm."""
    return topk_filtering(values, 512)


def top256_filtering(values: np.ndarray):
    """Receives the values in [n x c x (h*w)] shape and returns
    the topk values for each n according to their channel norm."""
    return topk_filtering(values, 256)


def top128_filtering(values: np.ndarray):
    """Receives the values in [n x c x (h*w)] shape and returns
    the topk values for each n according to their channel norm."""
    return topk_filtering(values, 128)


# Auto Enum of filtering functions
class PreSimFilteringFunc(Enum):
    top2048 = "top2048"
    top1024 = "top1024"
    top512 = "top512"
    top256 = "top256"
    top128 = "top128"
    none = "none"


def find_pre_sim_filtering(filtering: PreSimFilteringFunc) -> callable | None:
    if filtering == PreSimFilteringFunc.top2048:
        return top2048_filtering
    elif filtering is PreSimFilteringFunc.top1024:
        return top1024_filtering
    elif filtering is PreSimFilteringFunc.top512:
        return top512_filtering
    elif filtering is PreSimFilteringFunc.top256:
        return top256_filtering
    elif filtering is PreSimFilteringFunc.top128:
        return top128_filtering
    elif filtering is PreSimFilteringFunc.none:
        return None
    else:
        raise ValueError("Filtering {} not supported".format(filtering))
