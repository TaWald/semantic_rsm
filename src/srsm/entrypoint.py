from __future__ import annotations
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm
from srsm.core.bipartite_matching import google_linear_sum_assignment, hungarian_match, lapjv_match
from srsm.core.sim_calc import chunker, prepare_reps, semantic_sim_parallel, similarity_calculation
from srsm.core.sim_func import cosine_sim, inner_product, own_rbf
from srsm.util import file_io as io

import torch


def calculate_batchwise_semantic_similarity_matrices(
    representations: Path | list[Path] | dict[str, torch.Tensor],
    batch_size: int,
    sim_func: callable | list[callable],
    matching_func: callable,
    pre_sim_func=None,
    n_cores=32,
    subset: int = None,
) -> dict[str, list[np.ndarray]]:
    """Main function for calculating the semantic (permutation invariant) similarity matrices.
    This function is the main entry point for the semantic similarity calculation.
    - a path to a file containing representations (e.g. a .pkl file of a dictionary of layer_name: representations)
    - a list of paths to files containing representations (e.g. a list of .npy files -- Expects one layer per file and correct order)
    - a dictionary of layer_name: representations (e.g. a dictionary of layer_name: torch.Tensor) {"0": torch.Tensor, "1": torch.Tensor}
    Batch size:
    - If None, all representations are used at once.
    - If an integer, the representations are split into batches of the given size and one RSMs is returned for each batch.
    sim_func: callable or list of callables
    - The similarity function(s) to use for calculating the RSMs. If multiple are given, multiple RSMs are calculated.
    subset: int
    - If not None, only the first `subset` samples are used instead of all samples (for all layers).
    Returns a dictionary of layer_name: list of RSMs (one RSM per batch).
    """
    all_results = {}
    # Now we calculate the K values for each layer and save them
    if isinstance(representations, Path):
        representations = io.load(representations)
    elif isinstance(representations, list):
        representations = {str(cnt): rep for cnt, rep in enumerate(representations)}
    elif isinstance(representations["0"], torch.Tensor):
        pass
    else:
        raise NotImplementedError("Representations should be either a list of paths or a path to a file.")

    for layer_name, layer_representations in representations.items():
        # Only load the single layer to minimize memory usage.
        if isinstance(layer_representations, Path):
            layer_representations = list(io.load(layer_representations).values())[0]
        if subset is not None:
            layer_representations = layer_representations[:subset]
        layer_representations = prepare_reps(layer_representations)
        n_samples = layer_representations.shape[0] // batch_size
        layer_rep_gen = ((batch, batch) for batch in chunker(layer_representations, batch_size))

        # Given large enough batches we do multiprocessing inside the semantic similarity calculation.
        sem_sim_calc: callable = partial(
            semantic_sim_parallel,
            sim_func=sim_func,
            matching_func=matching_func,
            pre_sim_func=pre_sim_func,
            processes=n_cores,
        )
        results = [sem_sim_calc(*batch) for batch in tqdm(layer_rep_gen, total=n_samples, leave=False)]
        all_results[layer_name] = results
    return all_results


def calculate_batchwise_baseline_similarity_matrices(
    representations: Path | list[Path] | dict[str, torch.Tensor],
    batch_size: int | None,
    sim_func: callable | list[callable],
    n_cores=32,
    subset: int = None,
) -> dict[str, list[np.ndarray]]:
    """Main function for calculating baseline (spatio-semantic) similarity matrices.
    representations can be:
    - a path to a file containing representations (e.g. a .pkl file of a dictionary of layer_name: representations)
    - a list of paths to files containing representations (e.g. a list of .npy files -- Expects one layer per file and correct order)
    - a dictionary of layer_name: representations (e.g. a dictionary of layer_name: torch.Tensor) {"0": torch.Tensor, "1": torch.Tensor}
    Batch size:
    - If None, all representations are used at once.
    - If an integer, the representations are split into batches of the given size and one RSMs is returned for each batch.
    sim_func: callable or list of callables
    - The similarity function(s) to use for calculating the RSMs. If multiple are given, multiple RSMs are calculated.
    subset: int
    - If not None, only the first `subset` samples are used instead of all samples (for all layers).
    Returns a dictionary of layer_name: list of RSMs (one RSM per batch).
    """
    if not isinstance(sim_func, list):
        sim_func = [sim_func]
    all_results = {}
    # Now we calculate the K values for each layer and save them
    if isinstance(representations, Path):
        representations = io.load(representations)
    elif isinstance(representations, list):
        representations = {str(cnt): rep for cnt, rep in enumerate(representations)}
    elif isinstance(representations["0"], torch.Tensor):
        pass
    else:
        raise NotImplementedError("Representations should be either a list of paths or a path to a file.")

    for layer_name, layer_representations in representations.items():
        # Only load the single layer to minimize memory usage.
        if isinstance(layer_representations, Path):
            layer_representations = list(io.load(layer_representations).values())[0]
        if subset is not None:
            layer_representations = layer_representations[:subset]
        layer_representations = prepare_reps(layer_representations)
        if batch_size is None:
            n_samples = layer_representations.shape[0]
            layer_rep_gen = (layer_representations, layer_representations)
        else:
            n_samples = layer_representations.shape[0] // batch_size
            layer_rep_gen = ((batch, batch) for batch in chunker(layer_representations, batch_size))

        # Given large enough batches we do multiprocessing inside the semantic similarity calculation.
        sim_calc: callable = partial(
            similarity_calculation,
            sim_func=sim_func,
        )
        results = [sim_calc(*batch) for batch in tqdm(layer_rep_gen, total=n_samples, leave=False)]
        all_results[layer_name] = results
    return all_results


def get_or_calculate_RSM(
    out_dir: Path,
    representations: Path | list[Path] | dict[str, torch.Tensor],
    batch_size: int,
    sim_func: callable | None,
    matching_func: callable,
    pre_sim_func: callable | None,
    n_cores: int,
):
    """
    Retrieve or compute the permutation invariant Representational Similarity Matrices (RSM).
    This function checks if the RSMs for the given batch size already exist in the specified output directory.
    If they exist, it loads and returns them. Otherwise, it calculates the RSMs, saves them to the output directory,
    and then returns them.
    Args:
        out_dir (Path): The directory where the RSMs are stored or will be saved.
        representations (Path | list[Path] | dict[str, torch.Tensor]): The representations to be used for calculating the RSMs. Shaped: (n, c, h, w) or (n, t, d)
        batch_size (int): The size of the batches to be used for calculating the RSMs.
        sim_func (callable): The similarity function to be used for calculating the RSMs.
        n_cores (int): The number of CPU cores to be used for parallel processing.
    Returns:
        dict: The computed or loaded RSMs.
    """

    rsm_name = f"RSMs_{batch_size}_{matching_func.__name__}.pkl"
    rsm_path = out_dir / rsm_name
    if rsm_path.exists():
        print("RSM exists. Loading...")
        rsms = io.load_pickle(rsm_path)
    else:
        print("RSM does not exist. Calculating...")
        if sim_func is None:
            if matching_func in [google_linear_sum_assignment, lapjv_match, hungarian_match]:
                sim_func = [inner_product, cosine_sim, own_rbf]

        rsms = calculate_batchwise_semantic_similarity_matrices(
            representations=representations,
            batch_size=batch_size,
            sim_func=sim_func,
            matching_func=matching_func,
            pre_sim_func=pre_sim_func,
            n_cores=n_cores,
            subset=None,
        )
        io.save_pickle(rsms, rsm_path)
    return rsms


def get_or_calculate_baseline_RSM(
    out_dir: Path,
    representations: Path | list[Path] | dict[str, torch.Tensor],
    batch_size: int,
    sim_func: callable,
    n_cores: int,
):
    """
    Retrieve or compute the baseline Representational Similarity Matrices (RSM).
    This function checks if the RSMs for the given batch size already exist in the specified output directory.
    If they exist, it loads and returns them. Otherwise, it calculates the RSMs, saves them to the output directory,
    and then returns them.
    Args:
        out_dir (Path): The directory where the RSMs are stored or will be saved.
        representations (Path | list[Path] | dict[str, torch.Tensor]): The representations to be used for calculating the RSMs. Shaped: (n, c, h, w) or (n, t, d)
        batch_size (int): The size of the batches to be used for calculating the RSMs.
        sim_func (callable): The similarity function to be used for calculating the RSMs.
        n_cores (int): The number of CPU cores to be used for parallel processing.
    Returns:
        dict: The computed or loaded RSMs.
    """
    rsm_name = f"RSMs_{batch_size}_baseline.pkl"
    rsm_path = out_dir / rsm_name
    if rsm_path.exists():
        print("RSM exists. Loading...")
        rsms = io.load_pickle(rsm_path)
    else:
        print("RSM does not exist. Calculating...")
        rsms = calculate_batchwise_baseline_similarity_matrices(
            representations=representations,
            batch_size=batch_size,
            sim_func=sim_func,
            n_cores=n_cores,
            subset=None,
        )
        io.save_pickle(rsms, rsm_path)
    return rsms


def main():
    print("Hello!")


if __name__ == "__main__":
    main()
