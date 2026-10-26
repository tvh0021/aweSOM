"""Exact, chunked best-matching-unit (BMU) inference for a trained SOM.

The functions here are the reference classifier that the frozen model
artifact (:class:`aweSOM.model.SOMModel`) and the C++ header
``cpp/awesom_classify.hpp`` both reproduce.

Exactness contract:
    For every point, the squared distance to node ``i`` is accumulated in a
    float64 scalar, looping the features in ascending order
    (``acc = 0; acc += d * d`` with ``d = w[i, f] - x[f]``). Nodes are scanned
    in ascending index order and a node replaces the current best only if its
    distance is strictly smaller, so ties go to the lowest index, as with
    ``np.argmin``. For float64 input the labels equal ``Lattice.best_match``
    bitwise. Float32 input is promoted to float64 element by element; the
    result is exact for the promoted values but may differ from a float64
    version of the original data where two nodes are nearly tied.

Measured throughput (``benchmark()``, N = 2,000,000, F = 6, 2368 nodes,
float64, Apple M1 Max with 10 threads, numba 0.68): ``best_match_nodes``
1.61e6 points/s, ``Lattice.best_match`` 1.00e5 points/s, identical labels.
See ``cpp/README.md`` for the C++ figure.
"""

import time

import numpy as np
from numba import njit, prange

__all__ = ["best_match_nodes", "predict", "benchmark"]

DEFAULT_CHUNK = 2**20


@njit(parallel=True, cache=True)
def _bmu_kernel(
    weights: np.ndarray, data: np.ndarray, out_idx: np.ndarray, out_dist2: np.ndarray
) -> None:
    """Write the BMU index and squared distance of every row of ``data``.

    Args:
        weights (np.ndarray): float64 [nodes, F] lattice weights.
        data (np.ndarray): float64 or float32 [n, F] points (already scaled).
        out_idx (np.ndarray): int32 [n] output node indices.
        out_dist2 (np.ndarray): float64 [n] output squared distances.
    """
    n_points = data.shape[0]
    n_nodes = weights.shape[0]
    n_features = weights.shape[1]
    for p in prange(n_points):
        best = np.inf
        best_idx = 0
        for i in range(n_nodes):
            acc = 0.0
            for f in range(n_features):
                d = weights[i, f] - np.float64(data[p, f])
                acc += d * d
            if acc < best:
                best = acc
                best_idx = i
        out_idx[p] = best_idx
        out_dist2[p] = best


def _check_inputs(weights: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if weights.ndim != 2 or data.ndim != 2:
        raise ValueError("weights and data must be 2D arrays")
    if data.shape[1] != weights.shape[1]:
        raise ValueError(
            f"data has {data.shape[1]} features but the lattice has {weights.shape[1]}"
        )
    if weights.shape[0] == 0:
        raise ValueError("weights must contain at least one node")
    if data.dtype not in (np.float64, np.float32):
        data = data.astype(np.float64)
    return weights, data


def best_match_nodes(
    weights: np.ndarray, data: np.ndarray, chunk: int = DEFAULT_CHUNK
) -> tuple[np.ndarray, np.ndarray]:
    """Find the best-matching node of every point.

    The points are processed ``chunk`` rows at a time; apart from the outputs,
    the extra memory is at most one contiguous copy of a chunk (only made when
    ``data`` is not C-contiguous).

    Args:
        weights (np.ndarray): [nodes, F] lattice weights, node index = x + y * xdim.
        data (np.ndarray): [N, F] float64 or float32 points in the scaled space.
        chunk (int, optional): Number of points per kernel call. Defaults to 2**20.

    Returns:
        tuple[np.ndarray, np.ndarray]: int32 [N] node indices and float64 [N]
            squared Euclidean distances to those nodes.
    """
    weights, data = _check_inputs(weights, data)
    if chunk < 1:
        raise ValueError("chunk must be positive")
    n_points = data.shape[0]
    nodes_idx = np.empty(n_points, dtype=np.int32)
    dist2 = np.empty(n_points, dtype=np.float64)
    for start in range(0, n_points, chunk):
        stop = min(start + chunk, n_points)
        block = np.ascontiguousarray(data[start:stop])
        _bmu_kernel(weights, block, nodes_idx[start:stop], dist2[start:stop])
    return nodes_idx, dist2


def predict(
    weights: np.ndarray,
    node_cluster: np.ndarray,
    data: np.ndarray,
    chunk: int = DEFAULT_CHUNK,
    return_distance: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Assign each point the cluster id of its best-matching node.

    Args:
        weights (np.ndarray): [nodes, F] lattice weights.
        node_cluster (np.ndarray): [nodes] cluster id of each node (-1 = unassigned).
        data (np.ndarray): [N, F] float64 or float32 points in the scaled space.
        chunk (int, optional): Number of points per kernel call. Defaults to 2**20.
        return_distance (bool, optional): Also return the Euclidean BMU distance.
            Defaults to False.

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]: int32 [N] labels, and float64 [N]
            distances if ``return_distance`` is True.
    """
    node_cluster = np.asarray(node_cluster, dtype=np.int32).ravel()
    if node_cluster.shape[0] != np.shape(weights)[0]:
        raise ValueError("node_cluster must have one entry per node")
    nodes_idx, dist2 = best_match_nodes(weights, data, chunk=chunk)
    labels = node_cluster[nodes_idx]
    if return_distance:
        return labels, np.sqrt(dist2)
    return labels


def benchmark(N: int = 2_000_000, F: int = 6, nodes: int = 2368, seed: int = 0) -> dict:
    """Time ``best_match_nodes`` against ``Lattice.best_match`` on random data.

    Development hook, not used by the library. Both kernels are compiled on a
    small warm-up input first so that the timings exclude JIT compilation.

    Args:
        N (int, optional): Number of points. Defaults to 2_000_000.
        F (int, optional): Number of features. Defaults to 6.
        nodes (int, optional): Number of lattice nodes. Defaults to 2368 (64 x 37).
        seed (int, optional): Seed of the random inputs. Defaults to 0.

    Returns:
        dict: points per second of each kernel and the number of label disagreements.
    """
    import contextlib
    import io

    from .som import Lattice

    rng = np.random.default_rng(seed)
    weights = rng.uniform(-1.0, 1.0, size=(nodes, F))
    data = rng.normal(scale=0.5, size=(N, F))

    with contextlib.redirect_stdout(io.StringIO()):
        best_match_nodes(weights, data[:1000])
        Lattice.best_match(weights, data[:1000])

    t0 = time.perf_counter()
    idx_new, _ = best_match_nodes(weights, data)
    t_new = time.perf_counter() - t0

    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        idx_old = Lattice.best_match(weights, data)
        t_old = time.perf_counter() - t0

    result = {
        "N": N,
        "F": F,
        "nodes": nodes,
        "best_match_nodes_points_per_s": N / t_new,
        "lattice_best_match_points_per_s": N / t_old,
        "disagreements": int(np.count_nonzero(idx_new != idx_old.ravel().astype(np.int32))),
    }
    print(
        f"best_match_nodes:    {result['best_match_nodes_points_per_s']:.3e} points/s\n"
        f"Lattice.best_match:  {result['lattice_best_match_points_per_s']:.3e} points/s\n"
        f"label disagreements: {result['disagreements']}"
    )
    return result
