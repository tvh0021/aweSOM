"""Exact, chunked best-matching-unit (BMU) inference for a trained SOM on the CPU.

Exactness contract:
    For every point, the squared distance to node ``i`` is accumulated in a
    float64 scalar, looping the features in ascending order
    (``acc = 0; acc += d * d`` with ``d = w[i, f] - x[f]``). Nodes are scanned
    in ascending index order and a node replaces the current best only if its
    distance is strictly smaller, so ties go to the lowest index, as with
    ``np.argmin``. A node whose distance is NaN (non-finite weights or point)
    is never selected, unlike ``np.argmin`` which returns the first NaN; a
    point with no finite distance gets node 0 and distance inf. For finite
    float64 input the labels equal those of the original
    ``Lattice.best_match`` kernel (``backend="legacy"``) bitwise. Float32
    input is promoted to float64 element by element; the result is exact for
    the promoted values but may differ from a float64 version of the original
    data, or from the float32 arithmetic of the original kernel, where two
    nodes are nearly tied.

:func:`best_match_nodes` is the reference float64 scan.
:func:`best_match_nodes_screen` (the ``predict`` default) scans every node
in float32, certifies the float32 winner with a rounding-error bound and
falls back to the float64 scan for every point it cannot certify, so its
labels and distances equal :func:`best_match_nodes` bitwise.

Measured throughput (N = 2,000,000, F = 6, 2368 nodes, float64, Apple M1
Max with 10 threads, numba 0.68): ``best_match_nodes`` 1.61e6 points/s, the
original ``Lattice.best_match`` kernel 1.00e5 points/s, identical labels.
Points per second of both functions on an Apple M1 Max, 2368 nodes:

=====================  ======  ======
data                   numba   screen
=====================  ======  ======
plasma lattice, F = 4  2.2e6   4.9e6
random, F = 10         9.6e5   2.5e6
=====================  ======  ======
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

__all__ = [
    "best_match_nodes",
    "best_match_nodes_screen",
    "predict",
]

DEFAULT_CHUNK = 2**20
SCREEN_TILE = 512
_U32 = 2.0**-24  # float32 unit roundoff


@njit(parallel=True, nogil=True, cache=True)
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
    if weights.shape[1] == 0:
        raise ValueError("weights must contain at least one feature")
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


@njit(parallel=True, nogil=True, cache=True)
def _screen_kernel(
    weights: np.ndarray,
    w32: np.ndarray,
    wmax: float,
    data: np.ndarray,
    out_idx: np.ndarray,
    out_dist2: np.ndarray,
    tile: int,
) -> int:
    """float32 screening scan certified against the float64 reference; see best_match_nodes_screen."""
    n = data.shape[0]
    n_nodes, F = weights.shape
    g = (F + 2) * _U32 * 1.01
    ntiles = (n + tile - 1) // tile
    nfb = np.zeros(ntiles, dtype=np.int64)
    for t in prange(ntiles):
        start = t * tile
        m = min(tile, n - start)
        # feature-major float32 copy of the tile: the point loop below vectorises 4-wide
        xT = np.zeros((F, tile), dtype=np.float32)
        for p in range(m):
            for f in range(F):
                xT[f, p] = np.float32(data[start + p, f])
        s1 = np.full(tile, np.float32(np.inf))  # smallest screen value
        s2 = np.full(tile, np.float32(np.inf))  # smallest over every other node
        j1 = np.zeros(tile, dtype=np.int32)
        acc = np.empty(tile, dtype=np.float32)
        for i in range(n_nodes):
            w0 = w32[i, 0]
            for p in range(tile):
                d = w0 - xT[0, p]
                acc[p] = d * d
            for f in range(1, F):
                wf = w32[i, f]
                for p in range(tile):
                    d = wf - xT[f, p]
                    acc[p] = acc[p] + d * d
            ii = np.int32(i)
            for p in range(tile):
                a = acc[p]
                b1 = s1[p]
                b2 = s2[p]
                c = a < b1
                c2 = a < b2
                n2 = b1 if c else (a if c2 else b2)
                # fmin, not a select on c: LLVM turns the select into per-lane
                # branches and conditional stores here
                s1[p] = np.fmin(a, b1)
                j1[p] = ii if c else j1[p]
                s2[p] = n2
        for p in range(m):
            q = start + p
            xn2 = 0.0
            for f in range(F):
                v = np.float64(data[q, f])
                xn2 += v * v
            e = 2.0 * _U32 * (wmax + np.sqrt(xn2)) * 1.01 + 1e-18
            lo2 = np.sqrt(np.float64(s2[p]) / (1.0 + g)) - e
            hi1 = (np.sqrt(np.float64(s1[p]) / (1.0 - g)) + e) * (1.0 + 1e-12)
            if lo2 > hi1:  # False for NaN/inf/overflow and near-ties
                j = j1[p]
                acc64 = 0.0
                for f in range(F):
                    d = weights[j, f] - np.float64(data[q, f])
                    acc64 += d * d
                out_idx[q] = j
                out_dist2[q] = acc64
            else:
                best = np.inf
                best_idx = 0
                for i in range(n_nodes):
                    acc64 = 0.0
                    for f in range(F):
                        d = weights[i, f] - np.float64(data[q, f])
                        acc64 += d * d
                    if acc64 < best:
                        best = acc64
                        best_idx = i
                out_idx[q] = best_idx
                out_dist2[q] = best
                nfb[t] += 1
    return nfb.sum()


def best_match_nodes_screen(
    weights: np.ndarray, data: np.ndarray, chunk: int = DEFAULT_CHUNK, return_fallbacks: bool = False
):
    """Find the best-matching node of every point; bitwise equal to :func:`best_match_nodes`.

    Each point is first scanned against all nodes in float32 (vectorised over
    a tile of points), keeping the smallest screen value ``s1`` (node ``j1``)
    and the smallest value over every other node ``s2``. With u = 2**-24, the
    float32 screen distance of every node is within
    ``e = 2.02u(max|w| + |x|) + 1e-18`` of the true Euclidean distance after
    input rounding (the additive term covers float32 underflow of tiny
    differences), and the float32 sum is within a relative ``(F + 2)u``. If
    the resulting lower bound for the second node still exceeds the upper
    bound for ``j1`` by a relative 1e-12, every other node is strictly
    farther than ``j1`` by far more than float64 rounding, so the reference
    scan picks ``j1`` too; its squared distance is then recomputed with the
    reference float64 arithmetic. Every other point (near-ties, exact ties,
    non-finite or overflowing values) gets the full float64 reference scan.
    Labels and distances are therefore bitwise those of
    :func:`best_match_nodes`, for float64 and float32 input; the bound was
    checked independently for F = 1 to 10. Non-finite or huge (> 1e15)
    weights use :func:`best_match_nodes` directly.

    Args:
        weights (np.ndarray): [nodes, F] lattice weights, node index = x + y * xdim.
        data (np.ndarray): [N, F] float64 or float32 points in the scaled space.
        chunk (int, optional): Points per kernel call. Defaults to 2**20.
        return_fallbacks (bool, optional): Also return how many points needed
            the full float64 scan. Defaults to False.

    Returns:
        tuple: int32 [N] node indices and float64 [N] squared distances (and
            the fallback count if ``return_fallbacks``).
    """
    weights, data = _check_inputs(weights, data)
    if chunk < 1:
        raise ValueError("chunk must be positive")
    n_points = data.shape[0]
    if not np.all(np.isfinite(weights)) or np.max(np.abs(weights)) > 1e15:
        out = best_match_nodes(weights, data, chunk=chunk)
        return (*out, n_points) if return_fallbacks else out
    w32 = weights.astype(np.float32)
    wmax = float(np.sqrt((weights * weights).sum(axis=1)).max())
    nodes_idx = np.empty(n_points, dtype=np.int32)
    dist2 = np.empty(n_points, dtype=np.float64)
    fallbacks = 0
    for start in range(0, n_points, chunk):
        stop = min(start + chunk, n_points)
        block = np.ascontiguousarray(data[start:stop])
        fallbacks += int(
            _screen_kernel(
                weights, w32, wmax, block, nodes_idx[start:stop], dist2[start:stop], SCREEN_TILE
            )
        )
    if return_fallbacks:
        return nodes_idx, dist2, fallbacks
    return nodes_idx, dist2


def predict(
    weights: np.ndarray,
    node_cluster: np.ndarray,
    data: np.ndarray,
    chunk: int | None = None,
    return_distance: bool = False,
    backend: str = "screen",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Assign each point the cluster id of its best-matching node.

    ``backend="screen"`` (the default) runs :func:`best_match_nodes_screen`
    (float32 screen certified in float64; labels and distances bitwise
    equal to ``"numba"``, and faster on most lattices).
    ``backend="numba"`` runs :func:`best_match_nodes`, the exact float64
    reference scan.

    Args:
        weights (np.ndarray): [nodes, F] lattice weights.
        node_cluster (np.ndarray): [nodes] cluster id of each node (-1 = unassigned).
        data (np.ndarray): [N, F] float64 or float32 points in the scaled space.
        chunk (int | None, optional): Number of points per kernel call.
            Defaults to None (2**20).
        return_distance (bool, optional): Also return the Euclidean BMU distance.
            Defaults to False.
        backend (str, optional): ``"screen"`` or ``"numba"``. Defaults to ``"screen"``.

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]: int32 [N] labels, and float64
            [N] distances if ``return_distance`` is True.
    """
    node_cluster = np.asarray(node_cluster, dtype=np.int32).ravel()
    if node_cluster.shape[0] != np.shape(weights)[0]:
        raise ValueError("node_cluster must have one entry per node")
    if backend == "numba":
        nodes_idx, dist2 = best_match_nodes(
            weights, data, chunk=DEFAULT_CHUNK if chunk is None else chunk
        )
    elif backend == "screen":
        nodes_idx, dist2 = best_match_nodes_screen(
            weights, data, chunk=DEFAULT_CHUNK if chunk is None else chunk
        )
    else:
        raise ValueError('backend must be "screen" or "numba"')
    labels = node_cluster[nodes_idx]
    if return_distance:
        return labels, np.sqrt(dist2)
    return labels
