"""JAX-free pieces of the batch SOM map shared by :mod:`aweSOM.batch` and :mod:`aweSOM.stream`.

Radius schedule, neighbourhood matrix and the exact float64 quality measures
(quantization and topographic error). They are re-exported by
:mod:`aweSOM.batch`; this module exists so that the streaming CPU trainer
(:mod:`aweSOM.stream`) runs without JAX installed.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from numba import njit, prange

__all__ = [
    "radius_schedule",
    "neighbourhood_matrix",
    "quantization_error",
    "topographic_error",
    "quantization_error_fast",
    "topographic_error_fast",
]

def radius_schedule(epochs: int, r_max: float, r_min: float) -> np.ndarray:
    """Neighbourhood radius for each epoch.

    The schedule has the same proportions as the online trainer: ``r_max`` for
    the first half, a linear decrease over the next quarter, ``r_min`` for the
    rest.

    Rounding: the constant phase lasts ``ceil(epochs / 2)`` epochs and the linear
    phase ``min(ceil(epochs / 4), epochs - ceil(epochs / 2))`` epochs; the final
    phase takes the remainder. The linear phase takes the interior points of an
    evenly spaced grid from ``r_max`` to ``r_min`` (both endpoints excluded), so
    with 40 epochs there are exactly 20 epochs at ``r_max``, 10 strictly between,
    and 10 at ``r_min``. Radii are not rounded to integers;
    :func:`neighbourhood_matrix` accepts fractional radii.

    Args:
        epochs: Number of epochs (>= 1).
        r_max: Initial radius.
        r_min: Final radius.

    Returns:
        Float64 array of shape ``(epochs,)``.

    Raises:
        ValueError: If ``epochs < 1``.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    n_max = math.ceil(epochs / 2)
    n_lin = min(math.ceil(epochs / 4), epochs - n_max)
    n_min = epochs - n_max - n_lin
    linear = np.linspace(r_max, r_min, n_lin + 2)[1:-1]
    return np.concatenate(
        [np.full(n_max, float(r_max)), linear, np.full(n_min, float(r_min))]
    )


def _lattice_coordinates(xdim: int, ydim: int) -> np.ndarray:
    """Return integer (x, y) coordinates of each node, node i -> (i % xdim, i // xdim)."""
    i = np.arange(xdim * ydim)
    return np.stack([i % xdim, i // xdim], axis=1)


@functools.lru_cache(maxsize=8)
def _chebyshev_matrix(xdim: int, ydim: int) -> np.ndarray:
    """Return the Chebyshev lattice distance between every pair of nodes (no wrap).

    Cached; the returned array is read-only.
    """
    xy = _lattice_coordinates(xdim, ydim)
    d = np.abs(xy[:, None, :] - xy[None, :, :]).max(axis=2)
    d.flags.writeable = False
    return d


def _neighbourhood64(xdim: int, ydim: int, radius: float) -> np.ndarray:
    """Float64 version of :func:`neighbourhood_matrix`."""
    d = _chebyshev_matrix(xdim, ydim).astype(np.float64)
    if radius <= 0:
        return (d == 0).astype(np.float64)
    sigma = radius / 3.0
    h = np.exp(-(d**2) / (2.0 * sigma**2))
    h[d > radius] = 0.0
    return h


def neighbourhood_matrix(xdim: int, ydim: int, radius: float) -> np.ndarray:
    """Neighbourhood matrix of ``Lattice.Gamma`` with alpha = 1.

    ``H[c, i] = exp(-d^2 / (2 (radius / 3)^2))`` if ``d <= radius`` else 0, where
    ``d`` is the Chebyshev distance between nodes ``c`` and ``i`` on the
    ``xdim x ydim`` grid (node i at ``(i % xdim, i // xdim)``, no wrap). Row ``c``
    equals ``Lattice.Gamma(c, m2Ds, 1.0, radius)`` for an integer radius. For
    ``radius <= 0`` (where Gamma divides by zero) the limit is used: H is the
    identity. :func:`train_batch` uses the same matrix in float64.

    Args:
        xdim: Lattice x dimension.
        ydim: Lattice y dimension.
        radius: Neighbourhood radius (may be fractional).

    Returns:
        Float32 array of shape ``(xdim * ydim, xdim * ydim)``; symmetric.
    """
    return _neighbourhood64(xdim, ydim, radius).astype(np.float32)


@njit(parallel=True, cache=True)
def _top2_kernel(w: np.ndarray, x: np.ndarray, d1: np.ndarray, i1: np.ndarray, i2: np.ndarray) -> None:
    """First and second BMU of every row of x, ordered by (squared distance, index).

    Squared distances are summed over features in order, like the einsum of
    :func:`_sq_distances`; a strict ``<`` scan keeps the lower index on ties.
    """
    n_nodes, n_feat = w.shape
    for p in prange(x.shape[0]):
        b1 = np.inf
        b2 = np.inf
        j1 = 0
        j2 = 0
        for j in range(n_nodes):
            acc = 0.0
            for f in range(n_feat):
                diff = x[p, f] - w[j, f]
                acc += diff * diff
            if acc < b1:
                b2 = b1
                j2 = j1
                b1 = acc
                j1 = j
            elif acc < b2:
                b2 = acc
                j2 = j
        d1[p] = b1
        i1[p] = j1
        i2[p] = j2


def _top2(weights: np.ndarray, data: np.ndarray):
    w = np.ascontiguousarray(weights, np.float64)
    x = np.ascontiguousarray(data, np.float64)
    n = x.shape[0]
    d1 = np.empty(n)
    i1 = np.empty(n, np.int64)
    i2 = np.empty(n, np.int64)
    _top2_kernel(w, x, d1, i1, i2)
    return d1, i1, i2


def quantization_error_fast(weights: np.ndarray, data: np.ndarray) -> float:
    """:func:`quantization_error` with a compiled multithreaded scan (finite inputs)."""
    d1, _, _ = _top2(weights, data)
    block = _row_block(*np.shape(weights))
    total = 0.0
    q = np.sqrt(d1)
    for start in range(0, q.size, block):
        total += q[start : start + block].sum()
    return total / q.size


def topographic_error_fast(weights: np.ndarray, data: np.ndarray, xdim: int, ydim: int) -> float:
    """:func:`topographic_error` with a compiled multithreaded scan (finite inputs)."""
    if np.shape(weights)[0] != xdim * ydim:
        raise ValueError(f"weights has {np.shape(weights)[0]} nodes, expected {xdim * ydim}")
    _, first, second = _top2(weights, data)
    xy = _lattice_coordinates(xdim, ydim)
    cheb = np.abs(xy[first] - xy[second]).max(axis=1)
    return int((cheb > 1).sum()) / first.size


def _row_block(n_nodes: int, n_feat: int, max_elements: int = 1 << 23) -> int:
    """Rows per block so a ``rows x nodes x features`` float64 temporary stays near 64 MB."""
    return max(1, max_elements // max(1, n_nodes * n_feat))


def _sq_distances(w: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """Exact float64 squared distances ``(rows, nodes)`` by direct arithmetic."""
    diff = xb[:, None, :] - w[None, :, :]
    return np.einsum("ijk,ijk->ij", diff, diff)


def quantization_error(weights: np.ndarray, data: np.ndarray) -> float:
    """Mean Euclidean distance from each sample to its BMU (float64, exact).

    Args:
        weights: Lattice ``(nodes, F)``.
        data: Samples ``(N, F)``.

    Returns:
        The mean quantization error.
    """
    w = np.asarray(weights, np.float64)
    x = np.asarray(data, np.float64)
    block = _row_block(*w.shape)
    total = 0.0
    for start in range(0, x.shape[0], block):
        d = _sq_distances(w, x[start : start + block])
        total += np.sqrt(d.min(axis=1)).sum()
    return total / x.shape[0]


def topographic_error(
    weights: np.ndarray, data: np.ndarray, xdim: int, ydim: int
) -> float:
    """Fraction of samples whose two best-matching nodes are not lattice neighbours.

    The first and second BMUs are distinct nodes ordered by (distance, index).
    They are neighbours if their Chebyshev lattice distance is <= 1
    (8-neighbourhood, no wrap). Distances are exact float64.

    Args:
        weights: Lattice ``(xdim * ydim, F)``.
        data: Samples ``(N, F)``.
        xdim: Lattice x dimension.
        ydim: Lattice y dimension.

    Returns:
        The topographic error in [0, 1].
    """
    w = np.asarray(weights, np.float64)
    x = np.asarray(data, np.float64)
    if w.shape[0] != xdim * ydim:
        raise ValueError(f"weights has {w.shape[0]} nodes, expected {xdim * ydim}")
    xy = _lattice_coordinates(xdim, ydim)
    block = _row_block(*w.shape)
    errors = 0
    for start in range(0, x.shape[0], block):
        d = _sq_distances(w, x[start : start + block])
        rows = np.arange(d.shape[0])
        first = d.argmin(axis=1)
        d[rows, first] = np.inf
        second = d.argmin(axis=1)
        cheb = np.abs(xy[first] - xy[second]).max(axis=1)
        errors += int((cheb > 1).sum())
    return errors / x.shape[0]
