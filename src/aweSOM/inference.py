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

:func:`best_match_nodes_patch` prunes the scan with bounding balls over
``side x side`` lattice patches of neighbouring nodes. It requires the
lattice dimensions ``xdim`` and ``ydim``, and returns node indices and
squared distances bitwise equal to :func:`best_match_nodes` on finite,
tied, ``NaN``, and infinite inputs.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

__all__ = [
    "best_match_nodes",
    "best_match_nodes_patch",
    "best_match_nodes_screen",
    "narrow_label_dtype",
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
            # False for NaN/inf and near-ties. An overflowed runner-up s2 bounds
            # nothing (lo2 = inf), so it must be finite.
            if lo2 > hi1 and s2[p] < np.inf:
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

    Tie and non-finite contract (D3):
        Exact ties pick the lowest node index. The scan skips any node whose
        squared distance is ``NaN``. A row with no finite distance, including
        any row with a ``NaN``, ``+inf``, or ``-inf`` coordinate against finite
        weights, returns node ``0`` and squared distance ``inf``. A tile entry
        with runner-up ``s2 == inf``, whether from ``float32`` overflow or a
        single-node lattice, never certifies and falls back to the full
        ``float64`` reference scan. Non-finite weights or weights with
        ``max(|w|) > 1e15`` call :func:`best_match_nodes` directly. Node
        indices and squared distances equal :func:`best_match_nodes` bitwise
        in every case.

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


PATCH_TILE = 64


def lattice_patches(weights: np.ndarray, xdim: int, ydim: int, side: int | None = None) -> tuple:
    """Group the nodes into ``side x side`` lattice patches with a bounding ball each.

    Returns ``(wpT, perm, offs, centT, rad)``: weights in patch-major order,
    transposed to ``(F, nodes)``; ``perm`` maps that order to node indices
    (ascending within a patch); ``offs`` the patch boundaries; ``centT`` the
    ``(F, patches)`` centroids; ``rad`` the radii of the balls around them, inflated
    by a relative 1e-9 (plus 1e-300). Near squared-distance underflow these
    radii can lose containment; the search restarts unsafe rows without pruning.
    """
    w = np.ascontiguousarray(weights, dtype=np.float64)
    n_nodes = w.shape[0]
    if n_nodes != xdim * ydim:
        raise ValueError(f"weights has {n_nodes} nodes, expected {xdim} * {ydim}")
    if side is None:
        side = default_patch_side(n_nodes)
    side = max(1, int(side))
    perm = []
    offs = [0]
    for by in range(0, ydim, side):
        for bx in range(0, xdim, side):
            perm.extend(
                x + y * xdim
                for y in range(by, min(ydim, by + side))
                for x in range(bx, min(xdim, bx + side))
            )
            offs.append(len(perm))
    perm = np.asarray(perm, dtype=np.int64)
    offs = np.asarray(offs, dtype=np.int64)
    n_patch = len(offs) - 1
    cent = np.empty((n_patch, w.shape[1]))
    rad = np.empty(n_patch)
    for b in range(n_patch):
        m = w[perm[offs[b] : offs[b + 1]]]
        cent[b] = m.mean(axis=0)
        rad[b] = np.sqrt(((m - cent[b]) ** 2).sum(axis=1)).max()
    rad = rad * (1.0 + 1e-9) + 1e-300
    return np.ascontiguousarray(w[perm].T), perm, offs, np.ascontiguousarray(cent.T), rad


def default_patch_side(n_nodes: int) -> int:
    """Patch side used when none is given: 4 up to 8000 nodes, 5 above.

    From a sweep of sides 3 to 8 on synthetic five-feature data, side 4 was
    fastest at 91 x 64 nodes and side 5 was fastest at 160 x 100 nodes.
    """
    return 4 if n_nodes <= 8000 else 5


# Absolute allowance of the feature-group certificate. It exceeds every
# underflow error of a squared distance (at most F * 2**-1074), and keeps every
# bound quantity normal, so that only relative rounding errors remain.
FEATURE_ETA = 2.0**-1000


def _feature_slack(n_feat: int) -> float:
    """Relative error allowance of a computed squared distance over ``n_feat`` features.

    A computed ``sum((a - b)**2)`` has relative error at most ``(F + 2) * 2**-53``
    plus ``F * 2**-1074`` from underflow (in any summation order, with or without
    fused multiply-add); this is 512 times the relative part.
    """
    return (n_feat + 2) * 2.0**-44


def _feature_patches(weights: np.ndarray, leaf_size: int = 16) -> tuple:
    """Bounding balls for balanced feature-space groups, retaining original IDs.

    Split each group at the median of its widest feature. This only changes
    the search index, never the weights or their lattice positions. The radii
    are certified upper bounds: with ``e = _feature_slack(F)`` and computed
    squared radius ``r2``, the true squared radius is at most
    ``(r2 + FEATURE_ETA) / (1 - e)``, and ``rad`` is rounded above
    ``sqrt((r2 + FEATURE_ETA) * (1 + 2e))``, which exceeds it.

    Returns ``(wpT, perm, offs, centT, rad, w)`` for :func:`_feature_patch_kernel`.
    """
    w = np.ascontiguousarray(weights, dtype=np.float64)
    leaves = []

    def split(ids):
        if len(ids) <= leaf_size:
            leaves.append(np.sort(ids))
            return
        values = w[ids]
        axis = np.argmax(np.ptp(values, axis=0))
        mid = len(ids) // 2
        order = np.argpartition(values[:, axis], mid)
        split(ids[order[:mid]])
        split(ids[order[mid:]])

    split(np.arange(len(w), dtype=np.int64))
    perm = np.concatenate(leaves)
    offs = np.concatenate(([0], np.cumsum([len(ids) for ids in leaves])))
    cent = np.empty((len(leaves), w.shape[1]))
    r2 = np.empty(len(leaves))
    for b, ids in enumerate(leaves):
        members = w[ids]
        cent[b] = members.mean(axis=0)
        r2[b] = ((members - cent[b]) ** 2).sum(axis=1).max()
    eps = _feature_slack(w.shape[1])
    rad = np.nextafter(np.sqrt((r2 + FEATURE_ETA) * (1.0 + 2.0 * eps)), np.inf)
    return np.ascontiguousarray(w[perm].T), perm, offs, np.ascontiguousarray(cent.T), rad, w


@njit(parallel=True, nogil=True, cache=True)
def _feature_patch_kernel(wpT, perm, offs, centT, rad, w, data, out_idx, out_dist2) -> int:
    """Exact BMU search pruned by the certified balls of :func:`_feature_patches`.

    Bitwise equal to :func:`_bmu_kernel` for every input. A group is skipped
    only if every member's computed squared distance is provably larger than
    the current best, so neither a smaller distance nor an equal one with a
    lower index can be lost. With ``e = _feature_slack(F)`` and
    ``h = FEATURE_ETA``, a computed squared distance ``a`` to a point at true
    squared distance ``D`` satisfies ``D (1 - e) - h <= a <= D (1 + e) + h``.
    The group is skipped when ``(dc2 - h)(1 - e) > (rad + s)**2 (1 + 2**-48)``
    with ``s = sqrt((best + h)(1 + 2e))``; this implies ``|x - c| > R + s'``
    for the true distance to the centroid and the true radius ``R``, with
    ``s' = sqrt((best + h) / (1 - e))``, hence ``a > best`` for every member.
    Rows with a non-finite centroid distance (non-finite or huge queries)
    cannot be certified and are scanned in full with the reference arithmetic.
    Returns node evaluations.
    """
    n = data.shape[0]
    F, n_patch = centT.shape
    n_nodes = w.shape[0]
    eta = FEATURE_ETA
    eps = (F + 2) * 2.0**-44
    lo_scale = 1.0 - eps
    hi_scale = 1.0 + 2.0 * eps
    kappa = 1.0 + 2.0**-48
    max_size = 0
    for b in range(n_patch):
        max_size = max(max_size, offs[b + 1] - offs[b])
    ntiles = (n + PATCH_TILE - 1) // PATCH_TILE
    evals = np.zeros(ntiles, dtype=np.int64)
    for t in prange(ntiles):
        x = np.empty(F)
        dc2 = np.empty(n_patch)
        acc = np.empty(max_size)
        ev = 0
        for p in range(t * PATCH_TILE, min(n, (t + 1) * PATCH_TILE)):
            for f in range(F):
                x[f] = np.float64(data[p, f])
            for b in range(n_patch):
                d = centT[0, b] - x[0]
                dc2[b] = d * d
            for f in range(1, F):
                xf = x[f]
                for b in range(n_patch):
                    d = centT[f, b] - xf
                    dc2[b] += d * d
            first = 0
            dfirst = np.inf
            certified = True
            for b in range(n_patch):
                if not dc2[b] < np.inf:
                    certified = False
                if dc2[b] < dfirst:
                    dfirst = dc2[b]
                    first = b
            best = np.inf
            best_idx = 0
            if not certified:
                # reference scan of _bmu_kernel, ascending node order
                for i in range(n_nodes):
                    a = 0.0
                    for f in range(F):
                        d = w[i, f] - x[f]
                        a += d * d
                    if a < best:
                        best = a
                        best_idx = i
                ev += n_nodes
                out_idx[p] = best_idx
                out_dist2[p] = best
                continue
            sb = np.inf  # sqrt((best + eta) * (1 + 2 eps)), an upper bound
            for step in range(n_patch + 1):
                if step == 0:
                    b = first
                else:
                    b = step - 1
                    if b == first:
                        continue
                    lim = rad[b] + sb
                    if (dc2[b] - eta) * lo_scale > lim * lim * kappa:
                        continue
                lo = offs[b]
                s = offs[b + 1] - lo
                # reference arithmetic of _bmu_kernel per node: acc = 0; acc += d * d
                for k in range(s):
                    d = wpT[0, lo + k] - x[0]
                    acc[k] = d * d
                for f in range(1, F):
                    xf = x[f]
                    for k in range(s):
                        d = wpT[f, lo + k] - xf
                        acc[k] += d * d
                improved = False
                for k in range(s):
                    a = acc[k]
                    i = perm[lo + k]
                    # lexicographic (distance, index) minimum = strict ascending scan
                    if a < best or (a == best and i < best_idx):
                        best = a
                        best_idx = i
                        improved = True
                if improved:
                    sb = np.sqrt((best + eta) * hi_scale)
                ev += s
            out_idx[p] = best_idx
            out_dist2[p] = best
        evals[t] = ev
    return evals.sum()


@njit(parallel=True, nogil=True, cache=True)
def _patch_kernel(wpT, perm, offs, centT, rad, data, out_idx, out_dist2) -> int:
    """Exact BMU search pruned by patch balls; see best_match_nodes_patch. Returns node evaluations."""
    n = data.shape[0]
    F, n_patch = centT.shape
    # Original node order is needed if a row must restart after pruning.
    order = np.empty(len(perm), dtype=np.int64)
    for k in range(len(perm)):
        order[perm[k]] = k
    max_size = 0
    for b in range(n_patch):
        max_size = max(max_size, offs[b + 1] - offs[b])
    ntiles = (n + PATCH_TILE - 1) // PATCH_TILE
    evals = np.zeros(ntiles, dtype=np.int64)
    for t in prange(ntiles):
        x = np.empty(F)
        dc2 = np.empty(n_patch)
        acc = np.empty(max_size)
        ev = 0
        for p in range(t * PATCH_TILE, min(n, (t + 1) * PATCH_TILE)):
            for f in range(F):
                x[f] = np.float64(data[p, f])
            # squared distance to every patch centroid (vectorised over patches)
            for b in range(n_patch):
                d = centT[0, b] - x[0]
                dc2[b] = d * d
            for f in range(1, F):
                xf = x[f]
                for b in range(n_patch):
                    d = centT[f, b] - xf
                    dc2[b] += d * d
            first = 0
            dfirst = np.inf
            unsafe = False
            for b in range(n_patch):
                if not dc2[b] < np.inf:
                    unsafe = True
                if dc2[b] < dfirst:
                    dfirst = dc2[b]
                    first = b
            best = np.inf
            best_idx = 0
            sb = np.inf  # sqrt(best), inflated
            # the patch with the nearest centroid first, then every patch whose
            # ball can still hold a node at distance <= sqrt(best)
            steps = 0 if unsafe else n_patch + 1
            for step in range(steps):
                if step == 0:
                    b = first
                else:
                    b = step - 1
                    if b == first:
                        continue
                    lim = sb + rad[b]
                    if dc2[b] * (1.0 - 1e-10) > lim * lim:
                        continue
                lo = offs[b]
                s = offs[b + 1] - lo
                # reference arithmetic of _bmu_kernel per node: acc = 0; acc += d * d
                for k in range(s):
                    d = wpT[0, lo + k] - x[0]
                    acc[k] = d * d
                for f in range(1, F):
                    xf = x[f]
                    for k in range(s):
                        d = wpT[f, lo + k] - xf
                        acc[k] += d * d
                improved = False
                for k in range(s):
                    a = acc[k]
                    i = perm[lo + k]
                    # lexicographic (distance, index) minimum = strict ascending scan
                    if a < best or (a == best and i < best_idx):
                        best = a
                        best_idx = i
                        improved = True
                ev += s
                if improved:
                    sb = np.sqrt(best) * (1.0 + 1e-10)
                    # Relative margins cannot protect rounded-distance ties here.
                    # Restart the WHOLE row: an earlier patch may already be pruned.
                    if best <= FEATURE_ETA:
                        unsafe = True
                        break
            if unsafe:
                best = np.inf
                best_idx = 0
                for i in range(len(order)):
                    k = order[i]
                    a = 0.0
                    for f in range(F):
                        d = wpT[f, k] - x[f]
                        a += d * d
                    if a < best:
                        best = a
                        best_idx = i
                ev += len(order)
            out_idx[p] = best_idx
            out_dist2[p] = best
        evals[t] = ev
    return evals.sum()


def best_match_nodes_patch(
    weights: np.ndarray,
    data: np.ndarray,
    xdim: int,
    ydim: int,
    chunk: int = DEFAULT_CHUNK,
    side: int | None = None,
    return_evaluations: bool = False,
):
    """Find the best-matching node of every point; bitwise equal to :func:`best_match_nodes`.

    The lattice is cut into ``side x side`` patches of neighbouring nodes, and
    each patch gets the ball (centroid, radius) that contains its weights. For
    every point the squared distances to all centroids are computed, the patch
    with the nearest centroid is scanned first, and every other patch is
    scanned only if its ball can still hold a node at least as close as the
    best found so far (``|x - c| - r <= best``, checked with a relative margin
    of 1e-10, far above float64 rounding). Rows with a non-finite centroid
    distance or a current best squared distance <= 2**-1000 restart with a full
    ascending reference scan, since relative margins do not cover underflow.
    The nodes of a scanned patch use the
    reference float64 arithmetic of :func:`best_match_nodes`, and a node wins if
    its distance is smaller, or equal with a lower index, so the result is the
    lexicographic (distance, index) minimum: labels and distances are bitwise
    those of :func:`best_match_nodes` (NaN and inf rows included). Because a
    trained lattice is a smooth 2-D sheet, a point usually needs a few patches;
    the cost per point is about ``patches + scanned nodes`` distance
    evaluations instead of ``nodes``. Non-finite or huge (> 1e15) weights use
    :func:`best_match_nodes`.

    Tie and non-finite contract (D3):
        Exact ties within a patch or across patches pick the lowest node index
        through the lexicographic ``(distance, index)`` comparison, while the
        pruning bound retains every patch that can hold an equal distance. Any
        row with a ``NaN``, ``+inf``, or ``-inf`` coordinate produces a
        non-finite centroid distance and restarts with the full ascending
        ``float64`` scan, which skips ``NaN`` node distances and returns node
        ``0`` and squared distance ``inf`` when weights are finite. Non-finite
        weights or weights with ``max(|w|) > 1e15`` call
        :func:`best_match_nodes` directly. Node indices and squared distances
        equal :func:`best_match_nodes` bitwise in every case.

    Args:
        weights (np.ndarray): [xdim * ydim, F] lattice weights, node index = x + y * xdim.
        data (np.ndarray): [N, F] float64 or float32 points in the scaled space.
        xdim (int): Lattice x dimension.
        ydim (int): Lattice y dimension.
        chunk (int, optional): Points per kernel call. Defaults to 2**20.
        side (int | None, optional): Patch side in nodes. Defaults to
            :func:`default_patch_side`.
        return_evaluations (bool, optional): Also return the number of node
            distance evaluations (patch centroids not included).

    Returns:
        tuple: int32 [N] node indices and float64 [N] squared distances (and the
            evaluation count if ``return_evaluations``).
    """
    weights, data = _check_inputs(weights, data)
    if chunk < 1:
        raise ValueError("chunk must be positive")
    n_points = data.shape[0]
    if not np.all(np.isfinite(weights)) or np.max(np.abs(weights)) > 1e15:
        out = best_match_nodes(weights, data, chunk=chunk)
        return (*out, n_points * weights.shape[0]) if return_evaluations else out
    patches = lattice_patches(weights, xdim, ydim, side)
    nodes_idx = np.empty(n_points, dtype=np.int32)
    dist2 = np.empty(n_points, dtype=np.float64)
    evals = 0
    for start in range(0, n_points, chunk):
        stop = min(start + chunk, n_points)
        block = np.ascontiguousarray(data[start:stop])
        evals += int(_patch_kernel(*patches, block, nodes_idx[start:stop], dist2[start:stop]))
    if return_evaluations:
        return nodes_idx, dist2, evals
    return nodes_idx, dist2


def narrow_label_dtype(ids) -> np.dtype:
    """Smallest of uint8, uint16 and int32 that holds every value of ``ids``.

    uint8 or uint16 needs every id non-negative and at most 255 or 65535;
    otherwise (also for no ids) the result is int32.
    """
    ids = np.asarray(ids)
    if ids.size and ids.min() >= 0:
        top = ids.max()
        for dtype in (np.uint8, np.uint16):
            if top <= np.iinfo(dtype).max:
                return np.dtype(dtype)
    return np.dtype(np.int32)


def predict(
    weights: np.ndarray,
    node_cluster: np.ndarray,
    data: np.ndarray,
    chunk: int | None = None,
    return_distance: bool = False,
    backend: str = "screen",
    lattice_shape: tuple[int, int] | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Assign each point the cluster id of its best-matching node.

    ``backend="screen"`` is the default and runs :func:`best_match_nodes_screen`,
    scanning in float32 and certifying winners in float64. ``backend="patch"``
    runs :func:`best_match_nodes_patch`, pruning with bounding balls over lattice
    patches, and requires ``lattice_shape=(xdim, ydim)``; it is fastest on
    trained lattices of a few thousand nodes or more. ``backend="numba"`` runs
    :func:`best_match_nodes`, the reference float64 scan.

    Tie and non-finite contract (D3):
        Exact ties assign ``node_cluster[lowest_idx]`` using the lowest tied
        node index. Every backend skips nodes with ``NaN`` distances, and any
        row with no finite distance, including ``NaN``, ``+inf``, or ``-inf``
        coordinates against finite weights, receives ``node_cluster[0]`` and
        Euclidean distance ``inf``. All three backends ``"screen"``,
        ``"patch"``, and ``"numba"`` return bitwise-identical ``int32`` labels
        and ``float64`` Euclidean distances.

    Args:
        weights (np.ndarray): [nodes, F] lattice weights.
        node_cluster (np.ndarray): [nodes] cluster id of each node (-1 = unassigned).
        data (np.ndarray): [N, F] float64 or float32 points in the scaled space.
        chunk (int | None, optional): Number of points per kernel call.
            Defaults to None (2**20).
        return_distance (bool, optional): Also return the Euclidean BMU distance.
            Defaults to False.
        backend (str, optional): ``"screen"``, ``"patch"``, or ``"numba"``.
            Defaults to ``"screen"``.
        lattice_shape (tuple[int, int] | None, optional): ``(xdim, ydim)``,
            required by ``backend="patch"``.

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
    elif backend == "patch":
        if lattice_shape is None:
            raise ValueError('backend="patch" needs lattice_shape=(xdim, ydim)')
        nodes_idx, dist2 = best_match_nodes_patch(
            weights, data, *lattice_shape, chunk=DEFAULT_CHUNK if chunk is None else chunk
        )
    else:
        raise ValueError('backend must be "numba", "screen" or "patch"')
    labels = node_cluster[nodes_idx]
    if return_distance:
        return labels, np.sqrt(dist2)
    return labels
