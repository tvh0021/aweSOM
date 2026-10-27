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
    ``np.argmin``. A node whose distance is NaN (non-finite weights or point)
    is never selected, unlike ``np.argmin`` which returns the first NaN; a
    point with no finite distance gets node 0 and distance inf. For float64 input the labels equal ``Lattice.best_match``
    bitwise. Float32 input is promoted to float64 element by element; the
    result is exact for the promoted values but may differ from a float64
    version of the original data where two nodes are nearly tied.

Measured throughput (``benchmark()``, N = 2,000,000, F = 6, 2368 nodes,
float64, Apple M1 Max with 10 threads, numba 0.68): ``best_match_nodes``
1.61e6 points/s, ``Lattice.best_match`` 1.00e5 points/s, identical labels.
See ``cpp/README.md`` for the C++ figure.

:func:`best_match_nodes_jax` is the same search on a JAX device (CPU, CUDA,
ROCm or Apple Metal), imported lazily so ``aweSOM`` does not need JAX. With
float64 available it follows the contract above and its labels and distances
equal :func:`best_match_nodes` bitwise; without float64 (Apple Metal) it runs
the same direct-difference arithmetic in float32. Measured with
``planning/benchmarks/bench_inference_jax.py`` (N = 2,000,000, F = 4, 2368
nodes, default tile of 14,169 rows, Apple M1 Max): numba 2.2e6 points/s; JAX
0.12 CPU float64 5.1e5 points/s (bitwise-equal labels and distances); JAX
0.4.34 + jax-metal 0.1.1 float32 2.9e6 points/s (0 label disagreements). On
a CPU-only machine the numba path is the faster one.

:func:`best_match_nodes_kdtree` queries a ``scipy.spatial.cKDTree`` of the
nodes and then re-evaluates the chosen node's distance with the exact
float64 scan, so distances follow the contract above; only exact ties can
pick a different node. On random data in [0, 1) it reaches 8.0e6 points/s
at F = 4, but a trained lattice lies close to a 2D sheet in feature space,
so the tree visits hundreds of nodes per point on real data.

:func:`best_match_nodes_screen` (the ``predict`` default) scans every node
in float32, certifies the float32 winner with a rounding-error bound and
falls back to the float64 scan for every point it cannot certify, so its
labels and distances equal :func:`best_match_nodes` bitwise. Points per
second on an Apple M1 Max, 2368 nodes, ``planning/opt/inference-cpu``:

=====================  ======  ======  ======
data                   numba   kdtree  screen
=====================  ======  ======  ======
plasma lattice, F = 4  2.2e6   3.0e6   4.9e6
random, F = 10         9.6e5   5.9e5   2.5e6
=====================  ======  ======  ======
"""

from __future__ import annotations

import contextlib
import functools
import time
import warnings

import numpy as np
from numba import njit, prange

__all__ = [
    "best_match_nodes",
    "best_match_nodes_jax",
    "best_match_nodes_kdtree",
    "best_match_nodes_screen",
    "predict",
    "benchmark",
]

DEFAULT_CHUNK = 2**20
SCREEN_TILE = 512
KDTREE_LEAFSIZE = 64
_U32 = 2.0**-24  # float32 unit roundoff
JAX_TILE_ELEMENTS = 2**25


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


@njit(parallel=True, cache=True)
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


def _jax_x64_scope():
    """Return a context manager that enables float64 in JAX, or None if it cannot.

    Uses the scoped ``jax.enable_x64`` (recent JAX) or
    ``jax.experimental.enable_x64`` (JAX 0.4) so the global config is left as it
    was. As a last resort the global flag is switched on and restored on exit.
    """
    import jax

    if hasattr(jax, "enable_x64"):
        return jax.enable_x64(True)
    try:
        from jax.experimental import enable_x64
    except ImportError:
        enable_x64 = None
    if enable_x64 is not None:
        return enable_x64(True)

    @contextlib.contextmanager
    def _global_flag():
        old = bool(jax.config.read("jax_enable_x64"))
        jax.config.update("jax_enable_x64", True)
        try:
            yield
        finally:
            jax.config.update("jax_enable_x64", old)

    try:
        jax.config.read("jax_enable_x64")
    except Exception:
        return None
    return _global_flag()


def _jax_float64_scope(devices):
    """Return a fresh x64 scope if every device computes in float64 inside it, else None."""
    import jax

    if any(d.platform.upper() == "METAL" for d in devices):
        return None  # jax-metal has no float64; probing it can abort the process
    scope = _jax_x64_scope()
    if scope is None:
        return None
    try:
        with scope:
            for d in devices:
                probe = jax.device_put(np.ones(2, dtype=np.float64), d)
                if (probe * probe).dtype != np.float64:
                    return None
    except Exception:
        return None
    return _jax_x64_scope()


@functools.lru_cache(maxsize=None)
def _jax_tile_fn():
    """Build the jitted per-tile BMU search (compiled once per tile shape and dtype)."""
    import jax
    import jax.numpy as jnp

    def tile(weights, block, zero):
        n_features = weights.shape[1]
        # Same arithmetic as _bmu_kernel: direct differences and a sequential sum
        # over features, written out so XLA cannot reassociate it. XLA's CPU
        # backend contracts ``acc + d * d`` into a fused multiply-add, which
        # rounds once instead of twice. Adding a runtime zero to every product
        # leaves the contraction only ``d * d + 0``, which rounds to d * d.
        d = block[:, None, 0] - weights[None, :, 0]
        acc = d * d + zero
        for f in range(1, n_features):
            d = block[:, None, f] - weights[None, :, f]
            acc = acc + (d * d + zero)
        # _bmu_kernel never accepts a NaN distance; inf has the same effect here
        acc = jnp.where(jnp.isnan(acc), jnp.inf, acc)
        # argmin returns the first minimum, so ties go to the lowest node. A
        # single variadic (value, index) reduce would do both in one pass, but
        # jax-metal cannot lower it.
        best_idx = jnp.argmin(acc, axis=1).astype(jnp.int32)
        best = jnp.min(acc, axis=1)
        return best_idx, best

    return jax.jit(tile)


def best_match_nodes_jax(
    weights: np.ndarray,
    data: np.ndarray,
    chunk: int | None = None,
    precision: str = "highest",
    devices: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the best-matching node of every point with JAX, tile by tile.

    The points are cut into tiles of ``chunk`` rows (the last tile is padded, so
    one compiled shape serves the whole call). Each tile is copied to a device,
    searched there, and its results are copied into preallocated NumPy outputs;
    the device holds a ``chunk x nodes`` distance block at most, never one for
    the whole dataset. Tiles are dispatched asynchronously and, when several
    devices are given, round-robin over them (tile ``k`` on device
    ``k % len(devices)``), so copies and compute of consecutive tiles overlap.

    Arithmetic: squared distances are accumulated by direct differences,
    ``((x - w) ** 2)`` summed over features in ascending order, never through
    the ``||w||^2 - 2 x.w`` expansion. The node is chosen with ``jnp.argmin``,
    which returns the first (lowest) index among equal minima, like
    ``np.argmin``, so exact ties go to the lowest node as in
    :func:`best_match_nodes`. NaN distances are treated as infinite, which is
    what the strict ``<`` scan of the numba kernel does with them.

    * ``precision="highest"``: float64 when every device supports it (CPU,
      CUDA, ROCm; enabled through a scoped ``enable_x64``, the global JAX
      config is not changed). Labels and distances then equal
      :func:`best_match_nodes` bitwise, including float32 input, which is
      promoted to float64 on the host, with two qualifications: XLA flushes
      subnormal values to zero, so a squared distance below about 2e-308 is
      0 on the device (such a near-tie can pick a different node than the
      numba scan), and the guarantee holds for default XLA flags only
      (``--xla_cpu_enable_fast_math=true`` lets XLA reassociate the sum).
      If a device has no float64 (Apple Metal), a ``RuntimeWarning`` is
      issued and the float32 path is used.
    * ``precision="float32"``: weights and points are rounded to float32 and
      the same direct arithmetic runs in float32. The relative rounding error
      of a distance is about (F + 2) * 6e-8, so a point whose two nearest
      nodes are closer than that, or exactly tied after rounding, can get the
      other node. On random data (2368 nodes, F = 4, 2e6 points) no label
      changed, on CPU or Metal.

    Args:
        weights (np.ndarray): [nodes, F] lattice weights, node index = x + y * xdim.
        data (np.ndarray): [N, F] float64 or float32 points in the scaled space.
        chunk (int | None, optional): Points per tile. Defaults to
            ``2**25 // nodes`` (about 14,000 rows for 2368 nodes, so the
            distance block of one tile is at most 256 MB in float64).
        precision (str, optional): ``"highest"`` or ``"float32"``. Defaults to
            ``"highest"``.
        devices (list | None, optional): JAX devices to use. Defaults to
            ``jax.local_devices()``.

    Returns:
        tuple[np.ndarray, np.ndarray]: int32 [N] node indices and [N] squared
            Euclidean distances, float64 on the float64 path and float32 on the
            float32 path.
    """
    import jax

    weights, data = _check_inputs(weights, data)
    if precision not in ("highest", "float32"):
        raise ValueError('precision must be "highest" or "float32"')
    n_points, n_features = data.shape
    if chunk is None:
        chunk = max(1, JAX_TILE_ELEMENTS // weights.shape[0])
    if chunk < 1:
        raise ValueError("chunk must be positive")
    chunk = min(chunk, max(1, n_points))
    devices = list(jax.local_devices() if devices is None else devices)
    if not devices:
        raise ValueError("devices must contain at least one JAX device")

    scope = _jax_float64_scope(devices) if precision == "highest" else None
    if precision == "highest" and scope is None:
        warnings.warn(
            "best_match_nodes_jax: float64 is not available on "
            f"{devices[0].platform}; using float32 direct arithmetic",
            RuntimeWarning,
            stacklevel=2,
        )
    dtype = np.float64 if scope is not None else np.float32

    nodes_idx = np.empty(n_points, dtype=np.int32)
    dist2 = np.empty(n_points, dtype=dtype)
    if n_points == 0:
        return nodes_idx, dist2

    tile_fn = _jax_tile_fn()
    w_host = weights.astype(dtype)
    pad = np.zeros((chunk, n_features), dtype=dtype)
    with scope if scope is not None else contextlib.nullcontext():
        w_dev = [jax.device_put(w_host, d) for d in devices]
        zero_dev = [jax.device_put(np.zeros((), dtype=dtype), d) for d in devices]
        pending = []  # (start, stop, idx, dist) still in flight
        for k, start in enumerate(range(0, n_points, chunk)):
            stop = min(start + chunk, n_points)
            block = data[start:stop].astype(dtype)
            if stop - start < chunk:
                pad[: stop - start] = block
                pad[stop - start :] = 0
                block = pad
            dev = k % len(devices)
            out = tile_fn(w_dev[dev], jax.device_put(block, devices[dev]), zero_dev[dev])
            pending.append((start, stop, *out))
            if len(pending) > len(devices):
                _collect(pending.pop(0), nodes_idx, dist2)
        for item in pending:
            _collect(item, nodes_idx, dist2)
    return nodes_idx, dist2


def _collect(item, nodes_idx: np.ndarray, dist2: np.ndarray) -> None:
    start, stop, idx, d2 = item
    nodes_idx[start:stop] = np.asarray(idx)[: stop - start]
    dist2[start:stop] = np.asarray(d2)[: stop - start]


@njit(parallel=True, cache=True)
def _dist2_kernel(
    weights: np.ndarray, data: np.ndarray, idx: np.ndarray, out_dist2: np.ndarray
) -> None:
    """Exact squared distance of every point to its given node (same arithmetic as _bmu_kernel)."""
    n_features = weights.shape[1]
    for p in prange(data.shape[0]):
        i = idx[p]
        acc = 0.0
        for f in range(n_features):
            d = weights[i, f] - np.float64(data[p, f])
            acc += d * d
        out_dist2[p] = acc


def best_match_nodes_kdtree(
    weights: np.ndarray,
    data: np.ndarray,
    chunk: int = DEFAULT_CHUNK,
    workers: int = -1,
    leafsize: int = KDTREE_LEAFSIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the best-matching node of every point with a KD-tree over the nodes.

    A ``scipy.spatial.cKDTree`` is built on the lattice weights and queried
    ``chunk`` points at a time with ``workers`` threads. The squared distance
    of each point to its node is then recomputed with the exact float64 scan of
    :func:`best_match_nodes`, so the distance of the chosen node is bitwise
    equal to that function. The labels are equal except for ties and near-ties:
    the tree compares distances computed with its own rounding, so two nodes
    whose exact distances differ by about one float64 ulp, or are equal, can
    be resolved the other way. Weights and points must be finite (the tree
    rejects NaN); the brute-force scan is used for any point the tree cannot
    place (squared distances overflowing float64). Faster than the brute-force
    scan for a few features (about F <= 8 on 2368 nodes), slower above.

    Args:
        weights (np.ndarray): [nodes, F] lattice weights, node index = x + y * xdim.
        data (np.ndarray): [N, F] float64 or float32 points in the scaled space.
        chunk (int, optional): Points per query. Defaults to 2**20.
        workers (int, optional): Threads for the query, -1 for all cores. Defaults to -1.
        leafsize (int, optional): ``cKDTree`` leaf size. Defaults to 64 (scipy's
            16 is about 1.35x slower on 2368 nodes at F = 4 to 10).

    Returns:
        tuple[np.ndarray, np.ndarray]: int32 [N] node indices and float64 [N]
            squared Euclidean distances to those nodes.
    """
    from scipy.spatial import cKDTree

    weights, data = _check_inputs(weights, data)
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite for the kdtree backend")
    tree = cKDTree(weights, leafsize=leafsize)
    n_points = data.shape[0]
    nodes_idx = np.empty(n_points, dtype=np.int32)
    dist2 = np.empty(n_points, dtype=np.float64)
    for start in range(0, n_points, chunk):
        stop = min(start + chunk, n_points)
        block = np.ascontiguousarray(data[start:stop])
        _, idx = tree.query(block, k=1, workers=workers)
        missing = idx >= weights.shape[0]  # cKDTree's "no neighbour" sentinel
        if missing.any():
            idx[missing], _ = best_match_nodes(weights, block[missing])
        nodes_idx[start:stop] = idx
        _dist2_kernel(weights, block, nodes_idx[start:stop], dist2[start:stop])
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
    (CPU, float32 screen certified in float64; labels and distances bitwise
    equal to ``"numba"`` and the fastest CPU path measured).
    ``backend="numba"`` runs :func:`best_match_nodes` on the CPU, the exact
    reference scan that ``cpp/awesom_classify.hpp`` reproduces.
    ``backend="kdtree"`` runs :func:`best_match_nodes_kdtree` (CPU; ties may
    go to a higher node; points must be finite).
    ``backend="jax"`` runs :func:`best_match_nodes_jax` with
    ``precision="highest"`` on the default JAX devices: identical labels where
    the devices support float64, float32 arithmetic (with a warning) where they
    do not.

    Args:
        weights (np.ndarray): [nodes, F] lattice weights.
        node_cluster (np.ndarray): [nodes] cluster id of each node (-1 = unassigned).
        data (np.ndarray): [N, F] float64 or float32 points in the scaled space.
        chunk (int | None, optional): Number of points per kernel call (numba) or
            per tile (jax). Defaults to None: 2**20 on the CPU, ``2**25 // nodes``
            for jax.
        return_distance (bool, optional): Also return the Euclidean BMU distance.
            Defaults to False.
        backend (str, optional): ``"screen"``, ``"numba"``, ``"kdtree"`` or
            ``"jax"``. Defaults to ``"screen"``.

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]: int32 [N] labels, and [N]
            distances if ``return_distance`` is True (float64, or float32 from
            the jax backend without float64).
    """
    node_cluster = np.asarray(node_cluster, dtype=np.int32).ravel()
    if node_cluster.shape[0] != np.shape(weights)[0]:
        raise ValueError("node_cluster must have one entry per node")
    if backend == "numba":
        nodes_idx, dist2 = best_match_nodes(
            weights, data, chunk=DEFAULT_CHUNK if chunk is None else chunk
        )
    elif backend == "kdtree":
        nodes_idx, dist2 = best_match_nodes_kdtree(
            weights, data, chunk=DEFAULT_CHUNK if chunk is None else chunk
        )
    elif backend == "screen":
        nodes_idx, dist2 = best_match_nodes_screen(
            weights, data, chunk=DEFAULT_CHUNK if chunk is None else chunk
        )
    elif backend == "jax":
        nodes_idx, dist2 = best_match_nodes_jax(weights, data, chunk=chunk)
    else:
        raise ValueError('backend must be "numba", "kdtree", "screen" or "jax"')
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
