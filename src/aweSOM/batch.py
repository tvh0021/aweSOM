"""Multi-device batch SOM trainer built on JAX.

The batch map (Kohonen; Fort, Letremy & Cottrell, ESANN 2002) replaces the
online per-sample update with one fixed-point step per epoch:

    w_i = sum_j h(c_j, i) x_j / sum_j h(c_j, i)

where c_j is the best-matching unit (BMU) of sample x_j under the weights at the
start of the epoch and h is the neighbourhood function of ``Lattice.Gamma`` with
alpha = 1 (alpha cancels in the ratio). The update is factorised through per-node
sums: S_c = sum_{j: c_j = c} x_j and n_c = #{j: c_j = c}, so that num = H^T S and
den = H^T n with H[c, i] = h(c, i). Each device accumulates S and n over its rows
(``jax.ops.segment_sum`` inside a ``lax.scan`` over chunks), the partial sums are
combined with ``lax.psum``, and every device then applies the same update, so the
weights stay replicated. Nodes with den = 0 (no hit within the radius) keep their
weight, which is what the online rule does when all increments are zero.

Equivalence with the online trainer is statistical, not bitwise: the two
algorithms share stationary points but follow different trajectories. Use
:func:`quantization_error` and :func:`topographic_error` to compare maps.

Distances in the training pass use the expansion ||w||^2 - 2 x.w (||x||^2 is
dropped because it does not change the argmin) in the compute dtype (float32 by
default). In float32 this flips a BMU for roughly 1e-5 of the samples relative
to exact float64 arithmetic; :func:`quantization_error` and
:func:`topographic_error` use exact float64 direct arithmetic. The per-node
sums, counts, update and master weights are always float64, which makes the
result independent of the device count in practice. With ``dtype=np.float64``
one epoch matches a NumPy float64 reference to ~1e-15.

Multi-host / multi-GPU usage:
    The caller initialises the distributed runtime; this module never calls
    ``jax.distributed.initialize()``. With one process per GPU::

        import jax
        jax.distributed.initialize()          # auto-detects SLURM
        from aweSOM.batch import train_batch
        local = load_my_rows(jax.process_index(), jax.process_count())
        weights, info = train_batch(local, xdim, ydim)

    When ``jax.process_count() > 1``, ``data`` is the *process-local* slice of
    the training set. Each process pads its slice to a common length, and the
    global array is assembled with ``jax.make_array_from_process_local_data``.
    Every process must call :func:`train_batch` with the same arguments other
    than ``data``. The global row order is process-major (process 0's rows,
    then process 1's, ...), and the sampling initialisation draws rows from
    that global order, so a multi-process run starts from the same lattice as a
    single-process run on the concatenated data. The multi-process path is
    tested with two CPU processes on one machine (gloo collectives, uneven
    local row counts; weights equal a single-process run) but has not been
    run on GPUs or across nodes; see
    ``examples/slurm_scripts/submit_batch_gpu.slurm``.

Determinism:
    On GPU, scatter-add (``segment_sum``) uses atomics, so results can differ
    in the last bits between runs. Set
    ``XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true`` for bitwise
    reproducibility. ``precision="highest"`` prevents TF32 in the distance
    matmul.

Measured on Apple M1 Max, CPU (JAX 0.12.0.1, float32, N = 2,000,000, F = 6,
64 x 37 lattice, median epoch after compilation; compilation adds ~0.5 s):
    chunk 65536, 1 device:                  3.73 s per epoch, 5.4e5 points/s
    chunk 65536, 4 virtual CPU devices:     3.63 s per epoch, 5.5e5 points/s
    chunk 16384, 4 virtual CPU devices:     3.04 s per epoch, 6.6e5 points/s
    Virtual devices share the same cores (XLA already multithreads one CPU
    device), so they test sharding, not speed. The distance matmul + argmin
    is ~95% of the time; float64 accumulation costs the same as float32.

Example:
    >>> from aweSOM.batch import train_batch, quantization_error
    >>> weights, info = train_batch(data, 64, 37, epochs=40)
    >>> qe = quantization_error(weights, data)
"""

from __future__ import annotations

import functools
import math
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import NamedSharding, PartitionSpec as P
except ImportError as exc:  # pragma: no cover - exercised only without jax
    raise ImportError(
        "aweSOM.batch requires jax; pip install 'aweSOM[gpu]' or jax"
    ) from exc

if not hasattr(jax, "enable_x64") or not hasattr(jax, "shard_map"):
    raise ImportError(
        f"aweSOM.batch needs a recent jax with jax.enable_x64 and jax.shard_map (tested with 0.12); found jax {jax.__version__}. "
        "The Apple Metal plugin (jax-metal 0.1.1, jax 0.4.34) is too old and has no float64, "
        "so the batch trainer cannot run on Metal; use the CPU or a CUDA/ROCm build."
    )

__all__ = [
    "radius_schedule",
    "neighbourhood_matrix",
    "train_batch",
    "quantization_error",
    "topographic_error",
]

_AXIS = "data"
_PRECISIONS = {
    "highest": lax.Precision.HIGHEST,
    "high": lax.Precision.HIGH,
    "default": lax.Precision.DEFAULT,
}


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


@functools.lru_cache(maxsize=16)
def _make_epoch_step(mesh: Any, chunk: int, precision: Any) -> Callable:
    """Build (and cache) the jitted, sharded epoch step for a mesh and chunk size.

    The step maps (x, valid, w, h) -> (w_new, qe_sum, hits). ``x`` and ``valid``
    are sharded along the mesh axis, ``w`` (float64) and ``h`` (float64) are
    replicated, and all outputs are replicated. Distances are computed in the
    dtype of ``x``; sums, counts and the update are float64, so the result does
    not depend on the summation order (device count, chunking) beyond float64
    rounding. Must be traced and called with x64 enabled.
    """

    def local_step(x, valid, w, h):
        n_local, n_feat = x.shape
        n_nodes = w.shape[0]
        xs = x.reshape(n_local // chunk, chunk, n_feat)
        vs = valid.reshape(n_local // chunk, chunk)
        w_c = w.astype(x.dtype)
        w_sq = jnp.sum(w_c * w_c, axis=1)

        def body(carry, inputs):
            s_acc, n_acc, qe_acc = carry
            xc, vc = inputs
            d = w_sq[None, :] - 2.0 * jnp.dot(xc, w_c.T, precision=precision)
            bmu = jnp.argmin(d, axis=1)
            d_min = jnp.take_along_axis(d, bmu[:, None], axis=1)[:, 0]
            # ||x||^2 is only needed for the QE diagnostic, not for the argmin
            dist = jnp.sqrt(jnp.maximum(d_min + jnp.sum(xc * xc, axis=1), 0.0))
            v64 = vc.astype(jnp.float64)
            qe_acc = qe_acc + jnp.sum(v64 * dist.astype(jnp.float64))
            s_acc = s_acc + jax.ops.segment_sum(
                xc.astype(jnp.float64) * v64[:, None], bmu, num_segments=n_nodes
            )
            n_acc = n_acc + jax.ops.segment_sum(v64, bmu, num_segments=n_nodes)
            return (s_acc, n_acc, qe_acc), None

        init = (
            jnp.zeros((n_nodes, n_feat), jnp.float64),
            jnp.zeros((n_nodes,), jnp.float64),
            jnp.zeros((), jnp.float64),
        )
        # scan carries must be device-varying to accept per-device updates
        init = lax.pcast(init, _AXIS, to="varying")
        (s, n, qe), _ = lax.scan(body, init, (xs, vs))
        s = lax.psum(s, _AXIS)
        n = lax.psum(n, _AXIS)
        qe = lax.psum(qe, _AXIS)

        num = jnp.dot(h.T, s, precision=lax.Precision.HIGHEST)
        den = jnp.dot(h.T, n, precision=lax.Precision.HIGHEST)
        has_hits = den > 0
        safe_den = jnp.where(has_hits, den, 1.0)
        w_new = jnp.where(has_hits[:, None], num / safe_den[:, None], w)
        return w_new, qe, n

    sharded = jax.shard_map(
        local_step,
        mesh=mesh,
        in_specs=(P(_AXIS), P(_AXIS), P(), P()),
        out_specs=(P(), P(), P()),
    )
    return jax.jit(sharded)


def _allgather(x: np.ndarray) -> np.ndarray:
    """Stack ``x`` from every process along a new leading axis."""
    if jax.process_count() == 1:
        return np.asarray(x)[None]
    from jax.experimental import multihost_utils

    return np.asarray(multihost_utils.process_allgather(x, tiled=False))


def _initial_weights(
    local: np.ndarray,
    counts: np.ndarray,
    n_nodes: int,
    init: str,
    seed: int,
) -> np.ndarray:
    """Initial lattice drawn with ``np.random.default_rng(seed)`` like ``Lattice.fast_som``.

    ``"sampling"`` picks ``n_nodes`` random rows (with replacement) of the global
    data set, in process-major row order; ``"uniform"`` draws from U(0, 1).
    """
    rng = np.random.default_rng(seed)
    n_feat = local.shape[1]
    if init == "uniform":
        return rng.uniform(0.0, 1.0, (n_nodes, n_feat))
    if init != "sampling":
        raise ValueError(f"init must be 'sampling' or 'uniform', got {init!r}")
    ix = rng.integers(0, int(counts.sum()), n_nodes)
    if jax.process_count() == 1:
        return np.asarray(local[ix], dtype=np.float64)
    offset = int(counts[: jax.process_index()].sum())
    mine = (ix >= offset) & (ix < offset + local.shape[0])
    contrib = np.zeros((n_nodes, n_feat), np.float64)
    contrib[mine] = local[ix[mine] - offset]
    return _allgather(contrib).sum(axis=0)


def train_batch(
    data: np.ndarray,
    xdim: int,
    ydim: int,
    epochs: int = 40,
    r_max: Optional[float] = None,
    r_min: float = 8,
    chunk: int = 65536,
    seed: int = 42,
    init: str = "sampling",
    initial_weights: Optional[np.ndarray] = None,
    mesh: Any = None,
    devices: Optional[Sequence[Any]] = None,
    precision: str = "highest",
    dtype: Any = np.float32,
    verbose: bool = False,
    callback: Optional[Callable[[int, np.ndarray], None]] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train a SOM lattice with the batch map, data-parallel over a 1-D device mesh.

    Each epoch e with radius r_e computes the BMU of every sample under the
    current weights, accumulates per-node sums and hit counts on each device,
    combines them with ``psum``, and sets ``W = H(r_e)^T S / H(r_e)^T n`` (nodes
    with zero denominator keep their weight).

    Precision: samples and the distance pass use ``dtype``; the per-node sums,
    the counts, the update and the master copy of the weights are float64
    (inside a ``jax.enable_x64(True)`` scope, so the global JAX config is not
    changed). With float32 sums the summation order (device count, psum
    tree) changes the weights in the last bit, and near-tied BMUs amplify
    that to ~1e-4 within ten epochs; with float64 sums 1-, 2- and 4-device
    runs gave bitwise-identical weights in the tests. The extra cost is N * F
    float64 additions per epoch, small next to the N * nodes * F distance pass.

    Args:
        data: Training samples ``(N, F)``. In a multi-process run
            (``jax.process_count() > 1``) the process-local rows.
        xdim: Lattice x dimension.
        ydim: Lattice y dimension.
        epochs: Number of batch epochs.
        r_max: Initial radius. Defaults to ``max(xdim, ydim) + 1`` as in
            ``Lattice.fast_som``.
        r_min: Final radius; clamped to ``r_max``.
        chunk: Rows per device processed at once; the distance block is
            ``chunk x nodes``. Reduced automatically (to a multiple of 8) when a
            device holds fewer rows.
        seed: Seed of ``np.random.default_rng`` for the initial lattice.
        init: ``"sampling"`` (random data rows, as the online trainer) or
            ``"uniform"`` (U(0, 1)). Ignored if ``initial_weights`` is given.
        initial_weights: Optional starting lattice ``(xdim * ydim, F)``; must be
            identical on every process.
        mesh: Optional 1-D ``jax.sharding.Mesh`` with axis name ``"data"``.
        devices: Devices for the default mesh when ``mesh`` is None; defaults
            to ``jax.devices()``.
        precision: Matmul precision of the distance computation: ``"highest"``
            (no TF32), ``"high"`` or ``"default"``.
        dtype: Dtype of the samples and the distance computation
            (``np.float32`` or ``np.float64``).
        verbose: Print one line per epoch.
        callback: Called after every epoch as ``callback(epoch, weights)`` with
            the float64 weights after that epoch (epoch counted from 0).

    Returns:
        A tuple ``(weights, info)``. ``weights`` is float64 ``(xdim * ydim, F)``.
        ``info`` has keys ``radius`` (per epoch), ``qe`` (mean quantization error
        after each epoch; ``qe[e]`` is measured during the pass of epoch e + 1,
        the last one by an extra pass), ``qe_initial`` (of the initial lattice),
        ``hits`` (BMU counts of the returned weights), ``epoch_time`` (seconds
        per epoch; the first includes compilation), ``n_devices``, ``chunk``
        (effective rows per scan step) and ``n_samples``. The QE values use the
        ``||w||^2 - 2 x.w + ||x||^2`` expansion in ``dtype`` and are diagnostics;
        use :func:`quantization_error` for exact values.

    Raises:
        ValueError: On inconsistent shapes or arguments.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"dtype must be float32 or float64, got {dtype}")
    if precision not in _PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(_PRECISIONS)}, got {precision!r}")
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (N, F), got shape {data.shape}")
    n_nodes = xdim * ydim
    n_feat = data.shape[1]

    if mesh is None:
        devices = list(devices) if devices is not None else jax.devices()
        mesh = jax.make_mesh((len(devices),), (_AXIS,), devices=devices)
    if tuple(mesh.axis_names) != (_AXIS,):
        raise ValueError(f"mesh must be 1-D with axis name {_AXIS!r}, got {mesh.axis_names}")
    n_dev_local = len(mesh.local_devices)

    # every process pads its rows to the same length, a multiple of local devices * chunk;
    # padded rows have valid = 0 and do not contribute to sums, counts or QE
    counts = _allgather(np.array([data.shape[0]], np.int64))[:, 0]
    n_total = int(counts.sum())
    if n_total == 0:
        raise ValueError("data is empty")
    rows_per_dev = math.ceil(int(counts.max()) / n_dev_local)
    chunk = max(1, min(int(chunk), 8 * math.ceil(rows_per_dev / 8)))
    local_len = n_dev_local * chunk * math.ceil(rows_per_dev / chunk)

    if initial_weights is None:
        w0 = _initial_weights(data, counts, n_nodes, init, seed)
    else:
        w0 = np.array(initial_weights, np.float64)
        if w0.shape != (n_nodes, n_feat):
            raise ValueError(
                f"initial_weights must have shape {(n_nodes, n_feat)}, got {w0.shape}"
            )

    if r_max is None:
        r_max = max(xdim, ydim) + 1
    radii = radius_schedule(epochs, r_max, min(r_min, r_max))

    qe_after = np.zeros(epochs)
    times = np.zeros(epochs)
    with jax.enable_x64(True):
        x_local = np.zeros((local_len, n_feat), dtype)
        x_local[: data.shape[0]] = data
        v_local = np.zeros(local_len, dtype)
        v_local[: data.shape[0]] = 1.0
        sharding = NamedSharding(mesh, P(_AXIS))
        global_len = local_len * jax.process_count()
        x = jax.make_array_from_process_local_data(sharding, x_local, (global_len, n_feat))
        valid = jax.make_array_from_process_local_data(sharding, v_local, (global_len,))
        del x_local, v_local

        replicated = NamedSharding(mesh, P())
        step = _make_epoch_step(mesh, chunk, _PRECISIONS[precision])
        w = jax.device_put(w0, replicated)

        for e, r in enumerate(radii):
            t0 = time.perf_counter()
            h = jax.device_put(_neighbourhood64(xdim, ydim, r), replicated)
            w, qe_sum, _ = step(x, valid, w, h)
            # the pass of epoch e measures the weights that entered it
            qe_entering = float(qe_sum) / n_total
            times[e] = time.perf_counter() - t0
            if e > 0:
                qe_after[e - 1] = qe_entering
            else:
                qe_initial = qe_entering
            if verbose:
                print(
                    f"epoch {e + 1}/{epochs}  radius {r:.3f}  "
                    f"qe(entering) {qe_entering:.6g}  {times[e]:.3f} s",
                    flush=True,
                )
            if callback is not None:
                callback(e, np.asarray(w, np.float64))

        # one extra pass for the QE and hit counts of the final weights
        h = jax.device_put(np.eye(n_nodes), replicated)
        _, qe_sum, hits = step(x, valid, w, h)
        qe_after[-1] = float(qe_sum) / n_total
        weights = np.asarray(w, np.float64)
        hits = np.asarray(hits).round().astype(np.int64)

    info = {
        "radius": radii,
        "qe": qe_after,
        "qe_initial": qe_initial,
        "hits": hits,
        "epoch_time": times,
        "n_devices": mesh.size,
        "chunk": chunk,
        "n_samples": n_total,
    }
    return weights, info


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
