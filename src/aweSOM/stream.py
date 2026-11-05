"""Streaming CPU batch SOM trainer for data sets larger than memory.

:func:`train_batch_stream` runs the batch map of :func:`aweSOM.batch.train_batch`
(same radius schedule, same neighbourhood, same update
``w_i = sum_c h(c, i) S_c / sum_c h(c, i) n_c``, nodes with no hit keep their
weight) without JAX, reading the data one chunk at a time from a NumPy array, a
``numpy.memmap``, an ``h5py`` dataset, or a callable that yields chunks. Only
one chunk (plus one being prefetched) and the per-node accumulators are in
memory, so N is limited by the disk, not by RAM.

Per epoch and chunk:

1. Exact BMU search. ``bmu="screen"`` (default) is the float32 screen of
   :func:`aweSOM.inference.best_match_nodes_screen`, certified against the
   float64 reference, so every BMU equals the exact float64 BMU of
   :func:`aweSOM.inference.best_match_nodes` (ties to the lowest node).
   ``bmu="numba"`` is the float64 brute-force scan itself.
2. Accumulation of ``S_c = sum x`` and ``n_c`` in float64. The chunk is cut
   into ``groups`` contiguous row ranges, each range accumulates into its own
   slot, and the slots are summed in a fixed order at the end of the epoch, so
   the weights are bitwise independent of the thread count.
3. After the last chunk, the neighbourhood update: for every node, the sum over
   the nodes within the radius (Chebyshev distance on the lattice) is done
   directly in numba with a 1-D table of ``h(d)``; the dense ``nodes x nodes``
   matrix of :func:`aweSOM.batch.neighbourhood_matrix` is never formed (it is
   2 GB in float64 at 16,000 nodes).

Relation to :func:`aweSOM.batch.train_batch`: same initial lattice for the same
``seed`` (``init="sampling"`` draws the same rows), same schedule and update.
``train_batch`` finds BMUs with the float32 matmul expansion, which flips the
BMU of about 1e-5 of the samples against exact arithmetic; with
``train_batch(..., dtype=np.float64)`` the two agree to float64 rounding (see
``tests/stream_test.py``). :func:`aweSOM.batch.train_batch_kdtree` with
``bmu="numba"`` uses the same exact BMUs.

Per-epoch subsampling (``sample_fraction < 1``): each epoch uses a fresh,
stratified random subset of the rows. The row range is split into blocks of
``sample_block`` rows, the blocks into ``k`` equal strata, and one block is
drawn uniformly from each stratum (systematic stratified sampling), so every
part of the file (every snapshot or region, if the rows are stored in that
order) is represented in every epoch, and the reads are contiguous. The
update is then a ratio estimator of the full-data update. For node ``i`` the
weight is a weighted mean over ``n_eff,i`` samples (Kish effective sample size
of the neighbourhood weights) and its standard error is about
``sigma_i / sqrt(n_eff,i)``, with ``sigma_i`` the spread of the data around the
node (of the order of the quantization error). Block sampling of correlated
rows multiplies the variance by the design effect ``1 + (b - 1) rho``
(``b`` = rows per block, ``rho`` = intra-block correlation of the features).
With 10^8 rows per epoch and 10^4 nodes there are ~10^4 rows per node and, at
the final radius 8 (about 45 nodes of effective neighbourhood), ``n_eff ~
4e5``: the sampling noise of a weight is ~1.5e-3 of the local spread, far
below the quantization error. ``planning/benchmarks/bench_stream.py`` measures
the effect on QE and TE.

Sources and memory: ``data_source`` is anything with a 2-D ``shape`` that can
be sliced by rows (``ndarray``, ``memmap``, ``h5py.Dataset``), or a
zero-argument callable that returns a fresh iterable of ``(rows, F)`` chunks
on every call (the trainer calls it once per pass). Float32 and float64 data
are used as stored (no up-cast copy of the chunk); other dtypes are converted
to float64 chunk by chunk.

Example:
    >>> import numpy as np
    >>> from aweSOM.stream import train_batch_stream
    >>> x = np.load("features_f32.npy", mmap_mode="r")      # (N, F) float32
    >>> weights, info = train_batch_stream(x, 91, 64, epochs=40, sample_fraction=0.01)
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import resource
import sys
import time
from typing import Any, Callable, Iterable, Iterator, Optional

import numba
import numpy as np
from numba import njit, prange

from ._batch_core import radius_schedule
from .inference import (
    SCREEN_TILE,
    _bmu_kernel,
    _feature_patch_kernel,
    _feature_patches,
    _patch_kernel,
    _screen_kernel,
    lattice_patches,
    predict,
)

__all__ = [
    "NpyReader",
    "train_batch_stream",
    "read_checkpoint",
    "predict_stream",
    "stratified_sample",
    "neighbourhood_update",
    "peak_rss_bytes",
]

DEFAULT_CHUNK = 2**20
DEFAULT_GROUPS = 64


# --------------------------------------------------------------------- sources


def _is_h5(obj: Any) -> bool:
    return type(obj).__module__.startswith("h5py")


def _as_float(block: np.ndarray) -> np.ndarray:
    block = np.asarray(block)
    if block.dtype not in (np.float32, np.float64):
        block = block.astype(np.float64)
    return np.ascontiguousarray(block)


class _ArraySource:
    """Row-sliceable 2-D source (ndarray, memmap, h5py.Dataset)."""

    def __init__(self, data: Any):
        shape = tuple(data.shape)
        if len(shape) != 2:
            raise ValueError(f"data must be 2-D (N, F), got shape {shape}")
        self.data = data
        self.n, self.n_feat = int(shape[0]), int(shape[1])
        self.h5 = _is_h5(data)

    def read(self, start: int, stop: int) -> np.ndarray:
        return _as_float(self.data[start:stop])

    def rows(self, ix: np.ndarray) -> np.ndarray:
        """Rows ``ix`` (any order, repeats allowed), read in ascending order."""
        uniq, inv = np.unique(ix, return_inverse=True)
        if self.h5:
            got = np.asarray(self.data[uniq.tolist()]) if len(uniq) else np.empty((0, self.n_feat))
        else:
            got = np.asarray(self.data[uniq])
        return got[inv]

    def read_blocks(self, starts: np.ndarray, block: int) -> np.ndarray:
        """Concatenation of the row ranges ``[s, min(s + block, n))`` for ascending ``starts``."""
        if self.h5 or block >= 4096:
            parts = [self.data[s : min(s + block, self.n)] for s in starts.tolist()]
            return _as_float(np.concatenate(parts) if len(parts) > 1 else parts[0])
        rows = (starts[:, None] + np.arange(block)[None, :]).ravel()
        rows = rows[rows < self.n]
        return _as_float(np.take(self.data, rows, axis=0))


class _CallableSource:
    """Zero-argument callable returning a fresh iterable of (rows, F) chunks per call.

    Optional attributes of the callable, used when present:

    - ``n_feat`` or ``n_features``: number of columns. With the row count known
      as well (``n_samples``), no counting pass is made.
    - ``rows(ix)``: rows ``ix`` of the callable's row order (the order of the
      next call) without changing its state; used for the sampled initial
      lattice instead of a pass over the data.
    - ``lazy_chunks()``: one call (same state change as ``fn()``) yielding
      ``(rows, read)`` pairs, where ``read()`` returns the chunk. A rank of a
      multi-process run reads only the chunks it owns.
    - ``calls``: integer call counter; saved in checkpoints and restored on
      resume.
    """

    def __init__(
        self,
        fn: Callable[[], Iterable[np.ndarray]],
        n_samples: Optional[int],
        n_features: Optional[int] = None,
    ):
        self.fn = fn
        self.n = n_samples
        self.n_feat = n_features
        if self.n_feat is None:
            for name in ("n_feat", "n_features"):
                value = getattr(fn, name, None)
                if value is not None:
                    self.n_feat = int(value)
                    break
        if self.n is None or self.n_feat is None:
            n = 0
            for block in fn():
                block = np.asarray(block)
                if block.ndim != 2:
                    raise ValueError("chunks must be 2-D (rows, F)")
                if self.n_feat is not None and block.shape[1] != self.n_feat:
                    raise ValueError(f"n_features={self.n_feat} but a chunk has {block.shape[1]} columns")
                self.n_feat = block.shape[1]
                n += block.shape[0]
            if self.n is not None and self.n != n:
                raise ValueError(f"n_samples={self.n} but the source yielded {n} rows")
            self.n = n

    def lazy(self) -> Iterator[tuple[int, Callable[[], np.ndarray]]]:
        """One call of the source as ``(rows, read)`` pairs; ``read()`` returns the float chunk."""
        lazy = getattr(self.fn, "lazy_chunks", None)
        if callable(lazy):
            for m, read in lazy():
                yield int(m), (lambda read=read: _as_float(read()))
            return
        for block in self.fn():
            block = np.asarray(block)
            yield block.shape[0], (lambda block=block: _as_float(block))

    def rows(self, ix: np.ndarray) -> np.ndarray:
        """Rows ``ix`` as float64: the callable's ``rows`` if it has one, else one
        pass that reads only chunks holding requested rows and stops after the last."""
        own = getattr(self.fn, "rows", None)
        if callable(own):
            return np.asarray(own(ix), np.float64)
        order = np.argsort(ix, kind="stable")
        sorted_ix = ix[order]
        out = np.empty((len(ix), self.n_feat), np.float64)
        offset = 0
        pos = 0
        for m, read in self.lazy():
            if pos == len(ix):
                break
            hi = np.searchsorted(sorted_ix, offset + m, side="left")
            if hi > pos:
                block = read()
                out[order[pos:hi]] = block[sorted_ix[pos:hi] - offset]
                pos = hi
            offset += m
        return out


class NpyReader:
    """Read rows of a 2-D C-order ``.npy`` file with plain file reads (no memory map).

    A ``numpy.memmap`` keeps the pages it touched resident (they count in the
    process RSS until the OS evicts them); this reader copies each requested
    row range into a fresh array, so the process holds only the current
    chunks, like an ``h5py`` dataset. Supports ``shape``, ``dtype`` and
    ``reader[start:stop]``; other indices go through a temporary memmap.
    """

    def __init__(self, path: str):
        import threading

        self.path = str(path)
        with open(self.path, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
            self.offset = fh.tell()
        if fortran or len(shape) != 2:
            raise ValueError("NpyReader needs a 2-D C-order array")
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.row_bytes = self.shape[1] * self.dtype.itemsize
        self._fh = open(self.path, "rb", buffering=0)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, slice) and key.step in (None, 1):
            start, stop, _ = key.indices(self.shape[0])
            out = np.empty((max(0, stop - start), self.shape[1]), self.dtype)
            view = memoryview(out).cast("B")
            with self._lock:
                self._fh.seek(self.offset + start * self.row_bytes)
                got = 0
                while got < len(view):
                    k = self._fh.readinto(view[got:])
                    if not k:
                        raise IOError(f"short read from {self.path}")
                    got += k
            return out
        return np.array(np.load(self.path, mmap_mode="r")[key])


def _make_source(data_source: Any, n_samples: Optional[int], n_features: Optional[int] = None):
    if callable(data_source) and not hasattr(data_source, "shape"):
        return _CallableSource(data_source, n_samples, n_features)
    return _ArraySource(data_source)


def _stratified_block_starts(n: int, block: int, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """One random block from each of k equal strata of the ceil(n / block) blocks; ascending starts."""
    nb = math.ceil(n / block)
    k = min(nb, max(1, int(round(fraction * nb))))
    edges = (np.arange(k + 1, dtype=np.int64) * nb) // k
    picks = edges[:-1] + (rng.random(k) * (edges[1:] - edges[:-1])).astype(np.int64)
    return picks * block


def _epoch_chunks(
    source,
    chunk: int,
    fraction: float,
    block: int,
    rng: Optional[np.random.Generator],
    rank: int = 0,
    size: int = 1,
) -> Iterator[np.ndarray]:
    """Chunks of one pass: all rows, or a stratified block sample (array sources),
    or a Bernoulli row sample (callable sources).

    With ``size > 1`` only the chunks with index ``i % size == rank`` are
    yielded (round-robin ownership). The chunk boundaries and the random draws
    do not depend on ``rank`` or ``size``, so the ranks together see exactly
    the rows of a single-process pass.
    """
    if isinstance(source, _CallableSource):
        for i, (m, read) in enumerate(source.lazy()):
            keep = rng.random(m) < fraction if fraction < 1.0 else None
            if i % size != rank:
                continue
            c = read()
            if keep is not None:
                c = np.ascontiguousarray(c[keep])
            if c.shape[0]:
                yield c
        return
    if fraction >= 1.0:
        for i, start in enumerate(range(0, source.n, chunk)):
            if i % size == rank:
                yield source.read(start, min(start + chunk, source.n))
        return
    starts = _stratified_block_starts(source.n, block, fraction, rng)
    per = max(1, chunk // block)
    for i, lo in enumerate(range(0, len(starts), per)):
        if i % size == rank:
            yield source.read_blocks(starts[lo : lo + per], block)


def _prefetched(it: Iterator[np.ndarray], stats: dict) -> Iterator[np.ndarray]:
    """Read the next chunk in a background thread while the caller computes.

    ``stats["io_wait"]`` accumulates the time the consumer waited for data.
    """
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(next, it, None)
        while True:
            t0 = time.perf_counter()
            block = nxt.result()
            stats["io_wait"] += time.perf_counter() - t0
            if block is None:
                return
            nxt = pool.submit(next, it, None)
            yield block


# --------------------------------------------------------------------- kernels


@njit(parallel=True, nogil=True, cache=True)
def _accumulate_groups(data, idx, d2, s_g, n_g, q_g):
    """Add the rows of a chunk to per-group float64 node sums.

    Group g owns rows [g * m // G, (g + 1) * m // G) of the chunk and slots
    ``s_g[g], n_g[g], q_g[g]``; the result does not depend on the threads.
    """
    n_groups = s_g.shape[0]
    m, n_feat = data.shape
    for g in prange(n_groups):
        lo = (g * m) // n_groups
        hi = ((g + 1) * m) // n_groups
        q = 0.0
        for p in range(lo, hi):
            c = idx[p]
            for f in range(n_feat):
                s_g[g, c, f] += np.float64(data[p, f])
            n_g[g, c] += 1.0
            q += np.sqrt(d2[p])
        q_g[g] += q


@njit(parallel=True, nogil=True, cache=True)
def _neighbourhood_update_kernel(s, n, w, xdim, ydim, table, reach, out):
    n_nodes, n_feat = s.shape
    for i in prange(n_nodes):
        ix = i % xdim
        iy = i // xdim
        num = np.zeros(n_feat)
        den = 0.0
        for cy in range(max(0, iy - reach), min(ydim, iy + reach + 1)):
            dy = abs(cy - iy)
            for cx in range(max(0, ix - reach), min(xdim, ix + reach + 1)):
                c = cx + cy * xdim
                if n[c] == 0.0:
                    continue  # S_c = 0 as well
                d = max(abs(cx - ix), dy)
                h = table[d]
                if h == 0.0:
                    continue
                den += h * n[c]
                for f in range(n_feat):
                    num[f] += h * s[c, f]
        if den > 0.0:
            for f in range(n_feat):
                out[i, f] = num[f] / den
        else:
            for f in range(n_feat):
                out[i, f] = w[i, f]


def _h_table(xdim: int, ydim: int, radius: float) -> tuple[np.ndarray, int]:
    """h(d) for Chebyshev distance d = 0..max(xdim, ydim) - 1, as in ``batch._neighbourhood64``."""
    d = np.arange(max(xdim, ydim), dtype=np.float64)
    if radius <= 0:
        return (d == 0).astype(np.float64), 0
    sigma = radius / 3.0
    h = np.exp(-(d**2) / (2.0 * sigma**2))
    h[d > radius] = 0.0
    return h, int(min(math.floor(radius), len(d) - 1))


def neighbourhood_update(
    s: np.ndarray, n: np.ndarray, w: np.ndarray, xdim: int, ydim: int, radius: float
) -> np.ndarray:
    """Batch-map update ``H^T S / H^T n`` without forming H (nodes with no hit keep ``w``).

    Equal to ``neighbourhood_matrix(xdim, ydim, radius)`` in float64 applied as
    in :func:`aweSOM.batch.train_batch`, up to float64 summation order.
    """
    table, reach = _h_table(xdim, ydim, radius)
    out = np.empty_like(w, dtype=np.float64)
    _neighbourhood_update_kernel(
        np.ascontiguousarray(s, np.float64),
        np.ascontiguousarray(n, np.float64),
        np.ascontiguousarray(w, np.float64),
        int(xdim),
        int(ydim),
        table,
        reach,
        out,
    )
    return out


class _BMU:
    """Exact BMU search of one chunk against fixed weights.

    Patch search uses lattice groups, except in the first training epoch
    (``first_epoch``): sampled initialization has no lattice locality, so
    feature-space groups with certified bounds
    (:func:`aweSOM.inference._feature_patch_kernel`) are used there. Both retain
    original node IDs and reference arithmetic, including the lowest-index tie
    rule.

    ``"auto"`` times the screen and patch searches on the first rows of the
    first chunk and keeps the faster one for the pass (both are exact, with
    bitwise-equal results, so the choice only affects speed).
    """

    PROBE = 16384

    def __init__(self, weights: np.ndarray, kind: str, xdim: int, ydim: int,
                 first_epoch: bool = False):
        self.w = np.ascontiguousarray(weights, np.float64)
        self.kind = kind
        self.feature = first_epoch
        if kind in ("screen", "patch", "auto") and (
            not np.all(np.isfinite(self.w)) or np.abs(self.w).max() > 1e15
        ):
            self.kind = "numba"
        if self.kind in ("screen", "auto"):
            self.w32 = self.w.astype(np.float32)
            self.wmax = float(np.sqrt((self.w * self.w).sum(axis=1)).max())
        if self.kind in ("patch", "auto"):
            if self.feature:
                self.patches = _feature_patches(self.w)
            else:
                self.patches = lattice_patches(self.w, xdim, ydim)

    def _run(self, kind: str, block, idx, d2) -> int:
        if kind == "screen":
            return int(_screen_kernel(self.w, self.w32, self.wmax, block, idx, d2, SCREEN_TILE))
        if kind == "patch":
            kernel = _feature_patch_kernel if self.feature else _patch_kernel
            kernel(*self.patches, block, idx, d2)
            return 0
        _bmu_kernel(self.w, block, idx, d2)
        return 0

    def __call__(self, block: np.ndarray, idx: np.ndarray, d2: np.ndarray) -> int:
        if self.kind == "auto":
            m = min(self.PROBE, block.shape[0])
            times = {}
            for kind in ("screen", "patch"):
                t0 = time.perf_counter()
                self._run(kind, block[:m], idx[:m], d2[:m])
                times[kind] = time.perf_counter() - t0
            self.kind = min(times, key=times.get)
        return self._run(self.kind, block, idx, d2)


def peak_rss_bytes() -> int:
    """Peak resident set size of this process (``ru_maxrss``; bytes on macOS, KiB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


# --------------------------------------------------------------------- trainer


def _pass(
    source,
    bmu: _BMU,
    n_nodes: int,
    n_feat: int,
    chunk: int,
    fraction: float,
    block: int,
    rng: Optional[np.random.Generator],
    groups: int,
    prefetch: bool,
    rank: int = 0,
    size: int = 1,
) -> dict:
    """One pass: per-node float64 sums, counts and QE sum over all (or sampled) rows
    (over the chunks this rank owns when ``size > 1``)."""
    s_g = np.zeros((groups, n_nodes, n_feat))
    n_g = np.zeros((groups, n_nodes))
    q_g = np.zeros(groups)
    stats = {"io_wait": 0.0, "bmu": 0.0, "acc": 0.0, "fallbacks": 0, "rows": 0}
    it = _epoch_chunks(source, chunk, fraction, block, rng, rank, size)
    if prefetch:
        it = _prefetched(it, stats)
    idx = np.empty(0, np.int32)
    d2 = np.empty(0)
    t_read = time.perf_counter()
    for c in it:
        if not prefetch:
            stats["io_wait"] += time.perf_counter() - t_read
        m = c.shape[0]
        if idx.shape[0] < m:
            idx = np.empty(m, np.int32)
            d2 = np.empty(m)
        t0 = time.perf_counter()
        stats["fallbacks"] += bmu(c, idx[:m], d2[:m])
        t1 = time.perf_counter()
        _accumulate_groups(c, idx[:m], d2[:m], s_g, n_g, q_g)
        t2 = time.perf_counter()
        stats["bmu"] += t1 - t0
        stats["acc"] += t2 - t1
        stats["rows"] += m
        t_read = time.perf_counter()
    stats["s"] = s_g.sum(axis=0)
    stats["n"] = n_g.sum(axis=0)
    stats["qe_sum"] = float(q_g.sum())
    return stats


CHECKPOINT_FORMAT = "aweSOM.stream.checkpoint/1"
_RANK_FIELDS = ("rows", "fallbacks", "io_wait", "bmu", "acc", "pass_time")


def _git_commit() -> str:
    """HEAD of the git checkout that holds this file, or "" (no checkout, no git)."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _write_checkpoint(path: str, arrays: dict, meta: dict) -> None:
    """Write ``arrays`` and the JSON ``meta`` to one ``.npz`` (temporary file, then rename)."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, meta=np.array(json.dumps(meta)), **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_checkpoint(path: str) -> tuple[dict, dict]:
    """``(meta, arrays)`` of a :func:`train_batch_stream` checkpoint file."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    if meta.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a {CHECKPOINT_FORMAT} file")
    return meta, arrays


def train_batch_stream(
    data_source: Any,
    xdim: int,
    ydim: int,
    epochs: int = 40,
    chunk: int = DEFAULT_CHUNK,
    r_max: Optional[float] = None,
    r_min: float = 8,
    seed: int = 42,
    init: str = "sampling",
    initial_weights: Optional[np.ndarray] = None,
    bmu: str = "auto",
    sample_fraction: float = 1.0,
    sample_block: int = 4096,
    sample_seed: Optional[int] = None,
    final_pass: str = "full",
    n_samples: Optional[int] = None,
    groups: int = DEFAULT_GROUPS,
    prefetch: bool = True,
    n_threads: Optional[int] = None,
    verbose: bool = False,
    callback: Optional[Callable[[int, np.ndarray], None]] = None,
    n_features: Optional[int] = None,
    checkpoint: Optional[str] = None,
    resume: bool = False,
    comm: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train a SOM lattice with the batch map, streaming the data chunk by chunk.

    Args:
        data_source: ``(N, F)`` ndarray, ``numpy.memmap`` or ``h5py.Dataset``
            (sliced by rows), or a zero-argument callable returning a fresh
            iterable of ``(rows, F)`` arrays on every call (optional attributes:
            see ``_CallableSource``). Rows must be finite and scaled (as for
            every aweSOM trainer).
        xdim, ydim: Lattice dimensions; node ``i`` sits at ``(i % xdim, i // xdim)``.
        epochs: Number of batch epochs.
        chunk: Rows read and processed at once (array sources). Memory is about
            ``2 * chunk * F * itemsize + chunk * 12`` bytes for data, prefetch
            and BMU buffers, plus ``groups * nodes * (F + 1) * 8`` bytes of
            accumulators.
        r_max: Initial radius; defaults to ``max(xdim, ydim) + 1`` as in
            :func:`aweSOM.batch.train_batch`.
        r_min: Final radius; clamped to ``r_max``.
        seed: Seed of ``np.random.default_rng`` for the initial lattice; the
            same rows as :func:`aweSOM.batch.train_batch` are drawn.
        init: ``"sampling"`` or ``"uniform"``; ignored with ``initial_weights``.
        initial_weights: Optional ``(xdim * ydim, F)`` starting lattice.
        bmu: ``"auto"`` (default: per pass, the faster of screen and patch on
            the first 16384 rows), ``"screen"`` (float32 screen certified in
            float64), ``"patch"`` (bounding-ball pruned search over lattice
            patches, :func:`aweSOM.inference.best_match_nodes_patch`; the first
            epoch uses feature-space groups with certified bounds) or ``"numba"``
            (float64 brute force). All four give the exact float64 BMU
            (bitwise-equal labels), so the weights do not depend on the choice.
        sample_fraction: Fraction of the rows used per epoch (fresh stratified
            block sample every epoch; see the module docstring). 1.0 = all rows.
        sample_block: Rows per sampled block (array sources). Use 1 for i.i.d.
            rows from an in-memory array; >= 4096 for files.
        sample_seed: Seed of the per-epoch samples (default: ``seed``); epoch e
            uses ``np.random.default_rng([sample_seed, e])``.
        final_pass: ``"full"`` (QE and hits of the final weights over all rows,
            one extra full pass), ``"sample"`` (over one more sample) or
            ``"none"``.
        n_samples: Number of rows of a callable source if known. Without it, or
            without the column count, one counting pass is made.
        groups: Accumulator slots (fixed, so results do not depend on threads).
        prefetch: Read the next chunk in a background thread.
        n_threads: numba threads for this call (default: current setting).
        verbose: Print one line per epoch (rank 0).
        callback: ``callback(epoch, weights)`` after every epoch (rank 0).
        n_features: Columns of a callable source if known (also read from the
            callable's ``n_feat`` or ``n_features`` attribute).
        checkpoint: Path of a ``.npz`` file rewritten after every epoch (rank 0)
            with the weights, epoch, radii, QE, rows, fallbacks and timings,
            the sampling seed, the callable's ``calls`` counter (if it has one)
            and the git commit (:func:`read_checkpoint` reads it).
        resume: With ``checkpoint`` present on disk, continue after its last
            epoch (every rank reads it). Data, lattice, epochs, radii and
            sampling must match; the result is bitwise equal to an
            uninterrupted run with the same rank count.
        comm: mpi4py communicator (:func:`aweSOM.distributed.get_comm`) for a
            run with one rank per node. Rank 0 draws the initial lattice and
            broadcasts it; every rank searches the chunks with index
            ``i % size == rank``; the sums are gathered and added in rank order
            (:class:`aweSOM.distributed.Reducer`), then every rank applies the
            same update. A fixed rank count is bitwise reproducible; different
            rank counts differ in the last bits of the float64 sums. Every
            rank must call with the same arguments and the same data.

    Returns:
        ``(weights, info)``; ``weights`` float64 ``(xdim * ydim, F)`` (equal on
        every rank). ``info`` has the keys of :func:`aweSOM.batch.train_batch`
        (``radius``, ``qe``, ``qe_initial``, ``hits``, ``epoch_time``,
        ``chunk``, ``n_samples``) plus ``rows_per_epoch``, ``fallbacks`` (rows
        that needed the float64 rescan), ``io_wait``, ``bmu_time``,
        ``acc_time``, ``update_time``, ``reduce_time``, ``checkpoint_time``
        (seconds per epoch; io/bmu/acc of this rank), ``final_time``,
        ``n_threads``, ``peak_rss`` (bytes, this rank), ``ranks``, ``rank``,
        ``resumed_from`` (epochs read from the checkpoint) and ``rank_rows``,
        ``rank_fallbacks``, ``rank_io_wait``, ``rank_bmu``, ``rank_acc``,
        ``rank_pass_time`` (one list per epoch, one value per rank). ``qe`` is
        exact (float64 distance of the exact BMU); with sampling it is
        measured on the next epoch's sample. ``hits`` is empty when
        ``final_pass="none"``.
    """
    if bmu not in ("auto", "screen", "patch", "numba"):
        raise ValueError('bmu must be "auto", "screen", "patch" or "numba"')
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1]")
    if final_pass not in ("full", "sample", "none"):
        raise ValueError('final_pass must be "full", "sample" or "none"')
    if chunk < 1 or groups < 1 or sample_block < 1:
        raise ValueError("chunk, groups and sample_block must be positive")
    if init not in ("sampling", "uniform"):
        raise ValueError(f"init must be 'sampling' or 'uniform', got {init!r}")
    if resume and checkpoint is None:
        raise ValueError("resume=True needs a checkpoint path")
    red = None
    if comm is not None:
        from .distributed import Reducer

        red = Reducer(comm)
    rank, size = (red.rank, red.size) if red is not None else (0, 1)
    root = rank == 0
    old_threads = numba.get_num_threads()
    if n_threads is not None:
        numba.set_num_threads(int(n_threads))
    try:
        source = _make_source(data_source, n_samples, n_features)
        n_total, n_feat = source.n, source.n_feat
        if not n_total:
            raise ValueError("data is empty")
        n_nodes = xdim * ydim
        fn = getattr(source, "fn", None)

        if r_max is None:
            r_max = max(xdim, ydim) + 1
        radii = radius_schedule(epochs, r_max, min(r_min, r_max))
        s_seed = seed if sample_seed is None else sample_seed

        def epoch_rng(e: int):
            return np.random.default_rng([s_seed, e]) if sample_fraction < 1.0 else None

        keys = ("epoch_time", "io_wait", "bmu_time", "acc_time", "update_time", "reduce_time",
                "checkpoint_time")
        rec = {k: np.zeros(epochs) for k in keys}
        rows = np.zeros(epochs, np.int64)
        fallbacks = np.zeros(epochs, np.int64)
        bmu_kinds: list = []
        rank_rec: dict = {k: [] for k in _RANK_FIELDS}
        qe_after = np.zeros(epochs)
        qe_initial = math.nan
        start = 0
        expect = dict(xdim=int(xdim), ydim=int(ydim), n_feat=int(n_feat), n_samples=int(n_total),
                      epochs=int(epochs), sample_fraction=float(sample_fraction))
        if sample_fraction < 1.0:
            expect.update(sample_seed=int(s_seed), sample_block=int(sample_block))

        if resume and os.path.exists(checkpoint):
            meta, arrays = read_checkpoint(checkpoint)
            for k, v in expect.items():
                if meta.get(k) != v:
                    raise ValueError(f"checkpoint has {k}={meta.get(k)!r}, this run {v!r}")
            if not np.array_equal(arrays["radius"], radii):
                raise ValueError("checkpoint radius schedule differs (r_max, r_min or epochs)")
            start = int(meta["epoch"])
            w = np.array(arrays["weights"], np.float64)
            for k in keys:
                rec[k][:] = arrays[k]
            rows[:] = arrays["rows_per_epoch"]
            fallbacks[:] = arrays["fallbacks"]
            qe_after[:] = arrays["qe"]
            qe_initial = float(meta["qe_initial"])
            bmu_kinds = list(meta["bmu_kind"])
            rank_rec = {k: list(meta["rank_records"][k]) for k in _RANK_FIELDS}
            if meta.get("calls") is not None and fn is not None and hasattr(fn, "calls"):
                fn.calls = int(meta["calls"])
        else:
            w = None
            if initial_weights is not None:
                w = np.array(initial_weights, np.float64)
                if w.shape != (n_nodes, n_feat):
                    raise ValueError(f"initial_weights must have shape {(n_nodes, n_feat)}, got {w.shape}")
            elif init == "uniform":
                w = np.random.default_rng(seed).uniform(0.0, 1.0, (n_nodes, n_feat))
            else:
                ix = np.random.default_rng(seed).integers(0, n_total, n_nodes)
                # A callable without rows() advances its state on a pass: every rank
                # makes the same pass so that the ranks stay in step.
                stateful = fn is not None and not callable(getattr(fn, "rows", None))
                if root or stateful:
                    w = np.asarray(source.rows(ix), np.float64)
            if red is not None:
                w = red.bcast(w if root else None, (n_nodes, n_feat))

        commit = _git_commit() if (checkpoint is not None and root) else ""

        def save(e: int) -> None:
            calls = getattr(fn, "calls", None) if fn is not None else None
            meta = dict(format=CHECKPOINT_FORMAT, epoch=e + 1, **expect)
            meta.update(sample_seed=int(s_seed), sample_block=int(sample_block), seed=int(seed), chunk=int(chunk),
                        groups=int(groups), bmu=bmu, bmu_kind=bmu_kinds, qe_initial=qe_initial,
                        calls=None if calls is None else int(calls), ranks=size,
                        rank_records=rank_rec, git_commit=commit, version=_version())
            arrays = dict(weights=w, radius=radii, qe=qe_after, rows_per_epoch=rows,
                          fallbacks=fallbacks, **rec)
            _write_checkpoint(checkpoint, arrays, meta)

        for e in range(start, epochs):
            r = radii[e]
            t0 = time.perf_counter()
            searcher = _BMU(w, bmu, xdim, ydim, first_epoch=e == 0)
            st = _pass(source, searcher, n_nodes, n_feat, chunk, sample_fraction,
                       sample_block, epoch_rng(e), groups, prefetch, rank, size)
            st["pass_time"] = time.perf_counter() - t0
            bmu_kinds.append(searcher.kind)
            t_red = time.perf_counter()
            if red is not None:
                tot, per = red.reduce_pass(st)
            else:
                tot, per = st, {k: np.array([st[k]]) for k in _RANK_FIELDS}
            t1 = time.perf_counter()
            w = neighbourhood_update(tot["s"], tot["n"], w, xdim, ydim, r)
            t2 = time.perf_counter()
            qe_entering = tot["qe_sum"] / tot["rows"]
            if e > 0:
                qe_after[e - 1] = qe_entering
            else:
                qe_initial = qe_entering
            rec["epoch_time"][e] = t2 - t0
            rec["io_wait"][e] = st["io_wait"]
            rec["bmu_time"][e] = st["bmu"]
            rec["acc_time"][e] = st["acc"]
            rec["update_time"][e] = t2 - t1
            rec["reduce_time"][e] = t1 - t_red
            rows[e] = tot["rows"]
            fallbacks[e] = tot["fallbacks"]
            for k in _RANK_FIELDS:
                rank_rec[k].append(np.asarray(per[k]).tolist())
            if verbose and root:
                print(
                    f"epoch {e + 1}/{epochs}  radius {r:.3f}  rows {tot['rows']}  "
                    f"qe(entering) {qe_entering:.6g}  {t2 - t0:.3f} s "
                    f"(io wait {st['io_wait']:.2f}, bmu {st['bmu']:.2f}, update {t2 - t1:.3f}, "
                    f"fallbacks {tot['fallbacks']}, {searcher.kind}"
                    + (f", reduce {t1 - t_red:.3f}, ranks {size}" if red is not None else "") + ")",
                    flush=True,
                )
            if checkpoint is not None:
                if root:
                    t3 = time.perf_counter()
                    save(e)
                    rec["checkpoint_time"][e] = time.perf_counter() - t3
                if red is not None:
                    red.barrier()  # the file is complete before any rank can read it
            if callback is not None and root:
                callback(e, w.copy())

        t0 = time.perf_counter()
        hits = np.zeros(0, np.int64)
        if final_pass != "none":
            frac = 1.0 if final_pass == "full" else sample_fraction
            st = _pass(source, _BMU(w, bmu, xdim, ydim), n_nodes, n_feat, chunk, frac, sample_block,
                       np.random.default_rng([s_seed, epochs]) if frac < 1.0 else None,
                       groups, prefetch, rank, size)
            if red is not None:
                st, _ = red.reduce_pass(st)
            qe_after[-1] = st["qe_sum"] / st["rows"]
            hits = st["n"].round().astype(np.int64)
        else:
            qe_after[-1] = math.nan
        final_time = time.perf_counter() - t0
        threads = numba.get_num_threads()
    finally:
        numba.set_num_threads(old_threads)

    info = {
        "radius": radii,
        "qe": qe_after,
        "qe_initial": qe_initial,
        "hits": hits,
        "chunk": chunk,
        "n_samples": n_total,
        "rows_per_epoch": rows,
        "fallbacks": fallbacks,
        "bmu_kind": bmu_kinds,
        "final_time": final_time,
        "n_threads": threads,
        "peak_rss": peak_rss_bytes(),
        "ranks": size,
        "rank": rank,
        "resumed_from": start,
        **{f"rank_{k}": v for k, v in rank_rec.items()},
        **rec,
    }
    return w, info


def _version() -> str:
    from . import __version__

    return __version__


# --------------------------------------------------------------------- inference


def predict_stream(
    weights: np.ndarray,
    node_cluster: np.ndarray,
    data_source: Any,
    out: Any = None,
    out_distance: Any = None,
    chunk: int = DEFAULT_CHUNK,
    backend: str = "screen",
    lattice_shape: Optional[tuple[int, int]] = None,
    prefetch: bool = True,
    n_samples: Optional[int] = None,
) -> tuple[Any, dict[str, float]]:
    """Label every row of a streamed source with :func:`aweSOM.inference.predict`.

    Args:
        weights: ``(nodes, F)`` lattice.
        node_cluster: ``(nodes,)`` cluster id per node (use ``np.arange(nodes)``
            for BMU indices).
        data_source: As in :func:`train_batch_stream` (already scaled).
        out: Writable ``(N,)`` int32 array-like (``ndarray``, ``memmap``,
            ``h5py.Dataset``); allocated in memory if None.
        out_distance: Optional writable ``(N,)`` array for the BMU distance.
        chunk: Rows per chunk.
        backend: ``predict`` backend (default ``"screen"``; ``"patch"`` is
            exact too and faster on trained lattices, needs ``lattice_shape``).
        lattice_shape: ``(xdim, ydim)`` for ``backend="patch"``.
        prefetch: Read the next chunk in a background thread.
        n_samples: Rows of a callable source if known.

    Returns:
        ``(out, stats)`` with ``stats`` = rows, seconds, io_wait, compute.
    """
    source = _make_source(data_source, n_samples)
    if out is None:
        out = np.empty(source.n, np.int32)
    stats = {"io_wait": 0.0, "compute": 0.0, "write": 0.0}
    it = _epoch_chunks(source, chunk, 1.0, 1, None)
    if prefetch:
        it = _prefetched(it, stats)
    t_start = time.perf_counter()
    pos = 0
    for c in it:
        t0 = time.perf_counter()
        res = predict(weights, node_cluster, c, chunk=c.shape[0],
                      return_distance=out_distance is not None, backend=backend,
                      lattice_shape=lattice_shape)
        t1 = time.perf_counter()
        m = c.shape[0]
        if out_distance is not None:
            out[pos : pos + m], out_distance[pos : pos + m] = res
        else:
            out[pos : pos + m] = res
        pos += m
        stats["compute"] += t1 - t0
        stats["write"] += time.perf_counter() - t1
    stats["seconds"] = time.perf_counter() - t_start
    stats["rows"] = pos
    return out, stats


def stratified_sample(
    data_source: Any, n_target: int, block: int = 4096, seed: int = 0, chunk: int = DEFAULT_CHUNK
) -> np.ndarray:
    """Draw about ``n_target`` rows as a stratified block sample into memory.

    The rows are split into blocks of ``block`` rows, the blocks into equal
    strata, and one block is drawn from each stratum (the same scheme as
    ``train_batch_stream(..., sample_fraction=...)``). Returns an in-memory
    array in the source dtype (float32 stays float32).
    """
    source = _ArraySource(data_source)
    fraction = min(1.0, n_target / source.n)
    rng = np.random.default_rng(seed)
    parts = list(_epoch_chunks(source, chunk, fraction, block, rng)) if fraction < 1.0 else [
        source.read(0, source.n)
    ]
    return np.concatenate(parts) if len(parts) > 1 else parts[0]
