"""Benchmarks of the streaming batch trainer (aweSOM.stream) on synthetic 5-feature data.

Modes (``--mode``):

gen       Write N synthetic rows (float32, already scaled) to an ``.npy`` file
          that is later opened as a memmap. Rows are generated in blocks of
          2**20 with ``default_rng([seed, block])``, so the first 10^7 rows of
          the 10^9 file equal the 10^7 file.
stages    Train a lattice on the first 10^6 rows and save the weights after
          epoch 1 ("early", radius r_max, contracted lattice), the middle of
          the linear phase ("mid") and the end ("late") for the epoch timings.
epoch     Time single training epochs (BMU + accumulation + update) over N rows
          with fixed stage weights, from the memmap or from RAM; reports s/epoch,
          Mpts/s, Mpts/s per thread, node-distance evaluations/s per thread,
          I/O wait, screen fallbacks and peak RSS. ``--repeat`` passes, min reported.
train     Full training (``--epochs``, optional ``--sample-fraction`` or
          ``--fixed-sample``) on the first N rows, then exact QE and TE.
predict   Stream inference (``predict_stream``, ``--backend`` patch or screen) over N rows of
          the memmap into a memmap label file.
ioread    Sequential read throughput of the memmap (chunks copied, summed).

Every timing line records ``os.getloadavg()`` and ``numba.get_num_threads()``.
Results are appended as JSON lines to ``--out``.

The synthetic distribution: four anisotropic Gaussian blobs (85% of the rows),
a thin curved 2-D sheet in 5-D (noise 0.03, 10%) and a broad component (5%),
then aweSOM manual scaling ``(x - mean) / (2 std)`` with mean and std fitted
on a 2*10^6-row pilot sample.

Example:
    python planning/benchmarks/bench_stream.py --mode gen --N 100000000 --data /path/x1e8.npy
    python planning/benchmarks/bench_stream.py --mode stages --data /path/x1e8.npy --xdim 91 --ydim 64
    python planning/benchmarks/bench_stream.py --mode epoch --N 100000000 --data /path/x1e8.npy --xdim 91 --ydim 64
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

import numba
import numpy as np

from aweSOM.stream import (
    NpyReader,
    peak_rss_bytes,
    predict_stream,
    stratified_sample,
    train_batch_stream,
)

GEN_BLOCK = 2**20
N_FEAT = 5

_CENTERS = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [2.5, -1.0, 0.5, 0.0, -0.5],
        [-1.5, 2.0, -1.0, 1.0, 0.0],
        [0.5, 0.5, 2.5, -1.5, 1.0],
    ]
)
_STDS = np.array(
    [
        [1.0, 0.8, 0.6, 0.5, 0.4],
        [0.5, 0.5, 0.3, 0.6, 0.3],
        [0.4, 0.7, 0.5, 0.3, 0.5],
        [0.6, 0.3, 0.4, 0.5, 0.6],
    ]
)
_WEIGHTS = np.array([0.40, 0.20, 0.15, 0.10, 0.10, 0.05])  # 4 blobs, sheet, broad


def _raw_rows(rng: np.random.Generator, m: int) -> np.ndarray:
    comp = rng.choice(len(_WEIGHTS), m, p=_WEIGHTS)
    x = np.empty((m, N_FEAT))
    blob = comp < 4
    k = comp[blob]
    x[blob] = _CENTERS[k] + _STDS[k] * rng.standard_normal((k.size, N_FEAT))
    sheet = comp == 4
    ns = int(sheet.sum())
    u = rng.uniform(-1.0, 1.0, ns)
    v = rng.uniform(-1.0, 1.0, ns)
    s = np.stack(
        [2.0 * u, 2.0 * v, 0.8 * np.sin(np.pi * u) * np.cos(0.5 * np.pi * v),
         0.5 * u * v, np.full(ns, 3.0)], axis=1
    )
    x[sheet] = s + 0.03 * rng.standard_normal((ns, N_FEAT))
    broad = comp == 5
    x[broad] = 3.0 * rng.standard_normal((int(broad.sum()), N_FEAT))
    return x


def scaling(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Offset and scale of aweSOM manual scaling, fitted on a 2e6-row pilot sample."""
    pilot = _raw_rows(np.random.default_rng([seed, 2**31 - 1]), 2_000_000)
    return pilot.mean(axis=0), 2.0 * pilot.std(axis=0)


def gen_block(seed: int, b: int, m: int, offset: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((_raw_rows(np.random.default_rng([seed, b]), m) - offset) / scale).astype(np.float32)


def generate(path: str, n: int, seed: int = 0, workers: int = 8) -> None:
    offset, scale = scaling(seed)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, N_FEAT))
    nblocks = -(-n // GEN_BLOCK)

    def work(b):
        lo = b * GEN_BLOCK
        hi = min(n, lo + GEN_BLOCK)
        out[lo:hi] = gen_block(seed, b, GEN_BLOCK, offset, scale)[: hi - lo]

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        list(pool.map(work, range(nblocks)))
    out.flush()
    del out
    print(f"wrote {n} rows to {path} in {time.perf_counter() - t0:.1f} s", flush=True)


def _env() -> dict:
    return {"load": [round(v, 2) for v in os.getloadavg()], "threads": numba.get_num_threads()}


def _emit(out: str, rec: dict) -> None:
    rec = {**rec, **_env(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(json.dumps(rec), flush=True)
    if out:
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


class _Prefix:
    """First n rows of a row-sliceable source."""

    def __init__(self, src, n):
        self.src, self.shape, self.dtype = src, (n, src.shape[1]), src.dtype

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, _ = key.indices(self.shape[0])
            return self.src[start:stop]
        return self.src[key]


def _open(path: str, n: int | None, in_ram: bool, reader: bool = False):
    if reader and not in_ram:
        src = NpyReader(path)
        return _Prefix(src, n) if n is not None and n < src.shape[0] else src
    x = np.load(path, mmap_mode="r")
    if n is not None:
        x = x[:n]
    if in_ram:
        t0 = time.perf_counter()
        x = np.array(x)
        print(f"loaded {x.shape} into RAM in {time.perf_counter() - t0:.1f} s", flush=True)
    return x


def stages_path(results: str, xdim: int, ydim: int) -> Path:
    return Path(results) / f"stages_{xdim}x{ydim}.npz"


def mode_stages(a) -> None:
    x = _open(a.data, 1_000_000, in_ram=True)
    epochs = 40
    snaps = {}
    marks = {0: "early", 24: "mid", epochs - 1: "late"}

    def cb(e, w):
        if e in marks:
            snaps[marks[e]] = w

    t0 = time.perf_counter()
    w, info = train_batch_stream(x, a.xdim, a.ydim, epochs=epochs, callback=cb, verbose=a.verbose)
    radius = {name: float(info["radius"][min(e + 1, epochs - 1)]) for e, name in marks.items()}
    np.savez(stages_path(a.results, a.xdim, a.ydim), **snaps,
             **{f"radius_{k}": v for k, v in radius.items()})
    _emit(a.out, {"mode": "stages", "xdim": a.xdim, "ydim": a.ydim, "seconds": time.perf_counter() - t0,
                  "qe": info["qe"].tolist(), "fallbacks": info["fallbacks"].tolist()})


def mode_epoch(a) -> None:
    st = np.load(stages_path(a.results, a.xdim, a.ydim))
    x = _open(a.data, a.N, a.in_ram, a.reader)
    n = x.shape[0]
    nodes = a.xdim * a.ydim
    for stage in a.stages.split(","):
        w0 = st[stage]
        r = float(st[f"radius_{stage}"])
        best = None
        for rep in range(a.repeat):
            _, info = train_batch_stream(
                x, a.xdim, a.ydim, epochs=1, initial_weights=w0, r_max=r, r_min=r,
                final_pass="none", chunk=a.chunk, bmu=a.bmu, sample_fraction=a.sample_fraction,
            )
            t = float(info["epoch_time"][0])
            rec = {
                "mode": "epoch", "N": n, "xdim": a.xdim, "ydim": a.ydim, "stage": stage, "radius": r,
                "source": _src(a), "bmu": a.bmu, "rep": rep,
                "sample_fraction": a.sample_fraction, "rows": int(info["rows_per_epoch"][0]),
                "s_per_epoch": t, "io_wait": float(info["io_wait"][0]),
                "bmu_time": float(info["bmu_time"][0]), "acc_time": float(info["acc_time"][0]),
                "update_time": float(info["update_time"][0]),
                "fallbacks": int(info["fallbacks"][0]), "bmu_kind": info["bmu_kind"][0],
                "peak_rss_gb": peak_rss_bytes() / 1e9,
            }
            rows = rec["rows"]
            rec["mpts_s"] = rows / t / 1e6
            rec["mpts_s_per_thread"] = rec["mpts_s"] / numba.get_num_threads()
            rec["node_evals_s_per_thread"] = rows * nodes / t / numba.get_num_threads()
            _emit(a.out, rec)
            if best is None or t < best["s_per_epoch"]:
                best = rec
        _emit(a.out, {**best, "mode": "epoch_min", "repeat": a.repeat})


def mode_train(a) -> None:
    from aweSOM.batch import quantization_error_fast, topographic_error_fast

    x = _open(a.data, a.N, a.in_ram)
    t_s = 0.0
    if a.fixed_sample:
        t0 = time.perf_counter()
        train = stratified_sample(x, a.fixed_sample, block=a.sample_block, seed=a.seed)
        t_s = time.perf_counter() - t0
    else:
        train = x
    t0 = time.perf_counter()
    w, info = train_batch_stream(
        train, a.xdim, a.ydim, epochs=a.epochs, seed=a.seed, sample_fraction=a.sample_fraction,
        sample_block=a.sample_block, final_pass="none", chunk=a.chunk, verbose=a.verbose, bmu=a.bmu,
    )
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    xe = np.asarray(x[: a.eval_n])
    qe = quantization_error_fast(w, xe)
    te = topographic_error_fast(w, xe, a.xdim, a.ydim)
    t_eval = time.perf_counter() - t0
    tag = a.tag or f"N{x.shape[0]}_e{a.epochs}_f{a.sample_fraction}_fs{a.fixed_sample}_s{a.seed}"
    np.save(Path(a.results) / f"weights_{a.xdim}x{a.ydim}_{tag}.npy", w)
    _emit(a.out, {
        "mode": "train", "N": int(x.shape[0]), "xdim": a.xdim, "ydim": a.ydim, "epochs": a.epochs,
        "seed": a.seed, "sample_fraction": a.sample_fraction, "sample_block": a.sample_block,
        "fixed_sample": a.fixed_sample, "rows_per_epoch": int(info["rows_per_epoch"][0]),
        "train_seconds": t_train, "sample_seconds": t_s, "epoch_time": info["epoch_time"].tolist(),
        "fallbacks": info["fallbacks"].tolist(), "bmu_kind": info["bmu_kind"], "qe_train": info["qe"].tolist(),
        "eval_n": int(xe.shape[0]), "qe": qe, "te": te,
        "eval_seconds": t_eval, "peak_rss_gb": peak_rss_bytes() / 1e9, "tag": tag,
    })


def mode_predict(a) -> None:
    st = np.load(stages_path(a.results, a.xdim, a.ydim))
    w = st["late"]
    x = _open(a.data, a.N, a.in_ram, a.reader)
    labels_path = Path(a.scratch) / f"labels_{x.shape[0]}.npy"
    for rep in range(a.repeat):
        out = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.int32, shape=(x.shape[0],))
        _, stats = predict_stream(w, np.arange(w.shape[0]), x, out=out, chunk=a.chunk,
                                  backend=a.backend, lattice_shape=(a.xdim, a.ydim))
        out.flush()
        del out
        _emit(a.out, {"mode": "predict", "N": int(x.shape[0]), "xdim": a.xdim, "ydim": a.ydim,
                      "source": _src(a), "backend": a.backend, "rep": rep, **stats,
                      "mpts_s": stats["rows"] / stats["seconds"] / 1e6,
                      "peak_rss_gb": peak_rss_bytes() / 1e9})
    labels_path.unlink()


def _src(a) -> str:
    return "ram" if a.in_ram else ("read" if a.reader else "memmap")


def mode_ioread(a) -> None:
    x = _open(a.data, a.N, False, a.reader)
    for rep in range(a.repeat):
        t0 = time.perf_counter()
        tot = 0.0
        for s in range(0, x.shape[0], a.chunk):
            tot += float(np.ascontiguousarray(x[s : s + a.chunk])[:, 0].sum())
        t = time.perf_counter() - t0
        gb = x.shape[0] * x.shape[1] * 4 / 1e9
        _emit(a.out, {"mode": "ioread", "N": int(x.shape[0]), "rep": rep, "source": _src(a), "seconds": t, "GB_s": gb / t})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["gen", "stages", "epoch", "train", "predict", "ioread"])
    p.add_argument("--N", type=int, default=None, help="rows (gen: rows to write; else: first N rows)")
    p.add_argument("--data", required=True, help=".npy file (float32, N x 5)")
    p.add_argument("--xdim", type=int, default=91)
    p.add_argument("--ydim", type=int, default=64)
    p.add_argument("--stages", default="early,late")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--chunk", type=int, default=2**20)
    p.add_argument("--bmu", default="auto", help="training BMU search: auto, screen, patch, numba")
    p.add_argument("--backend", default="patch", help="predict backend: patch or screen")
    p.add_argument("--in-ram", action="store_true", help="load the N rows into memory first")
    p.add_argument("--reader", action="store_true", help="plain file reads (NpyReader) instead of a memmap")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-fraction", type=float, default=1.0)
    p.add_argument("--sample-block", type=int, default=4096)
    p.add_argument("--fixed-sample", type=int, default=0, help="train on a stratified in-RAM sample of this size")
    p.add_argument("--eval-n", type=int, default=10_000_000)
    p.add_argument("--tag", default="")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--results", default=".")
    p.add_argument("--scratch", default=".")
    p.add_argument("--out", default="")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    if a.threads:
        numba.set_num_threads(a.threads)
    if a.mode == "gen":
        generate(a.data, a.N, seed=0)
    else:
        globals()[f"mode_{a.mode}"](a)


if __name__ == "__main__":
    main()
