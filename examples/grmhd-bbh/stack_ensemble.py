#!/usr/bin/env python3
"""Train an ensemble of aweSOM realizations on a GRMHD feature file and stack them with SCE.

features.h5 -> realizations (seeds x lattice scales) -> SCE Gsum (mask-free) -> stacked labels

Each realization follows ``etk-analysis/scripts/train_som.py``: the first ``N`` rows,
the frozen median-IQR scaler for v2 or ``manual_scaling`` for legacy inputs,
a lattice of
``scale * number_of_nodes(N, F)`` nodes at height/width ratio 0.7, ``alpha_0 = 0.5`` with
decay, uniform (``1u``) initial weights, ``N`` online steps, and a U-matrix centroid
merge at ``merge_cost = 0.2``. Realizations differ in seed and lattice scale.

The ensemble goes through one SCE Gsum pass (``aweSOM.sce.loop_over_all_clusters``,
``write_masks=False``) and is then stacked two ways:

- ``anchored`` (``stacked_labels.npy``, recommended): ``make_sce_clusters.anchored_sce_labels``
  groups every realization's clusters by Jaccard index with the medoid realization's
  clusters and takes a plurality vote per cell (two passes; ``--weighting gsum`` weights votes by Gsum).
- ``gsum_ranges`` (``stacked_labels_gsum_ranges.npy``): ``make_sce_clusters.stack_sce_labels``,
  the make_sce_clusters CLI logic: the ranked Gsum curve is split where its smoothed
  relative derivative has local minima below ``--sce-threshold``, each range's
  Gsum-weighted masks are summed into one signal map, and each cell takes its strongest
  map. Ranges follow reproducibility, not identity, so one range can mix regimes.

Example (16 realizations on 10^6 rows, 4 at a time)::

    python stack_ensemble.py --input features_bbh_0003-180224-N10000000.h5 --N 1000000 \\
        --seeds 42 43 44 45 --lattice-scales 0.7 1 1.4 2 --jobs 4 --output-dir out/N1e6

Outputs in ``--output-dir``: ``realizations/`` (``labels.<tag>.npy`` uint8/uint16,
``som_object.<tag>.pkl`` without the training data, ``params.<tag>.json``),
``SCE/multimap_mappings.txt``, ``stacked_labels.npy`` + ``sce_signals.npy`` (anchored),
``stacked_labels_gsum_ranges.npy`` + ``sce_signals_gsum_ranges.npy``, and ``stack_summary.json``.
"""

import argparse
import contextlib
import itertools
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from aweSOM import Lattice
from aweSOM.grmhd_freeze import training_scaler
from aweSOM.make_sce_clusters import anchored_sce_labels, stack_sce_labels
from aweSOM.run_som import manual_scaling, number_of_nodes
from aweSOM.sce import find_number_of_clusters, loop_over_all_clusters

DEFAULT_INPUT = "/Volumes/ExternalNVMe/som-proto-f5/features_bbh_0003-180224-N10000000.h5"


def load_features(path: str, n: int):
    """First ``n`` rows of ``features`` (float32), the feature names, and the
    per-row volume weight (``weight_volume`` if present, else ones)."""
    with h5py.File(path, "r") as f:
        n = min(int(n), f["features"].shape[0])
        x = f["features"][:n]
        names = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["names"][()]]
        w = f["weight_volume"][:n] if "weight_volume" in f else np.ones(n)
    return x, names, w


def lattice_dims(n: int, f: int, ratio: float = 0.7, scale: float = 1.0) -> tuple[int, int]:
    """Lattice size of ``train_som.py`` (``initialize_lattice``) with the node count scaled.

    ``nodes = scale * number_of_nodes(n, f)``, ``xdim = int(sqrt(nodes / ratio))``,
    ``ydim = int(nodes / xdim)``; ``scale = 1`` gives 51 x 36 for (10^6, 5) and 91 x 64
    for (10^7, 5).
    """
    nodes = scale * number_of_nodes(n, f)
    xdim = int(np.sqrt(nodes / ratio))
    ydim = int(nodes / xdim)
    return xdim, ydim


def compact_labels(labels: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Renumber labels to 0..k-1 (SCE counts distinct ids) as uint8, or uint16 above 255.

    Returns the compact labels and the original id of each new id.
    """
    ids, inverse = np.unique(labels, return_inverse=True)
    dtype = np.uint8 if ids.size <= 256 else np.uint16
    return inverse.astype(dtype).reshape(labels.shape), [int(i) for i in ids]


def realization_tag(name: str, xdim: int, ydim: int, alpha_0: float, train: int, seed: int) -> str:
    """aweSOM file tag (``save_cluster_labels`` convention) plus the seed."""
    return f"{name}-{xdim}-{ydim}-{alpha_0}-{train}-1u-s{seed}"


def train_realization(
    data: np.ndarray,
    names: list[str],
    xdim: int,
    ydim: int,
    seed: int,
    alpha_0: float = 0.5,
    train: int = None,
    merge_cost: float = 0.2,
):
    """Train one lattice and label the training rows.

    Returns the Lattice, int32 labels (lattice cluster ids) and a timing dict.
    The U-matrix is computed from the final lattice (``history=False``).
    """
    train = data.shape[0] if train is None else int(train)
    som = Lattice(xdim=xdim, ydim=ydim, alpha_0=alpha_0, train=train, alpha_type="decay", sampling_type="uniform")
    som.seed = int(seed)
    t0 = time.perf_counter()
    som.train_lattice(data, names, history=False)
    t1 = time.perf_counter()
    proj = som.map_data_to_lattice()
    t2 = time.perf_counter()
    clusters = som.assign_cluster_to_lattice(smoothing=None, merge_cost=merge_cost)
    labels = som.assign_cluster_to_data(proj, clusters)
    t3 = time.perf_counter()
    timings = {"train": t1 - t0, "projection": t2 - t1, "clustering": t3 - t2}
    return som, labels, timings


def _job_list(seeds, scales, n_realizations):
    jobs = list(itertools.product(seeds, scales))
    if n_realizations is None:
        return jobs
    if n_realizations > len(jobs):
        raise ValueError(f"--n-realizations {n_realizations} exceeds len(seeds) x len(scales) = {len(jobs)}")
    return jobs[:n_realizations]


def run_realization(cfg: dict) -> dict:
    """Train, label and save one realization (skipped when its params JSON exists)."""
    out = Path(cfg["output_dir"]) / "realizations"
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(cfg["input"], "r") as fh:
        n = min(int(cfg["N"]), fh["features"].shape[0])
        f = fh["features"].shape[1]
    xdim, ydim = lattice_dims(n, f, cfg["ratio"], cfg["scale"])
    train = n if cfg["train"] is None else cfg["train"]
    tag = realization_tag(cfg["name"], xdim, ydim, cfg["alpha_0"], train, cfg["seed"])
    params_file = out / f"params.{tag}.json"
    if params_file.exists():
        return json.loads(params_file.read_text())

    t_start = time.perf_counter()
    x, names, _ = load_features(cfg["input"], n)
    with h5py.File(cfg["input"], "r") as h:
        scaler, scaling_meta = training_scaler(h, x, names, dtype=x.dtype)
    if scaler.kind == "manual":
        # Preserve historical float32 arithmetic before applying optional weights.
        data = manual_scaling(x)
        if "feature_weights" in scaling_meta:
            data = data * np.asarray(json.loads(scaling_meta["feature_weights"]), dtype=data.dtype)
    else:
        data = scaler.transform(x)
    del x
    log_path = out / f"train.{tag}.log"
    # the merge step prints a few lines per centroid pair; keep it out of the console
    with open(log_path, "w") as log, contextlib.redirect_stdout(log):
        som, labels, timings = train_realization(
            data, names, xdim, ydim, cfg["seed"], cfg["alpha_0"], train, cfg["merge_cost"]
        )
    compact, original_ids = compact_labels(labels)
    np.save(out / f"labels.{tag}.npy", compact)
    som.data_array = None  # keep the pickle small; the rows are the input file's prefix
    som.projection_1d = som.projection_2d = None
    with open(out / f"som_object.{tag}.pkl", "wb") as fh:
        pickle.dump(som, fh)
    counts = np.bincount(compact.astype(np.int64))
    params = {
        "tag": tag,
        "input_file": str(cfg["input"]),
        "n_samples": int(n),
        "feature_names": names,
        "seed": int(cfg["seed"]),
        "lattice_scale": float(cfg["scale"]),
        "lattice_xdim": int(xdim),
        "lattice_ydim": int(ydim),
        "total_nodes": int(xdim * ydim),
        "ratio": float(cfg["ratio"]),
        "alpha_0": float(cfg["alpha_0"]),
        "alpha_type": "decay",
        "sampling_type": "uniform",
        "train_steps": int(train),
        "merge_cost": float(cfg["merge_cost"]),
        "history": False,
        "scaling": scaler.kind,
        "scaler_offset": scaler.offset.tolist(),
        "scaler_scale": scaler.scale.tolist(),
        **scaling_meta,
        "n_clusters": int(len(original_ids)),
        "lattice_cluster_ids": original_ids,
        "cluster_counts": [int(c) for c in counts],
        "labels_file": str(out / f"labels.{tag}.npy"),
        "som_object_file": str(out / f"som_object.{tag}.pkl"),
        "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
        "wall_seconds": round(time.perf_counter() - t_start, 3),
        "numba_threads": int(_numba_threads()),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    params_file.write_text(json.dumps(params, indent=2))
    return params


def _numba_threads() -> int:
    import numba

    return numba.get_num_threads()


def _init_worker(threads: int):
    import numba

    numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))


def train_ensemble(cfg: dict, jobs: list[tuple[int, float]], n_procs: int = 1, threads: int = 2) -> list[dict]:
    """Train every (seed, scale) job, ``n_procs`` at a time; returns their params dicts in job order."""
    cfgs = [dict(cfg, seed=int(s), scale=float(sc)) for s, sc in jobs]
    if n_procs <= 1:
        _init_worker(threads)
        return [run_realization(c) for c in cfgs]
    with ProcessPoolExecutor(max_workers=n_procs, initializer=_init_worker, initargs=(threads,)) as pool:
        return list(pool.map(run_realization, cfgs))


def run_sce(label_files: list[str], sce_dir: str, n_cells: int) -> tuple[str, float]:
    """Mask-free SCE Gsum pass over ``label_files``; returns the mapping file and the seconds taken."""
    os.makedirs(sce_dir, exist_ok=True)
    mapping = os.path.join(sce_dir, "multimap_mappings.txt")
    open(mapping, "w").close()
    t0 = time.perf_counter()
    nids = find_number_of_clusters(label_files)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        loop_over_all_clusters(label_files, nids, np.array([n_cells]), sce_dir, write_masks=False)
    return mapping, time.perf_counter() - t0


def cluster_table(labels: np.ndarray, x: np.ndarray, names: list[str], w: np.ndarray) -> list[dict]:
    """Per-cluster row fraction, volume fraction and feature medians (physical units)."""
    rows = []
    wsum = float(w.sum())
    for c in np.unique(labels):
        m = labels == c
        rows.append(
            {
                "cluster": int(c),
                "count": int(m.sum()),
                "row_fraction": float(m.mean()),
                "volume_fraction": float(w[m].sum() / wsum),
                "medians": {nm: float(np.median(x[m, j])) for j, nm in enumerate(names)},
            }
        )
    return rows


def stack(output_dir: str, label_files: list[str], n_cells: int, threshold: float, smooth_fraction: int = 10,
          order: int = 4, weighting: str = "vote"):
    """SCE Gsum pass on ``label_files``, then both ways of turning it into labels.

    ``gsum_ranges``: ``stack_sce_labels`` (the make_sce_clusters CLI: Savitzky-Golay
    breaks of the ranked Gsum curve at ``threshold``, argmax of the range signal maps).
    ``anchored``: ``anchored_sce_labels`` (clusters grouped by overlap with the medoid
    realization, plurality vote with ``weighting``, two passes).

    Writes ``stacked_labels.npy`` / ``sce_signals.npy`` (anchored) and
    ``stacked_labels_gsum_ranges.npy`` / ``sce_signals_gsum_ranges.npy``.
    Returns ``{method: (labels, signals, info)}`` and a timing dict.
    """
    out = Path(output_dir)
    mapping, t_sce = run_sce(label_files, str(out / "SCE"), n_cells)
    t0 = time.perf_counter()
    ranged = stack_sce_labels(mapping, str(out), [n_cells], threshold, smooth_fraction, order)
    t1 = time.perf_counter()
    anchored = anchored_sce_labels(mapping, str(out), [n_cells], weighting=weighting)
    t2 = time.perf_counter()
    np.save(out / "stacked_labels.npy", anchored[0])
    np.save(out / "sce_signals.npy", anchored[1])
    np.save(out / "stacked_labels_gsum_ranges.npy", ranged[0])
    np.save(out / "sce_signals_gsum_ranges.npy", ranged[1])
    timings = {"sce_gsum": t_sce, "gsum_ranges": t1 - t0, "anchored": t2 - t1}
    return {"anchored": anchored, "gsum_ranges": ranged}, timings


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", default=DEFAULT_INPUT, help="feature HDF5 (datasets features, names[, weight_volume])")
    p.add_argument("--N", type=int, default=1_000_000, help="rows used for training (prefix of the file)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    p.add_argument("--lattice-scales", type=float, nargs="+", default=[0.7, 1.0, 1.4, 2.0],
                   help="multiples of the train_som.py node count, aspect ratio kept")
    p.add_argument("--n-realizations", type=int, default=None,
                   help="first n of the (seed, scale) product, seed-major (default: all)")
    p.add_argument("--merge-cost", type=float, default=0.2)
    p.add_argument("--alpha-0", type=float, default=0.5)
    p.add_argument("--ratio", type=float, default=0.7)
    p.add_argument("--train", type=int, default=None, help="online steps (default: N)")
    p.add_argument("--sce-threshold", type=float, default=-0.015,
                   help="make_sce_clusters derivative threshold")
    p.add_argument("--smooth-fraction", type=int, default=10)
    p.add_argument("--weighting", choices=["vote", "gsum"], default="vote",
                   help="vote weights of the anchored stack")
    p.add_argument("--name", default="bbh_0003_180224")
    p.add_argument("--output-dir", default="ensemble_out")
    p.add_argument("--jobs", type=int, default=1, help="realizations trained in parallel")
    p.add_argument("--threads", type=int, default=2, help="numba threads per realization")
    p.add_argument("--train-only", action="store_true", help="stop after training")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "input": args.input,
        "N": args.N,
        "ratio": args.ratio,
        "alpha_0": args.alpha_0,
        "train": args.train,
        "merge_cost": args.merge_cost,
        "name": args.name,
        "output_dir": str(out),
    }
    jobs = _job_list(args.seeds, args.lattice_scales, args.n_realizations)
    print(f"{len(jobs)} realizations, {args.jobs} at a time: {jobs}", flush=True)
    t0 = time.perf_counter()
    params = train_ensemble(cfg, jobs, args.jobs, args.threads)
    t_train = time.perf_counter() - t0
    for p in params:
        t = p["timings_seconds"]
        print(f"  {p['tag']}: {p['n_clusters']} clusters, train {t['train']:.1f} s, "
              f"projection {t['projection']:.1f} s, clustering {t['clustering']:.1f} s", flush=True)
    if args.train_only:
        return params

    label_files = [p["labels_file"] for p in params]
    n_cells = params[0]["n_samples"]
    results, t = stack(str(out), label_files, n_cells, args.sce_threshold, args.smooth_fraction,
                       weighting=args.weighting)
    x, names, w = load_features(args.input, n_cells)
    ranged_info = results["gsum_ranges"][2]
    anchored_info = results["anchored"][2]
    summary = {
        "args": vars(args),
        "realizations": [p["tag"] for p in params],
        "timings_seconds": {"ensemble_wall": round(t_train, 2), **{k: round(v, 3) for k, v in t.items()}},
        "anchored": {
            "medoid": anchored_info["medoid"],
            "group_sizes": [len(g) for g in anchored_info["groups"]],
            "clusters": cluster_table(results["anchored"][0], x, names, w),
        },
        "gsum_ranges": {
            "threshold": args.sce_threshold,
            "n_gsum_entries": len(ranged_info["gsum_values"]),
            "minimas": [int(m) for m in ranged_info["minimas"]],
            "cluster_ranges": [[int(a), int(b)] for a, b in ranged_info["cluster_ranges"]],
            "clusters": cluster_table(results["gsum_ranges"][0], x, names, w),
        },
    }
    (out / "stack_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"SCE Gsum {t['sce_gsum']:.2f} s, Gsum-range stack {t['gsum_ranges']:.2f} s, "
          f"anchored stack {t['anchored']:.2f} s", flush=True)
    for method in ("anchored", "gsum_ranges"):
        print(f"{method} stacked clusters:", flush=True)
        for r in summary[method]["clusters"]:
            med = ", ".join(f"{k} {v:+.2f}" for k, v in r["medians"].items())
            print(f"  cluster {r['cluster']}: rows {100 * r['row_fraction']:.2f} %, "
                  f"volume {100 * r['volume_fraction']:.2f} %, medians {med}", flush=True)
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
