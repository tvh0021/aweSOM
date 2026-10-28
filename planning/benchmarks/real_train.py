"""Online training on real BBH features: baseline fast_som (1069cff) vs the compiled trainer.

Data and parameters follow /Volumes/ExternalNVMe/etk-analysis/scripts/train_som.py: the first
n rows of the feature file, ``run_som.manual_scaling``, lattice from ``initialize_lattice(ratio
0.7)``, alpha_0 = 0.5, alpha_type "decay", sampling "uniform", train = n steps, merge 0.2.

1. Bitwise. The baseline draws its sample order from an unseeded ``np.random.default_rng()``,
   so it is not reproducible. For the bitwise check that call is patched to
   ``default_rng(42)`` (what the new trainer uses for seed 42), and both trainers start from
   the same initial lattice: ``lattice_history[0]`` of the pickle the baseline wrote for this
   configuration (passed as ``restart_lattice``, so neither draws an initial lattice). Then
   the final lattice, the U-matrix, every history snapshot, alpha, the merged clusters
   (merge 0.2) and the labels of all n points are compared. Timed ``--reps`` times each:
   baseline, new (history=True, the default), new (history=False).
2. Statistical. The unpatched baseline (its real behaviour) ``--seeds`` times against the
   new trainer with seeds 42, 43, ...; quantization and topographic error on the n training
   points, and the cluster count at merge 0.2. The stored baseline pickle is one more
   baseline sample.

usage: PYTHONPATH=<worktree>/src python planning/benchmarks/real_train.py [--cases 29x20,51x36] [--reps 3] [--seeds 3]
writes $AWESOM_RESULTS/train-<case>.json.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_common as rc  # noqa: E402

import aweSOM  # noqa: E402
from aweSOM.batch import quantization_error_fast, topographic_error_fast  # noqa: E402
from aweSOM.som import Lattice  # noqa: E402

CASES = {"29x20": ("bbh-29x20", 100_000), "51x36": ("bbh-51x36", 1_000_000)}
MERGE = 0.2


class patched_default_rng:
    """Make ``np.random.default_rng()`` (no seed) return ``default_rng(seed)``."""

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.orig = np.random.default_rng
        orig, seed = self.orig, self.seed
        np.random.default_rng = lambda *a, **k: orig(seed) if not a and not k else orig(*a, **k)

    def __exit__(self, *exc):
        np.random.default_rng = self.orig


def labels_of(som, data):
    idx = Lattice.best_match(som.lattice, data)
    p2 = Lattice.coordinate(idx.reshape(len(idx), 1), som.xdim)
    return Lattice.assign_cluster_to_data(p2, som.lattice_assigned_clusters)


def quality(w, data, xdim, ydim):
    return dict(qe=quantization_error_fast(w, data), te=topographic_error_fast(w, data, xdim, ydim))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="29x20,51x36")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    bsom = rc.load_baseline("src/aweSOM/som.py", "aweSOM_baseline_som")
    brun = rc.load_baseline("src/aweSOM/run_som.py", "aweSOM_baseline_run_som")
    BL = bsom.Lattice

    for case in args.cases.split(","):
        key, n = CASES[case]
        tag = rc.BBH_TAGS[key]
        with h5py.File(rc.BBH_H5, "r") as f:
            raw = f["features"][:n]
            names = [s.decode() if isinstance(s, bytes) else str(s) for s in f["names"][()]]
        scaled = brun.manual_scaling(raw)
        xdim, ydim = brun.initialize_lattice(scaled, 0.7)
        stored = pickle.load(open(os.path.join(rc.BBH_DIR, f"som_object.{tag}.pkl"), "rb"))
        L0 = np.array(stored.lattice_history[0], dtype=np.float64, copy=True)
        res = dict(case=case, n=n, F=scaled.shape[1], xdim=xdim, ydim=ydim, steps=n, alpha_0=0.5, merge=MERGE,
                   data_dtype=str(scaled.dtype), scaled_equals_pickle_data=bool(np.array_equal(scaled, stored.data_array)),
                   L0_source=f"som_object.{tag}.pkl lattice_history[0]", aweSOM=aweSOM.__file__, head=rc.git_head())
        rc.log(f"[{case}] n={n} lattice {xdim}x{ydim}, scaled == pickle data: {res['scaled_equals_pickle_data']}")

        def mk(cls):
            return cls(xdim=xdim, ydim=ydim, alpha_0=0.5, train=n, alpha_type="decay", sampling_type="uniform")

        def run_base():
            b = mk(BL)
            with patched_default_rng(42):
                b.train_lattice(scaled, names, restart_lattice=L0.copy())
            return b

        def run_new(history=True):
            s = mk(Lattice)
            s.train_lattice(scaled, names, restart_lattice=L0.copy(), history=history)
            return s

        # warm-up (JIT) for both trainers on a tiny problem
        with rc.quiet_fd():
            for cls in (BL, Lattice):
                w = cls(xdim=8, ydim=6, alpha_0=0.5, train=400, alpha_type="decay", sampling_type="uniform")
                w.train_lattice(scaled[:1000], names)

        tb, tbs, loadb, b = rc.timed(run_base, args.reps)
        rc.log(f"[{case}] baseline {tb:.2f} s ({tb / n * 1e6:.1f} us/step), load {loadb}")
        tn, tns, loadn, s = rc.timed(lambda: run_new(True), args.reps)
        rc.log(f"[{case}] new history=True {tn:.2f} s ({tn / n * 1e6:.2f} us/step), load {loadn}")
        th, ths, loadh, sh = rc.timed(lambda: run_new(False), args.reps)
        rc.log(f"[{case}] new history=False {th:.2f} s ({th / n * 1e6:.2f} us/step), load {loadh}")

        hist_equal = len(b.lattice_history) == len(s.lattice_history) and all(
            np.array_equal(x, y) for x, y in zip(b.lattice_history, s.lattice_history))
        umat_hist_equal = len(b.umat_history) == len(s.umat_history) and all(
            np.array_equal(x, y) for x, y in zip(b.umat_history, s.umat_history))
        with rc.quiet_fd():
            cb = b.assign_cluster_to_lattice(merge_cost=MERGE)
            cn = s.assign_cluster_to_lattice(merge_cost=MERGE)
            lb, ln = labels_of(b, scaled), labels_of(s, scaled)
        res["bitwise"] = dict(
            lattice_equal=bool(np.array_equal(b.lattice, s.lattice)),
            lattice_max_abs_diff=float(np.max(np.abs(b.lattice - s.lattice))),
            lattice_equal_history_false=bool(np.array_equal(b.lattice, sh.lattice)),
            umat_equal=bool(np.array_equal(b.umat, s.umat)),
            history_snapshots=len(s.lattice_history), lattice_history_equal=bool(hist_equal),
            umat_history_equal=bool(umat_hist_equal), alpha_equal=bool(b.alpha == s.alpha), epoch_equal=bool(b.epoch == s.epoch),
            clusters_equal=bool(np.array_equal(cb, cn)), n_clusters=int(len(np.unique(cn))),
            labels_equal=int(np.count_nonzero(lb == ln)), labels_total=int(n))
        res["timing"] = dict(
            baseline=dict(s_min=tb, s_all=tbs, load=loadb, us_per_step=tb / n * 1e6),
            new_history_true=dict(s_min=tn, s_all=tns, load=loadn, us_per_step=tn / n * 1e6, speedup=tb / tn),
            new_history_false=dict(s_min=th, s_all=ths, load=loadh, us_per_step=th / n * 1e6, speedup=tb / th))
        rc.log(f"[{case}] bitwise: {res['bitwise']}")

        # --- statistical comparison: unpatched baseline vs seeded new trainer
        stats = dict(baseline=[], new=[])
        q = quality(stored.lattice, scaled, xdim, ydim)
        with rc.quiet_fd():
            sp = BL(xdim=xdim, ydim=ydim)
            sp.lattice, sp.umat = stored.lattice, stored.umat
            q["n_clusters"] = int(len(np.unique(sp.assign_cluster_to_lattice(merge_cost=MERGE))))
        res["stored_pickle"] = q
        for k in range(args.seeds):
            t0 = time.perf_counter()
            with rc.quiet_fd():
                b = mk(BL)
                b.train_lattice(scaled, names)
                nc = len(np.unique(b.assign_cluster_to_lattice(merge_cost=MERGE)))
            tb_ = time.perf_counter() - t0
            st = dict(run=k, s=tb_, load=rc.uptime(), n_clusters=int(nc), **quality(b.lattice, scaled, xdim, ydim))
            stats["baseline"].append(st)
            t0 = time.perf_counter()
            with rc.quiet_fd():
                s = mk(Lattice)
                s.seed = 42 + k
                s.train_lattice(scaled, names, history=False)
                nc = len(np.unique(s.assign_cluster_to_lattice(merge_cost=MERGE)))
            tn_ = time.perf_counter() - t0
            sn = dict(seed=42 + k, s=tn_, load=rc.uptime(), n_clusters=int(nc), **quality(s.lattice, scaled, xdim, ydim))
            stats["new"].append(sn)
            rc.log(f"[{case}] stats {k}: baseline qe {st['qe']:.5f} te {st['te']:.5f} nc {st['n_clusters']} | "
                   f"new qe {sn['qe']:.5f} te {sn['te']:.5f} nc {sn['n_clusters']}")
        # reproducibility of the new trainer: same seed twice
        with rc.quiet_fd():
            r1, r2 = mk(Lattice), mk(Lattice)
            r1.train_lattice(scaled, names, history=False)
            r2.train_lattice(scaled, names, history=False)
        stats["new_seed42_repeat_bitwise"] = bool(np.array_equal(r1.lattice, r2.lattice))
        for side in ("baseline", "new"):
            for m in ("qe", "te"):
                v = np.array([r[m] for r in stats[side]])
                stats[f"{side}_{m}_mean"], stats[f"{side}_{m}_std"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
        res["statistical"] = stats
        print(rc.save_json(f"train-{case}.json", res), flush=True)


if __name__ == "__main__":
    main()
