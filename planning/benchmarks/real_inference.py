"""Inference on real trained lattices: baseline (1069cff) vs every new backend.

For each lattice + data pair the ground truth is what the baseline itself produced when the
pickle was written: ``projection_1d`` (BMU of every training point, from the baseline
``Lattice.best_match``) and, for the BBH runs, the saved ``labels.*.npy``. The baseline
pipeline (``best_match`` -> ``coordinate`` -> ``assign_cluster_to_data``) is also re-run
live on the first ``--baseline-n`` points for timing and to confirm it reproduces the pickle.

Per backend (numba, screen [default], kdtree, jax-cpu) the script reports node-index and
label equality over all points, max |distance - reference| (reference = NumPy float64
distance to the baseline's node), and throughput: min of ``--reps`` runs on all points with
all threads, plus a single-thread rate on ``--onethread-n`` points, which gives the projected
time for 1e10 points on a 48-core node by linear scaling (a projection, not a measurement).

usage: PYTHONPATH=<worktree>/src python planning/benchmarks/real_inference.py \
           [--cases plasma,bbh-29x20,bbh-51x36,bbh-91x64] [--reps 3] [--baseline-n 2097152]
writes $AWESOM_RESULTS/inference-<case>.json and prints a summary line per case to stderr.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_common as rc  # noqa: E402

import numba  # noqa: E402

import aweSOM  # noqa: E402
from aweSOM import inference as inf  # noqa: E402
from aweSOM.model import Scaler, SOMModel  # noqa: E402
from aweSOM.som import Lattice  # noqa: E402

N_PROJ = 1e10
CORES = 48


def ref_dist(lattice, data, idx, chunk=1 << 20):
    out = np.empty(len(idx))
    for s in range(0, len(idx), chunk):
        e = min(s + chunk, len(idx))
        d = lattice[idx[s:e]] - data[s:e].astype(np.float64)
        out[s:e] = np.sqrt(np.sum(d**2, axis=1))
    return out


def load_case(case):
    if case == "plasma":
        som = pickle.load(open(rc.PLASMA_PKL, "rb"))
        return som, None, None, None
    tag = rc.BBH_TAGS[case]
    som = pickle.load(open(os.path.join(rc.BBH_DIR, f"som_object.{tag}.pkl"), "rb"))
    labels = np.load(os.path.join(rc.BBH_DIR, f"labels.{tag}.npy"))
    import json

    summary = json.load(open(os.path.join(rc.BBH_DIR, f"summary.{tag}.json")))
    return som, labels, summary["merge_threshold"], summary["n_samples"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="plasma,bbh-29x20,bbh-51x36,bbh-91x64")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--baseline-n", type=int, default=2_097_152)
    ap.add_argument("--onethread-n", type=int, default=200_000)
    ap.add_argument("--backends", default="numba,screen,kdtree,jax")
    args = ap.parse_args()

    bsom = rc.load_baseline("src/aweSOM/som.py", "aweSOM_baseline_som")
    BL = bsom.Lattice
    nthreads = numba.get_num_threads()
    backends = args.backends.split(",")

    for case in args.cases.split(","):
        t_case = time.perf_counter()
        som, saved_labels, merge_thr, n_samples = load_case(case)
        lattice = np.ascontiguousarray(som.lattice, dtype=np.float64)
        data = np.ascontiguousarray(som.data_array)
        xdim, ydim = som.xdim, som.ydim
        N, F = data.shape
        clusters = np.asarray(som.lattice_assigned_clusters)
        node_cluster = clusters.ravel(order="F").astype(np.int32)
        proj = np.asarray(som.projection_1d).ravel().astype(np.int64)
        res = dict(case=case, N=N, F=F, nodes=xdim * ydim, xdim=xdim, ydim=ydim, data_dtype=str(data.dtype),
                   threads=nthreads, aweSOM=aweSOM.__file__, head=rc.git_head(), merge_threshold=merge_thr)
        rc.log(f"[{case}] N={N} F={F} nodes={xdim * ydim} dtype={data.dtype}")

        # --- reference labels: the baseline's own stored output
        ref_labels = node_cluster[proj]
        if saved_labels is not None:
            res["pickle_projection_labels_equal_saved_labels_npy"] = int(np.count_nonzero(ref_labels == saved_labels))
        ref_d = ref_dist(lattice, data, proj)

        # --- cluster assignment on the lattice: baseline vs new code vs stored
        thr = merge_thr if merge_thr is not None else 0.25
        b = BL(xdim=xdim, ydim=ydim)
        b.lattice, b.umat = lattice.copy(), np.asarray(som.umat).copy()
        n = Lattice(xdim=xdim, ydim=ydim)
        n.lattice, n.umat = lattice.copy(), np.asarray(som.umat).copy()
        with rc.quiet_fd():
            cb = b.assign_cluster_to_lattice(merge_cost=thr)
            cn = n.assign_cluster_to_lattice(merge_cost=thr)
        res["assign_cluster_to_lattice"] = dict(
            merge_cost=thr, baseline_equals_new=bool(np.array_equal(cb, cn)),
            baseline_equals_stored=bool(np.array_equal(cb, clusters)), n_clusters=int(len(np.unique(cn))))

        # --- baseline pipeline, live, on a prefix
        nb = min(N, args.baseline_n)
        sub = data[:nb]

        def baseline_pipeline(x):
            idx = BL.best_match(lattice, x)
            p2 = BL.coordinate(idx.reshape(len(idx), 1), xdim)
            return idx, BL.assign_cluster_to_data(p2, clusters)

        with rc.quiet_fd():
            baseline_pipeline(data[:2000])  # JIT warm-up
        tmin, ts, load, (bidx, blab) = rc.timed(lambda: baseline_pipeline(sub), args.reps)
        res["baseline"] = dict(n=nb, s_min=tmin, s_all=ts, load=load, pts_per_s=nb / tmin,
                               node_equal_pickle=int(np.count_nonzero(bidx.ravel().astype(np.int64) == proj[:nb])),
                               label_equal_ref=int(np.count_nonzero(blab == ref_labels[:nb])))
        rc.log(f"[{case}] baseline {nb / tmin:.3e} pts/s (min of {args.reps}), load {load}")

        # --- new: Lattice.best_match (legacy API, now delegating) on the same prefix
        with rc.quiet_fd():
            Lattice.best_match(lattice, data[:2000])
        tmin, ts, load, nidx = rc.timed(lambda: Lattice.best_match(lattice, sub), args.reps)
        res["lattice_best_match_new"] = dict(n=nb, s_min=tmin, s_all=ts, load=load, pts_per_s=nb / tmin,
                                             node_equal_pickle=int(np.count_nonzero(nidx.ravel().astype(np.int64) == proj[:nb])))

        # --- new predict, every backend, all points
        res["backends"] = {}
        node_fns = dict(numba=inf.best_match_nodes, screen=inf.best_match_nodes_screen,
                        kdtree=inf.best_match_nodes_kdtree, jax=inf.best_match_nodes_jax)
        for be in backends:
            kw = {} if be == "jax" else {}
            with rc.quiet_fd():
                inf.predict(lattice, node_cluster, data[:4096], backend=be, **kw)  # warm-up / compile
            tmin, ts, load, (lab, dist) = rc.timed(
                lambda: inf.predict(lattice, node_cluster, data, return_distance=True, backend=be), args.reps)
            with rc.quiet_fd():
                idx, _ = node_fns[be](lattice, data)
            r = dict(s_min=tmin, s_all=ts, load=load, pts_per_s=N / tmin,
                     node_equal=int(np.count_nonzero(idx.astype(np.int64) == proj)),
                     label_equal=int(np.count_nonzero(lab == ref_labels)),
                     max_abs_dist_diff=float(np.max(np.abs(dist - ref_d))),
                     n_dist_diff_nonzero=int(np.count_nonzero(dist != ref_d)))
            if be == "screen":
                with rc.quiet_fd():
                    _, _, nfb = inf.best_match_nodes_screen(lattice, data, return_fallbacks=True)
                r["float64_fallbacks"] = int(nfb)
            if r["node_equal"] != N:
                bad = np.flatnonzero(idx.astype(np.int64) != proj)
                d_ref = np.sum((lattice[proj[bad]] - data[bad].astype(np.float64)) ** 2, axis=1)
                d_new = np.sum((lattice[idx[bad]] - data[bad].astype(np.float64)) ** 2, axis=1)
                r["node_mismatch_examples"] = [dict(i=int(i), ref=int(proj[i]), new=int(idx[i]), d2_ref=float(a), d2_new=float(c))
                                               for i, a, c in list(zip(bad, d_ref, d_new))[:5]]
            res["backends"][be] = r
            rc.log(f"[{case}] {be}: {N / tmin:.3e} pts/s, nodes equal {r['node_equal']}/{N}, labels equal {r['label_equal']}/{N}, "
                   f"max|dd| {r['max_abs_dist_diff']:.3g}, load {load}")

        # --- SOMModel on raw features (BBH only): frozen manual scaler fitted on the training prefix
        if case.startswith("bbh"):
            import h5py

            with h5py.File(rc.BBH_H5, "r") as f:
                raw = f["features"][:n_samples]
            run_som = rc.load_baseline("src/aweSOM/run_som.py", "aweSOM_baseline_run_som")
            base_scaled = run_som.manual_scaling(raw)
            scaler = Scaler.fit(raw, kind="manual")
            new_scaled = scaler.transform(raw)
            lt = Lattice(xdim=xdim, ydim=ydim)
            lt.lattice, lt.lattice_assigned_clusters = lattice, clusters
            lt.data_array = data
            with rc.quiet_fd():
                model = SOMModel.from_lattice(lt, scaled_training_data=data, scaler=scaler, merge_cost=thr)
            tmin, ts, load, mlab = rc.timed(lambda: model.predict(raw), args.reps)
            res["somodel_raw"] = dict(
                raw_dtype=str(raw.dtype), baseline_manual_scaling_equals_pickle=bool(np.array_equal(base_scaled, data)),
                scaler_vs_pickle_max_abs=float(np.max(np.abs(new_scaled - data.astype(np.float64)))),
                label_equal=int(np.count_nonzero(mlab == ref_labels)), s_min=tmin, s_all=ts, load=load, pts_per_s=N / tmin)
            rc.log(f"[{case}] SOMModel.predict(raw): labels equal {res['somodel_raw']['label_equal']}/{N}, "
                   f"scaler max|d| {res['somodel_raw']['scaler_vs_pickle_max_abs']:.3g}")

        # --- single-thread rates and the 1e10-point projection
        n1 = min(N, args.onethread_n)
        x1 = np.ascontiguousarray(data[:n1])
        one = {}
        numba.set_num_threads(1)
        try:
            for name, fn in [
                ("baseline", lambda: baseline_pipeline(x1)),
                ("numba", lambda: inf.predict(lattice, node_cluster, x1, backend="numba")),
                ("screen", lambda: inf.predict(lattice, node_cluster, x1, backend="screen")),
                ("kdtree", lambda: node_cluster[inf.best_match_nodes_kdtree(lattice, x1, workers=1)[0]]),
            ]:
                tmin, ts, load, _ = rc.timed(fn, args.reps)
                rate = n1 / tmin
                one[name] = dict(n=n1, s_min=tmin, s_all=ts, load=load, pts_per_s=rate,
                                 projected_1e10_on_48_cores_s=N_PROJ / (CORES * rate))
                rc.log(f"[{case}] 1 thread {name}: {rate:.3e} pts/s -> 1e10 on 48 cores ~ {N_PROJ / (CORES * rate) / 60:.1f} min (projection)")
        finally:
            numba.set_num_threads(nthreads)
        res["one_thread"] = one
        res["wall_s"] = time.perf_counter() - t_case
        print(rc.save_json(f"inference-{case}.json", res), flush=True)
        del som, data


if __name__ == "__main__":
    main()
