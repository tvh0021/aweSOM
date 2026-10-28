"""SCE on real plasma label maps: baseline loop (1069cff sce.py) vs the contingency-table loop.

Inputs: the ``labels.*.npy`` maps (128^3 cells each) of the plasma ensemble in
planning/runs/plasma-2026-10-26 of the research checkout. ``--maps K`` takes the first K
files in sorted order (the "small ensemble"); ``--full`` adds one run over every map.
Both loops write ``multimap_mappings.txt`` and one ``mask-<run>-id<cid>.npy`` per cluster;
every output file is compared by SHA-256. Each loop is timed ``--reps`` times (min reported).
The opt-in mask-free mode (``write_masks=False``) is timed too.

Then the make_sce_clusters step on the new loop's output: Savitzky-Golay split at
threshold -0.015 (the CLI default), ``combine_separated_clusters`` (reads the mask files;
the baseline version crashes on NumPy 2 because of ``np.reshape(newshape=)``, ISSUES.md
item 3) vs ``combine_separated_clusters_from_labels`` (reads the label maps), compared bitwise.

usage: PYTHONPATH=<worktree>/src python planning/benchmarks/real_sce.py [--maps 8] [--reps 3] [--full]
writes $AWESOM_RESULTS/sce.json (work files under $AWESOM_RESULTS/sce-work, removed at the end).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import shutil
import sys

import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_common as rc  # noqa: E402

import aweSOM.make_sce_clusters as msc  # noqa: E402
import aweSOM.sce as sce  # noqa: E402


def digest(folder):
    out = {}
    for p in sorted(os.listdir(folder)):
        with open(os.path.join(folder, p), "rb") as f:
            out[p] = hashlib.sha256(f.read()).hexdigest()
    return out


def run_loop(mod, files, nids, dims, out, **kw):
    shutil.rmtree(out, ignore_errors=True)
    os.mkdir(out)
    open(os.path.join(out, "multimap_mappings.txt"), "w").close()
    mod.loop_over_all_clusters(files, nids, dims, out, **kw)


def compare(files, nids, dims, base, reps, tag):
    res = dict(maps=len(files), clusters_total=int(np.sum(nids)), cells=int(np.prod(dims)))
    tb, tbs, lb, _ = rc.timed(lambda: run_loop(base, files, nids, dims, "out_base"), reps)
    rc.log(f"[sce {tag}] baseline {tb:.2f} s, load {lb}")
    run_loop(sce, files[:2], nids[:2], dims, "out_warm")  # numba warm-up
    tn, tns, ln, _ = rc.timed(lambda: run_loop(sce, files, nids, dims, "out_new"), reps)
    rc.log(f"[sce {tag}] new {tn:.2f} s, load {ln}")
    tm, tms, lm, _ = rc.timed(lambda: run_loop(sce, files, nids, dims, "out_nomask", write_masks=False), reps)
    db, dn = digest("out_base"), digest("out_new")
    mm_base = open("out_base/multimap_mappings.txt").read()
    res.update(
        baseline=dict(s_min=tb, s_all=tbs, load=lb), new=dict(s_min=tn, s_all=tns, load=ln, speedup=tb / tn),
        new_no_masks=dict(s_min=tm, s_all=tms, load=lm, speedup=tb / tm),
        output_files=len(dn), all_files_identical=db == dn,
        multimap_identical=db["multimap_mappings.txt"] == dn["multimap_mappings.txt"],
        multimap_identical_no_masks=open("out_nomask/multimap_mappings.txt").read() == mm_base,
        differing_files=[k for k in db if db.get(k) != dn.get(k)][:10])
    rc.log(f"[sce {tag}] identical files: {res['all_files_identical']} ({len(dn)} files), speedup {tb / tn:.1f}x")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--full", action="store_true", help="also run every map once (baseline is slow)")
    args = ap.parse_args()

    base = rc.load_baseline("src/aweSOM/sce.py", "aweSOM_baseline_sce")
    assert not base.USE_JAX, "baseline must run its NumPy path"
    work = os.path.join(rc.RESULTS, "sce-work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    all_maps = sorted(glob.glob(os.path.join(rc.PLASMA_RUNS, "labels.*.npy")))
    for f in all_maps:
        os.symlink(f, os.path.join(work, os.path.basename(f)))
    cwd = os.getcwd()
    os.chdir(work)
    try:
        files = sorted(glob.glob("labels.*.npy"))
        nids_all = sce.find_number_of_clusters(files)
        assert np.array_equal(nids_all, base.find_number_of_clusters(files))
        n = round(np.load(files[0], mmap_mode="r").size ** (1 / 3))
        dims = np.array([n, n, n])
        res = dict(maps_dir=rc.PLASMA_RUNS, head=rc.git_head())
        k = args.maps
        res["small"] = compare(files[:k], nids_all[:k], dims, base, args.reps, f"{k} maps")
        res["small"]["files"] = files[:k]

        # make_sce_clusters on the new loop's output of the small ensemble
        gsum, map_list = msc.get_gsum_values("out_new/multimap_mappings.txt")
        win = len(gsum) // 10
        if win > 4:
            sm = savgol_filter(gsum, win, 4, deriv=0)
            deriv = savgol_filter(sm, win, 4, deriv=1) / sm
            ranges, _ = msc.get_sce_cluster_separation(deriv, -0.015)
        else:
            ranges = []
        if not ranges:  # too few clusters for the derivative split: one range per 1/4 of the ranking
            q = np.linspace(0, len(map_list), 5).astype(int)
            ranges = [[int(a), int(b)] for a, b in zip(q[:-1], q[1:])]
        dl = [int(d) for d in dims]
        with rc.quiet_fd():
            msc.combine_separated_clusters(map_list[:2], [[0, 2]], dl, "out_new")
            msc.combine_separated_clusters_from_labels(map_list[:2], [[0, 2]], dl, ".")
        tm, tms, lm, a = rc.timed(lambda: msc.combine_separated_clusters(map_list, ranges, dl, "out_new"), args.reps)
        tl, tls, ll, b = rc.timed(lambda: msc.combine_separated_clusters_from_labels(map_list, ranges, dl, "."), args.reps)
        res["combine"] = dict(ranges=ranges, from_masks=dict(s_min=tm, s_all=tms, load=lm),
                              from_labels=dict(s_min=tl, s_all=tls, load=ll, speedup=tm / tl),
                              bitwise_identical=bool(np.array_equal(a, b)) and a.dtype == b.dtype,
                              max_abs_diff=float(np.max(np.abs(a.astype(np.float64) - b))))
        rc.log(f"[combine] masks {tm:.2f} s, labels {tl:.2f} s, identical {res['combine']['bitwise_identical']}")

        if args.full:
            res["full"] = compare(files, nids_all, dims, base, 1, f"all {len(files)} maps")
        print(rc.save_json("sce.json", res), flush=True)
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
