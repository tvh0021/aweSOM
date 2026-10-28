"""Clump post-processing on the plasma current-sheet arrays: baseline vs fast.

Inputs: examples/plasma-turbulence/merged_sheets/{positive,negative,merged_double}_sheets_array.npy
(128^3, clump id per cell, -1 outside the sheets).

1. ``get_clumps`` from the baseline (1069cff) example script (pure-Python DFS) vs the new
   ``label_clumps`` (scipy.ndimage per value; cc3d when installed), on each array's sheet mask
   (``arr >= 0``) with connectivity 6, 10 and 26, and on the raw id arrays with connectivity 6.
   Equality: the clump id of every cell from ``label_clumps`` equals the index of its clump in
   the baseline's list.
2. ``merge_double_sheets`` (baseline) vs ``merge_double_sheets_fast`` on sub-cubes: dicts
   built as in plasma.ipynb with the baseline ``get_clumps`` (connectivity 10, largest clump
   = background dropped) from double = merged-double cells that are in neither the positive
   nor the negative array, positive and negative = those arrays. Equality: same keys in the
   same order, same cell sets, same value types, same in-place deletions.

usage: PYTHONPATH=<worktree>/src python planning/benchmarks/real_clumps.py [--reps 3] [--merge-sizes 24,32,48]
writes $AWESOM_RESULTS/clumps.json.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_common as rc  # noqa: E402

sys.path.insert(0, os.path.join(rc.REPO, "examples", "plasma-turbulence"))
import combine_clumps as cc  # noqa: E402  (worktree version: label_clumps, merge_double_sheets_fast)

SHEETS = os.path.join(rc.REPO, "examples", "plasma-turbulence", "merged_sheets")


def clumps_to_ids(clumps, shape):
    ids = np.full(shape, -1, dtype=np.int32)
    for k, cl in enumerate(clumps):
        ids[tuple(np.array(cl).T)] = k
    return ids


def canon(d):
    return {k: frozenset(tuple(int(c) for c in t) for t in v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--merge-sizes", default="24,32,48")
    args = ap.parse_args()
    base = rc.load_baseline("examples/plasma-turbulence/combine_clumps.py", "baseline_combine_clumps")
    arrays = {k: np.load(os.path.join(SHEETS, f"{k}_sheets_array.npy")) for k in ("positive", "negative", "merged_double")}
    try:
        import cc3d  # noqa: F401
        have_cc3d = True
    except ImportError:
        have_cc3d = False
    res = dict(head=rc.git_head(), cc3d_installed=have_cc3d, label_clumps=[], merge=[])

    cc.label_clumps(arrays["positive"][:8, :8, :8], cc.generate_neighbors(6))  # warm-up (numba)
    for name, arr in arrays.items():
        variants = [("mask", (arr >= 0).astype(np.int64), c) for c in (6, 10, 26)] + [("ids", arr, 6)]
        for kind, a, conn in variants:
            deltas = cc.generate_neighbors(conn)
            tb, tbs, lb, clumps = rc.timed(lambda: base.get_clumps(a, deltas), args.reps)
            tn, tns, ln, (ids, k) = rc.timed(lambda: cc.label_clumps(a, deltas), args.reps)
            same = bool(np.array_equal(clumps_to_ids(clumps, a.shape), ids)) and k == len(clumps)
            row = dict(array=name, input=kind, conn=conn, shape=a.shape, clumps=len(clumps), identical=same,
                       baseline=dict(s_min=tb, s_all=tbs, load=lb), new=dict(s_min=tn, s_all=tns, load=ln), speedup=tb / tn)
            res["label_clumps"].append(row)
            rc.log(f"[clumps] {name} {kind} conn{conn}: {len(clumps)} clumps, identical {same}, {tb:.2f} s -> {tn:.4f} s ({tb / tn:.0f}x), load {lb}")

    pos, neg, dbl = arrays["positive"], arrays["negative"], arrays["merged_double"]
    double_only = (dbl >= 0) & (pos < 0) & (neg < 0)
    for n in [int(s) for s in args.merge_sizes.split(",")]:
        comb = np.zeros((n, n, n), dtype=np.int64)
        comb[(neg[:n, :n, :n] >= 0)] = 1
        comb[(pos[:n, :n, :n] >= 0)] = 2
        comb[double_only[:n, :n, :n]] = 3
        clumps = base.get_clumps(comb, base.generate_neighbors(10))
        clumps.sort(key=len, reverse=True)
        D, P, N = {}, {}, {}
        for k, cl in list(enumerate(clumps))[1:]:
            v = comb[cl[0]]
            (D if v == 3 else P if v == 2 else N if v == 1 else {})[k] = cl
        snapshot = (D, P, N)
        holder = {}

        def run_base():
            a = copy.deepcopy(snapshot)
            holder["b"] = (base.merge_double_sheets(*a), a)

        def run_new():
            a = copy.deepcopy(snapshot)
            holder["n"] = (cc.merge_double_sheets_fast(*a), a)

        tb, tbs, lb, _ = rc.timed(run_base, args.reps)
        tn, tns, ln, _ = rc.timed(run_new, args.reps)
        (ob, ab), (on, an) = holder["b"], holder["n"]
        same = (list(ob) == list(on) and canon(ob) == canon(on)
                and [type(v) for v in ob.values()] == [type(v) for v in on.values()]
                and list(ab[1]) == list(an[1]) and list(ab[2]) == list(an[2]))
        row = dict(n=n, double=len(D), positive=len(P), negative=len(N),
                   merged=sum(isinstance(v, set) for v in ob.values()), identical=bool(same),
                   baseline=dict(s_min=tb, s_all=tbs, load=lb), new=dict(s_min=tn, s_all=tns, load=ln), speedup=tb / tn)
        res["merge"].append(row)
        rc.log(f"[merge] n={n}: D/P/N {len(D)}/{len(P)}/{len(N)}, merged {row['merged']}, identical {same}, "
               f"{tb:.2f} s -> {tn:.4f} s ({tb / tn:.0f}x), load {lb}")
    print(rc.save_json("clumps.json", res), flush=True)


if __name__ == "__main__":
    main()
