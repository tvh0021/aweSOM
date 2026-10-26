"""C2: SCE without mask files. Baseline = loop_over_all_clusters(write_masks=True) + make_sce_clusters
.combine_separated_clusters (reads the mask files); candidate = loop_over_all_clusters(write_masks=False)
+ combine_separated_clusters_from_labels. Both use this worktree's sce.py (C1 applied), interleaved.
Checks multimap_mappings.txt and the combined map are bitwise identical. Prints one JSON line.

usage: PYTHONPATH=<worktree>/src python bench_sce_combine.py [maps_dir] [reps]
"""
import contextlib, glob, io, json, os, shutil, sys, tempfile, time
import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MAPS = sys.argv[1] if len(sys.argv) > 1 else "/Users/tvh0021/git_repos/aweSOM-research/planning/runs/plasma-2026-10-26"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
with contextlib.redirect_stdout(io.StringIO()):
    import aweSOM.sce as sce
    import aweSOM.make_sce_clusters as msc
assert os.path.abspath(sce.__file__).startswith(ROOT), sce.__file__

os.makedirs(os.path.join(ROOT, "tmp"), exist_ok=True)
work = tempfile.mkdtemp(dir=os.path.join(ROOT, "tmp"))
for f in sorted(glob.glob(os.path.join(MAPS, "labels.*.npy"))):
    os.symlink(f, os.path.join(work, os.path.basename(f)))
os.chdir(work)
files = sorted(glob.glob("labels.*.npy"))
nids = sce.find_number_of_clusters(files)
n = round(np.load(files[0], mmap_mode="r").size ** (1 / 3))
dims = [n, n, n]


def ranges_for(mapping):
    gsum, map_list = msc.get_gsum_values(mapping)
    sm = savgol_filter(gsum, len(gsum) // 10, 4, deriv=0)
    d = savgol_filter(sm, len(gsum) // 10, 4, deriv=1) / sm
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            cr, _ = msc.get_sce_cluster_separation(d, -0.015)
        except IndexError:
            cr = []
    if len(cr) < 2:  # threshold finds no split on this ensemble: use 7 equal rank ranges instead
        edges = np.linspace(0, len(gsum), 8).astype(int)
        cr = [[int(a), int(b)] for a, b in zip(edges[:-1], edges[1:])]
    return map_list, cr


def baseline(out):
    shutil.rmtree(out, ignore_errors=True); os.mkdir(out); open(out + "/multimap_mappings.txt", "w").close()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        sce.loop_over_all_clusters(files, nids, np.array(dims), out)
    t1 = time.perf_counter()
    map_list, cr = ranges_for(out + "/multimap_mappings.txt")
    with contextlib.redirect_stdout(io.StringIO()):
        comb = msc.combine_separated_clusters(map_list, cr, dims, out)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1, comb, open(out + "/multimap_mappings.txt").read(), cr


def candidate(out):
    shutil.rmtree(out, ignore_errors=True); os.mkdir(out); open(out + "/multimap_mappings.txt", "w").close()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        sce.loop_over_all_clusters(files, nids, np.array(dims), out, write_masks=False)
    t1 = time.perf_counter()
    map_list, cr = ranges_for(out + "/multimap_mappings.txt")
    comb = msc.combine_separated_clusters_from_labels(map_list, cr, dims, ".")
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1, comb, open(out + "/multimap_mappings.txt").read(), cr


candidate("warm")
B, C = [], []
for r in range(REPS):
    b = baseline("out_base"); B.append(b[:2])
    c = candidate("out_cand"); C.append(c[:2])
same_comb = b[2].dtype == c[2].dtype and b[2].shape == c[2].shape and b[2].tobytes() == c[2].tobytes()
same_map = b[3] == c[3]
mask_bytes = sum(os.path.getsize(os.path.join("out_base", p)) for p in os.listdir("out_base") if p.startswith("mask-"))
os.chdir(ROOT); shutil.rmtree(work)
tb = [sum(x) for x in B]; tc = [sum(x) for x in C]
med = lambda v: float(np.median(v))
print(json.dumps({"claim": "C2", "maps": len(files), "cells": n ** 3, "clusters_total": int(np.sum(nids)),
                  "combined_clusters": len(c[4]), "reps": REPS, "mask_bytes_written_baseline": mask_bytes,
                  "baseline_loop_s": round(med([x[0] for x in B]), 3), "baseline_combine_s": round(med([x[1] for x in B]), 3),
                  "candidate_loop_s": round(med([x[0] for x in C]), 3), "candidate_combine_s": round(med([x[1] for x in C]), 3),
                  "baseline_total_s": [round(x, 3) for x in tb], "candidate_total_s": [round(x, 3) for x in tc],
                  "ratio_total": round(med(tb) / med(tc), 2),
                  "ratio_combine": round(med([x[1] for x in B]) / med([x[1] for x in C]), 2),
                  "mappings_identical": same_map, "combined_map_bitwise_identical": same_comb}))
