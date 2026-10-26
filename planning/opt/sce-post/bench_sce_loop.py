"""C1: SCE ensemble loop, baseline (commit 9f29e9b sce.py) vs this worktree's sce.py.

Runs both back to back, interleaved, on the same label maps, checks that
multimap_mappings.txt and every mask file are bitwise identical, prints one JSON line.

usage: PYTHONPATH=<worktree>/src python bench_sce_loop.py [maps_dir] [reps]
"""
import contextlib, glob, hashlib, importlib.util, io, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MAPS = sys.argv[1] if len(sys.argv) > 1 else "/Users/tvh0021/git_repos/aweSOM-research/planning/runs/plasma-2026-10-26"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BASE_COMMIT = "9f29e9b"


def load_module(name, src):
    path = os.path.join(tempfile.mkdtemp(dir=os.path.join(ROOT, "tmp")), name + ".py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


os.makedirs(os.path.join(ROOT, "tmp"), exist_ok=True)
base = load_module("sce_base", subprocess.check_output(["git", "-C", ROOT, "show", f"{BASE_COMMIT}:src/aweSOM/sce.py"], text=True))
import aweSOM.sce as cand  # noqa: E402

assert os.path.abspath(cand.__file__).startswith(ROOT), cand.__file__

work = tempfile.mkdtemp(dir=os.path.join(ROOT, "tmp"))
for f in sorted(glob.glob(os.path.join(MAPS, "labels.*.npy"))):
    os.symlink(f, os.path.join(work, os.path.basename(f)))
os.chdir(work)
files = sorted(glob.glob("labels.*.npy"))
nids = base.find_number_of_clusters(files)
dims = np.array(np.load(files[0], mmap_mode="r").shape)
if dims.size == 1:
    n = round(dims[0] ** (1 / 3)); dims = np.array([n, n, n])


def run(mod, out):
    shutil.rmtree(out, ignore_errors=True); os.mkdir(out)
    open(out + "/multimap_mappings.txt", "w").close()
    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        mod.loop_over_all_clusters(files, nids, dims, out)
    return time.perf_counter() - t


def digest(out):
    h = {}
    for p in sorted(os.listdir(out)):
        with open(os.path.join(out, p), "rb") as f:
            h[p] = hashlib.sha256(f.read()).hexdigest()
    return h


# find_number_of_clusters (distinct ids per file), interleaved
tfb, tfc = [], []
for r in range(REPS):
    t = time.perf_counter(); nb_ = base.find_number_of_clusters(files); tfb.append(time.perf_counter() - t)
    t = time.perf_counter(); nc_ = cand.find_number_of_clusters(files); tfc.append(time.perf_counter() - t)
nfind_same = bool(np.array_equal(nb_, nc_))

# warm-up (numba JIT / cache) on a two-run subset
with contextlib.redirect_stdout(io.StringIO()):
    os.mkdir("warm"); open("warm/multimap_mappings.txt", "w").close()
    cand.loop_over_all_clusters(files[:2], nids[:2], dims, "warm")
tb, tc = [], []
for r in range(REPS):
    tb.append(run(base, "out_base"))
    tc.append(run(cand, "out_cand"))
identical = digest("out_base") == digest("out_cand")
n_out = len(os.listdir("out_cand"))
os.chdir(ROOT)
shutil.rmtree(work)
mb, mc = float(np.median(tb)), float(np.median(tc))
print(json.dumps({"claim": "C1", "maps": len(files), "cells": int(np.prod(dims)), "clusters_total": int(np.sum(nids)),
                  "threads": int(__import__("numba").get_num_threads()), "reps": REPS,
                  "baseline_s": [round(x, 3) for x in tb], "candidate_s": [round(x, 3) for x in tc],
                  "baseline_median_s": round(mb, 3), "candidate_median_s": round(mc, 3), "ratio": round(mb / mc, 2),
                  "output_files": n_out, "bitwise_identical": identical,
                  "find_number_of_clusters_baseline_s": round(float(np.median(tfb)), 3),
                  "find_number_of_clusters_candidate_s": round(float(np.median(tfc)), 3), "find_number_of_clusters_same": nfind_same}))
