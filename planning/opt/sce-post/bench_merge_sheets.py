"""C4: combine_clumps.merge_double_sheets (pure Python, `tuple in list` membership) vs
merge_double_sheets_fast (ordinal-volume gathers). Inputs built as in plasma.ipynb from a real 128^3
SOM label map (double = labels 1|2, positive = 4, negative = 0, clumps with get_clumps/connectivity 10,
largest clump dropped), on sub-cubes. Checks merged dicts and the in-place deletions are identical.
Prints one JSON line.

usage: PYTHONPATH=<worktree>/src python bench_merge_sheets.py [sizes, default 24,32,40] [label_file]
"""
import contextlib, copy, io, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "examples", "plasma-turbulence"))
import combine_clumps as cc  # noqa: E402

SIZES = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "24,32,40").split(",")]
LAB = sys.argv[2] if len(sys.argv) > 2 else "/Users/tvh0021/git_repos/aweSOM/examples/plasma-turbulence/som_object.fiducial-5000-64-37-0.1-2097152-1u.pkl"
if LAB.endswith(".pkl"):
    LAB = "/Users/tvh0021/git_repos/aweSOM-research/planning/runs/plasma-2026-10-26/fiducial_labels_thr025.npy"
full = np.load(LAB).reshape(128, 128, 128).astype(np.int64)


def build(n):
    lab = full[:n, :n, :n]
    comb = lab.copy(); comb[lab == 1] = 2
    clumps = cc.get_clumps(comb, cc.generate_neighbors(10))
    clumps.sort(key=len, reverse=True)
    D, P, N = {}, {}, {}
    for k, cl in list(enumerate(clumps))[1:]:
        v = comb[cl[0]]
        (D if v == 2 else P if v == 4 else N if v == 0 else {})[k] = cl
    return D, P, N


def canon(d):
    return {k: frozenset(map(tuple, map(lambda t: map(int, t), v))) for k, v in d.items()}


res = {"claim": "C4", "label_file": os.path.basename(LAB), "rows": []}
for n in SIZES:
    D, P, N = build(n)
    args_b = copy.deepcopy((D, P, N)); args_c = copy.deepcopy((D, P, N))
    cc.merge_double_sheets_fast(*copy.deepcopy((D, P, N)))  # warm-up
    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        out_b = cc.merge_double_sheets(*args_b)
    tb = time.perf_counter() - t
    tcs = []
    for _ in range(3):
        a = copy.deepcopy((D, P, N))
        t = time.perf_counter(); out_c = cc.merge_double_sheets_fast(*a); tcs.append(time.perf_counter() - t)
    args_c = a
    same = (list(out_b) == list(out_c) and canon(out_b) == canon(out_c)
            and [type(v) for v in out_b.values()] == [type(v) for v in out_c.values()]
            and list(args_b[1]) == list(args_c[1]) and list(args_b[2]) == list(args_c[2]))
    tc = float(np.median(tcs))
    res["rows"].append({"n": n, "double": len(D), "positive": len(P), "negative": len(N),
                        "merged": sum(isinstance(v, set) for v in out_b.values()),
                        "baseline_s": round(tb, 3), "candidate_s": round(tc, 4), "ratio": round(tb / tc, 1), "identical": bool(same)})
print(json.dumps(res))
