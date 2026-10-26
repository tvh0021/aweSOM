"""C3: connected clumps of equal label. Baseline = combine_clumps.get_clumps (pure-Python DFS);
candidate = combine_clumps.label_clumps (scipy.ndimage.label per value, optional periodic merge).
Optional reference point: cc3d (if importable). Data: a real 128^3 SOM label map, sub-cubes for the
baseline. Equivalence: candidate ids == baseline clump list order, cell by cell. Prints one JSON line.

usage: PYTHONPATH=<worktree>/src[:<worktree>/.deps] python bench_clumps.py [label_file] [reps] [--full-baseline]
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "examples", "plasma-turbulence"))
import combine_clumps as cc  # noqa: E402

LAB = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/Users/tvh0021/git_repos/aweSOM-research/planning/runs/plasma-2026-10-26/labels.5000-63-38-0.1-100000-1u.npy"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
full = np.load(LAB).reshape(128, 128, 128).astype(np.int32)
try:
    import cc3d
except ImportError:
    cc3d = None


def med_time(f, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), r


def clumps_to_ids(clumps, shape):
    ids = np.full(shape, -1, dtype=np.int32)
    for k, cl in enumerate(clumps):
        ids[tuple(np.array(cl).T)] = k
    return ids


def periodic_reference(a, deltas):
    """get_clumps with wrapped neighbors (pure-Python DFS), numbered in first-cell C order."""
    ids = np.full(a.shape, -1, dtype=np.int32); n = 0
    for start in np.ndindex(a.shape):
        if ids[start] >= 0:
            continue
        stack = [start]; ids[start] = n; v = a[start]
        while stack:
            c = stack.pop()
            for d in deltas:
                nb = tuple((ci + di) % s for ci, di, s in zip(c, d, a.shape))
                if ids[nb] < 0 and a[nb] == v:
                    ids[nb] = n; stack.append(nb)
        n += 1
    return ids


res = {"claim": "C3", "label_file": os.path.basename(LAB), "reps": REPS, "rows": []}
cc.label_clumps(full[:8, :8, :8], cc.generate_neighbors(6))  # warm-up imports
for conn in (6, 10, 26):
    deltas = cc.generate_neighbors(conn)
    for n in (32, 48):
        sub = np.ascontiguousarray(full[:n, :n, :n])
        tb, clumps = med_time(lambda: cc.get_clumps(sub, deltas), 1)  # baseline is slow: one run
        tc, (ids, k) = med_time(lambda: cc.label_clumps(sub, deltas, use_cc3d=False), REPS)
        same = bool(np.array_equal(clumps_to_ids(clumps, sub.shape), ids)) and k == len(clumps)
        res["rows"].append({"conn": conn, "n": n, "clumps": k, "baseline_s": round(tb, 4), "candidate_s": round(tc, 5),
                            "ratio": round(tb / tc, 1), "identical": same})
    # periodic: equivalence against a periodic DFS on 24^3, timing on 128^3
    sub = np.ascontiguousarray(full[:24, :24, :24])
    ref = periodic_reference(sub, deltas)
    got, _ = cc.label_clumps(sub, deltas, periodic=True, use_cc3d=False)
    got_cc, _ = cc.label_clumps(sub, deltas, periodic=True)
    assert np.array_equal(ref, got_cc), conn
    tnp, (a0, _) = med_time(lambda: cc.label_clumps(full, deltas, use_cc3d=False), REPS)
    tp, (p0, kp) = med_time(lambda: cc.label_clumps(full, deltas, periodic=True, use_cc3d=False), REPS)
    row = {"conn": conn, "n": 128, "candidate_s": round(tnp, 4), "candidate_periodic_s": round(tp, 4),
           "periodic_clumps": kp, "periodic_identical_24cube": bool(np.array_equal(ref, got))}
    if cc3d is not None:  # label_clumps through its cc3d path (falls back to scipy where cc3d can't), renumbered
        t1, (a1, _) = med_time(lambda: cc.label_clumps(full, deltas), REPS)
        t2, (p1, _) = med_time(lambda: cc.label_clumps(full, deltas, periodic=True), REPS)
        t3, _ = med_time(lambda: cc.label_clumps(full, deltas, renumber=False), REPS)
        row.update({"label_clumps_cc3d_s": round(t1, 4), "label_clumps_cc3d_periodic_s": round(t2, 4),
                    "label_clumps_cc3d_no_renumber_s": round(t3, 4),
                    "cc3d_path_identical_to_scipy_path": bool(np.array_equal(a0, a1) and np.array_equal(p0, p1))})
    if cc3d is not None and conn in (6, 26):
        shifted = full + 1  # cc3d treats value 0 as background; SOM cluster 0 is a real cluster
        tcc, lab = med_time(lambda: cc3d.connected_components(shifted, connectivity=conn, out_dtype=np.uint32), REPS)
        row["cc3d_s"] = round(tcc, 4)
        mine = cc.label_clumps(full, deltas)[0].reshape(-1).astype(np.int64)
        theirs = lab.reshape(-1).astype(np.int64)
        n_pairs = np.unique(mine * (theirs.max() + 1) + theirs).size
        row["cc3d_same_partition"] = bool(n_pairs == np.unique(mine).size == np.unique(theirs).size)
        if conn == 6:
            try:
                tccp, labp = med_time(lambda: cc3d.connected_components(shifted, connectivity=6, periodic_boundary=True, out_dtype=np.uint32), REPS)
                minep = cc.label_clumps(full, deltas, periodic=True)[0].reshape(-1).astype(np.int64)
                tp_ = labp.reshape(-1).astype(np.int64)
                row["cc3d_periodic_s"] = round(tccp, 4)
                row["cc3d_periodic_same_partition"] = bool(np.unique(minep * (tp_.max() + 1) + tp_).size == np.unique(minep).size == np.unique(tp_).size)
            except Exception as e:  # periodic_boundary unsupported
                row["cc3d_periodic_error"] = repr(e)[:120]
    res["rows"].append(row)
if "--full-baseline" in sys.argv:  # one pure-Python DFS run on the whole 128^3 map (slow)
    deltas = cc.generate_neighbors(6)
    tb, clumps = med_time(lambda: cc.get_clumps(full, deltas), 1)
    tc, (ids, k) = med_time(lambda: cc.label_clumps(full, deltas, use_cc3d=False), REPS)
    res["rows"].append({"conn": 6, "n": 128, "baseline_s": round(tb, 2), "candidate_s": round(tc, 4), "ratio": round(tb / tc, 1),
                        "identical": bool(np.array_equal(clumps_to_ids(clumps, full.shape), ids)) and k == len(clumps)})
print(json.dumps(res))
