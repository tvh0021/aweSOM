"""Score trained models on the full grid of one iteration, at every merge cost.

    python score.py --it 135168 MODEL.npz [MODEL.npz ...] [--components all|none|0.1,0.2]
        [--save-labels 0.1,0.2]

Writes scores/<tag>__it<it>.json and scores/<tag>__it<it>_statlabels.npz (the
labels on the stats subsample, for cross-seed stability), and optionally the
full-grid labels at chosen merge costs (labels/<tag>__it<it>_mc<mc>.npy, int16).
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from common import COLS, RES, SCORES, col, limit_threads, load_grid, meta

limit_threads(6)
from aweSOM.inference import best_match_nodes_screen  # noqa: E402
from aweSOM.physics_score import physicality, reference_overlap, score_clusters  # noqa: E402

LOG2 = np.log10(2.0)
LABELS = RES / "labels"
NODES = RES / "nodes"


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def bmu_nodes(model, it, chunk=4_000_000):
    names = json.loads(str(model["info"]))["names"]
    w = model["weights"]
    mean, scale = model["mean"], model["scale"]
    cols = [col(it, nm) for nm in names]
    n = cols[0].shape[0]
    nodes = np.empty(n, np.int32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        x = np.stack([np.asarray(c[a:b], dtype=np.float64) for c in cols], 1)
        good = np.isfinite(x).all(1)
        x = (x - mean) / scale
        x[~good] = 0.0
        idx, _ = best_match_nodes_screen(w, x)
        idx = idx.astype(np.int32)
        idx[~good] = -1
        nodes[a:b] = idx
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--it", type=int, required=True)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--components", default="all")
    ap.add_argument("--save-labels", default="")
    a = ap.parse_args()
    SCORES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    NODES.mkdir(parents=True, exist_ok=True)
    it = a.it
    grid = load_grid(it)
    m = meta(it)
    sub = np.load(COLS / f"{it}_stats_rows.npy")
    ref = {T: np.load(COLS / f"{it}_sheet_T{T}.npy") for T in ("1", "3")}
    ridge = np.load(COLS / f"{it}_ridge.npy")
    # features on the stats rows only (score_clusters reads them at stats_rows)
    n = m["n"]
    feats = {}
    for nm in ("log_chi_B", "log_grad_ln_b2", "log_beta_inv"):
        v = np.full(n, np.nan, np.float32)
        v[sub] = col(it, nm)[sub]
        feats[nm] = v
    v = np.full(n, np.nan, np.float32)
    v[sub] = LOG2 + col(it, "log_beta_inv")[sub] + col(it, "log_theta")[sub]
    feats["log_sigma"] = v
    save_mc = [float(s) for s in a.save_labels.split(",") if s]
    for path in a.models:
        t0 = time.time()
        up0 = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
        model = np.load(path)
        info = json.loads(str(model["info"]))
        tag = info["tag"]
        npath = NODES / f"{tag}__it{it}.npy"
        if npath.exists():
            nodes = np.load(npath).astype(np.int32)
        else:
            nodes = bmu_nodes(model, it)
            np.save(npath, nodes.astype(np.int16))
        t_bmu = time.time() - t0
        mcs = info["merge_costs"]
        ncs = model["node_cluster"]
        if a.components == "all":
            comp_mc = set(mcs)
        elif a.components == "none":
            comp_mc = set()
        else:
            comp_mc = {float(s) for s in a.components.split(",")}
        labs = []
        for i, mc in enumerate(mcs):
            nc = np.append(ncs[i], -1).astype(np.int16)  # index -1 -> label -1
            labs.append(nc[nodes])
        out = dict(info=info, it=it, t_bmu=t_bmu, uptime=up0, mc={})
        statlabels = {}
        for i, mc in enumerate(mcs):
            lab = labs[i]
            k = int(ncs[i].max()) + 1
            others = [labs[j] for j in (i - 1, i + 1) if 0 <= j < len(mcs)]
            t1 = time.time()
            s = score_clusters(lab, grid, features=feats, reference=ref["1"], others=others, ridge=ridge,
                               n_clusters=k, components=mc in comp_mc, stats_rows=sub)
            s["reference_T3"] = reference_overlap(lab, ref["3"], grid.cell_volume(), k)
            s["physicality_T3"] = physicality(s["kappa"], s["reference_T3"]["best_jaccard"], s["stability"])
            s["t_score"] = time.time() - t1
            out["mc"][str(mc)] = jsonable(s)
            statlabels[f"mc{mc}"] = lab[sub]
            if mc in save_mc:
                np.save(LABELS / f"{tag}__it{it}_mc{mc}.npy", lab)
        np.savez_compressed(SCORES / f"{tag}__it{it}_statlabels.npz", **statlabels)
        out["t_total"] = time.time() - t0
        (SCORES / f"{tag}__it{it}.json").write_text(json.dumps(out))
        brief = {mc: (out["mc"][str(mc)]["n_clusters"], round(out["mc"][str(mc)]["reference"]["best_jaccard"], 3),
                      round(out["mc"][str(mc)]["physicality"], 3)) for mc in mcs}
        print(f"{tag} it{it}: bmu {t_bmu:.0f}s total {out['t_total']:.0f}s  (K, D_T1, P) {brief}", flush=True)


if __name__ == "__main__":
    main()
