"""Spatial post-processing study: clean full-grid labels with
aweSOM.postprocess.clean_labels at several settings and score before/after.

    python postproc_study.py --it 135168 --mc 0.1 TAG_seed0 TAG_seed1 ...

The labels come from score.py --save-labels (labels/<tag>__it<it>_mc<mc>.npy).
Cross-seed stability is computed between the cleaned labelings of the given tags.
"""

import argparse
import json
import subprocess
import time

import numpy as np

from common import COLS, RES, col, limit_threads, load_grid

limit_threads(4)
from aweSOM.physics_score import best_match_jaccard, physicality, reference_overlap, score_clusters  # noqa: E402
from aweSOM.postprocess import clean_labels  # noqa: E402
from score import LABELS, jsonable  # noqa: E402

SETTINGS = {
    "none": {},
    "min8": dict(min_size=8),
    "min27": dict(min_size=27),
    "min64": dict(min_size=64),
    "min125": dict(min_size=125),
    "maj1": dict(majority_iters=1),
    "maj2": dict(majority_iters=2),
    "maj1+min27": dict(majority_iters=1, min_size=27),
}
OUT = RES / "postproc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--it", type=int, required=True)
    ap.add_argument("--mc", type=float, required=True)
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--save", default="", help="settings whose cleaned labels to save (first tag only)")
    ap.add_argument("tags", nargs="+")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    it = a.it
    grid = load_grid(it)
    vol = grid.cell_volume()
    sub = np.load(COLS / f"{it}_stats_rows.npy")
    ref1 = np.load(COLS / f"{it}_sheet_T1.npy")
    ref3 = np.load(COLS / f"{it}_sheet_T3.npy")
    ridge = np.load(COLS / f"{it}_ridge.npy")
    n = ref1.size
    feats = {}
    for nm in ("log_chi_B", "log_beta_inv"):
        v = np.full(n, np.nan, np.float32)
        v[sub] = col(it, nm)[sub]
        feats[nm] = v
    save = set(s for s in a.save.split(",") if s)
    raw = {t: np.load(LABELS / f"{t}__it{it}_mc{a.mc}.npy") for t in a.tags}
    k = max(int(x.max()) + 1 for x in raw.values())
    results = {}
    for name in a.settings.split(","):
        kw = SETTINGS[name]
        res = {}
        cleaned_sub = {}
        for i, t in enumerate(a.tags):
            t0 = time.time()
            lab = raw[t]
            cl = clean_labels(lab, grid=grid, **kw) if kw else lab
            t_clean = time.time() - t0
            changed = lab != cl
            s = score_clusters(cl, grid, features=feats, reference=ref1, ridge=ridge, n_clusters=k,
                               components=True, stats_rows=sub)
            s["reference_T3"] = reference_overlap(cl, ref3, vol, k)
            s["changed_volume_frac"] = float(vol[changed].sum() / vol.sum())
            s["changed_points_frac"] = float(changed.mean())
            s["t_clean"] = t_clean
            s["t_score"] = time.time() - t0 - t_clean
            cleaned_sub[t] = cl[sub]
            if i == 0 and name in save:
                np.save(OUT / f"{t}__it{it}_mc{a.mc}_{name}.npy", cl)
            res[t] = s
            del cl, changed
            print(f"{name} {t}: changed {s['changed_volume_frac']:.4f} of volume, D1 {s['reference']['best_jaccard']:.3f}, "
                  f"clean {t_clean:.0f}s", flush=True)
        if len(a.tags) > 1:
            vs = vol[sub]
            for t in a.tags:
                js = [best_match_jaccard(cleaned_sub[t], cleaned_sub[u], vs, k, k) for u in a.tags if u != t]
                st = np.nanmean(js, axis=0)
                res[t]["stability"] = st
                res[t]["physicality"] = physicality(res[t]["kappa"], res[t]["reference"]["best_jaccard"],
                                                    st[res[t]["volume_frac"] > 0])
        results[name] = res
    up = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    tagbase = a.tags[0].replace("-seed0", "")
    (OUT / f"{tagbase}__it{it}_mc{a.mc}.json").write_text(json.dumps(jsonable(dict(
        it=it, mc=a.mc, tags=a.tags, settings={s: SETTINGS[s] for s in a.settings.split(",")}, uptime=up,
        results=results))))


if __name__ == "__main__":
    main()
