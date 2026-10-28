"""Collect scores/*.json into flat tables, with cross-seed stability.

    python aggregate.py            -> RES/tables/scoreboard.csv, seeds.csv
"""

import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import COLS, DX, RES, SCORES

from aweSOM.physics_score import best_match_jaccard

TABLES = RES / "tables"


def rows_of(path):
    d = json.loads(Path(path).read_text())
    info = d["info"]
    out = []
    for mc, s in d["mc"].items():
        vf = np.array(s["volume_frac"], float)
        live = vf > 0
        kap = np.array([np.nan if v is None else v for v in s["kappa"]], float)
        stab = np.array([np.nan if v is None else v for v in s["stability"]], float)
        ref = s["reference"]
        b = ref["best"]
        r = dict(
            tag=info["tag"], config=re.sub(r"-seed\d+", "", info["tag"]), variant=info["variant"],
            lattice=info["lattice"], nodes=info["xdim"] * info["ydim"], n=info["n"], alpha=info["alpha"],
            steps=info["steps"], seed=info["seed"], it=d["it"], mc=float(mc), K=int(live.sum()),
            C=float(np.nanmean(np.clip(kap[live], 0, 1))), D1=ref["best_jaccard"],
            D3=s["reference_T3"]["best_jaccard"], S_mc=float(np.nanmean(stab[live])),
            union1=ref["union_jaccard"], n_union1=len(ref["union"]),
            sheet=b, sheet_vf=vf[b], sheet_prec=ref["precision"][b], sheet_rec=ref["recall"][b],
            sheet_prec3=s["reference_T3"]["precision"][b], sheet_rec3=s["reference_T3"]["recall"][b],
            sheet_kappa=kap[b], sheet_ridge_enr=s["ridge_enrichment"][b],
            sheet_chi_med=s["features"]["log_chi_B"]["median"][b],
            sheet_beta_med=s["features"]["log_beta_inv"]["median"][b],
            t_train=info["t_train"],
        )
        # D: mean of the loose (T=1) and strict (T=3) sheet matches, see physicality()
        r["D"] = 0.5 * (r["D1"] + r["D3"])
        r["P_mc"] = float(np.cbrt(r["C"] * r["D"] * r["S_mc"]))
        r["P_mc_T1"] = float(np.cbrt(r["C"] * r["D1"] * r["S_mc"]))
        r["sheet3"] = s["reference_T3"]["best"]
        if "small_frac" in s:
            sf = np.array([np.nan if v is None else v for v in s["small_frac"]], float)
            r["speckle_vw"] = float(np.nansum(sf[live] * vf[live]))
            r["speckle_mean"] = float(np.nanmean(sf[live]))
            r["sheet_speckle"] = sf[b]
            r["n_comp"] = int(np.sum(s["n_components"]))
        out.append(r)
    return out


def seed_stability(rows):
    """Per (config, it, mc): mean over seed pairs of the mean best-match Jaccard."""
    vol_cache = {}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["config"], r["it"], r["mc"])].append(r)
    res = []
    for (cfg, it, mc), rs in groups.items():
        if len(rs) < 2:
            continue
        if it not in vol_cache:
            sub = np.load(COLS / f"{it}_stats_rows.npy")
            lev = np.load(COLS / f"{it}_level.npy", mmap_mode="r")[sub]
            vol_cache[it] = np.array([DX[int(lv)] ** 3 for lv in (5, 6, 7)])[lev - 5]
        vol = vol_cache[it]
        labs = {r["seed"]: np.load(SCORES / f"{r['tag']}__it{it}_statlabels.npz")[f"mc{mc}"] for r in rs}
        per_seed = {}
        for s, a in labs.items():
            js = []
            for s2, b in labs.items():
                if s2 == s:
                    continue
                j = best_match_jaccard(a, b, vol)
                vf = np.bincount(a[a >= 0], weights=vol[a >= 0], minlength=j.size)
                js.append(np.nanmean(j[vf > 0]))
            per_seed[s] = float(np.mean(js))
        for r in rs:
            r["S_seed"] = per_seed[r["seed"]]
            r["P_seed"] = float(np.cbrt(r["C"] * r["D"] * r["S_seed"]))
            r["P_seed_T1"] = float(np.cbrt(r["C"] * r["D1"] * r["S_seed"]))
            r["P_seed_T3"] = float(np.cbrt(r["C"] * r["D3"] * r["S_seed"]))
        P = [r["P_seed"] for r in rs]

        def ms(key):
            v = [r[key] for r in rs]
            return float(np.mean(v)), float(np.std(v))

        res.append(dict(config=cfg, it=it, mc=mc, seeds=len(rs), S_seed=np.mean(list(per_seed.values())),
                        S_seed_std=np.std(list(per_seed.values())),
                        P_seed=np.mean(P), P_seed_std=np.std(P), P_T1=ms("P_seed_T1")[0], P_T3=ms("P_seed_T3")[0],
                        D=ms("D")[0], D_std=ms("D")[1], C=np.mean([r["C"] for r in rs]),
                        C_std=np.std([r["C"] for r in rs]), D1=np.mean([r["D1"] for r in rs]),
                        D1_std=np.std([r["D1"] for r in rs]), D3=np.mean([r["D3"] for r in rs]),
                        D3_std=np.std([r["D3"] for r in rs]), sheet_vf=ms("sheet_vf")[0],
                        sheet_prec=ms("sheet_prec")[0], sheet_rec=ms("sheet_rec")[0],
                        K=np.mean([r["K"] for r in rs]), K_min=min(r["K"] for r in rs), K_max=max(r["K"] for r in rs)))
    return res


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(glob.glob(str(SCORES / "*.json"))):
        rows += rows_of(p)
    seeds = seed_stability(rows)
    write(TABLES / "scoreboard.csv", rows)
    if seeds:
        write(TABLES / "seeds.csv", seeds)
    print(f"{len(rows)} rows, {len(seeds)} seed groups")


if __name__ == "__main__":
    main()
