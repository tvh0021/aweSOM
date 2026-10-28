"""Markdown table of a post-processing study (postproc/<tagbase>__it<it>_mc<mc>.json).

    python show_postproc.py FILE.json

Per setting, means over the seeds in the file: changed volume, C, D1, D3,
D = (D1 + D3)/2, cross-seed stability S (only with >1 seed), P = cbrt(C D S),
speckle (volume fraction in components < 27 cells), and the sheet cluster's
component count and largest-component share.
"""

import json
import sys

import numpy as np


def arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


def main():
    d = json.load(open(sys.argv[1]))
    tags = d["tags"]
    print(f"it {d['it']}, merge cost {d['mc']}, {len(tags)} seed(s)\n")
    cols = ["setting", "changed vol", "K", "C", "D1", "D3", "D", "S_seed", "P", "speckle", "sheet comps",
            "sheet largest"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for name, res in d["results"].items():
        vals = {c: [] for c in cols[1:]}
        for t in tags:
            s = res[t]
            vf = arr(s["volume_frac"])
            live = vf > 0
            kap = arr(s["kappa"])
            C = float(np.nanmean(np.clip(kap[live], 0, 1)))
            D1 = s["reference"]["best_jaccard"]
            D3 = s["reference_T3"]["best_jaccard"]
            D = 0.5 * (D1 + D3)
            b = s["reference"]["best"]
            sf = arr(s["small_frac"])
            st = arr(s["stability"]) if "stability" in s else None
            S = float(np.nanmean(st[live])) if st is not None else np.nan
            vals["changed vol"].append(s["changed_volume_frac"])
            vals["K"].append(int(live.sum()))
            vals["C"].append(C)
            vals["D1"].append(D1)
            vals["D3"].append(D3)
            vals["D"].append(D)
            vals["S_seed"].append(S)
            vals["P"].append(float(np.cbrt(C * D * S)) if np.isfinite(S) else np.nan)
            vals["speckle"].append(float(np.nansum(sf[live] * vf[live])))
            vals["sheet comps"].append(s["n_components"][b])
            vals["sheet largest"].append(s["largest_frac"][b])
        row = [name]
        for c in cols[1:]:
            v = np.array(vals[c], float)
            if c in ("K", "sheet comps"):
                row.append(f"{np.mean(v):.0f}" if len(v) else "")
            elif np.all(np.isnan(v)):
                row.append("")
            else:
                row.append(f"{np.nanmean(v):.3f}" + (f" ± {np.nanstd(v):.3f}" if len(v) > 1 and c == "P" else ""))
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
