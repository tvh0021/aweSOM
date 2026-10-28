"""Per-cluster scoreboard of one scored model as Markdown.

    python cluster_table.py SCORE.json --mc 0.1

Columns: volume fraction; median [IQR] of the four diagnostics; neighbour
agreement kappa; components and speckle (volume share in components < 27
cells); sheet-likeness (precision / recall vs the T=1 metric mask, chi_B
ridge enrichment); stability across adjacent merge costs.
"""

import argparse
import json


def g(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("score")
    ap.add_argument("--mc", default="0.1")
    a = ap.parse_args()
    d = json.load(open(a.score))
    s = d["mc"][a.mc]
    feats = ["log_chi_B", "log_grad_ln_b2", "log_beta_inv", "log_sigma"]
    ref, ref3 = s["reference"], s["reference_T3"]
    print(f"{d['info']['tag']}, it {d['it']}, merge cost {a.mc}; sheet cluster (T=1 best match) = c{ref['best']}\n")
    cols = ["c", "vol frac"] + [f"{f} med [IQR]" for f in feats] + [
        "kappa", "comps", "speckle", "prec T1", "rec T1", "prec T3", "ridge enr", "S_mc"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    order = sorted(range(len(s["volume_frac"])), key=lambda k: -s["volume_frac"][k])
    for k in order:
        if s["volume_frac"][k] <= 0:
            continue
        row = [f"c{k}" + (" (sheet)" if k == ref["best"] else ""), g(s["volume_frac"][k], 3)]
        for f in feats:
            q = s["features"][f]
            row.append(f"{g(q['median'][k])} [{g(q['iqr'][k])}]")
        row += [g(s["kappa"][k]), str(s["n_components"][k]), g(s["small_frac"][k], 3),
                g(ref["precision"][k]), g(ref["recall"][k]), g(ref3["precision"][k]),
                g(s["ridge_enrichment"][k]), g(s["stability"][k])]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
