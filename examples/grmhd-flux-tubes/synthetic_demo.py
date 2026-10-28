"""Synthetic validation of aweSOM.tubes: curved Gaussian flux tubes in a 128^3 box.

    PYTHONPATH=<worktree>/src python synthetic_demo.py --out-dir <dir>

Cells are "labelled" by a threshold on |B| (standing in for a SOM cluster),
grouped with group_cells, described with tube_descriptors, selected with
select_tubes and scored against the truth masks with match_to_reference.
Writes synthetic_results.json and fig_synthetic.png.
"""

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aweSOM import tubes

KIND = ["arc (R=40, tilted)", "helix (r=5, pitch 90)", "bowed line", "arc (R=30, x-z)", "S-curve"]


def run(n, n_tubes, noise, threshold, seed):
    t0 = time.time()
    syn = tubes.make_synthetic_tubes(n=n, n_tubes=n_tubes, radius=3.0, noise=noise, seed=seed,
                                     threshold=threshold)
    t_make = time.time() - t0
    xyz, B, truth = syn["xyz"], syn["B"], syn["truth"]
    Bmag = np.linalg.norm(B, axis=1)
    t0 = time.time()
    ids = tubes.group_cells(Bmag > threshold, xyz, dx_per_level=1.0, connectivity=26)
    t_group = time.time() - t0
    t0 = time.time()
    d = tubes.tube_descriptors(ids, xyz, B)
    t_desc = time.time() - t0
    sel = tubes.select_tubes(d, min_cells=50, min_elongation=3.0, min_alignment=0.6)
    keep = np.where(np.isin(ids, sel), ids, -1)
    t0 = time.time()
    m = tubes.match_to_reference(keep, truth)
    m_all = tubes.match_to_reference(ids, truth)
    t_match = time.time() - t0
    per_tube = []
    for j in range(len(m.size_b)):
        a = m.best_a_for_b[j]
        k = int(m.labels_a[a]) if a >= 0 else -1
        per_tube.append({
            "tube": int(m.labels_b[j]), "kind": KIND[int(m.labels_b[j])],
            "truth_cells": int(m.size_b[j]), "matched_component": k,
            "iou": float(m.best_iou_b[j]), "recall": float(m.recall_b[j]),
            "component_cells": int(d["n_cells"][k]) if k >= 0 else 0,
            "elongation": float(d["elongation"][k]) if k >= 0 else None,
            "alignment": float(d["alignment"][k]) if k >= 0 else None,
            "alignment_local": float(d["alignment_local"][k]) if k >= 0 else None,
            "B_coherence": float(d["B_coherence"][k]) if k >= 0 else None,
            "tube_score": float(d["tube_score"][k]) if k >= 0 else None,
            "length_span": float(d["span"][k, 0]) if k >= 0 else None,
        })
    n_comp = int(ids.max() + 1)
    noise_comp = np.setdiff1d(np.arange(n_comp), m_all.labels_a[m_all.best_a_for_b[m_all.best_a_for_b >= 0]])
    res = {
        "n": n, "n_tubes": n_tubes, "noise_rms": noise, "threshold": threshold, "seed": seed,
        "selected_cells": int((Bmag > threshold).sum()), "n_components": n_comp,
        "n_selected_tubes": int(len(sel)), "n_mutual_matches": int(len(m.pairs)),
        "false_positives": int(len(sel) - len(m.pairs)),
        "unselected_components_cells_max": int(d["n_cells"][noise_comp].max()) if len(noise_comp) else 0,
        "per_tube": per_tube,
        "seconds": {"make": t_make, "group_cells": t_group, "tube_descriptors": t_desc,
                    "match_to_reference": t_match},
    }
    return syn, ids, d, sel, res


def figure(syn, ids, d, sel, res, path):
    xyz, truth = syn["xyz"], syn["truth"]
    fig = plt.figure(figsize=(13, 4.6))
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2 = fig.add_subplot(1, 3, 3)
    cols = plt.cm.tab10(np.arange(10))
    rng = np.random.default_rng(0)
    for j in range(res["n_tubes"]):
        rows = np.flatnonzero(truth == j)
        rows = rows[rng.random(len(rows)) < 0.25]
        ax0.scatter(*xyz[rows].T, s=1, color=cols[j], label=f"tube {j}: {KIND[j]}")
        c = syn["centre_lines"][j]
        ax0.plot(*c.T, color="k", lw=0.6)
    ax0.set_title("truth (|B| > 0.5 without noise)")
    colour_of = {p["matched_component"]: p["tube"] for p in res["per_tube"] if p["matched_component"] >= 0}
    for r, k in enumerate(sel):
        rows = np.flatnonzero(ids == k)
        rows = rows[rng.random(len(rows)) < 0.25]
        j = colour_of.get(int(k))
        ax1.scatter(*xyz[rows].T, s=1, color=cols[j] if j is not None else cols[(5 + r) % 10],
                    label=f"component {k}" + (f" -> tube {j}" if j is not None else " (unmatched)"))
    other = np.flatnonzero((ids >= 0) & ~np.isin(ids, sel))
    ax1.scatter(*xyz[other].T, s=2, color="0.6", label="rejected")
    ax1.set_title(f"recovered: {res['n_selected_tubes']} tubes of {res['n_components']} components")
    for ax in (ax0, ax1):
        ax.set_xlim(0, res["n"]); ax.set_ylim(0, res["n"]); ax.set_zlim(0, res["n"])
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.legend(loc="upper left", fontsize=6, markerscale=4, frameon=False)
    pt = res["per_tube"]
    x = np.arange(len(pt))
    w = 0.27
    ax2.bar(x - w, [p["iou"] for p in pt], w, label="IoU with truth", color="#3b6ea8")
    ax2.bar(x, [p["alignment"] or 0 for p in pt], w, label="alignment <|cos(B, e1)|>", color="#d9822b")
    ax2.bar(x + w, [p["alignment_local"] or 0 for p in pt], w, label="alignment, 4 slices", color="#6aa84f")
    ax2.set_xticks(x, [f"{p['tube']}\n{p['kind'].split(' (')[0]}" for p in pt], fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1, color="0.7", lw=0.5)
    ax2.legend(fontsize=7, loc="lower left")
    ax2.set_title(f"per tube (noise rms {res['noise_rms']}, threshold {res['threshold']})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--n-tubes", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-scan", type=float, nargs="*", default=[0.0, 0.05, 0.1, 0.15, 0.2])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    syn, ids, d, sel, res = run(args.n, args.n_tubes, args.noise, args.threshold, args.seed)
    figure(syn, ids, d, sel, res, f"{args.out_dir}/fig_synthetic.png")
    scan = []
    for nz in args.noise_scan:
        _, _, _, _, r = run(args.n, args.n_tubes, nz, args.threshold, args.seed)
        scan.append({"noise_rms": nz, "n_components": r["n_components"],
                     "n_selected_tubes": r["n_selected_tubes"], "n_mutual_matches": r["n_mutual_matches"],
                     "false_positives": r["false_positives"],
                     "iou": [p["iou"] for p in r["per_tube"]],
                     "alignment": [p["alignment"] for p in r["per_tube"]]})
    res["noise_scan"] = scan
    with open(f"{args.out_dir}/synthetic_results.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "noise_scan"}, indent=1))
    for s in scan:
        print(s)


if __name__ == "__main__":
    main()
