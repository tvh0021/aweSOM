"""Field-strength contrast of selected tubes against a surrounding shell.

    PYTHONPATH=<worktree>/src python tube_contrast.py --grid grid.h5 --B B.h5 \
        --components components_mc005_c0.h5 -o tube_contrast_mc005_c0.json

For each selected component, on its dominant level: the shell is the cells
within 2 cells (26-neighbourhood dilation, twice) of the component that are
not in it. Reports mean |B| (tube) / mean |B| (shell), the same for b^2
(smallb2), and the median log_beta_inv inside minus in the shell. A flux tube
should be a field concentration: ratio > 1.
"""

import argparse
import json

import h5py
import numpy as np
from scipy import ndimage

DX = {5: 0.25, 6: 0.125, 7: 0.0625}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--B", required=True)
    ap.add_argument("--components", required=True)
    ap.add_argument("--max-tubes", type=int, default=200)
    ap.add_argument("--shell", type=int, default=2)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    with h5py.File(args.grid, "r") as g:
        xyz = g["xyz"][:]
        level = g["level"][:]
        lbi = g["features"][:, 0]
    with h5py.File(args.B, "r") as f:
        Bmag = np.linalg.norm(f["B"][:], axis=1)
        b2 = f["smallb2"][:]
    with h5py.File(args.components, "r") as f:
        ids = f["ids"][:]
        sel = f["selected"][:][: args.max_tubes]
    per_level = {lv: np.flatnonzero(level == lv) for lv in DX}
    out = []
    for k in sel:
        rk = np.flatnonzero(ids == k)
        lv = int(np.bincount(level[rk]).argmax())
        h = DX[lv]
        rk = rk[level[rk] == lv]
        lo = xyz[rk].min(0) - (args.shell + 1) * h
        hi = xyz[rk].max(0) + (args.shell + 1) * h
        rows = per_level[lv]
        p = xyz[rows]
        rows = rows[np.all((p >= lo - 1e-6) & (p <= hi + 1e-6), axis=1)]
        idx = np.rint((xyz[rows] - lo) / h).astype(int)
        shape = tuple(idx.max(0) + 1)
        present = np.zeros(shape, bool)
        tube = np.zeros(shape, bool)
        present[tuple(idx.T)] = True
        tube[tuple(idx[ids[rows] == k].T)] = True
        grown = ndimage.binary_dilation(tube, structure=np.ones((3, 3, 3), bool), iterations=args.shell)
        shell = grown & ~tube & present
        in_shell = shell[tuple(idx.T)]
        in_tube = tube[tuple(idx.T)]
        rt, rs = rows[in_tube], rows[in_shell]
        out.append({"id": int(k), "level": lv, "n_tube": int(len(rt)), "n_shell": int(len(rs)),
                    "B_ratio": float(Bmag[rt].mean() / Bmag[rs].mean()),
                    "b2_ratio": float(b2[rt].mean() / b2[rs].mean()),
                    "dlog_beta_inv": float(np.median(lbi[rt]) - np.median(lbi[rs]))})
    r = np.array([o["B_ratio"] for o in out])
    dl = np.array([o["dlog_beta_inv"] for o in out])
    summary = {"components": args.components, "shell_cells": args.shell, "n": len(out),
               "B_ratio_quantiles": {q: float(np.quantile(r, float(q))) for q in ("0.1", "0.25", "0.5", "0.75", "0.9")} if len(r) else {},
               "fraction_B_ratio_gt_1": float(np.mean(r > 1)) if len(r) else None,
               "dlog_beta_inv_median": float(np.median(dl)) if len(dl) else None,
               "tubes": out}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "tubes"}, indent=1))


if __name__ == "__main__":
    main()
