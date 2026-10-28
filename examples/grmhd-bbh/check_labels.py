"""Check grid labels against the SOM's training sample.

The training rows of ``etkplot som-features`` are a random subset of each
level's owned grid points (quota per level proportional to dx^2), so

1. at every training row, the label predicted on the grid must equal the label
   saved at training time (the same point, the same features, the same model),
   up to float32 rounding ties; and the grid features there must equal the
   training features;
2. within each level, the grid's label fractions must match the training
   rows' fractions to within binomial sampling error;
3. over all levels, the grid's point (volume) fractions must match the
   training fractions weighted by ``weight`` (``weight_volume``).

    python check_labels.py --grid grid_LR_flux_180224.h5 --labels labels_LR_flux_180224.npy \
        --training-file features_bbh_0003-180224-N10000000.h5 \
        --training-labels labels.<tag>-mc0.05.npy -o label_check.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from aweSOM.sheet_stats import GridFileSource, level_lattice


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--training-file", required=True, type=Path)
    ap.add_argument("--training-labels", required=True, type=Path)
    ap.add_argument("--sheet-clusters", default=None, help="comma-separated, for a sheet-only summary line")
    ap.add_argument("-o", "--out", required=True, type=Path)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()

    src = GridFileSource(args.grid, args.labels)
    with h5py.File(args.training_file, "r") as h:
        t_xyz = h["xyz"][:]
        t_lev = h["level"][:]
        t_feat = h["features"][:]
        w_pts = h["weight"][:]
        w_vol = h["weight_volume"][:]
    t_lab = np.load(args.training_labels)
    n_lab = int(max(t_lab.max(), np.asarray(src.labels).max()) + 1)

    report = {"grid": str(args.grid), "labels": str(args.labels), "training_file": str(args.training_file),
              "training_labels": str(args.training_labels), "per_level": {}}
    g_counts, g_vol = {}, {}
    match_total = 0
    agree_total = 0
    for L in src.levels:
        lat = level_lattice(src, L)
        n = src.n_points(L)
        d = src.read(L, 0, n, ("xyz", "labels"))
        lab = np.asarray(d["labels"]).astype(np.int64)
        row_of = np.full(lat.size, -1, dtype=np.int32)
        row_of[lat.flat(d["xyz"])] = np.arange(n, dtype=np.int32)
        del d
        tm = t_lev == L
        rows = row_of[lat.flat(t_xyz[tm])]
        del row_of
        found = rows >= 0
        g_lab_at = lab[rows[found]]
        agree = g_lab_at == t_lab[tm][found]
        sel = src.row_range(L, 0, n)
        feats_at = src.h5["features"][sel][rows[found]] if isinstance(sel, slice) else None
        fdiff = float(np.abs(feats_at - t_feat[tm][found]).max()) if feats_at is not None else float("nan")
        del feats_at
        gc = np.bincount(lab, minlength=n_lab).astype(np.float64)
        tc = np.bincount(t_lab[tm], minlength=n_lab).astype(np.float64)
        n_t = int(tm.sum())
        pg, pt = gc / n, tc / n_t
        se = np.sqrt(np.maximum(pt * (1 - pt), 1.0 / n_t) / n_t)
        z = (pg - pt) / se
        report["per_level"][str(L)] = {
            "dx": lat.dx,
            "grid_points": n,
            "training_rows": n_t,
            "training_rows_found_on_grid": int(found.sum()),
            "label_agreement_at_training_rows": float(agree.mean()) if found.any() else float("nan"),
            "label_disagreements": int((~agree).sum()),
            "max_abs_feature_difference_at_training_rows": fdiff,
            "grid_fraction": pg.tolist(),
            "training_fraction": pt.tolist(),
            "z_score": z.tolist(),
            "max_abs_z": float(np.max(np.abs(z))),
        }
        print(f"level {L}: {found.sum()}/{n_t} training rows on the grid, label agreement {agree.mean():.7f}, "
              f"max |feature diff| {fdiff:.3g}, max |z| of per-level fractions {np.max(np.abs(z)):.2f}", flush=True)
        match_total += int(found.sum())
        agree_total += int(agree.sum())
        g_counts[L] = gc
        g_vol[L] = gc * lat.dx**3
    n_g = sum(c.sum() for c in g_counts.values())
    v_g = sum(v.sum() for v in g_vol.values())
    grid_pts = sum(g_counts.values()) / n_g
    grid_vol = sum(g_vol.values()) / v_g
    tr_pts = np.bincount(t_lab, weights=w_pts, minlength=n_lab) / w_pts.sum()
    tr_vol = np.bincount(t_lab, weights=w_vol, minlength=n_lab) / w_vol.sum()
    tr_sample = np.bincount(t_lab, minlength=n_lab) / len(t_lab)
    # grid reweighted to the training quotas: level L weight = quota_L / owned_L
    quota = {L: float((t_lev == L).sum()) for L in src.levels}
    grid_quota = sum(g_counts[L] * quota[L] / g_counts[L].sum() for L in src.levels) / sum(quota.values())
    report["overall"] = {
        "label_agreement_at_training_rows": agree_total / match_total,
        "training_rows_found_on_grid": match_total,
        "grid_volume_fraction": grid_vol.tolist(),
        "training_volume_weighted_fraction": tr_vol.tolist(),
        "grid_point_fraction": grid_pts.tolist(),
        "training_point_weighted_fraction": tr_pts.tolist(),
        "grid_reweighted_to_sample_quota": grid_quota.tolist(),
        "training_sample_fraction": tr_sample.tolist(),
        "max_abs_volume_fraction_difference": float(np.max(np.abs(grid_vol - tr_vol))),
        "max_rel_volume_fraction_difference": float(np.max(np.abs(grid_vol - tr_vol) / np.maximum(tr_vol, 1e-12))),
    }
    lines = ["| label | grid volume | training (weight_volume) | grid points | training (weight) | "
             "grid at sample quotas | training sample |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for c in range(n_lab):
        lines.append(f"| {c} | {grid_vol[c]:.4%} | {tr_vol[c]:.4%} | {grid_pts[c]:.4%} | {tr_pts[c]:.4%} | "
                     f"{grid_quota[c]:.4%} | {tr_sample[c]:.4%} |")
    if args.sheet_clusters:
        sc = [int(c) for c in args.sheet_clusters.split(",")]
        lines.append(f"| sheet {','.join(map(str, sc))} | {grid_vol[sc].sum():.4%} | {tr_vol[sc].sum():.4%} | "
                     f"{grid_pts[sc].sum():.4%} | {tr_pts[sc].sum():.4%} | {grid_quota[sc].sum():.4%} | "
                     f"{tr_sample[sc].sum():.4%} |")
    report["table_md"] = "\n".join(lines)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(report["table_md"], flush=True)
    print(f"overall label agreement at training rows: {report['overall']['label_agreement_at_training_rows']:.7f}; "
          f"wrote {args.out} ({report['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
