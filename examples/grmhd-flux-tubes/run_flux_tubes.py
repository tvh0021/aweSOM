"""Group SOM-labelled cells of the LR_flux grid into 3D components and describe them.

    PYTHONPATH=<worktree>/src python run_flux_tubes.py \
        --grid grid_LR_flux_180224.h5 --B B_LR_flux_180224.h5 \
        --labels labels_LR_flux_180224_mc0.1.npy --out-dir .

For each cell set (one SOM cluster, or a union) it runs group_cells
(26-connectivity, stitched across levels 5-7), tube_descriptors with the
aligned magnetic field, and select_tubes, and writes
``components_<set>.h5`` (component id per grid row as int32, -1 elsewhere, and
the descriptor arrays), plus ``flux_tubes_summary.json``.
"""

import argparse
import json
import resource
import time

import h5py
import numpy as np

import aweSOM
from aweSOM import tubes

DX = {5: 0.25, 6: 0.125, 7: 0.0625}


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS


def regions(centroid, holes, horizon_r):
    """Where a component's centroid sits relative to the holes.

    funnel       : within 15 M of a hole and within 35 deg of the hole's z axis
    near-hole    : within 6 M of a hole otherwise (mini-disk, horizon surroundings)
    between      : |z| < 6, projected onto the hole-hole segment at t in [0.25, 0.75],
                   within 8 M of it in the orbital plane
    disk/cavity  : |z| < 0.5 R (R = cylindrical radius from the centre of mass)
    high-latitude: everything else
    """
    out = []
    h1, h2 = holes
    seg = h2 - h1
    for c in centroid:
        d = np.linalg.norm(holes - c, axis=1)
        k = int(np.argmin(d))
        rel = c - holes[k]
        polar = np.degrees(np.arctan2(np.hypot(rel[0], rel[1]), abs(rel[2])))
        t = float((c - h1) @ seg / (seg @ seg))
        perp = np.linalg.norm((c - h1 - t * seg)[:2])
        if d[k] < 15 and polar < 35 and d[k] > horizon_r:
            out.append(f"funnel-{k + 1}")
        elif d[k] < 6:
            out.append(f"near-hole-{k + 1}")
        elif abs(c[2]) < 6 and 0.25 <= t <= 0.75 and perp < 8:
            out.append("between")
        elif abs(c[2]) < 0.5 * np.hypot(c[0], c[1]):
            out.append("disk/cavity")
        else:
            out.append("high-latitude")
    return np.array(out)


def quantiles(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {f"q{int(round(q * 100))}": float(np.quantile(x, q)) for q in qs} | {"max": float(x.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--B", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--sets", nargs="+", default=["c10=10", "c1=1", "c4=4", "magnetized=1,4,10", "sheet_c9=9"],
                    help="name=label[,label...]")
    ap.add_argument("--min-cells", type=int, default=100)
    ap.add_argument("--min-elongation", type=float, default=3.0)
    ap.add_argument("--min-alignment", type=float, default=0.7)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--summary", default="flux_tubes_summary.json")
    args = ap.parse_args()
    print("aweSOM from", aweSOM.__file__, flush=True)
    t0 = time.time()
    with h5py.File(args.grid, "r") as g:
        xyz = g["xyz"][:]
        level = g["level"][:]
        holes = g.attrs["horizon_centres"]
        horizon_r = float(np.max(g.attrs["horizon_radius"]))
        feats = g["features"]
        names = [n.decode() for n in g["names"][:]]
    with h5py.File(args.B, "r") as f:
        B = f["B"][:]
    labels = aweSOM.open_npy(args.labels, mmap_mode=None)
    assert len(labels) == len(xyz) == len(B)
    t_load = time.time() - t0
    print(f"loaded {len(xyz)} rows in {t_load:.0f}s, peak RSS {peak_rss_gb():.2f} GB", flush=True)

    summary = {"grid": args.grid, "B": args.B, "labels": args.labels, "aweSOM": aweSOM.__file__,
               "dx": DX, "connectivity": 26,
               "selection": {"min_cells": args.min_cells, "min_elongation": args.min_elongation,
                             "min_alignment": args.min_alignment},
               "load_seconds": t_load, "sets": {}}
    for spec in args.sets:
        name, vals = spec.split("=")
        vals = [int(v) for v in vals.split(",")]
        mask = np.isin(labels, vals)
        t1 = time.time()
        ids = tubes.group_cells(mask, xyz, level, DX, connectivity=26)
        t_group = time.time() - t1
        unst = tubes.group_cells(mask, xyz, level, DX, connectivity=26, stitch=False)
        n_unstitched = int(unst.max() + 1)
        del unst
        t1 = time.time()
        d = tubes.tube_descriptors(ids, xyz, B, level=level, dx_per_level=DX, n_segments=4)
        t_desc = time.time() - t1
        K = len(d["id"])
        sel = tubes.select_tubes(d, args.min_cells, args.min_elongation, args.min_alignment)
        big = d["n_cells"] >= args.min_cells
        reg = regions(d["centroid"], holes, horizon_r)
        vol_total = float(d["volume"].sum())
        # sensitivity of the tube count to the cuts
        grid_counts = {f"elong>={e:g},align>={a:g}": int(len(tubes.select_tubes(d, args.min_cells, e, a)))
                       for e in (2, 3, 5) for a in (0.6, 0.7, 0.8)}
        size_bins = [1, 10, 100, 1000, 10000, 100000, 10**9]
        hist = np.histogram(d["n_cells"], bins=size_bins)[0]

        def wmed(x, m):
            x, w = np.asarray(x)[m], d["volume"][m]
            ok = np.isfinite(x)
            if not ok.any():
                return None
            o = np.argsort(x[ok])
            cw = np.cumsum(w[ok][o])
            return float(x[ok][o][np.searchsorted(cw, 0.5 * cw[-1])])

        s = {
            "labels": vals, "cells": int(mask.sum()),
            "cells_per_level": {int(lv): int(np.sum(mask & (level == lv))) for lv in (5, 6, 7)},
            "volume_M3": vol_total,
            "n_components": K, "n_components_before_stitching": n_unstitched,
            "n_components_ge_min_cells": int(big.sum()),
            "largest_component_cell_fraction": float(d["n_cells"].max() / mask.sum()) if K else 0.0,
            "largest_component_volume_fraction": float(d["volume"].max() / vol_total) if K else 0.0,
            "size_hist_cells": {f"[{a},{b})": int(h) for a, b, h in zip(size_bins[:-1], size_bins[1:], hist)},
            "n_cells_quantiles_ge_min": quantiles(d["n_cells"][big]),
            "volume_quantiles_ge_min": quantiles(d["volume"][big]),
            "elongation_quantiles_ge_min": quantiles(d["elongation"][big]),
            "alignment_quantiles_ge_min": quantiles(d["alignment"][big]),
            "alignment_local_quantiles_ge_min": quantiles(d["alignment_local"][big]),
            "align_e3_quantiles_ge_min": quantiles(d["align_axes"][big, 2]),
            "flatness_quantiles_ge_min": quantiles(d["flatness"][big]),
            "B_coherence_quantiles_ge_min": quantiles(d["B_coherence"][big]),
            "volume_weighted_median_ge_min": {k: wmed(v, big) for k, v in {
                "elongation": d["elongation"], "alignment": d["alignment"],
                "alignment_local": d["alignment_local"], "align_e3": d["align_axes"][:, 2],
                "flatness": d["flatness"], "B_coherence": d["B_coherence"]}.items()},
            "n_tubes": int(len(sel)),
            "tube_volume_M3": float(d["volume"][sel].sum()),
            "tube_count_sensitivity": grid_counts,
            "regions_tubes": {r: int(np.sum(reg[sel] == r)) for r in np.unique(reg)},
            "regions_ge_min_cells": {r: int(np.sum(reg[big] == r)) for r in np.unique(reg)},
            "regions_volume_fraction": {r: float(d["volume"][reg == r].sum() / vol_total) for r in np.unique(reg)},
            "top_tubes": [
                {"id": int(k), "n_cells": int(d["n_cells"][k]), "volume": float(d["volume"][k]),
                 "centroid": [round(float(v), 3) for v in d["centroid"][k]],
                 "length_span": float(d["span"][k, 0]), "extents": [float(v) for v in d["extents"][k]],
                 "elongation": float(d["elongation"][k]), "alignment": float(d["alignment"][k]),
                 "alignment_local": float(d["alignment_local"][k]), "B_coherence": float(d["B_coherence"][k]),
                 "mean_B": float(d["mean_B"][k]), "tube_score": float(d["tube_score"][k]),
                 "region": str(reg[k]),
                 "cells_per_level": {lv: int(c[k]) for lv, c in d["cells_per_level"].items()}}
                for k in sel[:20]],
            "seconds": {"group_cells": t_group, "tube_descriptors": t_desc},
            "peak_rss_gb_so_far": peak_rss_gb(),
        }
        summary["sets"][name] = s
        print(f"[{name}] {s['cells']} cells -> {K} components ({n_unstitched} before stitching), "
              f"{int(big.sum())} with >= {args.min_cells} cells, {len(sel)} tubes; largest holds "
              f"{100 * s['largest_component_volume_fraction']:.1f}% of the volume; "
              f"group {t_group:.1f}s, describe {t_desc:.1f}s, peak RSS {peak_rss_gb():.2f} GB", flush=True)
        with h5py.File(f"{args.out_dir}/components_{name}.h5", "w") as f:
            f.create_dataset("ids", data=ids.astype(np.int32), compression="lzf")
            f.create_dataset("selected", data=sel)
            f.create_dataset("region", data=np.array([r.encode() for r in reg]))
            for k, v in d.items():
                if isinstance(v, dict):
                    for lv, c in v.items():
                        f.create_dataset(f"{k}/{lv}", data=c)
                else:
                    f.create_dataset(k, data=v)
            f.attrs.update(labels=vals, label_file=args.labels, connectivity=26,
                           min_cells=args.min_cells, min_elongation=args.min_elongation,
                           min_alignment=args.min_alignment)
        del ids, d
    summary["total_seconds"] = time.time() - t0
    summary["peak_rss_gb"] = peak_rss_gb()
    with open(f"{args.out_dir}/{args.summary}", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"done in {summary['total_seconds']:.0f}s, peak RSS {summary['peak_rss_gb']:.2f} GB")


if __name__ == "__main__":
    main()
