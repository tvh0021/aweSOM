"""Label GRMHD snapshots block by block with a frozen SOM model.

For each iteration and each level block (``aweSOM.io``), this script reads the
inputs, computes the model features with the thresholds frozen in the model
(``aweSOM.features_grmhd``), predicts the labels, and then:

* appends labels, xyz and level (and, with ``--save-features``, the
  features) to ``<out>/it<iteration>.h5``;
* records per-block statistics in that file: cells per cluster, histograms of
  each feature for the medians, and the midplane cells for the quick look;
* rewrites ``<out>/it<iteration>_stats.json`` (per level: cluster volume
  fractions, sheet-cluster cells, feature medians) and
  ``<out>/it<iteration>_quicklook.png`` (midplane labels and log_chi_B)
  after each level and at the end.

It can be resumed: blocks recorded in the iteration file are skipped and rows
left over from an interrupted block are cut. The file records the settings that
decide what a block holds (slab width, levels, halo, reader, source and model
names, ...) and refuses to resume with different ones, since block keys name
only the slab start. It also records a signature of the source's layout
(levels and boxes, or components) for the iteration; if the source has changed
since (a level or box that arrived after ``--settle`` seconds of quiet), the
rows are discarded and the iteration is redone, because which cells a coarse
block owns depends on the finer levels. In ``--watch`` mode an iteration is
taken up only once every file it refers to can be opened; read errors are
retried at the next poll, and a completed iteration whose layout changes is
redone. Memory is bounded by one block (``--slab`` x points of one patch, all
inputs plus feature temporaries) plus the reader's cache (``--cache-gb``).

Examples::

    # one iteration of the BBH Carpet run, levels 5-7
    python analyze_chunks.py --source idx3d_0003.json --model model.h5 \\
        --iterations 180224 --levels 5 6 7 --spheres-from grid_LR_flux_180224.h5 -o out/

    # follow a CarpetX run as it writes Silo output
    python analyze_chunks.py --source run/data/silo --model model.h5 --watch -o out/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from aweSOM.features_grmhd import INPUTS, FeatureThresholds, finish_features, raw_features
from aweSOM.io import CarpetHDF5Reader, CarpetXReader
from aweSOM.io.carpet import looks_like_carpet
from aweSOM.model import SOMModel

# fixed histogram bins for the medians: (lo, hi) per feature, 0.005 wide
HIST_RANGE = {
    "bz_frac": (0.0, 1.0),
    "pol_frac": (0.0, 1.0),
    "log_beta_inv": (-12.0, 12.0),
    "log_theta": (-12.0, 12.0),
    "log_chi_B": (-3.5, 3.5),
    "log_grad_ln_b2": (-3.5, 3.5),
    "asinh_div_v": (-8.0, 8.0),
}
BIN = 0.005
MID_COLS = ("x", "y", "label", "log_chi_B", "level")


def _hist_edges(name):
    lo, hi = HIST_RANGE[name]
    return lo, hi, int(round((hi - lo) / BIN))


def _hist(x: np.ndarray, name: str) -> np.ndarray:
    """Counts in fixed bins plus underflow (first) and overflow (last)."""
    lo, hi, n = _hist_edges(name)
    i = np.floor((x - lo) / BIN).astype(np.int64) + 1
    i = np.clip(i, 0, n + 1)
    return np.bincount(i, minlength=n + 2)


def _median(h: np.ndarray, name: str) -> float:
    """Median from a histogram (bin centre; NaN if empty)."""
    tot = h.sum()
    if tot == 0:
        return float("nan")
    lo, _, _ = _hist_edges(name)
    k = int(np.searchsorted(np.cumsum(h), 0.5 * tot))
    return float(lo + (k - 1 + 0.5) * BIN)


class IterationFile:
    """``it<iteration>.h5``: rows, plus a record per finished block."""

    def __init__(self, path: Path, n_features: int, save_features: bool, config: Optional[dict] = None,
                 layout: Optional[str] = None):
        self.path = path
        self.f = h5py.File(path, "a")
        f = self.f
        self.restarted = False
        if "labels" in f and config is not None and len(f["blocks"]):
            stored = json.loads(f.attrs.get("config", "null"))
            if stored != config:
                diff = {k: (stored.get(k) if stored else None, config[k]) for k in config
                        if not stored or stored.get(k) != config[k]}
                f.close()
                raise SystemExit(f"{path}: the blocks on file were made with a different configuration "
                                 f"(stored, now): {diff}; use the same settings or a fresh output directory")
        if config is not None:
            f.attrs["config"] = json.dumps(config, sort_keys=True)
        if "labels" in f and layout is not None and len(f["blocks"]) and f.attrs.get("layout", "") != layout:
            # the source changed since these blocks were read (a level or box arrived late): which cells
            # a coarse block owns depends on the finer levels, so none of the stored rows can be kept
            del f["blocks"]
            f.create_group("blocks")
            f.attrs["complete"] = False
            self.restarted = True
        if layout is not None:
            f.attrs["layout"] = layout
        if "labels" not in f:
            opts = dict(chunks=True, compression="lzf")
            f.create_dataset("labels", (0,), np.int32, maxshape=(None,), **opts)
            f.create_dataset("xyz", (0, 3), np.float32, maxshape=(None, 3), **opts)
            f.create_dataset("level", (0,), np.int8, maxshape=(None,), **opts)
            f.create_dataset("finite", (0,), bool, maxshape=(None,), **opts)
            if save_features:
                f.create_dataset("features", (0, n_features), np.float32, maxshape=(None, n_features), **opts)
            f.create_dataset("midplane", (0, len(MID_COLS)), np.float32, maxshape=(None, len(MID_COLS)), **opts)
            f.create_group("blocks")
            f.attrs["complete"] = False
        self.save_features = "features" in f
        # resume: keep only the rows of finished blocks
        rows = mids = 0
        for name in [k for k, g in f["blocks"].items() if "row_stop" not in g.attrs]:
            del f["blocks"][name]  # interrupted while recording the block
        for g in f["blocks"].values():
            rows = max(rows, int(g.attrs["row_stop"]))
            mids = max(mids, int(g.attrs["mid_stop"]))
        for name in ("labels", "xyz", "level", "finite", "features"):
            if name in f and f[name].shape[0] != rows:
                f[name].resize(rows, axis=0)
        if f["midplane"].shape[0] != mids:
            f["midplane"].resize(mids, axis=0)
        self.rows, self.mids = rows, mids

    def done(self) -> set:
        return {g.attrs["key"] for g in self.f["blocks"].values()}

    def append(self, key: str, level: int, labels, xyz, finite, features, mid, stats: Dict[str, np.ndarray], attrs: dict):
        f = self.f
        n = len(labels)
        a, b = self.rows, self.rows + n
        for name, val in (("labels", labels), ("xyz", xyz), ("level", np.full(n, level, np.int8)), ("finite", finite)):
            f[name].resize(b, axis=0)
            f[name][a:b] = val
        if self.save_features:
            f["features"].resize(b, axis=0)
            f["features"][a:b] = features
        m0, m1 = self.mids, self.mids + len(mid)
        f["midplane"].resize(m1, axis=0)
        f["midplane"][m0:m1] = mid
        g = f["blocks"].create_group(key.replace("/", "_"))
        for k, v in stats.items():
            g.create_dataset(k, data=v, compression="lzf" if np.size(v) > 64 else None)
        g.attrs.update(attrs)
        g.attrs.update(key=key, level=level, row_start=a, row_stop=b, mid_start=m0, mid_stop=m1)
        f.flush()  # rows first, block record last: a crash leaves rows that resume cuts
        self.rows, self.mids = b, m1

    def close(self):
        self.f.close()


def summarise(itf: IterationFile, model: SOMModel, sheets: List[int]) -> dict:
    """Per-level and total statistics from the block records."""
    names = tuple(model.feature_names)
    n_cl = int(model.node_cluster.max()) + 1
    per = {}
    for g in itf.f["blocks"].values():
        L = int(g.attrs["level"])
        s = per.setdefault(L, {"counts": np.zeros(n_cl + 1, np.int64), "hist": {n: 0 for n in names},
                               "cells": 0, "nonfinite": 0, "blocks": 0, "dx": float(g.attrs.get("dx", np.nan))})
        s["counts"] += g["cluster_counts"][()]
        for n in names:
            s["hist"][n] = s["hist"][n] + g[f"hist_{n}"][()]
        s["cells"] += int(g.attrs["row_stop"] - g.attrs["row_start"])
        s["nonfinite"] += int(g.attrs["nonfinite"])
        s["blocks"] += 1
    out = {"levels": {}, "sheet_clusters": sheets}
    tot_counts = np.zeros(n_cl + 1, np.int64)
    tot_vol = np.zeros(n_cl + 1)
    tot_hist = {n: 0 for n in names}
    for L, s in sorted(per.items()):
        c = s["counts"]  # index 0 = label -1 (unassigned), k + 1 = cluster k
        finite = int(c.sum())
        vol = s["dx"] ** 3
        tot_counts += c
        tot_vol += c * vol
        for n in names:
            tot_hist[n] = tot_hist[n] + s["hist"][n]
        out["levels"][str(L)] = {
            "dx": s["dx"],
            "blocks": s["blocks"],
            "cells": s["cells"],
            "nonfinite_cells": s["nonfinite"],
            "volume_fraction": {str(k - 1): float(c[k] / max(finite, 1)) for k in range(len(c)) if c[k]},
            "sheet_cells": int(sum(c[k + 1] for k in sheets)),
            "feature_median": {n: _median(np.asarray(s["hist"][n]), n) for n in names},
        }
    vt = tot_vol.sum()
    out["total"] = {
        "cells": int(sum(s["cells"] for s in per.values())),
        "nonfinite_cells": int(sum(s["nonfinite"] for s in per.values())),
        "volume_fraction": {str(k - 1): float(tot_vol[k] / vt) for k in range(len(tot_vol)) if tot_counts[k]} if vt else {},
        "sheet_cells": int(sum(tot_counts[k + 1] for k in sheets)),
        "feature_median": {n: _median(np.asarray(tot_hist[n]), n) if np.size(tot_hist[n]) > 1 else float("nan")
                           for n in names},
    }
    return out


def quicklook(itf: IterationFile, png: Path, n_clusters: int, sheets: List[int], title: str) -> None:
    """Midplane maps of the labels and log_chi_B, finer levels drawn over coarser ones."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    mid = itf.f["midplane"][()]
    if len(mid) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    base = plt.get_cmap("tab20", 20)
    cmap = ListedColormap([base(k % 20) for k in range(max(n_clusters, 1))]).with_extremes(bad=(0, 0, 0, 0))
    for ax, col, kw, lab in (
        (axes[0], 2, dict(cmap=cmap, vmin=-0.5, vmax=n_clusters - 0.5, interpolation="nearest"), "cluster"),
        (axes[1], 3, dict(cmap="magma", vmin=-3, vmax=1.6, interpolation="nearest"), "log_chi_B"),
    ):
        im = None
        levels = sorted(np.unique(mid[:, 4]).astype(int))
        box = [np.inf, -np.inf, np.inf, -np.inf]
        for L in levels:
            p = mid[mid[:, 4] == L]
            xs, ys = np.unique(p[:, 0]), np.unique(p[:, 1])
            dx = np.min(np.diff(xs)) if len(xs) > 1 else 1.0
            dy = np.min(np.diff(ys)) if len(ys) > 1 else 1.0
            i = np.rint((p[:, 0] - xs[0]) / dx).astype(int)
            j = np.rint((p[:, 1] - ys[0]) / dy).astype(int)
            img = np.full((j.max() + 1, i.max() + 1), np.nan)
            img[j, i] = p[:, col]
            if L != levels[-1]:
                # each point is drawn as a pixel of its own level's size, so a
                # coarser level's pixels stop half a coarse cell short of the
                # finer level's region: grow them by one pixel (the finer
                # level is drawn over the rest)
                img = _grow(img)
            ext = (xs[0] - dx / 2, xs[0] + (i.max() + 0.5) * dx, ys[0] - dy / 2, ys[0] + (j.max() + 0.5) * dy)
            im = ax.imshow(np.ma.masked_invalid(img), origin="lower", extent=ext, **kw)
            box = [min(box[0], ext[0]), max(box[1], ext[1]), min(box[2], ext[2]), max(box[3], ext[3])]
            ax.add_patch(plt.Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], fill=False, lw=0.5, ec="0.5"))
        if im is not None:
            fig.colorbar(im, ax=ax, label=lab, shrink=0.85)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
        ax.set_aspect("equal")
        ax.set_title(lab + (f" (sheet clusters {sheets})" if col == 2 and sheets else ""))
    fig.suptitle(title)
    tmp = png.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    os.replace(tmp, png)


def _grow(img: np.ndarray) -> np.ndarray:
    """NaN pixels next to (8-neighbourhood) finite ones take a neighbour's value."""
    out = img.copy()
    pad = np.pad(img, 1, constant_values=np.nan)
    ny, nx = img.shape
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            if dj == di == 0:
                continue
            nb = pad[1 + dj : 1 + dj + ny, 1 + di : 1 + di + nx]
            fill = np.isnan(out) & np.isfinite(nb)
            out[fill] = nb[fill]
    return out


def _midplane(block, z0: float, labels, logchi, xyz) -> np.ndarray:
    """Owned cells of ``block`` in the grid plane nearest z = z0 (within half a cell)."""
    k = int(np.rint((z0 - block.xyz_base[2]) / block.spacing[2])) - int(block.xyz_offset[2])
    nz = block.interior_shape[2]
    if not 0 <= k < nz:
        return np.zeros((0, len(MID_COLS)), np.float32)
    zk = block.xyz_base[2] + (block.xyz_offset[2] + k) * block.spacing[2]
    if abs(zk - z0) > 0.5 * block.spacing[2] * (1 + 1e-9):
        return np.zeros((0, len(MID_COLS)), np.float32)
    # owned points come in C order (x, y, z); pick those with z index k
    idx = np.nonzero(block.owned_mask)[2]
    sel = idx == k
    out = np.empty((int(sel.sum()), len(MID_COLS)), np.float32)
    out[:, 0] = xyz[sel, 0]
    out[:, 1] = xyz[sel, 1]
    out[:, 2] = labels[sel]
    out[:, 3] = logchi[sel]
    out[:, 4] = block.level
    return out


def run_config(args, thr: FeatureThresholds) -> dict:
    """The settings that decide which rows a block holds; a file is only resumed with the same ones."""
    return {
        "slab": int(args.slab), "levels": None if args.levels is None else [int(L) for L in args.levels],
        "halo": int(thr.halo), "save_features": bool(args.save_features), "midplane_z": float(args.midplane_z),
        "reader": args.reader, "carpet_backend": args.carpet_backend, "fill": args.fill,
        "source": Path(args.source).name, "model": Path(args.model).name,
        "hole_centres": getattr(args, "bh", None),
    }


def source_layout(reader, it: int, levels=None) -> str:
    """A signature of the blocks the source holds for ``it`` (levels, boxes or
    components): processing is complete only for the layout it saw. Raises
    OSError while a referenced file is missing or unreadable."""
    if isinstance(reader, CarpetXReader):
        snap = reader.snapshot(it)
        lv = sorted(snap.levels) if levels is None else [int(L) for L in levels]
        entries = [(rl, b.c, [int(v) for v in b.lo], [int(v) for v in b.hi])
                   for rl in lv if rl in snap.levels for b in snap.levels[rl].boxes]
    else:
        ref = reader.reference_var or INPUTS[0]
        lv = reader.levels(it, ref) if levels is None else [int(L) for L in levels]
        if getattr(reader, "backend", None) == "native":
            entries = [(rl, c.c, Path(c.path).name, c.dataset, list(c.shape))
                       for rl in lv for c in reader._cat.components(ref, it, rl)]
        else:
            root = Path(reader.path)
            root = root if root.is_dir() else Path(json.loads(root.read_text())["directory"])
            entries = [(p.name, p.stat().st_size) for p in sorted(root.glob("*.h5"))] + [("levels", lv)]
    return hashlib.sha1(json.dumps(entries, sort_keys=True).encode()).hexdigest() + f":{len(entries)}"


def process_iteration(reader, model: SOMModel, thr: FeatureThresholds, it: int, args, sheets: List[int],
                      layout: Optional[str] = None) -> dict:
    names = tuple(model.feature_names)
    centres = getattr(args, "bh", None)
    if "pol_frac" in names:
        if centres is None or not np.all(np.isfinite(centres)):
            raise ValueError("pol_frac requires --bh with six finite coordinates for this iteration")
        centres = np.asarray(centres, dtype=float).reshape(2, 3)
    out = Path(args.out)
    if layout is None:
        layout = source_layout(reader, it, args.levels)
    itf = IterationFile(out / f"it{it:08d}.h5", len(names), args.save_features, {**run_config(args, thr), "feature_names": list(names)},
                        layout)
    if itf.f.attrs.get("complete", False):
        print(f"[chunks] it={it}: already complete", flush=True)
        itf.close()
        return {}
    if itf.restarted:
        print(f"[chunks] it={it}: the source's blocks changed since the file was written; starting it over",
              flush=True)
    done = itf.done()
    n_cl = int(model.node_cluster.max()) + 1
    jchi = names.index("log_chi_B")
    timers = {"read": 0.0, "features": 0.0, "predict": 0.0, "write": 0.0}
    cells = 0
    t_start = time.time()
    t_read0 = time.time()
    last_level = None
    levels = args.levels
    it_blocks = reader.iter_level_blocks(INPUTS, it, levels=levels, halo=thr.halo, slab=args.slab, skip=done)
    if done:
        print(f"[chunks] it={it}: resuming, {len(done)} blocks done, {itf.rows} rows kept", flush=True)
    title = f"{Path(args.source).name} it={it}"
    try:
        for block in it_blocks:
            t1 = time.time()
            timers["read"] += t1 - t_read0
            if last_level is not None and block.level != last_level:
                _flush_outputs(itf, model, sheets, out, it, title)
            last_level = block.level
            axes = None
            if "pol_frac" in names:
                axes = tuple(block.xyz_base[d] +
                             (block.xyz_offset[d] + np.arange(block.shape[d]) - block.halo[d]) * block.spacing[d]
                             for d in range(3))
            raw = raw_features(block.data, block.spacing, thr.accuracy, names, thr.pipeline,
                               xyz=axes, hole_centres=centres)
            inner = block.interior
            owned = block.owned_mask
            picked = {n: v[inner][owned] for n, v in raw.items()}
            b2 = block.data["smallb2"][inner][owned]
            del raw
            feats = finish_features(picked, b2, thr, names)
            del picked, b2
            xyz64 = block.xyz()
            t2 = time.time()
            timers["features"] += t2 - t1
            labels = model.predict(feats, backend=args.predict_backend)
            t3 = time.time()
            timers["predict"] += t3 - t2
            finite = np.all(np.isfinite(feats), axis=1)
            lab_f = labels[finite]
            counts = np.bincount(lab_f + 1, minlength=n_cl + 1)[: n_cl + 1].astype(np.int64)
            hists = {f"hist_{n}": _hist(feats[finite, j].astype(np.float64), n) for j, n in enumerate(names)}
            mid = _midplane(block, args.midplane_z, labels, feats[:, jchi], xyz64)
            itf.append(
                block.key, block.level, labels, xyz64.astype(np.float32), finite, feats, mid,
                {"cluster_counts": counts, **hists},
                {"nonfinite": int((~finite).sum()), "dx": float(block.spacing[0]), "time": block.time},
            )
            cells += len(labels)
            t_read0 = time.time()
            timers["write"] += t_read0 - t3
            if args.verbose:
                print(f"[chunks] it={it} {block.key}: {len(labels)} cells, {cells / (t_read0 - t_start):.3g} cells/s", flush=True)
            del block, feats, labels, xyz64
    except BaseException:
        itf.close()
        raise
    itf.f.attrs["complete"] = True
    stats = _flush_outputs(itf, model, sheets, out, it, title)
    wall = time.time() - t_start
    run = {"iteration": it, "cells_this_run": cells, "wall_s": wall, "cells_per_s": cells / wall if wall else None,
           "timers_s": timers, "peak_rss_gb": _peak_rss_gb(), "rows_total": itf.rows,
           "cache": {"hits": getattr(reader.cache, "hits", None), "misses": getattr(reader.cache, "misses", None)}}
    stats["run"] = run
    (out / f"it{it:08d}_stats.json").write_text(json.dumps(stats, indent=1))
    itf.close()
    print(f"[chunks] it={it}: {cells} cells in {wall:.1f}s ({run['cells_per_s']:.3g} cells/s), "
          f"read {timers['read']:.1f}s features {timers['features']:.1f}s predict {timers['predict']:.1f}s "
          f"write {timers['write']:.1f}s, peak RSS {run['peak_rss_gb']:.2f} GB", flush=True)
    return stats


def _flush_outputs(itf, model, sheets, out, it, title):
    stats = summarise(itf, model, sheets)
    stats["complete"] = bool(itf.f.attrs.get("complete", False))
    (out / f"it{it:08d}_stats.json").write_text(json.dumps(stats, indent=1))
    quicklook(itf, out / f"it{it:08d}_quicklook.png", int(model.node_cluster.max()) + 1, sheets, title)
    return stats


def _peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


def _ready(reader, it: int, settle: float) -> bool:
    """Whether the files of ``it`` have not changed for ``settle`` seconds.
    (Whether every file they refer to exists is checked by :func:`source_layout`.)"""
    root = Path(reader.path)
    if isinstance(reader, CarpetXReader):
        paths = [reader._file(it)]
        d = Path(str(paths[0]) + ".dir")
        if d.is_dir():
            paths += list(d.iterdir())
    else:
        root = root if root.is_dir() else Path(json.loads(root.read_text())["directory"])
        paths = list(root.glob("*.h5"))
    newest = max((p.stat().st_mtime for p in paths if p.exists()), default=0)
    return time.time() - newest > settle


def make_reader(args):
    spheres = None
    if args.spheres_from:
        with h5py.File(args.spheres_from, "r") as f:
            spheres = list(zip(f.attrs["horizon_centres"], f.attrs["horizon_radius"]))
    cache = args.cache_gb * 1e9
    kind = args.reader
    if kind == "auto":
        kind = "carpet" if looks_like_carpet(args.source) else "carpetx"
    if kind == "carpet":
        return CarpetHDF5Reader(args.source, backend=args.carpet_backend, spheres=spheres, parfile=args.parfile,
                                etkplot_path=args.etkplot_path, cache_bytes=cache)
    return CarpetXReader(args.source, spheres=spheres, fill=args.fill, cache_bytes=cache)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, help="Carpet index/directory or CarpetX Silo/openPMD directory")
    ap.add_argument("--model", required=True, help="frozen SOMModel (.h5) with feature_* metadata")
    ap.add_argument("-o", "--out", required=True, help="output directory")
    ap.add_argument("--iterations", type=int, nargs="+", help="iterations to process")
    ap.add_argument("--watch", action="store_true", help="keep polling the source for new iterations")
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between polls (--watch)")
    ap.add_argument("--settle", type=float, default=120.0, help="seconds an iteration's files must be unchanged")
    ap.add_argument("--max-idle", type=float, default=None, help="stop watching after this many idle seconds")
    ap.add_argument("--levels", type=int, nargs="+", help="levels to process (default: all)")
    ap.add_argument("--slab", type=int, default=64, help="x points per block")
    ap.add_argument("--reader", choices=("auto", "carpet", "carpetx"), default="auto")
    ap.add_argument("--carpet-backend", choices=("auto", "etkplot", "native"), default="native")
    ap.add_argument("--etkplot-path", help="directory holding the etkplot package")
    ap.add_argument("--parfile", help="Carpet: horizons from this parfile's trajectory (etkplot backend)")
    ap.add_argument("--bh", type=float, nargs=6, help="explicit hole centres x1 y1 z1 x2 y2 z2 for pol_frac; fixed for the requested iterations")
    ap.add_argument("--spheres-from", help="HDF5 file with horizon_centres/horizon_radius attributes")
    ap.add_argument("--fill", choices=("coarse", "nan"), default="coarse", help="CarpetX: halo cells outside the level")
    ap.add_argument("--cache-gb", type=float, default=1.0, help="reader cache budget")
    ap.add_argument("--midplane-z", type=float, default=0.0, help="z of the quick-look plane")
    ap.add_argument("--predict-backend", default="screen")
    ap.add_argument("--save-features", action="store_true", help="also store the float32 features")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    model = SOMModel.load(args.model)
    thr = FeatureThresholds.from_model(model)
    if "pol_frac" in model.feature_names and (args.bh is None or not np.all(np.isfinite(args.bh))):
        ap.error("pol_frac requires --bh with six finite coordinates")
    sheets = [int(s) for s in model.metadata.get("sheet_clusters", "").split() if s.lstrip("-").isdigit()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[chunks] model {args.model}: {model.xdim}x{model.ydim}, thresholds {thr}, sheet clusters {sheets}", flush=True)
    reader = make_reader(args)
    print(f"[chunks] reader {type(reader).__name__} ({getattr(reader, 'backend', getattr(reader, 'format', ''))})", flush=True)
    if not args.watch:
        for it in args.iterations or reader.iterations():
            process_iteration(reader, model, thr, it, args, sheets)
        return 0
    idle_since = time.time()
    while True:
        reader = make_reader(args)  # re-list the source
        todo = []
        for it in args.iterations or reader.iterations():
            try:
                if not _ready(reader, it, args.settle):
                    continue
                layout = source_layout(reader, it, args.levels)  # fails while a referenced file is missing
            except OSError as e:
                print(f"[chunks] it={it}: not readable yet ({e})", flush=True)
                continue
            if _complete(out / f"it{it:08d}.h5", layout):
                continue
            todo.append((it, layout))
        for it, layout in todo:
            try:
                process_iteration(reader, model, thr, it, args, sheets, layout)
            except OSError as e:
                print(f"[chunks] it={it}: read failed ({e}); will retry", flush=True)
        if todo:
            idle_since = time.time()
        elif args.max_idle is not None and time.time() - idle_since > args.max_idle:
            print("[chunks] idle; stopping", flush=True)
            return 0
        time.sleep(args.poll)


def _complete(path: Path, layout: Optional[str] = None) -> bool:
    """Whether ``path`` was completed, for the source layout ``layout`` if given."""
    if not path.exists():
        return False
    with h5py.File(path, "r") as f:
        done = bool(f.attrs.get("complete", False))
        return done and (layout is None or f.attrs.get("layout", "") == layout)


if __name__ == "__main__":
    sys.exit(main())
