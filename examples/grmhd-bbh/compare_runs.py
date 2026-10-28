"""Compare current-sheet statistics of GRMHD BBH runs.

Each ``--run NAME=GRID.h5[:LABELS.npy]`` names an etkplot grid file
(``som-features --grid-out``). Without LABELS the labels are predicted with
``--model`` (a frozen :class:`aweSOM.model.SOMModel`, see ``build_model.py``)
and saved as ``<out>/labels_<NAME>_<IT>.npy`` (uint8, 255 = non-finite
features). Sheet clusters come from ``--sheet-clusters`` or from the model's
``sheet_clusters`` metadata.

For every run, :func:`aweSOM.sheet_stats.sheet_statistics` writes
``<out>/runs/<NAME>/<NAME>_sheet_stats.json``, ``..._components.csv`` and
``..._planes.npz``; then the comparison table (``comparison.md``,
``comparison.json``) and the figures (``figures/*.png``) are made from those
files. ``--reuse`` skips the prediction and statistics of any run whose outputs
exist, so a later call with more runs only processes the new ones.

    python compare_runs.py --model som_model.h5 --out OUT \
        --run LR_flux=grid_LR_flux_180224.h5 \
        --run LR=grid_LR_180224.h5 --run LR_flux_CBD_2=grid_LR_flux_CBD_2_180224.h5 --reuse
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

import aweSOM
from aweSOM.sheet_stats import GridFileSource, SheetConfig, sheet_statistics, write_outputs

# categorical slots 1-3 of the reference palette (validated all-pairs for <= 3 series)
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#008300", "#e87ba4", "#eda100"]
SHEET_COLOR = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


def parse_run(spec):
    if "=" not in spec:
        raise SystemExit(f"--run needs NAME=GRID.h5[:LABELS.npy], got {spec!r}")
    name, rest = spec.split("=", 1)
    labels = None
    if ":" in rest and rest.rsplit(":", 1)[1].endswith(".npy"):
        rest, labels = rest.rsplit(":", 1)
    return name, Path(rest), (Path(labels) if labels else None)


# --------------------------------------------------------------------------- labels


def predict_labels(grid, model, out_path, chunk_rows=4_000_000, log=print):
    """Predict uint8 labels for every grid point, level by level, in chunks."""
    from numpy.lib.format import open_memmap

    src = GridFileSource(grid)
    n = src.n_total
    names = src.names
    if list(model.feature_names) != names:
        raise SystemExit(f"model features {model.feature_names} != grid features {names}")
    out = open_memmap(out_path.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8, shape=(n,))
    t0 = time.perf_counter()
    n_bad = 0
    for a in range(0, n, chunk_rows):
        b = min(a + chunk_rows, n)
        X = src.h5["features"][a:b]
        ok = np.all(np.isfinite(X), axis=1)
        lab = np.full(b - a, 255, dtype=np.uint8)
        if ok.any():
            pred = model.predict(X[ok], chunk=2**20, backend="screen")
            if pred.min() < 0 or pred.max() > 254:
                raise SystemExit("labels do not fit in uint8 (or a node is unassigned)")
            lab[ok] = pred.astype(np.uint8)
        n_bad += int((~ok).sum())
        out[a:b] = lab
        el = time.perf_counter() - t0
        log(f"[predict] {b:,}/{n:,} rows, {el:.0f}s, {el / b * 1e6:.2f} us/pt, load {os.getloadavg()[0]:.1f}")
    out.flush()
    del out
    os.replace(out_path.with_suffix(".tmp.npy"), out_path)
    src.close()
    return {"seconds": round(time.perf_counter() - t0, 1), "n_points": n, "n_nonfinite": n_bad,
            "us_per_point": round((time.perf_counter() - t0) / n * 1e6, 3)}


def label_fractions(grid, labels_path):
    """Per-level and total point counts and volume fractions of every label."""
    src = GridFileSource(grid, labels_path)
    out = {"per_level": {}, "volume": {}, "points": {}}
    vol_tot = 0.0
    vol_by = {}
    pts_by = {}
    for L in src.levels:
        n = src.n_points(L)
        lab = np.asarray(src.read(L, 0, n, ("labels",))["labels"])
        counts = np.bincount(lab, minlength=256)
        p = src.read(L, 0, min(n, 4096), ("xyz",))["xyz"].astype(np.float64)
        dx = min(np.min(np.diff(np.unique(p[:, k]))) for k in range(3) if np.unique(p[:, k]).size > 1)
        ids = np.flatnonzero(counts)
        out["per_level"][str(L)] = {"dx": float(dx), "n_points": n,
                                    "fraction": {str(int(c)): counts[c] / n for c in ids}}
        for c in ids:
            vol_by[int(c)] = vol_by.get(int(c), 0.0) + counts[c] * dx**3
            pts_by[int(c)] = pts_by.get(int(c), 0) + int(counts[c])
        vol_tot += n * dx**3
    n_tot = sum(pts_by.values())
    out["volume"] = {str(c): v / vol_tot for c, v in sorted(vol_by.items())}
    out["points"] = {str(c): v / n_tot for c, v in sorted(pts_by.items())}
    src.close()
    return out


# --------------------------------------------------------------------------- table


def _fmt(v, kind="g"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if kind == "pct":
        return f"{100 * v:.3f} %"
    if kind == "int":
        return f"{int(v):,}"
    return f"{v:.4g}"


def table_rows(s):
    """(label, value, kind) rows of the comparison table for one summary."""
    vol, comp, th, feat = s["volume"], s["components"], s["thickness"], s["features"]
    rows = [
        ("iteration", s["attrs"].get("iteration"), "int"),
        ("time [M]", s["attrs"].get("time"), "g"),
        ("sheet clusters", ",".join(map(str, s["sheet_clusters"])), "str"),
        ("owned volume [M^3]", vol["total_owned"], "g"),
        ("sheet volume [M^3]", vol["sheet"], "g"),
        ("sheet volume fraction", vol["sheet_fraction"], "pct"),
    ]
    for L, p in vol["per_level"].items():
        rows.append((f"  level {L} (dx {p['dx']:g}) fraction", p["sheet_fraction"], "pct"))
    rows += [
        ("components (all)", comp["n_components"], "int"),
        ("components > 0.01 M^3", comp["n_above_volume"].get("0.01"), "int"),
        ("components > 1 M^3", comp["n_above_volume"].get("1"), "int"),
        (f"components >= {comp['min_cells_resolved']} cells", comp["n_resolved"], "int"),
        ("median component volume [M^3]", comp["volume_percentiles"].get("50"), "g"),
        ("volume-weighted median component volume [M^3]", comp["volume_weighted_median"], "g"),
        ("largest component / sheet volume", comp["largest_fraction_of_sheet_volume"], "pct"),
        ("10 largest / sheet volume", comp["top10_fraction_of_sheet_volume"], "pct"),
        ("local thickness, EDT ridge median (area-weighted) [M]", th["edt"]["median_area_weighted"], "g"),
    ]
    for L, p in th["edt"]["per_level"].items():
        rows.append((f"  level {L} EDT thickness median [M] / [cells]",
                     f"{_fmt(p['median'])} / {_fmt(p['median_in_cells'])}", "str"))
    rows += [
        ("component thickness, inertia median [M]", th["inertia"]["median"], "g"),
        ("component thickness, inertia volume-weighted median [M]", th["inertia"]["median_volume_weighted"], "g"),
        ("component thickness, EDT median [M]", th["edt"]["component_median"], "g"),
        ("aspect L1/L3, median", th["aspect"]["median_13"], "g"),
        ("aspect L2/L3, median", th["aspect"]["median_23"], "g"),
    ]
    for name in ("log_chi_B", "log_beta_inv", "log_sigma"):
        if name in feat:
            f = feat[name]
            rows.append((f"{name} median inside / outside",
                         f"{_fmt(f['inside_percentiles']['50'])} / {_fmt(f['outside_percentiles']['50'])}", "str"))
    if "between" in s:
        for w, b in s["between"]["widths"].items():
            rows.append((f"between holes (w = {w} M): share of sheet volume", b["fraction_of_sheet_volume"], "pct"))
            rows.append((f"between holes (w = {w} M): sheet filling / enhancement",
                         f"{_fmt(b['sheet_filling'], 'pct')} / {_fmt(b['enhancement'])}", "str"))
    return rows


def write_table(summaries, out):
    names = list(summaries)
    per = {n: table_rows(s) for n, s in summaries.items()}
    labels = [r[0] for r in per[names[0]]]
    lines = ["| quantity | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    js = {}
    for i, lab in enumerate(labels):
        cells = []
        for n in names:
            rows = per[n]
            row = rows[i] if i < len(rows) and rows[i][0] == lab else next((r for r in rows if r[0] == lab), None)
            if row is None:
                cells.append("n/a")
                continue
            _, v, kind = row
            cells.append(str(v) if kind == "str" else _fmt(v, kind))
            js.setdefault(lab, {})[n] = v
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    (out / "comparison.json").write_text(json.dumps(js, indent=1, default=str) + "\n")
    return "\n".join(lines)


# --------------------------------------------------------------------------- figures


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _legend(fig, axes, names):
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=RUN_COLORS[i % len(RUN_COLORS)], lw=2) for i in range(len(names))]
    if len(names) > 1:
        fig.legend(handles, names, loc="outside upper center", ncol=len(names), frameon=False, fontsize=9)


def _step(ax, edges, y, color, ls="-", label=None):
    ax.stairs(y, edges, color=color, lw=2, ls=ls, label=label)


def make_figures(summaries, planes, out, dpi=160):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle, Rectangle

    out.mkdir(parents=True, exist_ok=True)
    names = list(summaries)
    color = {n: RUN_COLORS[i % len(RUN_COLORS)] for i, n in enumerate(names)}
    title_runs = ", ".join(names)
    paths = {}

    # component volumes
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for n, s in summaries.items():
        h = s["components"]["volume_hist"]
        e = np.asarray(h["edges"])
        _step(axes[0], e, np.asarray(h["count"], float), color[n])
        sv = s["volume"]["sheet"]
        _step(axes[1], e, np.asarray(h["volume"]) / sv if sv else np.zeros(len(e) - 1), color[n])
    axes[0].set(xscale="log", yscale="log", xlabel="component volume [M$^3$]", ylabel="number of components")
    axes[1].set(xscale="log", xlabel="component volume [M$^3$]", ylabel="share of sheet volume per bin")
    for ax in axes:
        _style(ax)
    axes[0].set_title("Connected sheet components", fontsize=10, color=INK, loc="left")
    axes[1].set_title("Where the sheet volume sits", fontsize=10, color=INK, loc="left")
    _legend(fig, axes, names)
    paths["component_volume"] = out / "component_volume.png"
    fig.savefig(paths["component_volume"], dpi=dpi)
    plt.close(fig)

    # thickness
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    dxs = set()
    for n, s in summaries.items():
        th = s["thickness"]
        e = np.asarray(th["edt"]["hist"]["edges"])
        a = np.asarray(th["edt"]["hist"]["area"])
        _step(axes[0], e, a / a.sum() if a.sum() else a, color[n])
        c = np.asarray(th["inertia"]["hist"]["count"], float)
        _step(axes[1], e, c / c.sum() if c.sum() else c, color[n])
        dxs |= {p["dx"] for p in s["volume"]["per_level"].values()}
    for ax in axes:
        for dx in sorted(dxs):
            ax.axvline(dx, color=INK2, lw=0.8, ls=":")
            ax.text(dx, 1.0, f" dx={dx:g}", transform=ax.get_xaxis_transform(), fontsize=7, color=INK2,
                    va="top", rotation=90)
        ax.set_xscale("log")
        _style(ax)
    axes[0].set(xlabel="local thickness 2d - dx/2 at ridge cells [M]", ylabel="area-weighted share")
    axes[1].set(xlabel="component thickness sqrt(12 lambda_min) [M]", ylabel="share of resolved components")
    axes[0].set_title("Local thickness (distance transform)", fontsize=10, color=INK, loc="left")
    axes[1].set_title("Component thickness (inertia tensor)", fontsize=10, color=INK, loc="left")
    _legend(fig, axes, names)
    paths["thickness"] = out / "thickness.png"
    fig.savefig(paths["thickness"], dpi=dpi)
    plt.close(fig)

    # features inside / outside
    feats = [f for f in ("log_chi_B", "log_beta_inv", "log_sigma") if all(f in s["features"] for s in summaries.values())]
    if feats:
        fig, axes = plt.subplots(1, len(feats), figsize=(4.2 * len(feats), 3.8), constrained_layout=True, squeeze=False)
        for ax, f in zip(axes[0], feats):
            for n, s in summaries.items():
                h = s["features"][f]
                e = np.asarray(h["edges"])
                w = np.diff(e)
                for key, ls in (("inside_volume", "-"), ("outside_volume", "--")):
                    y = np.asarray(h[key])
                    _step(ax, e, y / y.sum() / w if y.sum() else y, color[n], ls=ls)
            ax.set(xlabel=f, ylabel="volume-weighted pdf")
            _style(ax)
        axes[0][0].set_title("solid: inside sheets, dashed: outside", fontsize=10, color=INK, loc="left")
        _legend(fig, axes, names)
        paths["features"] = out / "features_inside_outside.png"
        fig.savefig(paths["features"], dpi=dpi)
        plt.close(fig)

    # radial profiles and the profile along the hole-hole segment
    if all("radial" in s for s in summaries.values()):
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
        for n, s in summaries.items():
            r = s["radial"]
            e = np.asarray(r["edges"])
            for ax, key in zip(axes[:3], ("hole1", "hole2", "com")):
                tot = np.asarray(r[key]["total_volume"])
                sh = np.asarray(r[key]["sheet_volume"])
                with np.errstate(invalid="ignore", divide="ignore"):
                    _step(ax, e, np.where(tot > 0, sh / tot, np.nan), color[n])
            p = s["between"]["segment_profile"]
            tot = np.asarray(p["total_volume"])
            sh = np.asarray(p["sheet_volume"])
            with np.errstate(invalid="ignore", divide="ignore"):
                _step(axes[3], np.asarray(p["s_edges"]), np.where(tot > 0, sh / tot, np.nan), color[n])
        for ax, key in zip(axes[:3], ("hole 1", "hole 2", "centre of mass")):
            ax.set(xlabel=f"distance from {key} [M]", ylabel="sheet volume / volume", xlim=(0, 50))
            _style(ax)
        w = summaries[names[0]]["between"]["default_width"]
        axes[3].set(xlabel="s along hole 1 -> hole 2", ylabel="sheet volume / volume")
        axes[3].axvspan(0, 1, color=GRID, alpha=0.4, lw=0)
        axes[3].set_title(f"box |a|,|b| <= {w:g} M around the axis", fontsize=9, color=INK2, loc="left")
        for ax in axes:
            _style(ax)
        _legend(fig, axes, names)
        paths["radial"] = out / "radial_profiles.png"
        fig.savefig(paths["radial"], dpi=dpi)
        plt.close(fig)

    # z = 0 slices
    if planes:
        k = len(names)
        fig, axes = plt.subplots(1, k, figsize=(5.6 * k, 5.4), constrained_layout=True, squeeze=False)
        bgv = np.concatenate([np.asarray(p[L]["background"]).ravel() for p in planes.values() for L in p])
        bgv = bgv[np.isfinite(bgv)]
        vmin, vmax = np.percentile(bgv, [1, 99]) if bgv.size else (0, 1)
        grey = plt.get_cmap("Greys_r").copy()
        grey.set_bad((0, 0, 0, 0))
        red = ListedColormap([SHEET_COLOR])
        for ax, n in zip(axes[0], names):
            pl = planes.get(n, {})
            for L in sorted(pl):
                p = pl[L]
                ext = np.asarray(p["extent"])
                bg = np.asarray(p["background"], dtype=np.float64)
                shv = np.asarray(p["sheet"])
                if L != max(pl):
                    # close the blank half-cell strip along the edges of finer boxes
                    bg, shv = _fill_unowned(bg, shv)
                im = ax.imshow(bg.T, origin="lower", extent=ext, cmap=grey, vmin=vmin, vmax=vmax,
                               interpolation="nearest")
                sh = np.where(shv.T == 1, 1.0, np.nan)
                ax.imshow(sh, origin="lower", extent=ext, cmap=red, vmin=0, vmax=1, interpolation="nearest")
                if L != min(pl):
                    ax.add_patch(Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], fill=False,
                                           ec=INK2, lw=0.6, ls="--"))
            s = summaries[n]
            if "holes" in s:
                rad = s["attrs"].get("horizon_radius", [1.0, 1.0])
                for (hx, hy, _), rr in zip(s["holes"]["positions"], rad):
                    ax.add_patch(Circle((hx, hy), rr, fc="white", ec=INK, lw=0.8, zorder=3))
            f = s["volume"]["sheet_fraction"]
            ax.set_title(f"{n}: z = 0, sheet cells red (volume fraction {100 * f:.2f} %)", fontsize=10,
                         color=INK, loc="left")
            ax.set(xlabel="x [M]", ylabel="y [M]")
            if pl:
                ext = np.array([pl[L]["extent"] for L in pl])
                ax.set_xlim(ext[:, 0].min(), ext[:, 1].max())
                ax.set_ylim(ext[:, 2].min(), ext[:, 3].max())
            ax.set_aspect("equal")
            ax.tick_params(colors=INK2, labelsize=8)
        cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.01)
        cb.set_label(summaries[names[0]]["config"]["plane_background"], color=INK)
        paths["slices"] = out / "slices_z0.png"
        fig.savefig(paths["slices"], dpi=dpi)
        plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def _fill_unowned(bg, sheet):
    """Fill unowned cells (sheet == -1) that touch an owned cell with that cell's values.

    A coarse cell on the edge of a finer box is owned by the finer level, which
    covers only half a fine cell beyond its last point; without this one-cell
    rim the half-coarse-cell strip in between would be blank."""
    from scipy import ndimage

    bad = sheet < 0
    if not bad.any() or bad.all():
        return bg, sheet
    dist, idx = ndimage.distance_transform_edt(bad, return_indices=True)
    rim = bad & (dist <= 1.5)
    bg, sheet = bg.copy(), sheet.copy()
    src = tuple(i[rim] for i in idx)
    bg[rim] = bg[src]
    sheet[rim] = sheet[src]
    return bg, sheet


def load_planes(path):
    z = np.load(path)
    out = {}
    for key in z.files:
        lev, field = key.split("_", 1)
        out.setdefault(int(lev[1:]), {})[field] = z[key]
    return out


# --------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=GRID.h5[:LABELS.npy]")
    ap.add_argument("--model", type=Path, default=None, help="SOMModel file (.h5 or text) for labels")
    ap.add_argument("--sheet-clusters", default=None, help="comma-separated sheet cluster ids")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="", help="suffix of the per-run output directory (e.g. a cluster subset)")
    ap.add_argument("--reuse", action="store_true", help="reuse existing labels and per-run statistics")
    ap.add_argument("--chunk-rows", type=int, default=4_000_000)
    ap.add_argument("--connectivity", type=int, default=6, choices=[6, 18, 26])
    ap.add_argument("--min-cells", type=int, default=64)
    ap.add_argument("--between-widths", default="2,4,8")
    ap.add_argument("--between-default", type=float, default=4.0)
    ap.add_argument("--plane-z", type=float, default=0.0)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"aweSOM from {aweSOM.__file__}", flush=True)

    model = None
    if args.model is not None:
        from aweSOM.model import SOMModel

        model = SOMModel.load(args.model)
    if args.sheet_clusters:
        clusters = [int(c) for c in args.sheet_clusters.split(",")]
    elif model is not None and "sheet_clusters" in model.metadata:
        clusters = [int(c) for c in model.metadata["sheet_clusters"].split(",")]
    else:
        raise SystemExit("give --sheet-clusters or a --model with sheet_clusters metadata")
    print(f"sheet clusters: {clusters}", flush=True)
    cfg = SheetConfig(connectivity=args.connectivity, min_cells=args.min_cells,
                      between_widths=tuple(float(w) for w in args.between_widths.split(",")),
                      between_default=args.between_default, plane_z=args.plane_z, chunk_rows=args.chunk_rows)

    summaries, planes = {}, {}
    for spec in args.run:
        name, grid, labels = parse_run(spec)
        run_dir = args.out / "runs" / (name + (f"_{args.tag}" if args.tag else ""))
        stats_json = run_dir / f"{name}_sheet_stats.json"
        if args.reuse and stats_json.exists():
            s = json.loads(stats_json.read_text())
            if s["sheet_clusters"] == sorted(clusters):
                print(f"[{name}] reusing {stats_json}", flush=True)
                summaries[name] = s
                planes[name] = load_planes(run_dir / f"{name}_planes.npz")
                continue
        info = {"grid": str(grid), "load_start": os.getloadavg()}
        src = GridFileSource(grid)
        it = src.attrs.get("iteration", "x")
        src.close()
        if labels is None:
            labels = args.out / f"labels_{name}_{it}.npy"
            if not (args.reuse and labels.exists()):
                if model is None:
                    raise SystemExit(f"[{name}] no labels and no --model")
                print(f"[{name}] predicting labels -> {labels}", flush=True)
                info["predict"] = predict_labels(grid, model, labels, args.chunk_rows)
                info["predict"]["load_end"] = os.getloadavg()
                info["predict"]["model"] = str(args.model)
        info["labels"] = str(labels)
        t0 = time.perf_counter()
        info["label_fractions"] = label_fractions(grid, labels)
        info["label_fractions_seconds"] = round(time.perf_counter() - t0, 1)
        src = GridFileSource(grid, labels)
        res = sheet_statistics(src, clusters, config=cfg)
        src.close()
        info["load_end"] = os.getloadavg()
        info["peak_rss_gb"] = round(rss_gb(), 3)
        res.summary["run"] = {"name": name, **info}
        paths = write_outputs(res, run_dir, name)
        print(f"[{name}] wrote {paths}", flush=True)
        summaries[name] = res.summary
        planes[name] = res.planes

    md = write_table(summaries, args.out)
    figs = make_figures(summaries, planes, args.out / "figures")
    print(md, flush=True)
    print(f"figures: {figs}", flush=True)
    print(f"peak RSS {rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
