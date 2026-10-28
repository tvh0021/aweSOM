"""Figures for the report.

    python figures.py slice --it 135168 --labels A.npy B.npy ... --titles ... --out fig.png
    python figures.py mc --out fig.png CONFIG [CONFIG ...]
    python figures.py variants --out fig.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from common import COLS, DX, RES, load_grid  # noqa: E402

# reference categorical palette (dataviz skill), fixed order; clusters past 8 fold to grey
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#b9b8b2"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def plane_composite(grid, values, z=0.0, fill=np.nan):
    """[(extent, 2D array)] per level at the plane z, coarse first."""
    out = []
    for g in sorted(grid.groups):
        m = np.full(grid.shape[g], fill, dtype=np.float32)
        m.ravel()[grid.flat[g]] = values[grid.rows[g]]
        k = int(round((z - grid.origin[g][2]) / DX[g]))
        sl = m[:, :, k].T
        lo = grid.origin[g]
        nx, ny = grid.shape[g][:2]
        ext = (lo[0] - DX[g] / 2, lo[0] + (nx - 0.5) * DX[g], lo[1] - DX[g] / 2, lo[1] + (ny - 0.5) * DX[g])
        out.append((ext, sl))
    return out


def label_cmap(order):
    """Colour each cluster id by its rank in ``order`` (largest-volume first)."""
    k = max(order) + 1
    cols = [OTHER] * k
    for r, c in enumerate(order[: len(SERIES)]):
        cols[c] = SERIES[r]
    return ListedColormap(cols), k


def slice_fig(a):
    grid = load_grid(a.it)
    labs = [np.load(p) for p in a.labels]
    sheet = np.load(COLS / f"{a.it}_sheet_T1.npy")
    vol = grid.cell_volume()
    k = max(int(x.max()) + 1 for x in labs)
    vf = np.bincount(labs[0][labs[0] >= 0], weights=vol[labs[0] >= 0], minlength=k)
    order = list(np.argsort(-vf))
    if a.sheet_cluster is not None:  # the sheet cluster always gets the first (blue) slot
        order.remove(a.sheet_cluster)
        order.insert(0, a.sheet_cluster)
    cmap, k = label_cmap(order)
    n = len(labs) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.6), constrained_layout=True)
    win = [float(v) for v in a.window.split(",")]
    for ax, lab, title in zip(axes, labs, a.titles):
        for ext, sl in plane_composite(grid, lab.astype(np.float32), fill=np.nan):
            sl = np.where(sl < 0, np.nan, sl)
            ax.imshow(sl, origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=k - 0.5, interpolation="nearest")
        ax.set_title(title, color=INK)
    ax = axes[-1]
    for ext, sl in plane_composite(grid, sheet.astype(np.float32), fill=np.nan):
        ax.imshow(np.where(np.isnan(sl), np.nan, sl), origin="lower", extent=ext, cmap=ListedColormap(["#f0efec", INK]),
                  vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("metric sheets (|C|/|B|>1, θ>90°)", color=INK)
    for ax in axes:
        ax.set_xlim(win[0], win[1])
        ax.set_ylim(win[2], win[3])
        ax.set_xlabel("x [M]")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y [M]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[r]) for r in range(min(len(order), len(SERIES)))]
    names = [f"c{c}" + (" (sheet)" if c == a.sheet_cluster else "") for c in order[: len(SERIES)]]
    fig.legend(handles, names, loc="outside lower center", ncol=min(8, len(names)), frameon=False)
    fig.suptitle(a.suptitle, color=INK)
    fig.savefig(a.out, dpi=200)


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def mc_fig(a):
    rows = read_csv(RES / "tables" / "scoreboard.csv")
    seeds = {(r["config"], r["it"], r["mc"]): r for r in read_csv(RES / "tables" / "seeds.csv")}
    metrics = [("C", "coherence C"), ("D", "sheet match D (mean of T=1, T=3)"), ("S", "stability S (seeds)"), ("P", "physicality P")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True, sharex=True)
    for ci, cfg in enumerate(a.configs):
        rs = sorted([r for r in rows if r["config"] == cfg and r["it"] == str(a.it)], key=lambda r: float(r["mc"]))
        mcs = sorted({float(r["mc"]) for r in rs})
        for ax, (m, lab) in zip(axes, metrics):
            ys, es = [], []
            for mc in mcs:
                g = seeds.get((cfg, str(a.it), str(mc)))
                if m == "S":
                    v = [float(g["S_seed"])] if g else []
                elif m == "P":
                    v = [float(r["P_seed"]) for r in rs if float(r["mc"]) == mc and r.get("P_seed")]
                else:
                    v = [float(r[m]) for r in rs if float(r["mc"]) == mc]
                ys.append(np.mean(v) if v else np.nan)
                es.append(np.std(v) if len(v) > 1 else 0)
            ax.errorbar(mcs, ys, yerr=es, color=SERIES[ci], lw=2, marker="o", ms=5, capsize=2,
                        label=a.labels[ci] if a.labels else cfg)
            ax.set_title(lab, color=INK)
            ax.set_xlabel("merge cost")
            ax.grid(axis="y", color="#e4e3df", lw=0.6)
    axes[0].set_ylabel(f"score (it {a.it}, mean ± sd over seeds)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=len(a.configs), frameon=False)
    fig.savefig(a.out, dpi=200)


def variants_fig(a):
    seeds = read_csv(RES / "tables" / "seeds.csv")
    sel = [r for r in seeds if r["it"] == str(a.it) and float(r["mc"]) == a.mc and r["config"] in a.configs]
    sel.sort(key=lambda r: a.configs.index(r["config"]))
    metrics = [("C", "C_std", "coherence C"), ("D", "D_std", "sheet match D (mean of T=1, T=3)"),
               ("S_seed", None, "stability S (seeds)"), ("P_seed", "P_seed_std", "physicality P")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True, sharey=True)
    y = np.arange(len(sel))
    for ax, (m, sd, title) in zip(axes, metrics):
        v = [float(r[m]) for r in sel]
        e = [float(r[sd]) if sd else 0 for r in sel]
        ax.barh(y, v, xerr=e, color=SERIES[0], height=0.6, error_kw=dict(ecolor=MUTED, lw=1))
        for yi, vi, ei in zip(y, v, e):
            ax.text(vi + ei + 0.02, yi, f"{vi:.3f}", va="center", color=INK, fontsize=8)
        ax.set_title(title, color=INK)
        ax.set_xlim(0, 1.05)
        ax.grid(axis="x", color="#e4e3df", lw=0.6)
    axes[0].set_yticks(y, [a.labels[a.configs.index(r["config"])] if a.labels else r["config"] for r in sel])
    axes[0].invert_yaxis()
    fig.suptitle(f"Feature variants at merge cost {a.mc}, it {a.it} (3 seeds, mean ± sd)", color=INK)
    fig.savefig(a.out, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["slice", "mc", "variants"])
    ap.add_argument("--it", type=int, default=135168)
    ap.add_argument("--mc", type=float, default=0.1)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--titles", nargs="*")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--sheet-cluster", type=int)
    ap.add_argument("--window", default="-22,-6,-12,4")
    ap.add_argument("--suptitle", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"slice": slice_fig, "mc": mc_fig, "variants": variants_fig}[a.kind](a)


if __name__ == "__main__":
    main()
