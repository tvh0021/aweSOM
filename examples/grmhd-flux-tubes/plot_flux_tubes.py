"""Figures for the LR_flux flux-tube run (reads run_flux_tubes.py's outputs).

    PYTHONPATH=<worktree>/src python plot_flux_tubes.py --grid grid_LR_flux_180224.h5 \
        --B B_LR_flux_180224.h5 --labels labels_LR_flux_180224_mc0.05.npy --dir . \
        --candidate mc005_c0 --candidate-labels 0 --contrast sheet_c9 c10

Writes fig_real_top10.png, fig_real_slice.png and fig_real_distributions.png.
"""

import argparse

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

DX = {5: 0.25, 6: 0.125, 7: 0.0625}
SET_COLOURS = ["#3b6ea8", "#d9822b", "#6aa84f", "#8e5ea2"]


def load_set(path):
    with h5py.File(path, "r") as f:
        d = {k: f[k][()] for k in f if not isinstance(f[k], h5py.Group)}
        d["region"] = np.array([r.decode() for r in d["region"]])
    return d


def top10_figure(xyz, holes, comp, out, title):
    ids, sel = comp["ids"], comp["selected"]
    top = sel[:10]
    if len(top) < 10:  # fill with the best-scoring remaining big components
        big = np.flatnonzero(comp["n_cells"] >= 100)
        rest = big[~np.isin(big, top)]
        rest = rest[np.argsort(-np.nan_to_num(comp["tube_score"][rest]))]
        top = np.concatenate([top, rest[:10 - len(top)]])
    rows = np.flatnonzero(np.isin(ids, top))
    cid = ids[rows]
    rng = np.random.default_rng(0)
    keep = np.zeros(len(rows), bool)
    for k in top:
        r = np.flatnonzero(cid == k)
        keep[r if len(r) <= 6000 else rng.choice(r, 6000, replace=False)] = True
    rows, cid = rows[keep], cid[keep]
    rank = {int(k): i for i, k in enumerate(top)}
    col = plt.cm.tab10(np.array([rank[int(k)] for k in cid]))
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(*xyz[rows].T, s=0.5, c=col)
    ax.scatter(*holes.T, s=40, c="k", marker="o")
    ax.set_xlabel("x [M]"); ax.set_ylabel("y [M]"); ax.set_zlabel("z [M]")
    for j, (a, b, lab) in enumerate([(0, 1, "xy (orbital plane)"), (0, 2, "xz")]):
        axp = fig.add_subplot(1, 3, 2 + j)
        axp.scatter(xyz[rows, a], xyz[rows, b], s=0.3, c=col, rasterized=True)
        axp.scatter(holes[:, a], holes[:, b], s=30, c="k")
        axp.set_aspect("equal")
        axp.set_xlabel("xyz"[a] + " [M]"); axp.set_ylabel("xyz"[b] + " [M]")
        axp.set_title(f"projection on {lab}")
    handles = [plt.Line2D([], [], ls="", marker="o", color=plt.cm.tab10(i),
                          label=f"#{i + 1} id {k}: {comp['region'][k]}, L={comp['span'][k, 0]:.1f} M, "
                                f"el {comp['elongation'][k]:.1f}, al {comp['alignment'][k]:.2f}"
                                + ("" if k in sel else " (fails cut)"))
               for i, k in enumerate(top)]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return top


def slice_figure(xyz, level, labels, lbi, B, comp, k, out, cand_labels, title, label_name=""):
    ids = comp["ids"]
    rows_k = np.flatnonzero(ids == k)
    lv = np.bincount(level[rows_k]).argmax()
    h = DX[int(lv)]
    c = comp["centroid"][k]
    # axis-aligned plane through the centroid holding the most tube cells
    best = None
    for n in range(3):
        cn = xyz[rows_k[0], n] + h * np.round((c[n] - xyz[rows_k[0], n]) / h)
        cnt = np.sum((level[rows_k] == lv) & (np.abs(xyz[rows_k, n] - cn) < h / 2))
        if best is None or cnt > best[2]:
            best = (n, cn, cnt)
    n, cn, _ = best
    a, b = [i for i in range(3) if i != n]
    half = 0.6 * comp["span"][k, 0] + 1.0
    win = ((level == lv) & (np.abs(xyz[:, n] - cn) < h / 2)
           & (np.abs(xyz[:, a] - c[a]) < half) & (np.abs(xyz[:, b] - c[b]) < half))
    r = np.flatnonzero(win)
    ia = np.rint((xyz[r, a] - xyz[r, a].min()) / h).astype(int)
    ib = np.rint((xyz[r, b] - xyz[r, b].min()) / h).astype(int)
    shape = (ia.max() + 1, ib.max() + 1)
    ext = (xyz[r, a].min() - h / 2, xyz[r, a].max() + h / 2, xyz[r, b].min() - h / 2, xyz[r, b].max() + h / 2)

    def img(v, fill=np.nan):
        im = np.full(shape, fill, dtype=float)
        im[ia, ib] = v
        return im.T

    lab_img = img(labels[r])
    tube_img = img((ids[r] == k).astype(float), 0.0)
    cand_img = img(np.isin(labels[r], cand_labels).astype(float), 0.0)
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    nc = int(labels.max()) + 1
    cmap = ListedColormap(plt.cm.tab20(np.arange(nc) % 20))
    im0 = axs[0].imshow(lab_img, origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=nc - 0.5,
                        interpolation="nearest")
    cb = fig.colorbar(im0, ax=axs[0], ticks=range(nc), fraction=0.046)
    cb.set_label(f"SOM cluster ({label_name})")
    axs[0].contour(cand_img, levels=[0.5], colors="k", linewidths=0.6, extent=ext, origin="lower")
    axs[0].contour(tube_img, levels=[0.5], colors="w", linewidths=1.4, extent=ext, origin="lower")
    axs[0].set_title(f"clusters; black: candidate cluster, white: component {k}")
    im1 = axs[1].imshow(img(lbi[r]), origin="lower", extent=ext, cmap="viridis", interpolation="nearest")
    fig.colorbar(im1, ax=axs[1], fraction=0.046).set_label(r"$\log_{10}\beta^{-1}$")
    axs[1].contour(tube_img, levels=[0.5], colors="w", linewidths=1.4, extent=ext, origin="lower")
    step = max(1, shape[0] // 28)
    Ba, Bb = img(B[r, a]), img(B[r, b])
    nrm = np.hypot(Ba, Bb)
    X, Y = np.meshgrid(np.linspace(ext[0] + h / 2, ext[1] - h / 2, shape[0]),
                       np.linspace(ext[2] + h / 2, ext[3] - h / 2, shape[1]))
    s = (slice(None, None, step), slice(None, None, step))
    axs[1].quiver(X[s], Y[s], (Ba / nrm)[s], (Bb / nrm)[s], color="w", alpha=0.8, scale=40, width=0.002)
    axs[1].set_title("in-plane field direction (unit arrows)")
    for ax in axs:
        ax.set_xlabel("xyz"[a] + " [M]"); ax.set_ylabel("xyz"[b] + " [M]")
    e1 = comp["axes"][k, 0]
    fig.suptitle(f"{title}: level {lv}, plane {'xyz'[n]} = {cn:.4g}; component {k} "
                 f"(L = {comp['span'][k, 0]:.1f} M, e1 = ({e1[0]:.2f}, {e1[1]:.2f}, {e1[2]:.2f}), "
                 f"alignment {comp['alignment'][k]:.2f})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def distributions_figure(sets, out, min_cells):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    axs = axs.ravel()
    for i, (name, d) in enumerate(sets.items()):
        col = SET_COLOURS[i % len(SET_COLOURS)]
        big = d["n_cells"] >= min_cells
        v = np.sort(d["volume"])[::-1]
        axs[0].loglog(v, np.arange(1, len(v) + 1), color=col, lw=2, label=f"{name} ({len(v)} comps)")
        kw = dict(histtype="step", lw=2, color=col, density=True, label=f"{name} ({big.sum()})")
        axs[1].hist(np.log10(d["elongation"][big]), bins=40, range=(0, 1.5), **kw)
        axs[2].hist(d["alignment"][big], bins=40, range=(0, 1), **kw)
        axs[3].hist(d["align_axes"][big, 2], bins=40, range=(0, 1), **kw)
        axs[4].hist(np.log10(d["flatness"][big]), bins=40, range=(0, 1.5), **kw)
        axs[5].scatter(d["elongation"][big], d["alignment"][big], s=4, color=col, alpha=0.5, label=name)
    axs[0].set_xlabel("component volume [M$^3$]"); axs[0].set_ylabel("number of components >= volume")
    axs[1].set_xlabel(r"$\log_{10}$ elongation (extent$_1$/extent$_2$)")
    axs[2].set_xlabel(r"alignment $\langle|\cos(B, e_1)|\rangle$")
    axs[3].set_xlabel(r"$\langle|\cos(B, e_3)|\rangle$ (field along the shortest axis)")
    axs[4].set_xlabel(r"$\log_{10}$ flatness (extent$_2$/extent$_3$)")
    axs[5].set_xscale("log"); axs[5].set_xlabel("elongation"); axs[5].set_ylabel("alignment")
    axs[5].axvline(3, color="0.5", lw=0.8, ls="--"); axs[5].axhline(0.7, color="0.5", lw=0.8, ls="--")
    for ax in axs[1:5]:
        ax.set_ylabel("density of components")
    for ax in axs:
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"component descriptors (components with >= {min_cells} cells; dashed: tube cuts)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--B", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--candidate", default="mc005_c0")
    ap.add_argument("--candidate-labels", type=int, nargs="+", default=[0])
    ap.add_argument("--contrast", nargs="*", default=["sheet_c9", "c10"])
    ap.add_argument("--min-cells", type=int, default=100)
    ap.add_argument("--label-name", default="merge_cost 0.05")
    args = ap.parse_args()
    with h5py.File(args.grid, "r") as g:
        xyz = g["xyz"][:]
        level = g["level"][:]
        lbi = g["features"][:, 0]
        holes = g.attrs["horizon_centres"]
    with h5py.File(args.B, "r") as f:
        B = f["B"][:]
    from aweSOM.npy_parts import open_npy
    labels = open_npy(args.labels, mmap_mode=None)
    comp = load_set(f"{args.dir}/components_{args.candidate}.h5")
    top = top10_figure(xyz, holes, comp, f"{args.dir}/fig_real_top10.png",
                       f"LR_flux it 180224: top 10 components of {args.candidate} by tube score")
    slice_figure(xyz, level, labels, lbi, B, comp, int(top[0]), f"{args.dir}/fig_real_slice.png",
                 args.candidate_labels, f"{args.candidate}", label_name=args.label_name)
    sets = {args.candidate: comp}
    for name in args.contrast:
        sets[name] = load_set(f"{args.dir}/components_{name}.h5")
    distributions_figure(sets, f"{args.dir}/fig_real_distributions.png", args.min_cells)


if __name__ == "__main__":
    main()
