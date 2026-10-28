"""Proposal figure: the current sheet aweSOM finds between the two BBH jets.

Three stages, each writing its products to ``--out`` so that later stages (and
the figure) can be rerun in seconds:

``labels``  freeze the trained 91x64 lattice (re-clustered at ``--merge-cost``)
            with the frozen training scaler, label every owned grid point of the
            snapshot, and pick the current-sheet cluster from per-cluster feature
            medians (highest log_chi_B and log_grad_ln_b2). Writes
            ``labels_<tag>.npy`` and ``clusters_<tag>.json``.
``sample``  sample the grid's labels and log_beta_inv onto the co-rotating
            meridional and orbital planes (nearest owned grid point of the finest
            level that has one within half a cell diagonal, one KD-tree per level),
            and cut the sheet-cluster cells of the central box for the 3D panel.
            Needs the raw-field planes from ``sample_raw_planes.py`` (run with the
            etkplot interpreter) for the frame and the B field. Writes
            ``planes_<tag>.npz`` and ``sheet3d_<tag>.npz``.
``plot``    draw the figure from those .npz files: PDF + 300-dpi PNG.

Run with the aweSOM interpreter, e.g.::

    PYTHONPATH=<worktree>/src nice -n 10 python proposal_figure.py labels \
        --grid grid_LR_flux_180224.h5 --train-features features_...h5 \
        --som som_object....pkl --out RESULTS --tag it180224
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import h5py
import numpy as np

FEATURES = ["log_beta_inv", "log_theta", "log_chi_B", "log_grad_ln_b2", "asinh_div_v"]
DX = {5: 0.25, 6: 0.125, 7: 0.0625}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- labels


def build_model(som_path: Path, train_features: Path, merge_cost: float):
    import aweSOM
    from aweSOM.model import Scaler, SOMModel

    log(f"aweSOM from {aweSOM.__file__}")
    with h5py.File(train_features, "r") as h:
        names = [n.decode() for n in h["names"][()]]
        raw = h["features"][()]  # float32, the rows train_som.py trained on
        mean_file = h["scaler/mean"][()]
        sigma_file = h["scaler/sigma"][()]
    assert names == FEATURES, names
    # train_som.py scaled with run_som.manual_scaling(x) = (x - mean) / (2 std),
    # with mean and std reduced along axis 0 of the float32 array. That reduction
    # accumulates row by row in float32 and is off by up to ~1% from the float64
    # scaler/mean, scaler/sigma stored in the file, so recompute it the same way:
    # this is the scaler the lattice was actually trained in.
    mean32 = np.mean(raw, axis=0)
    std32 = np.std(raw, axis=0)
    scaler = Scaler("manual", mean32.astype(np.float64), 2.0 * std32.astype(np.float64))
    log(f"training scaler (float32 stats): mean {mean32}, 2 std {2 * std32}")
    log(f"file scaler (float64 stats):     mean {mean_file}, 2 sigma {2 * sigma_file}")
    del raw

    t0 = time.perf_counter()
    with open(som_path, "rb") as f:
        lattice = pickle.load(f)
    log(f"loaded lattice {lattice.xdim}x{lattice.ydim} in {time.perf_counter() - t0:.1f}s")
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        lattice.assign_cluster_to_lattice(smoothing=None, merge_cost=merge_cost)
    n_cl = len(np.unique(lattice.lattice_assigned_clusters))
    log(f"re-clustered at merge_cost={merge_cost}: {n_cl} clusters")
    data = getattr(lattice, "data_array", None)
    # the BMU-distance thresholds are not used here; a 10^6-row subsample keeps
    # from_lattice fast
    sub = None if data is None else np.asarray(data[:1_000_000])
    model = SOMModel.from_lattice(
        lattice, scaled_training_data=sub, scaler=scaler, merge_cost=merge_cost,
        metadata={"dataset": "bbh_0003_180224", "som": som_path.name},
    )
    return model, scaler, data


def stage_labels(args) -> None:
    out = Path(args.out)
    model, scaler, train_scaled = build_model(Path(args.som), Path(args.train_features),
                                              args.merge_cost)
    info: dict = {"merge_cost": args.merge_cost, "som": str(args.som),
                  "grid": str(args.grid), "train_features": str(args.train_features),
                  "lattice": [model.xdim, model.ydim]}

    # consistency check: scaler vs the lattice's own scaled training data
    with h5py.File(args.train_features, "r") as h:
        raw_train = h["features"][:1_000_000]
    info["scaler_offset"] = scaler.offset.tolist()
    info["scaler_scale"] = scaler.scale.tolist()
    if train_scaled is not None:
        d = np.max(np.abs(scaler.transform(raw_train) - train_scaled[:1_000_000]))
        log(f"max |scaler(raw) - lattice.data_array| on 1e6 rows: {d:.2e}")
        info["scaler_check_maxabs"] = float(d)
        if d > 1e-5:
            raise SystemExit("scaler does not reproduce the lattice's training data")
    raw_train = raw_train[:200_000]

    # optional cross-check against saved training labels at the same merge cost
    if args.ref_labels and Path(args.ref_labels).exists():
        ref = np.load(args.ref_labels, mmap_mode="r")[:200_000]
        mine = model.predict(raw_train, backend="screen")
        agree = float(np.mean(ref == mine))
        log(f"agreement with {Path(args.ref_labels).name} on 2e5 training rows: {agree:.5f}")
        info["ref_label_agreement"] = agree

    t0 = time.perf_counter()
    with h5py.File(args.grid, "r") as h:
        n = h["features"].shape[0]
        level = h["level"][()]
        labels = np.empty(n, dtype=np.uint8)
        step = 1 << 23
        for s in range(0, n, step):
            x = h["features"][s:s + step]
            labels[s:s + step] = model.predict(x, chunk=2**20, backend="screen")
            log(f"  predicted {min(s + step, n):,}/{n:,}")
    t_pred = time.perf_counter() - t0
    log(f"predicted {n:,} grid points in {t_pred:.0f}s")
    np.save(out / f"labels_{args.tag}.npy", labels)
    info["predict_seconds"] = round(t_pred, 1)
    info["n_points"] = int(n)

    # per-cluster statistics on the grid: medians and volume fractions
    vol = np.zeros(8)
    for l, dx in DX.items():
        vol[l] = dx**3
    cell_vol = vol[level]
    total_vol = float(cell_vol.sum())
    ids = np.unique(labels)
    stats = {}
    with h5py.File(args.grid, "r") as h:
        feats = h["features"][()]
    for c in ids:
        m = labels == c
        med = np.median(feats[m], axis=0)
        stats[int(c)] = {
            "n_points": int(m.sum()),
            "point_fraction": float(m.mean()),
            "volume_M3": float(cell_vol[m].sum()),
            "volume_fraction": float(cell_vol[m].sum() / total_vol),
            "median": dict(zip(FEATURES, map(float, med))),
            "points_per_level": {int(l): int(np.sum(m & (level == l))) for l in DX},
        }
    del feats
    chi = {c: s["median"]["log_chi_B"] for c, s in stats.items()}
    grad = {c: s["median"]["log_grad_ln_b2"] for c, s in stats.items()}
    sheet = max(chi, key=chi.get)
    info["sheet_cluster"] = int(sheet)
    info["sheet_is_also_max_grad"] = bool(sheet == max(grad, key=grad.get))
    info["total_volume_M3"] = total_vol
    info["clusters"] = stats
    with open(out / f"clusters_{args.tag}.json", "w") as f:
        json.dump(info, f, indent=1)
    log(f"sheet cluster {sheet}: volume fraction {stats[sheet]['volume_fraction']:.4%}, "
        f"point fraction {stats[sheet]['point_fraction']:.4%}")
    for c, s in stats.items():
        log(f"  c{c}: vol {s['volume_fraction']:.4%}  " +
            "  ".join(f"{k}={v:+.2f}" for k, v in s["median"].items()))


# --------------------------------------------------------------------------- sample


def _plane_coords(xyz: np.ndarray, origin, e1, e2, e3) -> np.ndarray:
    d = xyz.astype(np.float64) - origin
    return np.stack([d @ e1, d @ e2, d @ e3], axis=-1)


def stage_sample(args) -> None:
    from scipy.spatial import cKDTree

    out = Path(args.out)
    raw = dict(np.load(args.raw_planes))
    labels = np.load(Path(out) / f"labels_{args.tag}.npy", mmap_mode="r")
    with open(Path(out) / f"clusters_{args.tag}.json") as f:
        info = json.load(f)
    sheet = info["sheet_cluster"]
    hw, n = args.half_width, args.n

    t0 = time.perf_counter()
    with h5py.File(args.grid, "r") as h:
        xyz = h["xyz"][()]
        level = h["level"][()]
        lbi = h["features"][:, 0]  # log_beta_inv
        chi = h["features"][:, 2]  # log_chi_B
    log(f"read grid ({len(level):,} points) in {time.perf_counter() - t0:.1f}s")

    res: dict = {"sheet_cluster": sheet, "half_width": hw, "holes": raw["holes"],
                 "time": raw["time"], "it": raw["it"]}
    for kind in ("meridional", "orbital"):
        o, e1, e2 = raw[f"{kind}_origin"], raw[f"{kind}_e1"], raw[f"{kind}_e2"]
        e3 = np.cross(e1, e2)
        a = np.linspace(-hw, hw, n)
        P = o + a[:, None, None] * e1 + a[None, :, None] * e2  # (n, n, 3), [i, j] at (a_i, b_j)
        P = P.reshape(-1, 3)
        lab = np.full(len(P), 255, np.uint8)
        lev = np.full(len(P), -1, np.int8)
        val_lbi = np.full(len(P), np.nan, np.float32)
        todo = np.ones(len(P), bool)
        for l in sorted(DX, reverse=True):  # finest level first
            dx = DX[l]
            r = 0.5 * np.sqrt(3.0) * dx * 1.001
            idx = np.flatnonzero(level == l)
            # only the slab of this level's points that can be nearest to the plane
            keep = []
            for s in range(0, len(idx), 1 << 22):
                ii = idx[s:s + (1 << 22)]
                c = _plane_coords(xyz[ii], o, e1, e2, e3)
                m = (np.abs(c[:, 2]) <= r) & (np.abs(c[:, 0]) <= hw + r) & (np.abs(c[:, 1]) <= hw + r)
                keep.append(ii[m])
            keep = np.concatenate(keep)
            if keep.size == 0:
                continue
            tree = cKDTree(xyz[keep].astype(np.float64))
            q = np.flatnonzero(todo)
            dist, j = tree.query(P[q], k=1, distance_upper_bound=r, workers=4)
            hit = np.isfinite(dist)
            g = keep[j[hit]]
            lab[q[hit]] = labels[g]
            lev[q[hit]] = l
            val_lbi[q[hit]] = lbi[g]
            todo[q[hit]] = False
            log(f"{kind}: level {l}: {keep.size:,} slab points, {hit.sum():,} plane points")
        res[f"{kind}_a"] = a
        res[f"{kind}_b"] = a.copy()
        res[f"{kind}_labels"] = lab.reshape(n, n)
        res[f"{kind}_level"] = lev.reshape(n, n)
        res[f"{kind}_log_beta_inv"] = val_lbi.reshape(n, n)
        res[f"{kind}_origin"], res[f"{kind}_e1"], res[f"{kind}_e2"] = o, e1, e2
        log(f"{kind}: {np.mean(lab == 255):.4%} of plane points without a grid point")
    # raw-field planes (for streamlines and the sigma = 1 contour), on their own grid
    for kind in ("meridional", "orbital"):
        e1, e2 = raw[f"{kind}_e1"], raw[f"{kind}_e2"]
        B = np.stack([raw[f"{kind}_B{c}"] for c in "xyz"], axis=-1).astype(np.float64)
        res[f"{kind}_raw_a"] = raw[f"{kind}_a"]
        res[f"{kind}_raw_b"] = raw[f"{kind}_b"]
        res[f"{kind}_B1"] = (B @ e1).astype(np.float32)
        res[f"{kind}_B2"] = (B @ e2).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            res[f"{kind}_log_sigma"] = np.log10(raw[f"{kind}_smallb2"] / raw[f"{kind}_rho_b"]).astype(np.float32)
            res[f"{kind}_raw_log_beta_inv"] = np.log10(
                raw[f"{kind}_smallb2"] / (2 * raw[f"{kind}_P"])).astype(np.float32)
    np.savez_compressed(out / f"planes_{args.tag}.npz", **res)
    log(f"wrote planes_{args.tag}.npz ({time.perf_counter() - t0:.0f}s)")

    # 3D: sheet-cluster volume per voxel of a uniform co-rotating box
    o, e1, e2 = raw["orbital_origin"], raw["orbital_e1"], raw["orbital_e2"]
    ez = np.cross(e1, e2)  # orbital axis
    vx, (hx, hz) = args.voxel, (args.box_xy, args.box_z)
    nx, nz = int(round(2 * hx / vx)), int(round(2 * hz / vx))
    acc_v = np.zeros(nx * nx * nz)
    acc_s = np.zeros(nx * nx * nz)
    acc_chi = np.zeros(nx * nx * nz)
    cellvol = np.zeros(8)
    for l, dx in DX.items():
        cellvol[l] = dx**3
    for s in range(0, len(level), 1 << 22):
        sl = slice(s, s + (1 << 22))
        c = _plane_coords(xyz[sl], o, e1, e2, ez)
        m = (np.abs(c[:, 0]) < hx) & (np.abs(c[:, 1]) < hx) & (np.abs(c[:, 2]) < hz)
        c = c[m]
        ix = np.floor((c[:, 0] + hx) / vx).astype(np.int64)
        iy = np.floor((c[:, 1] + hx) / vx).astype(np.int64)
        iz = np.floor((c[:, 2] + hz) / vx).astype(np.int64)
        k = (ix * nx + iy) * nz + iz
        w = cellvol[level[sl][m]]
        is_s = labels[sl][m] == sheet
        acc_v += np.bincount(k, weights=w, minlength=acc_v.size)
        acc_s += np.bincount(k[is_s], weights=w[is_s], minlength=acc_v.size)
        acc_chi += np.bincount(k[is_s], weights=(w * chi[sl][m])[is_s], minlength=acc_v.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (acc_s / acc_v).reshape(nx, nx, nz).astype(np.float32)
        chim = (acc_chi / acc_s).reshape(nx, nx, nz).astype(np.float32)
    np.savez_compressed(out / f"sheet3d_{args.tag}.npz", sheet_fraction=frac, log_chi_B=chim,
                        voxel=vx, box_xy=hx, box_z=hz, holes=raw["holes"], origin=o, e1=e1, e2=e2,
                        ez=ez, sheet_cluster=sheet)
    log(f"wrote sheet3d_{args.tag}.npz: {np.sum(frac > 0.5):,} voxels > 50% sheet "
        f"({time.perf_counter() - t0:.0f}s)")


# --------------------------------------------------------------------------- plot

# Display groups of the 12 clusters of the 91x64 lattice at merge_cost = 0.1, from
# their feature medians on the it 180224 grid (clusters_it180224.json). Cluster ids
# are those of the frozen lattice, so they are the same at every iteration.
# Colours: Okabe-Ito vermillion for the sheet, muted tints for context; the set
# passes the CVD (deutan/protan/tritan) and normal-vision separation checks of the
# dataviz validator for all pairs.
GROUPS = [
    # (label, colour, cluster ids)
    ("current sheet (c9)", "#D55E00", [9]),
    ("sheet envelope (c2)", "#EFAE70", [2]),
    ("hot, magnetized: funnel, corona", "#7FB5DD", [1, 3, 4, 6, 10]),
    ("cold disk, weak-field floor", "#E6E6E6", [0, 5, 7, 8, 11]),
]
HOLE_RADIUS = 1.035  # horizon-mask radius of the grid (sqrt(r+^2 + a^2), m = 0.5, chi = 0.9)


def _group_image(labels: np.ndarray) -> np.ndarray:
    img = np.full(labels.shape, np.nan)
    for g, (_, _, ids) in enumerate(GROUPS):
        img[np.isin(labels, ids)] = g
    return img


def _holes_in_plane(P: dict, kind: str) -> np.ndarray:
    o, e1, e2 = P[f"{kind}_origin"], P[f"{kind}_e1"], P[f"{kind}_e2"]
    d = P["holes"] - o
    return np.stack([d @ e1, d @ e2], axis=-1)


def _draw_holes(ax, xy, labels=True):
    import matplotlib.patches as mpatches

    for k, (x, y) in enumerate(xy):
        ax.add_patch(mpatches.Circle((x, y), HOLE_RADIUS, fc="black", ec="white", lw=0.5, zorder=5))
        if labels:
            ax.annotate(f"BH{k + 1}", (x, y), xytext=(0, -9), textcoords="offset points",
                        ha="center", va="top", fontsize=8, color="black", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))


def stage_plot(args) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out = Path(args.out)
    P = dict(np.load(out / f"planes_{args.tag}.npz"))
    V = dict(np.load(out / f"sheet3d_{args.tag}.npz"))
    hw = float(P["half_width"])
    t = float(P["time"])

    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "font.family": "sans-serif",
        "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.dpi": 300,
    })
    cmap_cat = ListedColormap([c for _, c, _ in GROUPS])
    cmap_cat.set_bad("white")
    norm_cat = BoundaryNorm(np.arange(len(GROUPS) + 1) - 0.5, len(GROUPS))
    ext = [-hw, hw, -hw, hw]

    fig = plt.figure(figsize=(7.0, 6.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharex=ax_a, sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, 0], projection="3d")
    ax_d = fig.add_subplot(gs[1, 1])

    mer_holes = _holes_in_plane(P, "meridional")
    orb_holes = _holes_in_plane(P, "orbital")

    # (a) magnetization on the meridional plane, with in-plane B streamlines
    lbi = P["meridional_log_beta_inv"]
    im = ax_a.imshow(lbi.T, origin="lower", extent=ext, cmap="cividis",
                     vmin=args.vmin, vmax=args.vmax, interpolation="nearest", rasterized=True)
    ra, rb = P["meridional_raw_a"], P["meridional_raw_b"]
    B1, B2 = P["meridional_B1"].T, P["meridional_B2"].T  # (b, a) for streamplot
    speed = np.hypot(B1, B2)
    ok = np.isfinite(speed)
    B1 = np.where(ok, B1, 0.0)
    B2 = np.where(ok, B2, 0.0)
    ax_a.streamplot(ra, rb, B1, B2, density=args.stream_density, color="black", linewidth=0.4,
                    arrowsize=0.5, arrowstyle="-|>", broken_streamlines=True, zorder=3)
    _draw_holes(ax_a, mer_holes)
    cb = fig.colorbar(im, ax=ax_a, location="right", shrink=0.92, pad=0.02, extend="both")
    cb.set_label(r"$\log_{10}\beta^{-1} = \log_{10}(b^2/2P)$")
    cb.ax.tick_params(labelsize=8)
    ax_a.set_title("(a) Magnetization and field lines", loc="left")
    ax_a.set_xlabel(r"$x'$ [M]  (BH1 $\rightarrow$ BH2)")
    ax_a.set_ylabel(r"$z$ [M]  (orbital axis)")

    # sigma = 1 contour (jet funnels) for panels (a), (b)
    lsig = P["meridional_log_sigma"].T
    for ax, col in ((ax_a, "white"), (ax_b, "black")):
        ax.contour(ra, rb, np.where(np.isfinite(lsig), lsig, -9), levels=[0.0], colors=col,
                   linewidths=0.8, linestyles="--", zorder=4)

    # (b) aweSOM clusters on the meridional plane
    img = _group_image(P["meridional_labels"])
    ax_b.imshow(img.T, origin="lower", extent=ext, cmap=cmap_cat, norm=norm_cat,
                interpolation="nearest", rasterized=True)
    _draw_holes(ax_b, mer_holes)
    for spec in args.annotate or []:
        x, y, tx, ty, text = spec.split(",", 4)
        ax_b.annotate(text, xy=(float(x), float(y)), xytext=(float(tx), float(ty)), fontsize=8,
                      ha="center", va="center", zorder=7,
                      arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black", shrinkA=1, shrinkB=1,
                                      mutation_scale=7),
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.3", lw=0.4))
    ax_b.set_title("(b) aweSOM clusters, meridional plane", loc="left")
    ax_b.set_xlabel(r"$x'$ [M]  (BH1 $\rightarrow$ BH2)")
    ax_b.tick_params(labelleft=False)
    for ax in (ax_a, ax_b):
        ax.set_xlim(-hw, hw)
        ax.set_ylim(-args.zmax, args.zmax)
        ax.set_aspect("equal")

    # (d) aweSOM clusters on the orbital plane
    img = _group_image(P["orbital_labels"])
    ax_d.imshow(img.T, origin="lower", extent=ext, cmap=cmap_cat, norm=norm_cat,
                interpolation="nearest", rasterized=True)
    _draw_holes(ax_d, orb_holes)
    ax_d.axhline(0.0, color="black", lw=0.6, ls=":", zorder=4)
    ax_d.text(0.0, -0.8, "plane of (a), (b)", fontsize=7.5, ha="center", va="top", zorder=6,
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax_d.set_title("(d) aweSOM clusters, orbital plane", loc="left")
    ax_d.set_xlabel(r"$x'$ [M]  (BH1 $\rightarrow$ BH2)")
    ax_d.set_ylabel(r"$y'$ [M]")
    ax_d.set_xlim(-hw, hw)
    ax_d.set_ylim(-hw, hw)
    ax_d.set_aspect("equal")

    # (c) 3D: voxels that are mostly sheet cluster, coloured by log chi_B
    frac, chim = V["sheet_fraction"], V["log_chi_B"]
    vx, hx, hz = float(V["voxel"]), float(V["box_xy"]), float(V["box_z"])
    sel = np.argwhere(np.nan_to_num(frac) > args.voxel_threshold)
    rng = np.random.default_rng(0)
    if len(sel) > args.max_voxels:
        sel = sel[rng.choice(len(sel), args.max_voxels, replace=False)]
    xs = -hx + (sel[:, 0] + 0.5) * vx
    ys = -hx + (sel[:, 1] + 0.5) * vx
    zs = -hz + (sel[:, 2] + 0.5) * vx
    cs = chim[sel[:, 0], sel[:, 1], sel[:, 2]]
    sc = ax_c.scatter(xs, ys, zs, c=cs, cmap="viridis", vmin=args.chi_min, vmax=args.chi_max,
                      s=0.6, marker="o", linewidths=0, alpha=0.8, depthshade=False,
                      rasterized=True)
    o, e1, e2 = V["origin"], V["e1"], V["e2"]
    ez = V["ez"]
    hp = np.stack([(V["holes"] - o) @ e1, (V["holes"] - o) @ e2, (V["holes"] - o) @ ez], -1)
    for k, (x, y, z) in enumerate(hp):
        ax_c.scatter([x], [y], [z], s=28, c="black", depthshade=False, zorder=10)
        ax_c.plot([x, x], [y, y], [z, z + hz], color="#0072B2", lw=1.6, zorder=11)
        ax_c.plot([x, x], [y, y], [z, z - hz], color="#0072B2", lw=1.6, zorder=11)
        ax_c.text(x, y, z + hz + 1.0, f"BH{k + 1} jet axis", fontsize=7.5, ha="center",
                  color="#0072B2")
    ax_c.set_xlim(-hx, hx)
    ax_c.set_ylim(-hx, hx)
    ax_c.set_zlim(-hz, hz)
    ax_c.set_box_aspect((2 * hx, 2 * hx, 2 * hz))
    ax_c.view_init(elev=args.elev, azim=args.azim)
    ax_c.set_xlabel(r"$x'$ [M]", labelpad=-6)
    ax_c.set_ylabel(r"$y'$ [M]", labelpad=-6)
    ax_c.set_zlabel(r"$z$ [M]", labelpad=-8)
    ax_c.tick_params(pad=-3, labelsize=7)
    ax_c.set_xticks([-20, 0, 20])
    ax_c.set_yticks([-20, 0, 20])
    ax_c.set_zticks([-10, 0, 10])
    for axis in (ax_c.xaxis, ax_c.yaxis, ax_c.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]["linewidth"] = 0.3
    ax_c.set_title("(c) Current-sheet cluster in 3D", loc="left")
    cb = fig.colorbar(sc, ax=ax_c, location="right", shrink=0.62, pad=0.13, extend="both")
    cb.set_label(r"$\log_{10}\chi_B$ (field curvature)")
    cb.solids.set_alpha(1.0)
    cb.ax.tick_params(labelsize=8)

    handles = [Patch(fc=c, ec="0.5", lw=0.3, label=lab) for lab, c, _ in GROUPS]
    handles.append(Line2D([], [], color="black", ls="--", lw=0.8,
                          label=r"$\sigma = b^2/\rho = 1$ (jet funnel edge)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, title="aweSOM clusters (b, d)",
               title_fontsize=8, frameon=False, handlelength=1.4, columnspacing=1.2, fontsize=8)
    fig.suptitle(f"BBH LR_flux, it {int(P['it'])} (t = {t:.0f} M), co-rotating frame",
                 fontsize=9, x=0.01, ha="left")

    stem = out / f"fig_current_sheet_between_jets_{args.tag}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    log(f"wrote {stem}.pdf/.png ({len(sel):,} voxels in 3D)")


# --------------------------------------------------------------------------- main


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    a = sub.add_parser("labels")
    a.add_argument("--grid", required=True)
    a.add_argument("--train-features", required=True)
    a.add_argument("--som", required=True)
    a.add_argument("--merge-cost", type=float, default=0.1)
    a.add_argument("--ref-labels", default=None,
                   help="saved training labels at the same merge cost, for a cross-check")
    a.add_argument("--out", required=True)
    a.add_argument("--tag", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--grid", required=True)
    s.add_argument("--raw-planes", required=True, help="output of sample_raw_planes.py")
    s.add_argument("--half-width", type=float, default=24.0)
    s.add_argument("--n", type=int, default=769, help="plane points per side (769: 0.0625 M)")
    s.add_argument("--voxel", type=float, default=0.25, help="3D voxel size [M]")
    s.add_argument("--box-xy", type=float, default=24.0)
    s.add_argument("--box-z", type=float, default=12.0)
    s.add_argument("--out", required=True)
    s.add_argument("--tag", required=True)

    f = sub.add_parser("plot")
    f.add_argument("--out", required=True)
    f.add_argument("--tag", required=True)
    f.add_argument("--vmin", type=float, default=-1.0, help="log beta_inv colour range")
    f.add_argument("--vmax", type=float, default=3.0)
    f.add_argument("--chi-min", type=float, default=0.0)
    f.add_argument("--chi-max", type=float, default=1.5)
    f.add_argument("--zmax", type=float, default=24.0, help="vertical extent of (a), (b)")
    f.add_argument("--stream-density", type=float, default=1.1)
    f.add_argument("--voxel-threshold", type=float, default=0.25,
                   help="sheet volume fraction a 3D voxel needs to be drawn")
    f.add_argument("--max-voxels", type=int, default=80_000)
    f.add_argument("--elev", type=float, default=18.0)
    f.add_argument("--azim", type=float, default=-62.0)
    f.add_argument("--annotate", action="append",
                   help="arrow label in panel (b): 'x,y,text_x,text_y,text' (plane coords, M)")

    args = p.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {"labels": stage_labels, "sample": stage_sample, "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    main()
