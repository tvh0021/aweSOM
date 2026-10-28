"""Per-iteration column store for the study: grid features, C_B, chi_B contrast,
chi_B ridge mask, metric sheet masks, the voxel grid, the stats subsample and
the training-sample -> grid row map.

    python prep.py 180224
"""

import json
import pickle
import sys
import time

import h5py
import numpy as np
from scipy import ndimage

from common import B2_MIN, COLS, DATA, DX, GRID, LEVELS, METRIC_NPZ_135168, SAMPLE, limit_threads

limit_threads(4)
from aweSOM.physics_score import VoxelGrid, ridge_mask  # noqa: E402

it = int(sys.argv[1])
COLS.mkdir(parents=True, exist_ok=True)
t0 = time.time()
log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)  # noqa: E731
meta = {"it": it}

with h5py.File(GRID[it], "r") as f:
    names = [n.decode() for n in f["names"][()]]
    level = f["level"][:]
    for j, nm in enumerate(names):
        np.save(COLS / f"{it}_{nm}.npy", np.ascontiguousarray(f["features"][:, j]))
    xyz = f["xyz"][:]
    meta["winsor_log_chi_B"] = f.attrs["winsor_log_chi_B"].tolist()
n = level.size
meta["n"] = int(n)
meta["names"] = names
np.save(COLS / f"{it}_level.npy", level)
log(f"features {len(names)} columns")

grid = VoxelGrid.from_points(xyz, DX, level)
del xyz
for g in grid.groups:
    grid.flat[g] = grid.flat[g].astype(np.int32)
with open(COLS / f"{it}_voxelgrid.pkl", "wb") as fh:
    pickle.dump(grid, fh, protocol=4)
meta["shape"] = {str(g): grid.shape[g] for g in grid.groups}
meta["origin"] = {str(g): grid.origin[g].tolist() for g in grid.groups}
log(f"voxel grid {meta['shape']}")
starts = {g: (grid.rows[g].start, grid.rows[g].stop) for g in grid.groups}
assert all(isinstance(grid.rows[g], slice) for g in grid.groups)

# --- C_B and the sheets metric
cb = np.concatenate([np.load(DATA / "extra" / f"extra_it{it}_rl{rl}_log_one_minus_C_B.npy") for rl in LEVELS])
assert cb.size == n
np.save(COLS / f"{it}_log_one_minus_C_B.npy", cb)
# order check: every 97th point's log_grad_ln_b2 against the grid's
lo_w, hi_w = (-np.inf, np.inf)
with h5py.File(GRID[it], "r") as f:
    lo_w, hi_w = f.attrs["winsor_log_grad_ln_b2"]
jg = names.index("log_grad_ln_b2")
maxdiff = 0.0
for rl in LEVELS:
    chk = np.load(DATA / "extra" / f"extra_it{it}_rl{rl}_check_lg.npy")
    rows = starts[rl][0] + chk[:, 0].astype(np.int64)
    g = np.load(COLS / f"{it}_log_grad_ln_b2.npy", mmap_mode="r")[rows]
    ok = (g > lo_w) & (g < hi_w)
    maxdiff = max(maxdiff, float(np.abs(g[ok] - chk[ok, 1]).max()))
meta["order_check_max_abs_diff"] = maxdiff
log(f"C_B; order check max |diff| = {maxdiff:.3g}")
assert maxdiff < 1e-5

if it == 135168:
    z = np.load(METRIC_NPZ_135168)
    ratio = np.concatenate([z[f"ratio_m_rl{rl}"] for rl in LEVELS])
    th3 = np.concatenate([z[f"th3_rl{rl}"] for rl in LEVELS])
    b2 = np.concatenate([z[f"b2_rl{rl}"] for rl in LEVELS])
    del z
else:
    ld = lambda k: np.concatenate([np.load(DATA / "extra" / f"extra_it{it}_rl{rl}_{k}.npy") for rl in LEVELS])  # noqa: E731
    ratio, th3, b2 = ld("ratio_m"), ld("th3"), ld("b2")
assert ratio.size == n
with np.errstate(invalid="ignore"):
    for T in (1.0, 3.0):
        m = (ratio > T) & (th3 > 90) & (b2 >= B2_MIN[it])
        np.save(COLS / f"{it}_sheet_T{T:g}.npy", m)
        vol = grid.cell_volume()
        meta[f"sheet_T{T:g}_volume_frac"] = float(vol[m].sum() / vol.sum())
        meta[f"sheet_T{T:g}_points"] = int(m.sum())
np.save(COLS / f"{it}_ratio_m.npy", ratio.astype(np.float32))
del ratio, th3, b2
log(f"metric sheets: T1 {meta['sheet_T1_volume_frac']:.4f}, T3 {meta['sheet_T3_volume_frac']:.4f} of volume")

# --- chi_B contrast and ridge
jc = names.index("log_chi_B")
chi = np.load(COLS / f"{it}_log_chi_B.npy")
vol = grid.cell_volume()
rng = np.random.default_rng(12345)
sub = np.sort(rng.choice(n, 3_000_000, replace=False))
np.save(COLS / f"{it}_stats_rows.npy", sub)
order = np.argsort(chi[sub])
cw = np.cumsum(vol[sub][order])
q90 = float(chi[sub][order][np.searchsorted(cw, 0.9 * cw[-1])])
meta["ridge_threshold_log_chi_B_q90"] = q90
contrast = np.empty(n, np.float32)
ridge = np.zeros(n, bool)
for g in grid.groups:
    m = np.full(grid.shape[g], np.nan, np.float32)
    m.ravel()[grid.flat[g]] = chi[grid.rows[g]]
    valid = np.isfinite(m)
    s = ndimage.uniform_filter(np.where(valid, m, 0).astype(np.float32), 5, mode="constant")
    c = ndimage.uniform_filter(valid.astype(np.float32), 5, mode="constant")
    contrast[grid.rows[g]] = (m - s / c).ravel()[grid.flat[g]]
    del s, c
    r = ridge_mask(m, threshold=q90)
    ridge[grid.rows[g]] = r.ravel()[grid.flat[g]]
    del m, r, valid
    log(f"level {g}: contrast, ridge")
np.save(COLS / f"{it}_chi_contrast.npy", contrast)
np.save(COLS / f"{it}_ridge.npy", ridge)
meta["ridge_volume_frac"] = float(vol[ridge].sum() / vol.sum())

# --- training sample rows -> grid rows
with h5py.File(SAMPLE[it], "r") as f:
    sx = f["xyz"][:]
    sl = f["level"][:]
    sf = f["features"][:]
    scb = f["extra/log_one_minus_C_B"][:]
smap = np.full(sl.size, -1, np.int64)
for g in grid.groups:
    inv = np.full(int(np.prod(grid.shape[g])), -1, np.int64)
    inv[grid.flat[g]] = np.arange(starts[g][0], starts[g][1])
    sel = np.flatnonzero(sl == g)
    ijk = np.rint((sx[sel] - grid.origin[g]) / DX[g]).astype(np.int64)
    smap[sel] = inv[np.ravel_multi_index(tuple(ijk.T), grid.shape[g])]
    del inv
assert (smap >= 0).all()
np.save(COLS / f"{it}_sample_to_grid.npy", smap)
diff_f = 0.0
for j, nm in enumerate(names):
    gcol = np.load(COLS / f"{it}_{nm}.npy", mmap_mode="r")[smap]
    diff_f = max(diff_f, float(np.abs(sf[:, j] - gcol).max()))
cbg = cb[smap]
ok = np.isfinite(scb) & np.isfinite(cbg)
meta["sample_vs_grid_max_abs_diff_features"] = diff_f
meta["sample_vs_grid_max_abs_diff_C_B"] = float(np.abs(scb[ok] - cbg[ok]).max())
meta["sample_C_B_nonfinite"] = int((~np.isfinite(scb)).sum())
log(f"sample map: features max|diff| {diff_f:.3g}, C_B max|diff| {meta['sample_vs_grid_max_abs_diff_C_B']:.3g}")
(COLS / f"{it}_meta.json").write_text(json.dumps(meta, indent=1))
log("done")
