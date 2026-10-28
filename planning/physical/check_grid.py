"""Grid geometry per level, and order check of the sheets-metric npz (it 135168)
against grid_135168.h5 via log_grad_ln_b2 (unclipped values only)."""

import sys
import zipfile

import h5py
import numpy as np

grid = sys.argv[1]
npz = sys.argv[2] if len(sys.argv) > 2 else None
with h5py.File(grid, "r") as f:
    level = f["level"][:]
    names = [n.decode() for n in f["names"][()]]
    lo_w, hi_w = f.attrs["winsor_log_grad_ln_b2"]
    starts = {}
    for rl in (5, 6, 7):
        idx = np.flatnonzero(level == rl)
        a, b = idx[0], idx[-1] + 1
        assert b - a == idx.size
        starts[rl] = (a, b)
        xyz = f["xyz"][a:b]
        dx = 0.25 / 2 ** (rl - 5)
        lo, hi = xyz.min(0), xyz.max(0)
        ijk = np.rint((xyz - lo) / dx)
        err = np.abs((xyz - lo) / dx - ijk).max()
        shape = (ijk.max(0) + 1).astype(int)
        print(rl, b - a, "lo", lo, "hi", hi, "shape", shape, "voxels %.3g" % np.prod(shape),
              "fill %.3f" % ((b - a) / np.prod(shape)), "max idx err %.2g" % err)
        if npz is not None:
            z = np.load(npz)
            ref = z[f"log_grad_ln_b2_rl{rl}"]
            g = f["features"][a:b, names.index("log_grad_ln_b2")]
            ok = (g > lo_w) & (g < hi_w) & np.isfinite(ref)
            d = np.abs(g[ok] - ref[ok])
            print("   npz n=%d grid n=%d  max|diff| %.3g  median %.3g" % (ref.size, g.size, d.max(), np.median(d)))
