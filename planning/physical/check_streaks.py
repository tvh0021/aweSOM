"""Look for axis-aligned constant streaks in a grid column (copied/extrapolated data).

    python check_streaks.py --it 135168 --name log_chi_B

Per level: the volume share of cells whose value is bit-identical to the
neighbour along x, y or z (real turbulent data almost never repeats exactly).
"""

import argparse

import numpy as np

from common import col, load_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--it", type=int, default=135168)
    ap.add_argument("--name", default="log_chi_B")
    a = ap.parse_args()
    grid = load_grid(a.it)
    v = np.asarray(col(a.it, a.name), dtype=np.float32)
    tot_rep, tot = 0.0, 0.0
    for g in sorted(grid.groups):
        m = np.full(grid.shape[g], np.nan, np.float32)
        m.ravel()[grid.flat[g]] = v[grid.rows[g]]
        ok = ~np.isnan(m)
        rep = np.zeros(m.shape, bool)
        for ax in range(3):
            sl_a = [slice(None)] * 3
            sl_b = [slice(None)] * 3
            sl_a[ax] = slice(1, None)
            sl_b[ax] = slice(None, -1)
            eq = (m[tuple(sl_a)] == m[tuple(sl_b)]) & ok[tuple(sl_a)] & ok[tuple(sl_b)]
            rep[tuple(sl_a)] |= eq
            rep[tuple(sl_b)] |= eq
        n = int(ok.sum())
        r = int(rep.sum())
        # where along z are the repeats concentrated?
        zr = rep.sum((0, 1)) / np.maximum(ok.sum((0, 1)), 1)
        kz = int(round((0 - grid.origin[g][2]) / grid.spacing[g]))
        print(f"level {g}: {n} cells, repeated-value share {r / n:.4f}; at z=0 plane {zr[kz]:.4f}; "
              f"max plane share {zr.max():.4f} at k={int(zr.argmax())}", flush=True)
        dv = grid.spacing[g] ** 3
        tot_rep += r * dv
        tot += n * dv
    print(f"volume share with repeated values: {tot_rep / tot:.4f}")


if __name__ == "__main__":
    main()
