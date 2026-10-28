"""Check the imported production lattice against the flux-tubes labels (same grid, same iteration).

    python prod_check.py MINE.npy THEIRS.npy

Prints the share of points with identical labels (and of the volume, if the
180224 voxel grid is available), plus per-label counts that differ.
"""

import sys

import numpy as np

from common import load_grid


def main():
    a = np.load(sys.argv[1], mmap_mode="r")
    b = np.load(sys.argv[2], mmap_mode="r")
    print("shapes", a.shape, b.shape, "dtypes", a.dtype, b.dtype)
    a = np.asarray(a, np.int32)
    b = np.asarray(b, np.int32)
    same = a == b
    print(f"identical labels: {same.mean():.6f} of points")
    try:
        vol = load_grid(180224).cell_volume()
        print(f"identical labels: {vol[same].sum() / vol.sum():.6f} of volume")
    except Exception as e:  # noqa: BLE001
        print("no volume weights:", e)
    k = int(max(a.max(), b.max())) + 1
    ca = np.bincount(a[a >= 0], minlength=k)
    cb = np.bincount(b[b >= 0], minlength=k)
    for c in range(k):
        print(f"c{c}: mine {ca[c]}, theirs {cb[c]}, rel diff {(ca[c] - cb[c]) / max(cb[c], 1):+.2e}")


if __name__ == "__main__":
    main()
