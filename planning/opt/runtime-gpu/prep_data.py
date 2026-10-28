"""Write W.bin, X.bin, ref_idx.bin, ref_d2.bin for bench_cpp into OUTDIR.

python prep_data.py OUTDIR real            # fiducial plasma-turbulence lattice and data (F=4)
python prep_data.py OUTDIR uniform F N     # uniform(-1,1) weights, normal(0,0.5) points: worst case
python prep_data.py OUTDIR sheet F N       # 64x37 lattice on a smooth 2D sheet in F dims, points near it
Reference labels/distances come from aweSOM.inference.best_match_nodes (numba).
"""

import pickle
import sys
from pathlib import Path

import numpy as np

from aweSOM.inference import best_match_nodes

PKL = "/Users/tvh0021/git_repos/aweSOM/examples/plasma-turbulence/som_object.fiducial-5000-64-37-0.1-2097152-1u.pkl"


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    kind = sys.argv[2]
    rng = np.random.default_rng(0)
    if kind == "real":
        lat = pickle.load(open(PKL, "rb"))
        X = np.ascontiguousarray(np.asarray(lat.data_array, dtype=np.float64))
        W = np.ascontiguousarray(np.asarray(lat.lattice, dtype=np.float64).reshape(-1, X.shape[1]))
    else:
        F, N = int(sys.argv[3]), int(sys.argv[4])
        if kind == "uniform":
            W = rng.uniform(-1, 1, size=(2368, F))
            X = rng.normal(scale=0.5, size=(N, F))
        else:
            u, v = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 37))
            uv = np.stack([u.ravel(), v.ravel()], 1)
            A = rng.normal(size=(2, F)) * 3
            ph = rng.uniform(0, 2 * np.pi, F)
            W = np.sin(uv @ A + ph)
            pts = rng.uniform(0, 1, size=(N, 2))
            X = np.sin(pts @ A + ph) + rng.normal(scale=0.1, size=(N, F))
        W, X = np.ascontiguousarray(W), np.ascontiguousarray(X)
    idx, d2 = best_match_nodes(W, X)
    W.tofile(out / "W.bin")
    X.tofile(out / "X.bin")
    idx.astype(np.int32).tofile(out / "ref_idx.bin")
    d2.tofile(out / "ref_d2.bin")
    print(W.shape, X.shape)


if __name__ == "__main__":
    main()
