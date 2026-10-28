"""Write Bx, By, Bz, rho_b, smallb2 aligned row-for-row with a som-features grid file.

Run with the etk-analysis environment:

    PYTHONPATH=/Volumes/ExternalNVMe/etk-analysis \
    /Volumes/ExternalNVMe/etk-analysis/.venv/bin/python extract_B.py \
        --index /Volumes/ExternalNVMe/som-proto-f5/idx3d_0003.json \
        --parfile /Volumes/ExternalNVMe/simulation-data/michi-bbh/output-0003/parfile.par \
        --grid grid_LR_flux_180224.h5 --it 180224 -o B_LR_flux_180224.h5

The row order is reproduced the way ``etkplot.somfeatures._level_rows`` writes the
grid file: level by level, patch by patch (``ownership.level_ownership`` order),
x-slab by x-slab, owned points in C order. Every row's coordinates and level are
checked against the grid file.
"""

import argparse
import time
from types import SimpleNamespace

import h5py
import numpy as np
from etkplot import ownership
from etkplot.somfeatures import _read_slab, _slabs
from etkplot.source import open_source

VARS = ["Bx", "By", "Bz", "rho_b", "smallb2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--parfile", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--it", type=int, default=180224)
    ap.add_argument("--levels", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--ref-var", default="smallb2", help="som-features' reference variable")
    ap.add_argument("--slab", type=int, default=64)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    src = open_source(args.index)
    opts = SimpleNamespace(parfile=args.parfile, horizon_mask=True)
    hz = ownership.horizons(src, args.ref_var, args.it, opts)
    with h5py.File(args.grid, "r") as g, h5py.File(args.out, "w") as f:
        n_total = g["xyz"].shape[0]
        gxyz, glevel = g["xyz"], g["level"]
        dB = f.create_dataset("B", (n_total, 3), np.float32, chunks=(1 << 18, 3))
        dsc = {v: f.create_dataset(v, (n_total,), np.float32, chunks=(1 << 18,))
               for v in ("rho_b", "smallb2")}
        row = 0
        max_dxyz = 0.0
        for rl in args.levels:
            own = ownership.level_ownership(src, args.ref_var, args.it, rl, opts, horizons=hz)
            print(f"rl={rl}: {own.count} owned, {len(own.patches)} patches, dx {own.dx}",
                  flush=True)
            for p in own.patches:
                plane = int(np.prod(p.component.shape[1:]))
                for a, b in _slabs(p.component.shape[0], args.slab):
                    owned = p.component[a:b] >= 0
                    count = int(np.count_nonzero(owned))
                    if count == 0:
                        continue
                    vals = {v: _read_slab(src, v, args.it, rl, p, a, b, 0)[owned] for v in VARS}
                    flat = np.flatnonzero(owned) + a * plane
                    xyz = p.xyz(flat).astype(np.float32)
                    d = float(np.max(np.abs(gxyz[row:row + count] - xyz)))
                    max_dxyz = max(max_dxyz, d)
                    if d > 1e-4 or np.any(glevel[row:row + count] != rl):
                        raise RuntimeError(f"row {row}: xyz/level mismatch ({d})")
                    dB[row:row + count] = np.stack(
                        [vals["Bx"], vals["By"], vals["Bz"]], axis=1).astype(np.float32)
                    for v in dsc:
                        dsc[v][row:row + count] = vals[v].astype(np.float32)
                    row += count
            print(f"rl={rl}: done, rows {row}, {time.time() - t0:.0f}s", flush=True)
        if row != n_total:
            raise RuntimeError(f"wrote {row} rows, grid has {n_total}")
        f.attrs.update(iteration=args.it, index=args.index, grid=args.grid,
                       max_abs_dxyz=max_dxyz, variables=",".join(VARS),
                       note="Bx,By,Bz as in the Carpet 3D output; rows match the grid file rows")
    print(f"wrote {args.out}: {row} rows, max |dxyz| {max_dxyz:g}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
