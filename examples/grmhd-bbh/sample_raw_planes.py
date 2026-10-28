"""Sample raw fields of the BBH run on the co-rotating meridional and orbital planes.

Run with the etkplot interpreter (it does not import aweSOM)::

    PYTHONPATH=/Volumes/ExternalNVMe/etk-analysis nice -n 10 \
        /Volumes/ExternalNVMe/etk-analysis/.venv/bin/python sample_raw_planes.py \
        --index /Volumes/ExternalNVMe/som-proto-f5/idx3d_0003.json \
        --parfile .../output-0003/parfile.par --it 180224 -o raw_planes_it180224.npz

For each plane (``meridional``: e1 from hole 1 to hole 2, e2 the orbital axis;
``orbital``: e1 the same, e2 in the orbital plane) through the centre of mass,
writes the frame (origin, e1, e2), the hole positions, the in-plane coordinates,
and Bx, By, Bz, smallb2, rho_b, P sampled linearly from the finest level that
holds data at each point.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from etkplot.orbit import corot_frame, load_trajectory
from etkplot.resample import sample_plane
from etkplot.source import open_source

VARS = ["Bx", "By", "Bz", "smallb2", "rho_b", "P"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", required=True)
    p.add_argument("--parfile", required=True)
    p.add_argument("--it", type=int, required=True)
    p.add_argument("--half-width", type=float, default=24.0)
    p.add_argument("--n", type=int, default=601)
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()

    t0 = time.perf_counter()
    src = open_source(a.index)
    traj = load_trajectory(a.parfile)
    t = src.times("rho_b")[a.it]
    holes = traj.positions(t)
    out = {"it": a.it, "time": t, "holes": holes, "half_width": a.half_width,
           "masses": traj.masses}
    for kind in ("meridional", "orbital"):
        fr = corot_frame(traj, t, kind)
        out[f"{kind}_origin"], out[f"{kind}_e1"], out[f"{kind}_e2"] = fr.origin, fr.e1, fr.e2
        for v in VARS:
            s = sample_plane(src, v, a.it, fr, half_width=a.half_width, n=a.n)
            out[f"{kind}_{v}"] = s.data.astype(np.float32)
            out[f"{kind}_level"] = s.level.astype(np.int8)
            out[f"{kind}_a"], out[f"{kind}_b"] = s.a, s.b
            print(f"{kind} {v}: done ({time.perf_counter() - t0:.1f}s)", flush=True)
    np.savez_compressed(a.out, **out)
    print(f"t = {t}, holes = {holes.tolist()}, wrote {a.out} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
