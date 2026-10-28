"""Per-owned-point side fields on levels 5-7 of one iteration, in the same owned
order as ``som-features --grid-out`` (patches in read_level order, C order):

- ``log_one_minus_C_B`` (etkplot.features, 5^3 box), always;
- with ``--metric``: the sheets metric of etk-analysis/scripts/sheets_metric.py
  (``ratio_m`` = |C|/|B| in M^-1, ``th3`` reversal angle in degrees at +-3 cells,
  ``b2``), computed by that script's own ``fields`` function.

Writes ``<out>/extra_it<it>_rl<rl>_<key>.npy`` (float32) and a check file with
every 97th point's log_grad_ln_b2 (before winsorizing) so the order can be
verified against the grid file. Run with the etk-analysis venv:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Volumes/ExternalNVMe/etk-analysis \
      /Volumes/ExternalNVMe/etk-analysis/.venv/bin/python extra_fields.py \
      idx3d_0003.json --it 180224 --parfile parfile.par --b2-min 2.2196638e-08 \
      -o OUT --metric

Read-only use of etk-analysis; nothing there is modified.
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "/Volumes/ExternalNVMe/etk-analysis/scripts")
import sheets_metric as sm  # noqa: E402

from etkplot import open_source  # noqa: E402
from etkplot import features as F  # noqa: E402
from etkplot import metric as M  # noqa: E402
from etkplot import ownership  # noqa: E402
from etkplot.somfeatures import _read_slab, _slabs  # noqa: E402

LEVELS = (5, 6, 7)
STRIDE = 97


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("--it", type=int, required=True)
    ap.add_argument("--parfile", required=True)
    ap.add_argument("--b2-min", type=float, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--metric", action="store_true")
    ap.add_argument("--slab", type=int, default=32)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    src = open_source(args.index)
    opt = SimpleNamespace(accuracy=sm.ACC, parfile=args.parfile, b2_min=args.b2_min, s_v=1.0)
    st = M.Spacetime.from_parfile(args.parfile)
    t = src.times("rho_b")[args.it]
    hz = ownership.horizons(src, "smallb2", args.it, opt)
    halo = sm.HALO if args.metric else max(F.C_B_HALF, F.halo(["log_grad_ln_b2"], opt))
    variables = sm.VARS
    t0 = time.time()
    for rl in LEVELS:
        own = ownership.level_ownership(src, "smallb2", args.it, rl, opt, hz)
        keys = ["log_one_minus_C_B"] + (["ratio_m", "th3", "b2"] if args.metric else [])
        parts = {k: [] for k in keys}
        check = []
        start = 0
        for p in own.patches:
            for a, b in _slabs(p.component.shape[0], args.slab):
                owned = p.component[a:b] >= 0
                cnt = int(np.count_nonzero(owned))
                if cnt == 0:
                    continue
                arr = {v: _read_slab(src, v, args.it, rl, p, a, b, halo) for v in variables}
                inner = slice(halo, halo + b - a)
                ev = np.zeros(arr["Bx"].shape, bool)
                ev[inner] = owned
                fe = F.compute(["log_one_minus_C_B", "log_grad_ln_b2"], arr, p.delta, opt)
                parts["log_one_minus_C_B"].append(fe["log_one_minus_C_B"][ev].astype(np.float32))
                lg = fe["log_grad_ln_b2"][ev]
                sel = (np.arange(start, start + cnt) % STRIDE) == 0
                check.append(np.stack([np.flatnonzero(sel) + start, lg[sel]], 1))
                if args.metric:
                    origin = np.array(p.origin, float).copy()
                    origin[0] += (a - halo) * p.delta[0]
                    f = sm.fields(arr, origin, np.asarray(p.delta, float), t, st, opt, ev)
                    for k in ("ratio_m", "th3", "b2"):
                        parts[k].append(f[k].astype(np.float32))
                start += cnt
        assert start == own.count, (start, own.count)
        for k, v in parts.items():
            np.save(args.out / f"extra_it{args.it}_rl{rl}_{k}.npy", np.concatenate(v))
        np.save(args.out / f"extra_it{args.it}_rl{rl}_check_lg.npy", np.concatenate(check))
        print(f"rl {rl}: {own.count} owned points, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
