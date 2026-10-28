"""Check analyze_chunks.py output against an etkplot grid file of the same iteration.

The grid file (``som-features --grid-out``) holds every owned point's
thresholded features, xyz and level in the same order as the chunked driver
writes them. This script predicts the grid file's labels with the same model
and compares, chunk by chunk: labels (must be identical), level and xyz
(identical), and the features if the driver saved them.

    python compare_with_grid.py --grid grid_LR_flux_180224.h5 --chunks out/it00180224.h5 --model model.h5
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import h5py
import numpy as np

from aweSOM.model import SOMModel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grid", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", type=int, default=1 << 22, help="rows per comparison chunk")
    ap.add_argument("-o", "--out", help="write the result as JSON here")
    args = ap.parse_args(argv)
    model = SOMModel.load(args.model)
    t0 = time.time()
    res = {"grid": args.grid, "chunks": args.chunks}
    with h5py.File(args.grid, "r") as g, h5py.File(args.chunks, "r") as c:
        n_g, n_c = g["features"].shape[0], c["labels"].shape[0]
        res.update(rows_grid=int(n_g), rows_chunks=int(n_c), complete=bool(c.attrs.get("complete", False)))
        if n_g != n_c:
            res["error"] = "row counts differ"
        n = min(n_g, n_c)
        diff = {"labels": 0, "level": 0, "xyz": 0, "features": 0}
        per_level = {}
        has_f = "features" in c
        for a in range(0, n, args.rows):
            b = min(a + args.rows, n)
            gf = g["features"][a:b]
            lab_g = model.predict(gf)
            lab_c = c["labels"][a:b]
            lev = g["level"][a:b]
            bad = lab_g != lab_c
            diff["labels"] += int(bad.sum())
            diff["level"] += int(np.count_nonzero(lev != c["level"][a:b]))
            diff["xyz"] += int(np.count_nonzero(np.any(g["xyz"][a:b] != c["xyz"][a:b], axis=1)))
            if has_f:
                cf = c["features"][a:b]
                same = (gf == cf) | (np.isnan(gf) & np.isnan(cf))
                diff["features"] += int(np.count_nonzero(~np.all(same, axis=1)))
            for L in np.unique(lev):
                s = per_level.setdefault(int(L), {"rows": 0, "label_diff": 0})
                m = lev == L
                s["rows"] += int(m.sum())
                s["label_diff"] += int(np.count_nonzero(bad & m))
        res["differing_rows"] = diff
        res["features_compared"] = has_f
        res["per_level"] = per_level
    res["seconds"] = time.time() - t0
    res["identical"] = (res["rows_grid"] == res["rows_chunks"]) and not any(diff.values())
    txt = json.dumps(res, indent=1)
    print(txt)
    if args.out:
        open(args.out, "w").write(txt)
    return 0 if res["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
