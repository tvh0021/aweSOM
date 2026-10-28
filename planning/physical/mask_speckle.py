"""How speckled are the reference sheet masks themselves?

    python mask_speckle.py --it 135168

Connected components (faces, per level) of the T=1 and T=3 metric sheet masks
and of the chi_B ridge: the volume share in components smaller than 8 / 27
cells. If the reference itself is speckled, removing small label components
lowers the Jaccard with it by construction.
"""

import argparse
import json

import numpy as np

from common import COLS, RES, limit_threads, load_grid

limit_threads(2)
from aweSOM.physics_score import component_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--it", type=int, default=135168)
    a = ap.parse_args()
    grid = load_grid(a.it)
    cv = {g: grid.spacing[g] ** grid.ndim for g in grid.groups}
    out = {}
    for name in ("sheet_T1", "sheet_T3", "ridge"):
        m = np.load(COLS / f"{a.it}_{name}.npy")
        lab = np.where(m, 1, -1).astype(np.int16)
        maps = grid.to_maps(lab, -1)
        r = {}
        for small in (8, 27):
            s = component_stats(maps, 2, cv, small=small)
            r[f"small_frac_{small}"] = float(s["small_frac"][1])
            r["n_components"] = int(s["n_components"][1])
            r["largest_frac"] = float(s["largest_frac"][1])
        out[name] = r
        print(name, r, flush=True)
        del maps
    (RES / "tables" / f"mask_speckle_{a.it}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
