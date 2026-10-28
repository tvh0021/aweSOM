"""Shared paths and loaders for the 2026-10-28 physicality study."""

import json
import os
import pickle
from pathlib import Path

import numpy as np

RES = Path("/Volumes/ExternalNVMe/awesom-results/2026-10-28/physical")
DATA = RES / "data"
COLS = DATA / "cols"
MODELS = RES / "models"
SCORES = RES / "scores"
FIGS = RES / "figures"

GRID = {
    180224: Path("/Volumes/ExternalNVMe/awesom-results/2026-10-28/lr-flux-180224/grid_LR_flux_180224.h5"),
    135168: DATA / "grid_135168.h5",
}
SAMPLE = {
    180224: Path("/Volumes/ExternalNVMe/som-proto-f5/features_bbh_0003-180224-N10000000.h5"),
    135168: Path("/Volumes/ExternalNVMe/som-proto-f5/features_bbh_0003-135168-N10000000.h5"),
}
METRIC_NPZ_135168 = Path("/Volumes/ExternalNVMe/som-proto-f5/sheets-metric/sheets_3d_it135168.npz")
B2_MIN = {180224: 2.2196638299940852e-08, 135168: 2.0437879837720358e-08}
LEVELS = (5, 6, 7)
DX = {5: 0.25, 6: 0.125, 7: 0.0625}

BASE = ["log_beta_inv", "log_theta", "log_chi_B", "log_grad_ln_b2", "asinh_div_v"]
VARIANTS = {
    "base5": BASE,
    "nodiv4": BASE[:4],
    "cb_swap5": BASE[:4] + ["log_one_minus_C_B"],
    "cb_add6": BASE + ["log_one_minus_C_B"],
    "chic_swap5": BASE[:4] + ["chi_contrast"],
    "chic_add6": BASE + ["chi_contrast"],
}
MERGE_COSTS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)


def col(it, name, mmap=True):
    return np.load(COLS / f"{it}_{name}.npy", mmap_mode="r" if mmap else None)


def load_grid(it):
    with open(COLS / f"{it}_voxelgrid.pkl", "rb") as fh:
        return pickle.load(fh)


def meta(it):
    return json.loads((COLS / f"{it}_meta.json").read_text())


def limit_threads(n=3):
    for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(k, str(n))
