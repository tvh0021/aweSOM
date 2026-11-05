"""Synthetic seven-feature rows from the gmm_bbh7 mixture (tests/fixtures/gmm_bbh7.npz).

Same generator as the Z-scale 1e9 kit (studies/2026-11-03/Z-scale-1e9-gp/kit/scale_1e9.py):
block ``b`` of ``GEN_BLOCK`` rows comes from ``default_rng([seed, b])`` with elementwise
float32 arithmetic only, so the rows do not depend on the machine or the thread count.
"""
from pathlib import Path

import numpy as np

GMM = Path(__file__).with_name("fixtures") / "gmm_bbh7.npz"
GEN_BLOCK = 2**20
DATA_SEED = 20261103


def block(b: int, seed: int = DATA_SEED) -> np.ndarray:
    """Rows of block ``b``: (GEN_BLOCK, 7) float32."""
    z = np.load(GMM)
    w = z["weights"].astype(np.float64)
    mu = np.ascontiguousarray(z["means"], np.float32)
    L = np.ascontiguousarray(z["cholesky"], np.float32)
    lo, hi = z["clip_lo"].astype(np.float32), z["clip_hi"].astype(np.float32)
    k, f = mu.shape
    cumw = np.cumsum(w / w.sum())
    cumw[-1] = 1.0
    rng = np.random.default_rng([seed, b])
    comp = np.searchsorted(cumw, rng.random(GEN_BLOCK), side="right")
    np.minimum(comp, k - 1, out=comp)
    zz = rng.standard_normal((f, GEN_BLOCK), dtype=np.float32)
    x = np.empty((f, GEN_BLOCK), np.float32)
    tmp = np.empty(GEN_BLOCK, np.float32)
    for i in range(f):
        np.take(np.ascontiguousarray(mu[:, i]), comp, out=x[i], mode="clip")
        for j in range(i + 1):
            np.take(np.ascontiguousarray(L[:, i, j]), comp, out=tmp, mode="clip")
            np.multiply(tmp, zz[j], out=tmp)
            np.add(x[i], tmp, out=x[i])
        np.clip(x[i], lo[i], hi[i], out=x[i])
    return np.ascontiguousarray(x.T)


def rows(n: int, seed: int = DATA_SEED) -> np.ndarray:
    """First ``n`` rows, (n, 7) float32."""
    parts = [block(b, seed) for b in range(-(-n // GEN_BLOCK))]
    return np.concatenate(parts)[:n]
