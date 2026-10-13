"""Time a compiled (numba) single-threaded prototype of the SOM training loop."""
import time
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def train_steps(lattice, xk, xdim, alpha, nsize):
    # same math as fast_som (BMU + Gaussian/Chebyshev-cutoff update), but only touches nodes inside the window
    n_nodes, n_feat = lattice.shape
    ydim = n_nodes // xdim
    sigma2 = 2.0 * (nsize / 3.0) ** 2
    for t in range(xk.shape[0]):
        x = xk[t]
        best = np.inf; c = 0
        for j in range(n_nodes):
            s = 0.0
            for f in range(n_feat):
                d = lattice[j, f] - x[f]; s += d * d
            if s < best:
                best = s; c = j
        cx = c % xdim; cy = c // xdim
        for iy in range(max(0, cy - nsize), min(ydim, cy + nsize + 1)):
            for ix in range(max(0, cx - nsize), min(xdim, cx + nsize + 1)):
                dist = max(abs(ix - cx), abs(iy - cy))
                h = alpha * np.exp(-dist * dist / sigma2)
                j = ix + iy * xdim
                for f in range(n_feat):
                    lattice[j, f] -= (lattice[j, f] - x[f]) * h
    return lattice

rng = np.random.default_rng(0)
xk = rng.normal(size=(100_000, 4)) * 0.5
lat = xk[rng.integers(0, len(xk), 64 * 37)].copy()
train_steps(lat.copy(), xk[:10], 64, 0.1, 65)
for nsize in (65, 8):
    t0 = time.perf_counter(); train_steps(lat.copy(), xk, 64, 0.1, nsize); dt = time.perf_counter() - t0
    print(f"njit training, nsize={nsize:>2}: {dt:.3f} s for 100k steps = {dt / 1e5 * 1e6:.1f} us/step")
