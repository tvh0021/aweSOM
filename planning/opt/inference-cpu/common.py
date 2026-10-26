"""Shared helpers: real fiducial data loader, interleaved timing."""
import pickle
import time

import numpy as np

PKL = "/Users/tvh0021/git_repos/aweSOM/examples/plasma-turbulence/som_object.fiducial-5000-64-37-0.1-2097152-1u.pkl"


def load_real():
    import aweSOM  # noqa: F401  (needed to unpickle)
    with open(PKL, "rb") as fh:
        lat = pickle.load(fh)
    w = np.ascontiguousarray(lat.lattice, dtype=np.float64)
    x = np.ascontiguousarray(lat.data_array, dtype=np.float64)
    return w, x


def synthetic(F, n=2_097_152, nodes=2368, seed=0):
    """Spatially coherent synthetic data: smooth random fields on a 128^3 box.

    Lattice = random sample of data rows (density-matched stand-in for a trained SOM).
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    side = round(n ** (1 / 3))
    cols = []
    for f in range(F):
        fld = gaussian_filter(rng.normal(size=(side,) * 3), sigma=3, mode="wrap")
        fld = (fld - fld.mean()) / fld.std()
        cols.append(fld.ravel())
    x = np.ascontiguousarray(np.stack(cols, axis=1))
    w = np.ascontiguousarray(x[rng.choice(x.shape[0], nodes, replace=False)] + 0.0)
    return w, x


def interleaved(fns: dict, reps=7, warm=1):
    """Run each fn in turn, reps times (round robin). Returns medians (s) and all samples."""
    for _ in range(warm):
        for f in fns.values():
            f()
    samples = {k: [] for k in fns}
    cpu = {k: [] for k in fns}
    for _ in range(reps):
        for k, f in fns.items():
            c0 = time.process_time()
            t0 = time.perf_counter()
            f()
            samples[k].append(time.perf_counter() - t0)
            cpu[k].append(time.process_time() - c0)
    med = {k: float(np.median(v)) for k, v in samples.items()}
    interleaved.cpu_median = {k: float(np.median(v)) for k, v in cpu.items()}
    return med, samples
