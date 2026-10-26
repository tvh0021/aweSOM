"""Shared setup for the training benchmarks: real fiducial plasma data and the reference kernel."""
import pickle, sys, time
import numpy as np
from numba import njit

PKL = "/Users/tvh0021/git_repos/aweSOM/examples/plasma-turbulence/som_object.fiducial-5000-64-37-0.1-2097152-1u.pkl"


def load_fiducial():
    with open(PKL, "rb") as fh:
        fid = pickle.load(fh)
    return fid


def median_interleaved(fns, reps=5, warm=1):
    """Run every fn in fns round-robin; return the per-fn list of times (warm-up excluded)."""
    times = {k: [] for k in fns}
    for r in range(warm + reps):
        for k, fn in fns.items():
            t = time.perf_counter(); fn(); dt = time.perf_counter() - t
            if r >= warm:
                times[k].append(dt)
    return times


# --- verbatim copy of the step kernel at commit 9f29e9b (reference) ---
@njit()
def _pairwise_sum_ref(a, start, n):
    if n < 8:
        res = -0.0
        for i in range(start, start + n):
            res += a[i]
        return res
    elif n <= 128:
        r0 = a[start]; r1 = a[start + 1]; r2 = a[start + 2]; r3 = a[start + 3]
        r4 = a[start + 4]; r5 = a[start + 5]; r6 = a[start + 6]; r7 = a[start + 7]
        i = 8
        while i < n - (n % 8):
            r0 += a[start + i]; r1 += a[start + i + 1]; r2 += a[start + i + 2]; r3 += a[start + i + 3]
            r4 += a[start + i + 4]; r5 += a[start + i + 5]; r6 += a[start + i + 6]; r7 += a[start + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[start + i]
            i += 1
        return res
    else:
        n2 = n // 2
        n2 -= n2 % 8
        return _pairwise_sum_ref(a, start, n2) + _pairwise_sum_ref(a, start + n2, n - n2)


@njit()
def _neighborhood_weight_ref(dist, alpha, nsize):
    sigma = nsize / 3
    return alpha * np.exp(-((dist) ** 2) / (2 * sigma**2))


@njit()
def som_train_steps_ref(lattice, xk, xdim, alphas, nsizes, k_start, k_stop):
    n_nodes, n_features = lattice.shape
    ydim = n_nodes // xdim
    s = np.empty(n_nodes)
    squ = np.empty(n_features)
    for k in range(k_start, k_stop):
        for i in range(n_nodes):
            if n_features < 8:
                acc = -0.0
                for f in range(n_features):
                    diff = lattice[i, f] - xk[k, f]
                    acc += diff * diff
                s[i] = acc
            else:
                for f in range(n_features):
                    diff = lattice[i, f] - xk[k, f]
                    squ[f] = diff * diff
                s[i] = _pairwise_sum_ref(squ, 0, n_features)
        c = 0
        best = s[0]
        if not np.isnan(best):
            for i in range(1, n_nodes):
                if s[i] < best:
                    best = s[i]
                    c = i
                elif np.isnan(s[i]):
                    c = i
                    break
        alpha = alphas[k]
        nsize = nsizes[k]
        cx = c % xdim
        cy = c // xdim
        for y in range(max(0, cy - nsize), min(ydim, cy + nsize + 1)):
            for x in range(max(0, cx - nsize), min(xdim, cx + nsize + 1)):
                g = _neighborhood_weight_ref(float(max(abs(x - cx), abs(y - cy))), alpha, nsize)
                if g != 0.0:
                    i = x + y * xdim
                    for f in range(n_features):
                        diff = lattice[i, f] - xk[k, f]
                        lattice[i, f] -= diff * g


def online_setup(data, xdim, ydim, alpha0, n_steps, seed=42):
    """The arrays fast_som builds for a fresh run (sampling init, decay schedule)."""
    from aweSOM.som import _som_schedule
    n_in, n_feat = data.shape
    rng = np.random.default_rng(seed)
    lattice0 = data[rng.integers(0, n_in, xdim * ydim), :].astype(np.float64, copy=True)
    nsize_max = max(xdim, ydim) + 1
    nsize_min = 8 if n_steps // 4 > nsize_max - 8 else nsize_max - n_steps // 4
    nsize_step = n_steps // 4
    nsize_freq = max(1, nsize_step // max(1, nsize_max - nsize_min))
    alphas, nsizes, _, _ = _som_schedule(n_steps, 0, float(alpha0), 1, max(1, n_steps // 25),
                                         nsize_max, nsize_min, nsize_step, nsize_freq)
    idx = np.arange(n_in); rng.shuffle(idx)
    if n_steps > n_in:
        idx = np.tile(idx, n_steps // n_in + 1)
    xk = np.ascontiguousarray(data[idx[:n_steps]], dtype=np.float64)
    return lattice0, xk, alphas, nsizes
