"""Usage: python bench_inference_jax.py [N] [F] [chunk ...]

Time the numba BMU kernel (aweSOM.inference.best_match_nodes, float64) against
the JAX path (best_match_nodes_jax) on random data and a 64 x 37 lattice.
Each kernel runs once to compile, then the median of three runs is reported.
The JAX path runs with precision="highest" (float64 where available, else a
float32 fallback) and precision="float32", for each tile size given (default:
the library default). Label disagreements are counted against numba."""

import platform
import sys
import time
import warnings

import numpy as np

import jax
from aweSOM.inference import best_match_nodes, best_match_nodes_jax

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
F = int(sys.argv[2]) if len(sys.argv) > 2 else 4
CHUNKS = [int(c) for c in sys.argv[3:]] or [None]
NODES = 64 * 37
REPEAT = 3

rng = np.random.default_rng(0)
weights = rng.uniform(-1.0, 1.0, size=(NODES, F))
data = rng.normal(scale=0.5, size=(N, F))


def timeit(fn):
    out = fn()  # compile
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


print(
    f"N={N} F={F} nodes={NODES} | jax {jax.__version__} on {jax.devices()} | "
    f"{platform.machine()} {platform.mac_ver()[0] or platform.platform()}"
)
(ref_idx, ref_d2), t = timeit(lambda: best_match_nodes(weights, data))
print(f"{'numba float64':32s} {N / t:10.3e} points/s  {t:7.3f} s")

warnings.simplefilter("ignore", RuntimeWarning)
for chunk in CHUNKS:
    for precision in ("highest", "float32"):
        (idx, d2), t = timeit(
            lambda: best_match_nodes_jax(weights, data, chunk=chunk, precision=precision)
        )
        n_diff = int(np.count_nonzero(idx != ref_idx))
        d2_equal = bool(np.array_equal(d2, ref_d2))
        label = f"jax {precision} ({d2.dtype}) chunk={chunk}"
        print(
            f"{label:32s} {N / t:10.3e} points/s  {t:7.3f} s  "
            f"label disagreements {n_diff} ({n_diff / N:.2e})  d2 bitwise equal {d2_equal}"
        )
