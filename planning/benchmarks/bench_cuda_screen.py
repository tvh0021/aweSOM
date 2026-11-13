"""Time certified CUDA screening on the current device against the CPU reference.

Run after the device tests, for example::

    PYTHONPATH=src python planning/benchmarks/bench_cuda_screen.py --rows 1000000

Includes warmup, bitwise checks against the CPU float64 search and the median
synchronous search time, excluding construction but including host/device copies. No cluster jobs are submitted.
"""
import argparse
import json
import time

import numpy as np

from aweSOM._gpu_screen import GpuScreenSearcher, _cuda_screen_reference
from aweSOM.inference import best_match_nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=1_000_000)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    for case in ['random', 'smooth', 'contracted']:
        w = rng.normal(size=(10000, 7))
        if case != 'random':
            a, b = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
            w = np.column_stack([np.sin(a.ravel() * (f + 1)) + b.ravel() for f in range(7)])
        if case == 'contracted':
            w *= 1e-4
        x = (w[rng.integers(len(w), size=args.rows)] +
             rng.normal(size=(args.rows, 7)) * 0.02).astype(args.dtype)
        idx, d2 = np.empty(args.rows, np.int32), np.empty(args.rows)
        search = GpuScreenSearcher(w, 100, 100)
        search(x, idx, d2)  # Compilation and allocation are outside the timer.
        ref_i, ref_d = best_match_nodes(w, x)
        np.testing.assert_array_equal(idx, ref_i)
        np.testing.assert_array_equal(d2.view(np.uint64), ref_d.view(np.uint64))
        elapsed = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            search(x, idx, d2)
            elapsed.append(time.perf_counter() - start)
        seconds = float(np.median(elapsed))
        _, _, fb = _cuda_screen_reference(w, x[:1024], return_fallbacks=True)
        print(json.dumps(dict(case=case, rows=args.rows, dtype=args.dtype, seconds=seconds,
                              mrows_per_s=args.rows / seconds / 1e6,
                              sampled_fallback_fraction=fb / min(args.rows, 1024))))


if __name__ == '__main__':
    main()
