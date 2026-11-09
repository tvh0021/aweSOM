"""Compare exact CUDA and certified CUDA screening on the current device.

Run after the device tests, for example::

    PYTHONPATH=src python planning/benchmarks/bench_cuda_screen.py --rows 1000000

Includes warmup, bitwise checks and median synchronous search time, excluding
construction but including host/device copies. No cluster jobs are submitted.
"""
import argparse
import json
import time

import numpy as np

from aweSOM._gpu import GpuSearcher
from aweSOM._gpu_screen import GpuScreenSearcher, _cuda_screen_reference


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
        ref_i, ref_d = np.empty(args.rows, np.int32), np.empty(args.rows)
        idx, d2 = np.empty_like(ref_i), np.empty_like(ref_d)
        timings = {}
        for name, cls in [('cuda', GpuSearcher), ('cuda_screen', GpuScreenSearcher)]:
            search = cls(w, 100, 100)
            search(x, idx, d2)  # Compilation and allocation are outside the timer.
            if name == 'cuda':
                ref_i[:], ref_d[:] = idx, d2
            else:
                np.testing.assert_array_equal(idx, ref_i)
                np.testing.assert_array_equal(d2.view(np.uint64), ref_d.view(np.uint64))
            elapsed = []
            for _ in range(args.repeat):
                start = time.perf_counter()
                search(x, idx, d2)
                elapsed.append(time.perf_counter() - start)
            timings[name] = float(np.median(elapsed))
        _, _, fb = _cuda_screen_reference(w, x[:1024], return_fallbacks=True)
        print(json.dumps(dict(case=case, rows=args.rows, dtype=args.dtype,
                              seconds=timings, speedup=timings['cuda'] / timings['cuda_screen'],
                              sampled_fallback_fraction=fb / min(args.rows, 1024))))


if __name__ == '__main__':
    main()
