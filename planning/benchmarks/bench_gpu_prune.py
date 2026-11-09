"""GH200 parity/latency probe. Run with PYTHONPATH=src; requires CuPy/CUDA."""
import argparse
import json
import time

import numpy as np

from aweSOM._gpu import GpuSearcher
from aweSOM._gpu_prune import PruneGpuSearcher, _cuda_prune_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=200000)
    parser.add_argument('--side', type=int, default=100)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--weights', help='Optional trained float64 .npy lattice')
    parser.add_argument('--data', help='Optional scaled .npy rows')
    parser.add_argument('--hint', help='Optional previous int32 BMUs in .npy format')
    args = parser.parse_args()
    rng = np.random.default_rng(41)
    u, v = np.meshgrid(np.linspace(-1, 1, args.side), np.linspace(-1, 1, args.side))
    w = np.column_stack([u.ravel(), v.ravel()] + [
        (0.2 * np.sin((f + 1) * u + v)).ravel() for f in range(5)])
    if args.weights:
        w = np.load(args.weights)
    previous = rng.integers(len(w), size=args.rows, dtype=np.int32)
    x = w[previous] + rng.normal(scale=0.002, size=(args.rows, w.shape[1]))
    if args.data:
        x = np.load(args.data)[:args.rows]
    i, d = np.empty(len(x), np.int32), np.empty(len(x))
    full = GpuSearcher(w, args.side, args.side)
    prune = PruneGpuSearcher(w, (args.side, args.side))
    full(x, i, d)
    ri, rd = i.copy(), d.copy()
    hint = np.load(args.hint)[:len(x)] if args.hint else (ri if args.data else previous)
    results = {}
    for name, search, hint in [('cuda', full, None), ('prune', prune, None),
                               ('prune_hint', prune, hint),
                               ('prune_wrong_hint', prune, np.full(len(x), len(w) - 1, np.int32))]:
        def run():
            if name == 'cuda':
                search(x, i, d)
            else:
                search(x, i, d, hint)
        run()  # compile/warm up outside timing
        times = []
        for _ in range(args.repeat):
            t = time.perf_counter()
            run()
            times.append(time.perf_counter() - t)
        np.testing.assert_array_equal(i, ri)
        np.testing.assert_array_equal(d.view(np.uint64), rd.view(np.uint64))
        record = dict(seconds=min(times), rows_per_second=len(x) / min(times))
        if name != 'cuda':
            _, _, counts = _cuda_prune_reference(w, x[:1000], (args.side, args.side),
                hint=None if hint is None else hint[:1000], return_evaluations=True)
            record['nodes_checked_fraction'] = float(counts.mean() / len(w))
        results[name] = record
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
