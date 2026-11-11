"""Time the CUDA labelling path in three layers on synthetic rows.

1. ``kernel``: the search kernel alone on a device-resident pool, 2**20-row
   launches on one stream, timed with ``cp.cuda.Event``.
2. ``searcher``: synchronous ``GpuSearcher.__call__`` on host rows (staging,
   copies of idx and d2, drain per call), as in a direct searcher call.
3. ``predict_stream``: labels only (no distances), the queued path, from a
   callable source that serves views of the host pool.

Rows cycle through one pool (``--pool`` rows, host and device), so 1e9 rows
need no 28 GB input. The lattice is a smooth 100x100 sheet and rows sit near
nodes, like trained data. The script prints one JSON line per layer and checks
that ``predict_stream`` labels equal ``node_cluster[idx]`` of the searcher on
the first pool cycle. It reads no system information. Example::

    PYTHONPATH=src python planning/benchmarks/bench_label_path.py --rows 1000000000
"""
import argparse
import json
import time

import numpy as np

from aweSOM.stream import predict_stream


def lattice(nodes_side=100, features=7):
    a, b = np.meshgrid(np.linspace(-1, 1, nodes_side), np.linspace(-1, 1, nodes_side))
    return np.column_stack([np.sin(a.ravel() * (f + 1)) + b.ravel() for f in range(features)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=float, default=1e8)
    parser.add_argument('--pool', type=int, default=2**24, help='rows in the reused pool')
    parser.add_argument('--chunk', type=int, default=2**20, help='predict_stream chunk')
    parser.add_argument('--backend', choices=['cuda', 'cuda_screen'], default='cuda_screen')
    parser.add_argument('--clusters', type=int, default=10, help='node_cluster = node %% clusters')
    parser.add_argument('--out-dtype', choices=['int32', 'uint8'], default='int32')
    parser.add_argument('--layers', default='kernel,searcher,predict_stream')
    args = parser.parse_args()
    import cupy as cp
    from aweSOM._gpu import GpuSearcher
    from aweSOM._gpu_screen import GpuScreenSearcher

    rows = int(args.rows)
    sub = 2**20
    # A whole, positive number of predict_stream chunks, so the source cycles
    # through the entire pool (the kernel loop handles any pool size).
    pool_rows = max(1, args.pool // args.chunk) * args.chunk
    rng = np.random.default_rng(42)
    w = lattice()
    pool = (w[rng.integers(len(w), size=pool_rows)] +
            rng.normal(size=(pool_rows, w.shape[1])) * 0.02).astype(np.float32)
    clusters = (np.arange(len(w)) % args.clusters).astype(np.int32)
    cls = GpuScreenSearcher if args.backend == 'cuda_screen' else GpuSearcher
    search = cls(w, 100, 100)
    idx, d2 = np.empty(pool_rows, np.int32), np.empty(pool_rows)
    search(pool, idx, d2)  # compile, allocate staging, reference for the label check
    layers = args.layers.split(',')
    common = dict(backend=args.backend, rows=rows, pool=pool_rows, nodes=len(w),
                  features=w.shape[1])

    def report(layer, seconds, **extra):
        print(json.dumps(dict(common, layer=layer, seconds=seconds,
                              mrows_per_s=rows / seconds / 1e6,
                              ns_per_row=seconds / rows * 1e9, **extra)), flush=True)

    if 'kernel' in layers:
        x_dev = cp.asarray(pool)
        i_dev, d_dev = cp.empty(sub, cp.int32), cp.empty(sub, cp.float64)
        stream = cp.cuda.Stream(non_blocking=True)
        kernel = search.kernels[0]

        def launch(start, n):
            kernel(((n + search.block - 1) // search.block,), (search.block,),
                   (search.w, x_dev[start:start + n], i_dev, d_dev, np.int32(n),
                    np.int32(search.nodes), np.int32(search.tile)),
                   shared_mem=search.shared, stream=stream)

        cp.cuda.get_current_stream().synchronize()
        launch(0, sub)
        stream.synchronize()
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record(stream)
        done = 0
        while done < rows:
            offset = done % pool_rows
            n = min(sub, rows - done, pool_rows - offset)
            launch(offset, n)
            done += n
        end.record(stream)
        end.synchronize()
        report('kernel', cp.cuda.get_elapsed_time(begin, end) / 1e3)
        del x_dev, i_dev, d_dev

    if 'searcher' in layers:
        start_t = time.perf_counter()
        done = 0
        while done < rows:
            n = min(pool_rows, rows - done)
            search(pool[:n], idx[:n], d2[:n])
            done += n
        report('searcher', time.perf_counter() - start_t, call_rows=pool_rows)
        search(pool, idx, d2)  # restore the full-pool reference

    if 'predict_stream' in layers:
        chunk = args.chunk
        per_pool = pool_rows // chunk

        def source():
            for start in range(0, rows, chunk):
                o = (start // chunk) % per_pool * chunk
                yield pool[o:o + min(chunk, rows - start)]
        source.n_feat = w.shape[1]
        out = np.empty(rows, args.out_dtype)
        out[:] = 0  # fault the pages outside the timer
        start_t = time.perf_counter()
        _, stats = predict_stream(w, clusters, source, out=out, chunk=chunk,
                                  backend=args.backend, n_samples=rows)
        seconds = time.perf_counter() - start_t
        m = min(rows, pool_rows)
        same = bool(np.array_equal(out[:m], clusters[idx[:m]].astype(args.out_dtype)))
        report('predict_stream', seconds, chunk=chunk, out_dtype=args.out_dtype,
               labels_match_searcher=same,
               stats={k: v for k, v in stats.items() if k != 'rows'})
        if not same:
            raise SystemExit('predict_stream labels differ from the searcher')


if __name__ == '__main__':
    main()
