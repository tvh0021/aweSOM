"""Exact BMU and transfer probe; run beside scale_1e9.py and gmm_bbh7.npz."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def bits(a, b):
    return bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--kit', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--rows', type=int, default=20_000_000)
    parser.add_argument('--check-rows', type=int, default=1_000_000)
    args = parser.parse_args()
    args.out.mkdir()  # Refuse any existing run, including a failed one.
    import cupy as cp
    import numba
    import aweSOM
    from aweSOM._gpu import GpuSearcher, _pinned
    from aweSOM.inference import _bmu_kernel
    from aweSOM.stream import peak_rss_bytes, train_batch_stream
    import importlib.util
    spec = importlib.util.spec_from_file_location('scale_generator', args.kit / 'scale_1e9.py')
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    threads = int(os.environ.get('NUMBA_NUM_THREADS', '8'))
    numba.set_num_threads(threads)
    source_root = Path(aweSOM.__file__).resolve().parents[2]
    if (source_root / 'SOURCE_COMMIT').exists():
        commit = (source_root / 'SOURCE_COMMIT').read_text().strip()
    else:
        commit = subprocess.check_output(['git', '-C', str(source_root),
                                          'rev-parse', 'HEAD'], text=True).strip()
    record = dict(status='running', rows=args.rows, check_rows=args.check_rows,
                  threads=threads, ranks=1, nodes=[os.environ.get('SLURMD_NODENAME', 'unknown')],
                  commit=commit,
                  cupy=cp.__version__, driver=cp.cuda.runtime.driverGetVersion(),
                  runtime=cp.cuda.runtime.runtimeGetVersion(), checks=[], rates=[], epochs=[],
                  kernel_b='not implemented; exact float64 kernel has zero fallbacks')

    def save():
        record['peak_rss_bytes'] = peak_rss_bytes()
        p = args.out / 'probe.json.tmp'
        p.write_text(json.dumps(record, indent=2, default=lambda x: x.tolist()) + '\n')
        p.replace(args.out / 'probe.json')

    save()
    try:
        mix = gen.Mixture(args.kit / 'gmm_bbh7.npz')
        x, provenance = gen.generate(mix, max(args.rows, args.check_rows), 20261103, threads, np.float32)
        record['generator'] = provenance
        trained = np.load(args.kit / 'trained-8238.npy')
        assert trained.shape == (91 * 64, 7)
        record['trained_weights_sha256'] = hashlib.sha256(trained.tobytes()).hexdigest()
        initial = x[np.random.default_rng(42).choice(len(x), size=91 * 64, replace=False)].astype(np.float64)
        np.save(args.out / 'sampled-initial.npy', initial)
        for name, w in [('trained', trained), ('sampled', initial)]:
            subset = x[:args.check_rows]
            ri, rd = np.empty(len(subset), np.int32), np.empty(len(subset))
            _bmu_kernel(w, subset[:1], ri[:1], rd[:1])
            start = time.perf_counter()
            _bmu_kernel(w, subset, ri, rd)
            cpu_seconds = time.perf_counter() - start
            search = GpuSearcher(w, 91, 64)
            idx, d2 = np.empty_like(ri), np.empty_like(rd)
            search(subset, idx, d2)
            passed = np.array_equal(idx, ri) and bits(d2, rd)
            record['checks'].append(dict(lattice=name, bitwise=passed, cpu_seconds=cpu_seconds,
                                          fallbacks=0))
            save()
            if not passed:
                raise AssertionError(f'{name}: CUDA differs from CPU')
        for features in [4, 7, 10]:
            # F=4 uses a projection; F=10 appends the first three GMM columns.
            data = np.ascontiguousarray(x[:args.rows, :4]) if features == 4 else x[:args.rows]
            if features == 10:
                data = np.concatenate([data, data[:, :3]], axis=1)
            for xdim, ydim in [(91, 64), (160, 100)]:
                w = data[np.random.default_rng(42).choice(len(data), size=xdim * ydim,
                                                         replace=False)].astype(np.float64)
                for overlap in [False, True]:
                    search = GpuSearcher(w, xdim, ydim, overlap=overlap)
                    idx, d2 = np.empty(len(data), np.int32), np.empty(len(data))
                    warm = min(len(data), 2 * search.chunk)
                    search(data[:warm], idx[:warm], d2[:warm])
                    start = time.perf_counter()
                    search(data, idx, d2)
                    seconds = time.perf_counter() - start
                    record['rates'].append(dict(features=features, lattice=[xdim, ydim],
                        overlap=overlap, rows=len(data), seconds=seconds,
                        mrows_per_second=len(data) / seconds / 1e6, fallbacks=0))
                    save()
                del search, idx, d2
            del data
        # Measure the API's actual pageable and pinned H2D completion times.
        size = 1 << 30
        owner, pinned = _pinned(cp, (size,), np.uint8)
        pinned[:] = 1
        pageable = np.ones(size, np.uint8)
        device = cp.empty(size, np.uint8)
        stream = cp.cuda.Stream(non_blocking=True)
        record['h2d'] = []
        for kind, host in [('pageable', pageable), ('pinned', pinned)]:
            device.set(host, stream=stream)
            stream.synchronize()
            start = time.perf_counter()
            for _ in range(3):
                device.set(host, stream=stream)
            stream.synchronize()
            seconds = (time.perf_counter() - start) / 3
            record['h2d'].append(dict(kind=kind, bytes=size, seconds=seconds,
                                      gb_per_second=size / seconds / 1e9))
        # Record trainer timing keys and a real multi-epoch CUDA/CPU comparison.
        opts = dict(xdim=48, ydim=34, epochs=5, chunk=2**18, final_pass='none',
                    prefetch=False, n_threads=threads)
        cpu, _ = train_batch_stream(x[:200_000], bmu='numba', **opts)
        gpu, stats = train_batch_stream(x[:200_000], bmu='cuda', **opts)
        record['epochs'] = {k: v for k, v in stats.items() if k in
                            ('epoch_time', 'bmu_time', 'acc_time', 'update_time', 'rows_per_epoch', 'fallbacks')}
        record['training_bitwise'] = bits(cpu, gpu)
        assert record['training_bitwise'], 'training weights differ'
        record['status'] = 'passed'
    except BaseException as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
