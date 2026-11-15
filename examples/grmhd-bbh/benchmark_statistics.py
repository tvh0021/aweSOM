"""Local statistics I/O benchmark; writes only under the supplied directory.

Run with the repository's src on PYTHONPATH. The reference is frozen in tests.
"""
import argparse
import importlib.util
import json
import resource
from pathlib import Path
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--rows', type=int, default=3_000_000)
    p.add_argument('--uncached', action='store_true', help='macOS F_NOCACHE for the HDF5 input')
    a = p.parse_args(); a.directory.mkdir(parents=True, exist_ok=True)
    old = load('reference', ROOT/'tests/production_pool_statistics_reference.py')
    new = load('production', Path(__file__).with_name('production_pool.py'))
    rng = np.random.default_rng(8014); chunk = 131072
    path = a.directory/'pool.h5'
    with h5py.File(path, 'w') as h:
        raw = h.create_dataset('raw', (a.rows, 7), dtype='f4')
        for start in range(0, a.rows, chunk):
            raw[start:start+chunk] = rng.normal(size=(min(chunk,a.rows-start), 7)).astype('f4')
    arrays = {}
    for name, dtype in [('nodes','i4'), ('labels','i4'), ('distances','f8')]:
        arrays[name] = np.lib.format.open_memmap(a.directory/f'{name}.npy', mode='w+', dtype=dtype, shape=(a.rows,))
    arrays['nodes'][:] = rng.integers(0, 10000, a.rows, dtype='i4')
    arrays['labels'][:] = arrays['nodes'] % 10
    arrays['distances'][:] = rng.random(a.rows)
    for x in arrays.values(): x.flush()
    old.scatter_values(np.array([0], dtype='i4'), np.array([1.]), np.array([0]), np.zeros(1))
    report = dict(rows=a.rows, features=7, chunk=chunk, cache='F_NOCACHE HDF5 input' if a.uncached else 'OS cache uncontrolled')
    with h5py.File(path) as h:
        if a.uncached:
            import fcntl
            fcntl.fcntl(h.id.get_vfd_handle(), 48, 1)
        report['hdf5'] = dict(chunks=h['raw'].chunks, compression=h['raw'].compression, dtype=str(h['raw'].dtype))
        outputs = []
        for run, variant in enumerate(['old', 'ram-buffer', 'new', 'new', 'ram-buffer', 'old']):
            module = new if variant == 'new' else old
            parts = {}
            def timed(name, fn, *args, **kwargs):
                wall, cpu = time.perf_counter(), time.process_time()
                result = fn(*args, **kwargs)
                pair = parts.setdefault(name, [0.,0.]); pair[0] += time.perf_counter()-wall; pair[1] += time.process_time()-cpu
                return result
            class Raw:
                shape = h['raw'].shape
                dtype = h['raw'].dtype
                def __len__(self): return a.rows
                def __getitem__(self, key): return timed('raw_read', h['raw'].__getitem__, key)
            original = module.grouped_values if module is old else None
            if original:
                module.grouped_values = lambda *args, **kwargs: timed('grouped_including_reads', original, *args, **kwargs)
            open_memmap = np.lib.format.open_memmap
            if variant == 'ram-buffer':
                # Change only scratch placement; keep column reads and grouping.
                (a.directory/'bench-statistics.npy').touch()
                np.lib.format.open_memmap = lambda path, mode, dtype, shape: np.empty(shape, dtype=dtype)
            percentile, median = np.percentile, np.median
            np.percentile = lambda *args, **kwargs: timed('percentile', percentile, *args, **kwargs)
            np.median = lambda *args, **kwargs: timed('median', median, *args, **kwargs)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            wall, cpu = time.perf_counter(), time.process_time()
            result = module.export_statistics(Raw(), arrays['nodes'], arrays['distances'], arrays['labels'], 10001, a.directory, 'bench', chunk)
            record = dict(implementation=variant, wall=time.perf_counter()-wall, cpu=time.process_time()-cpu, parts=parts)
            np.lib.format.open_memmap = open_memmap
            np.percentile, np.median = percentile, median
            end_usage = resource.getrusage(resource.RUSAGE_SELF)
            record['block_input_operations'] = end_usage.ru_inblock - usage.ru_inblock
            record['block_output_operations'] = end_usage.ru_oublock - usage.ru_oublock
            report[f'run{run}'] = record; print(json.dumps(record), flush=True)
            if original: module.grouped_values = original
            outputs.append(result)
        for result in outputs[1:]:
            assert outputs[0][0].tobytes() == result[0].tobytes()
            assert np.float64(outputs[0][1]).tobytes() == np.float64(result[1]).tobytes()
            assert outputs[0][2] == result[2]
    report['bitwise_equal'] = True
    (a.directory/'timings.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
