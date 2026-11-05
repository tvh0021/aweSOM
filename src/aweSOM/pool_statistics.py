"""Exact external-sort summaries with bounded RAM and private scratch files."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
import numpy as np
from numba import njit

DTYPE = np.dtype([('value', 'f8'), ('label', 'i8')])


@contextmanager
def scratch_directory(scratch=None, fallback=None):
    root = scratch or os.environ.get('SLURM_TMPDIR') or os.environ.get('TMPDIR') or fallback
    with tempfile.TemporaryDirectory(prefix='awesom-stat-', dir=root) as directory:
        yield Path(directory)


@njit(cache=True)
def _merge(a, b, out):
    i = j = k = 0
    while i < len(a) and j < len(b) and k < len(out):
        if a[i]['value'] <= b[j]['value']:
            out[k] = a[i]; i += 1
        else:
            out[k] = b[j]; j += 1
        k += 1
    return i, j, k


def sorted_blocks(blocks, directory, chunk):
    """Sort runs, then merge pairs using three fixed-size buffers."""
    paths = []
    for i, (values, labels) in enumerate(blocks):
        data = np.empty(len(values), DTYPE)
        data['value'], data['label'] = values, labels
        data.sort(order='value', kind='quicksort')
        path = directory / f'run-0-{i}'
        data.tofile(path); paths.append(path)
    generation = 0
    while len(paths) > 1:
        generation += 1; merged = []
        for i in range(0, len(paths), 2):
            if i+1 == len(paths): merged.append(paths[i]); continue
            path = directory / f'run-{generation}-{i}'
            with paths[i].open('rb') as fa, paths[i+1].open('rb') as fb, path.open('xb') as dst:
                a = np.fromfile(fa, DTYPE, chunk); b = np.fromfile(fb, DTYPE, chunk)
                buffer = np.empty(chunk, DTYPE)
                while len(a) and len(b):
                    na, nb, count = _merge(a, b, buffer)
                    buffer[:count].tofile(dst)
                    a, b = a[na:], b[nb:]
                    if not len(a): a = np.fromfile(fa, DTYPE, chunk)
                    if not len(b): b = np.fromfile(fb, DTYPE, chunk)
                for rest, source in ((a, fa), (b, fb)):
                    while len(rest):
                        rest.tofile(dst); rest = np.fromfile(source, DTYPE, chunk)
            paths[i].unlink(); paths[i+1].unlink(); merged.append(path)
        paths = merged
    if not paths: raise ValueError('empty statistics input')
    return paths[0]


@njit(cache=True)
def _scan(block, seen, targets, selected, sums, ties, state):
    for row in block:
        value, label = row['value'], row['label']
        rank = state[0]
        if rank and value != state[2]:
            sums += ties * (state[1] + rank + 1)
            ties[:] = 0; state[1] = rank
        ties[label] += 1
        state[2] = value
        for group in (label, len(seen)-1):
            pos = seen[group]
            for q in range(targets.shape[1]):
                for side in range(2):
                    if pos == targets[group, q, side]: selected[group, q, side] = value
            seen[group] += 1
        state[0] += 1


def summarize(blocks, counts, probabilities, directory, chunk):
    """Return per-label and pooled percentiles, medians and doubled rank sums."""
    counts = np.asarray(counts, dtype='i8')
    sizes = np.r_[counts, counts.sum()]
    positions = np.maximum(sizes[:, None]-1, 0) * np.asarray(probabilities)[None, :]
    targets = np.stack([np.floor(positions), np.ceil(positions)], axis=-1).astype('i8')
    selected = np.full(targets.shape, np.nan)
    sums = np.zeros(len(counts)); ties = np.zeros(len(counts)); state = np.zeros(3)
    seen = np.zeros(len(sizes), dtype='i8')
    path = sorted_blocks(blocks, directory, chunk)
    try:
        with path.open('rb') as source:
            while True:
                data = np.fromfile(source, DTYPE, chunk)
                if not len(data): break
                _scan(data, seen, targets, selected, sums, ties, state)
        sums += ties * (state[1] + state[0] + 1)
    finally: path.unlink()
    lo, hi = selected[..., 0], selected[..., 1]
    fraction = positions - targets[..., 0]
    result = np.where(fraction >= .5, hi-(hi-lo)*(1-fraction), lo+(hi-lo)*fraction)
    # np.median adds its two middle values before dividing.
    median = (lo + hi) / 2
    result[sizes == 0] = np.nan
    return result, median, sums


def release_mapping(array):
    """Drop mapped pages after a sequential read or committed write on Linux."""
    import mmap
    mapping = getattr(array, '_mmap', None)
    if mapping is not None and hasattr(mapping, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
        array.flush()
        mapping.madvise(mmap.MADV_DONTNEED)


def read_slice(array, start, stop):
    result = np.array(array[start:stop])
    release_mapping(array)
    return result
