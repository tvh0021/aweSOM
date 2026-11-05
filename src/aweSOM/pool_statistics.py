"""Exact external-sort summaries with bounded RAM and private scratch files."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
import numpy as np
from numba import njit

DTYPE = np.dtype([('value', 'f8'), ('label', 'i8')])
MERGE_BUFFER_RECORDS = 2**26


def scratch_root(scratch=None, fallback=None):
    """Resolve the same scratch root for capacity checks and temporary files."""
    return Path(scratch or os.environ.get('SLURM_TMPDIR') or os.environ.get('TMPDIR')
                or fallback or tempfile.gettempdir()).resolve()


@contextmanager
def scratch_directory(scratch=None, fallback=None):
    root = scratch_root(scratch,fallback)
    with tempfile.TemporaryDirectory(prefix='awesom-stat-', dir=root) as directory:
        yield Path(directory)


@njit(cache=True)
def _sift(heap, size, values, positions):
    parent = 0
    while 2*parent+1 < size:
        child = 2*parent+1
        if child+1 < size:
            left, right = heap[child], heap[child+1]
            lv, rv = values[left,positions[left]], values[right,positions[right]]
            if rv < lv or (rv == lv and right < left): child += 1
        top, bottom = heap[parent], heap[child]
        tv, bv = values[top,positions[top]], values[bottom,positions[bottom]]
        if tv < bv or (tv == bv and top < bottom): break
        heap[parent], heap[child] = bottom, top
        parent = child


@njit(cache=True)
def _drain(values, labels, positions, lengths, heap, size, output):
    count = 0
    while size and count < len(output):
        run = heap[0]; pos = positions[run]
        output[count]['value'] = values[run,pos]
        output[count]['label'] = labels[run,pos]
        count += 1; positions[run] += 1
        if positions[run] == lengths[run]: return count, run
        _sift(heap,size,values,positions)
    return count, -1


def sorted_blocks(blocks, directory, chunk):
    """Write sorted runs once and consume them in one buffered k-way merge.

    One file descriptor serves every run. Merge buffers hold at most 2**26
    records with at most 2**15 records buffered per run. Refuse more than
    2**18 runs to retain a fixed upper bound on buffers and metadata.
    """
    path = directory/'runs.bin'; offsets = []; lengths = []; packed = None
    with path.open('xb') as dst:
        for values, labels in blocks:
            values, labels = np.asarray(values), np.asarray(labels)
            is_packed = values.dtype == np.float32
            if packed is not None and packed != is_packed: raise ValueError('mixed run dtypes')
            packed = is_packed
            if not len(values): continue
            if len(offsets) == 2**18: raise ValueError('too many statistics runs; increase chunk')
            offsets.append(dst.tell()); lengths.append(len(values))
            if packed:
                if np.any(labels < 0) or np.any(labels > 0xFFFFFFFF):
                    raise ValueError('packed labels must fit uint32')
                bits = (values + np.float32(0)).view(np.uint32)
                keys = np.where(bits >> np.uint32(31), ~bits,
                                bits | np.uint32(0x80000000)).astype(np.uint64)
                keys <<= np.uint64(32); keys |= labels.astype(np.uint64)
                keys.sort(); keys.tofile(dst)
            else:
                idx = np.argsort(values,kind='stable')
                records = np.empty(len(values),DTYPE)
                records['value'],records['label'] = values[idx], labels[idx]
                records.tofile(dst)
    if not lengths: raise ValueError('empty statistics input')
    runs = len(lengths); width = min(chunk,2**15,max(1,MERGE_BUFFER_RECORDS//runs))
    values = np.empty((runs,width),dtype='f8'); labels = np.empty((runs,width),dtype='i8')
    positions = np.zeros(runs,dtype='i8'); available = np.zeros(runs,dtype='i8')
    remaining = np.array(lengths,dtype='i8'); offsets = np.array(offsets,dtype='i8')
    output = np.empty(min(chunk,2**18),DTYPE)
    dtype = np.dtype('u8') if packed else DTYPE
    try:
        with path.open('rb') as source:
            def refill(run):
                count = min(width,int(remaining[run]))
                source.seek(int(offsets[run]))
                data = np.fromfile(source,dtype,count)
                if len(data) != count: raise IOError('truncated statistics run')
                offsets[run] += count*dtype.itemsize; remaining[run] -= count
                if packed:
                    bits = (data >> np.uint64(32)).astype(np.uint32)
                    bits = np.where(bits >> np.uint32(31), bits & np.uint32(0x7FFFFFFF), ~bits)
                    values[run,:count] = bits.view(np.float32)
                    labels[run,:count] = (data & np.uint64(0xFFFFFFFF)).astype('i8')
                else:
                    values[run,:count],labels[run,:count] = data['value'],data['label']
                positions[run] = 0; available[run] = count
                return count
            for run in range(runs): refill(run)
            heap = np.array(sorted(range(runs),key=lambda r:(values[r,0],r)),dtype='i8')
            size = runs
            while size:
                count, empty = _drain(values,labels,positions,available,heap,size,output)
                yield output[:count]
                if empty >= 0:
                    if not refill(empty):
                        size -= 1; heap[0] = heap[size]
                    if size: _sift(heap,size,values,positions)
    finally:
        path.unlink()


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
    for data in sorted_blocks(blocks,directory,chunk):
        _scan(data, seen, targets, selected, sums, ties, state)
    sums += ties * (state[1] + state[0] + 1)
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


def feature_summary(job):
    """Read one HDF5 feature in a spawned, single-threaded statistics worker."""
    filename, dataset, label_path, column, counts, chunk, scratch = job
    import h5py
    import numba
    numba.set_num_threads(1)
    labels = np.load(label_path,mmap_mode='r')
    with h5py.File(filename,'r') as source, scratch_directory(scratch) as directory:
        raw = source[dataset]
        blocks = ((raw[a:a+chunk,column],read_slice(labels,a,a+chunk))
                  for a in range(0,len(raw),chunk))
        return summarize(blocks,counts,[.25,.5,.75],directory,chunk)


def parallel_features(raw, labels, counts, chunk, scratch, workers):
    """Map independent features within the enclosing fit's CPU allocation."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    jobs = [(raw.file.filename,raw.name,str(labels.filename),j,counts,chunk,str(scratch))
            for j in range(raw.shape[1])]
    previous = os.environ.get('NUMBA_NUM_THREADS')
    os.environ['NUMBA_NUM_THREADS'] = '1'
    try:
        with ProcessPoolExecutor(max_workers=min(workers,raw.shape[1]),
                                 mp_context=mp.get_context('spawn')) as executor:
            return list(executor.map(feature_summary,jobs))
    finally:
        if previous is None: os.environ.pop('NUMBA_NUM_THREADS',None)
        else: os.environ['NUMBA_NUM_THREADS'] = previous
