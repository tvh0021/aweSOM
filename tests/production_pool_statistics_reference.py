"""Frozen statistics from 0a006be for differential checks."""
import numpy as np

def grouped_values(keys, values, n_groups, out, chunk):
    """Counting scatter into one disk-backed float64 buffer; O(chunk+groups) RAM."""
    counts = np.zeros(n_groups, np.int64)
    for a in range(0,len(keys),chunk):
        counts += np.bincount(keys[a:a+chunk], minlength=n_groups)
    edges = np.r_[0, np.cumsum(counts)]
    positions = edges[:-1].copy()
    buf = np.lib.format.open_memmap(out, mode='w+', dtype='f8', shape=(len(keys),))
    for a in range(0,len(keys),chunk):
        scatter_values(np.asarray(keys[a:a+chunk]), np.asarray(values[a:a+chunk], dtype='f8'), positions, buf)
    return buf, edges


from numba import njit

@njit(cache=False)
def scatter_values(keys, values, positions, buf):
    for i in range(len(keys)):
        k = keys[i]
        buf[positions[k]] = values[i]
        positions[k] += 1


def export_statistics(raw, nodes, distances, labels, n_nodes, out, tag, chunk):
    scratch = out / f'{tag}-statistics.npy'
    buf, edges = grouped_values(nodes, distances, n_nodes, scratch, chunk)
    q = np.full(n_nodes, np.nan)
    for i in range(n_nodes):
        if edges[i+1] > edges[i]:
            q[i] = np.percentile(buf[edges[i]:edges[i+1]], 99., overwrite_input=True)
    reject = float(np.percentile(buf, 99.9, overwrite_input=True))
    del buf
    medians = {int(c): [] for a in range(0,len(labels),chunk) for c in np.unique(labels[a:a+chunk])}
    class Column:
        def __init__(self, j): self.j = j
        def __getitem__(self, key): return raw[key,self.j]
    for j in range(raw.shape[1]):
        buf, edges = grouped_values(labels, Column(j), int(max(medians))+1, scratch, chunk)
        for c in medians:
            medians[c].append(float(np.median(buf[edges[c]:edges[c+1]], overwrite_input=True)))
        del buf
    scratch.unlink()
    return q, reject, medians


class PhaseClock:
    """Emit completed stages immediately, even if a later stage never finishes."""
    def __init__(self, tag):
        self.tag, self.records = tag, {}
        self.wall, self.cpu = time.perf_counter(), time.process_time()

    def finish(self, stage):
        wall, cpu = time.perf_counter(), time.process_time()
        self.records[stage] = dict(wall_seconds=wall-self.wall, cpu_seconds=cpu-self.cpu)
        print(json.dumps(dict(tag=self.tag, stage=stage, **self.records[stage])), flush=True)
        self.wall, self.cpu = wall, cpu


def thread_budget(args):
    """Split the allocation without exceeding Numba's startup thread ceiling."""
    import numba
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', available))
    workers = min(args.seed_workers, len(args.seeds))
    threads = args.threads_per_seed
    if threads is None:
        threads = min(allocated // max(1, workers), numba.config.NUMBA_NUM_THREADS)
    if args.seed_workers < 1 or workers < 1 or threads < 1 or workers*threads > allocated or threads > numba.config.NUMBA_NUM_THREADS:
        raise ValueError('invalid seed thread budget or NUMBA_NUM_THREADS ceiling')
    return dict(seed_workers=workers, threads_per_seed=threads, allocated_cpus=allocated)


def execution_info():
    import numba
    return dict(numba_threads=numba.get_num_threads(), numba_thread_ceiling=numba.config.NUMBA_NUM_THREADS,
                pid=os.getpid(), hostname=os.uname().nodename,
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                thread_environment={k:os.environ.get(k) for k in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                    'SLURM_CPUS_PER_TASK', 'SLURM_JOB_ID')})

