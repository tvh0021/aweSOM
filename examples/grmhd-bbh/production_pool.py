"""Train and verify frozen SOMs from equal, level-stratified snapshot quotas.

Inputs are read only. Outputs require a fresh directory. See --help for overrides.
The orchestration runs each fit in a fresh worker, as in cluster/kit/check7.py.

Sheet-candidate selection (user-approved consensus rule, synthesis opus.md D1-D15):
each fit records its candidates, clusters whose one-vs-rest rank effects on the
training rows satisfy E(log_chi_B) > 0, E(log_grad_ln_b2) > 0 and E(log_beta_inv) < 0.
After every seed of a lattice is fitted, main() selects once for the group with the
top-two rule (user-approved 2026-11-06): each seed's first two candidates (largest
E(log_chi_B)) are eligible; the first pick is the tuple, one eligible candidate per
seed, that is mutual best by Jaccard for every seed pair and has the largest minimum
E(log_chi_B); the unused eligible candidates join as a second pick only if they are
mutual best too. The audit also records what the previous first-candidates-only rule
gives. The IDs are frozen in each model as selected_sheet_clusters (alias
sheet_clusters for older readers). A "sheet" proper needs the offline
spatial check, which this driver does not run.
"""
from __future__ import annotations

import copy
import functools
import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
# CPU set at start, saved before anything can initialise MPI (`python -m mpi4py` imports MPI
# only on an exception; MPI_Init happens in get_comm). Linux only. See restore_affinity.
AFFINITY_AT_START = frozenset(os.sched_getaffinity(0)) if sys.platform.startswith('linux') else None
# Bound native libraries before NumPy is imported, including in spawned workers.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import h5py
import numpy as np

NAMES = ['log_beta_inv', 'log_theta', 'log_chi_B', 'log_grad_ln_b2',
         'asinh_div_v', 'bz_frac', 'pol_frac']
UNAVAILABLE = {'status': 'not available', 'reason': 'not recorded by the extraction'}


def node_name():
    """This node's name from Slurm (SLURMD_NODENAME is set in every task, per node), or 'local'
    outside Slurm. No system call: no host-state reads on RIT RC (user rule 2026-11-06, extended 2026-11-12)."""
    return os.environ.get('SLURMD_NODENAME') or 'local'


def on_tigris(*paths):
    """True on RIT RC (Tigris or SPORC): SLURM_CLUSTER_NAME is tigris or sporc (any case)
    or a path lies under /shared/rc/ or /home/thccrg/. Free-space checks are skipped there
    (no filesystem reads, user rule 2026-11-06, extended to SPORC 2026-11-12)."""
    if os.environ.get('SLURM_CLUSTER_NAME', '').lower() in {'tigris', 'sporc'}: return True
    return any(m in (os.path.abspath(p) + '/') for p in paths for m in ('/shared/rc/', '/home/thccrg/'))


def plain(x):
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if isinstance(x, np.ndarray): return plain(x.tolist())
    if isinstance(x, np.generic): return plain(x.item())
    if isinstance(x, bytes): return x.decode()
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def dump(path, data):
    tmp = Path(str(path) + '.partial')
    with tmp.open('x') as f:
        json.dump(plain(data), f, indent=2, allow_nan=False)
    publish(tmp, path)


def publish(tmp, path):
    os.link(tmp, path)  # atomic, refuses existing target
    Path(tmp).unlink()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def commit(root):
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def provenance():
    """Verify the deployed driver and imported package before creating run outputs."""
    spec = importlib.util.find_spec('aweSOM')
    if spec is None or spec.origin is None:
        raise ValueError('cannot locate imported aweSOM source')
    source = Path(spec.origin).resolve().parent
    manifest_path = Path(__file__).with_name('production_pool-source.json')
    files = {str(p.relative_to(source)): digest(p) for p in source.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.nbc', '.nbi'}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not re.fullmatch(r'[0-9a-f]{40}', manifest['commit']):
            raise ValueError('invalid source manifest commit')
        if files != manifest['files'] or digest(Path(__file__)) != manifest['driver_sha256']:
            raise ValueError('source manifest hash mismatch: aweSOM package or production_pool.py')
        return manifest['commit']
    # Only the imported package's own checkout can supply fallback provenance.
    root = source.parents[1]
    if not (root / '.git').exists() or Path(__file__).resolve() != root/'examples/grmhd-bbh/production_pool.py':
        raise ValueError('production_pool-source.json required outside the aweSOM git checkout')
    return commit(root)


def randomized_allocate(counts, total, rng):
    """Dependent rounding: fixed total, floor/ceil counts, exact marginal means."""
    c = np.asarray(counts, dtype=np.int64)
    if total < np.count_nonzero(c) or total > c.sum():
        raise ValueError('epoch budget smaller than populated strata or exceeds pool')
    exact = c * (total / int(c.sum()))
    q = np.floor(exact).astype(np.int64)
    f = exact-q
    # Pairwise pivotal rounding preserves each fractional seat's expectation.
    active = list(np.flatnonzero((f > 0) & (f < 1)))
    while len(active) >= 2:
        i, j = active.pop(), active.pop()
        up, down = min(1-f[i], f[j]), min(f[i], 1-f[j])
        if rng.random() < down/(up+down):
            f[i] += up; f[j] -= up
        else:
            f[i] -= down; f[j] += down
        for k in (i,j):
            if f[k] < 1e-12: f[k] = 0
            elif f[k] > 1-1e-12: f[k] = 1
            else: active.append(k)
    if active: f[active[0]] = total-int(q.sum())-int(np.rint(f).sum()-np.rint(f[active[0]]))
    q += np.rint(f).astype(np.int64)
    assert q.sum() == total and np.all(q <= c)
    return q


def validated(path):
    status = path.parent / 'status'
    attempts = sorted(status.iterdir(), key=lambda p: p.stat().st_mtime_ns) if status.exists() else []
    latest = attempts[-1] / 'final.tsv' if attempts else None
    if latest is None or not latest.is_file() or latest.read_text().split('\t')[:2] != ['VALIDATED', '0']:
        raise ValueError(f'Bad status: latest extraction must be VALIDATED/0: {path}')
    return str(latest)


def allocate(counts, total):
    """Largest-remainder quotas, retaining the extractor's selected distribution."""
    c = np.asarray(counts, dtype=np.int64)
    if total > c.sum() or total < 0: raise ValueError('quota exceeds available rows')
    exact = c * (total / int(c.sum()))
    q = np.floor(exact).astype(np.int64)
    order = np.argsort(-(exact-q), kind='stable')
    q[order[:total-int(q.sum())]] += 1
    if np.any((c > 0) & (q == 0)):
        raise ValueError('pool thinning would erase a populated level')
    return q


def inspect(args):
    records = []
    for path in args.inputs:
        if not path.is_file(): raise ValueError(f'Missing input: {path}')
        marker = validated(path)
        with h5py.File(path, 'r') as h:
            attrs = plain(dict(h.attrs))
            names = plain(h['names'][:])
            expected = NAMES[:6] if args.allow_six_features else NAMES
            if names != expected: raise ValueError(f'feature names differ: {path}: {names}')
            if attrs.get('pipeline') != 'v2': raise ValueError(f'pipeline differs: {path}')
            for key in ['b2_min', 's_v']:
                if not np.isfinite(attrs.get(key, np.nan)) or attrs[key] <= 0:
                    raise ValueError(f'missing or invalid {key}: {path}')
            n = len(h['features'])
            if not n or h['features'].shape != (n, len(names)): raise ValueError('bad feature shape')
            for key in ['level', 'weight', 'weight_volume']:
                if h[key].shape != (n,): raise ValueError(f'bad {key} shape')
            counts = {}
            for a in range(0, n, args.chunk):
                lv, ct = np.unique(h['level'][a:a+args.chunk], return_counts=True)
                for l, c in zip(lv, ct): counts[int(l)] = counts.get(int(l), 0) + int(c)
            extraction = {k: plain(v[:]) for k,v in h['level_counts'].items()}
            if 'level' not in extraction:
                raise ValueError('extraction level_counts must identify levels')
            levels = extraction['level']
            if len(set(levels)) != len(levels) or set(counts)-set(levels):
                raise ValueError('observed levels differ from extraction level_counts')
            if any(len(v) != len(levels) for v in extraction.values()):
                raise ValueError('bad extraction level_counts shape')
            for i, level in enumerate(levels):
                observed = counts.get(level, 0)
                if 'kept' in extraction and observed != extraction['kept'][i]:
                    raise ValueError(f'level {level}: observed count differs from extraction kept')
                if extraction.get('owned', [0]*len(levels))[i] > 0 and observed == 0:
                    if 'kept' not in extraction or extraction['kept'][i] != 0:
                        raise ValueError(f'populated extraction level {level} absent without explanation')
            records.append(dict(path=str(path.resolve()), snapshot=str(attrs['iteration']),
                                attrs=attrs, names=names, rows=n, counts=counts, status=marker,
                                extraction_counts=extraction, observed_level_counts={l: counts.get(l,0) for l in levels},
                                sha256=digest(path)))
    ids = [r['snapshot'] for r in records]
    if len(set(ids)) != len(ids): raise ValueError('duplicate snapshot IDs')
    if set(ids) & set(args.held_out): raise ValueError('held-out snapshot appears in training inputs')
    if len(records) < 2: raise ValueError('pool needs at least two snapshots')
    anchor = next((r for r in records if r['snapshot'] == args.anchor), None)
    if anchor is None: raise ValueError('anchor must be a training input')
    mismatches = []
    # Matching names/pipeline are mandatory even in the legacy diagnostic override.
    for r in records:
        for key in ['b2_min', 's_v', 'accuracy', 'epsilon', 'exponent', 'gamma_K',
                    'C_B_box', 'direction_norm', 'horizon_mask', 'zslab', 'hole_boxes', 'winsor_quantiles']:
            if r['attrs'].get(key) != anchor['attrs'].get(key):
                mismatches.append([r['snapshot'], key, r['attrs'].get(key), anchor['attrs'].get(key)])
        origin = r['attrs'].get('thresholds_from')
        if origin and Path(origin).name != Path(anchor['path']).name:
            raise ValueError(f'thresholds_from names a different anchor: {origin}')
        # Extractor index names include the segment, which changes across time.
        def run_identity(record):
            return re.sub(r'_\d{4}_som\.json$', '_som.json', str(record['attrs'].get('source')))
        if run_identity(r) != run_identity(anchor):
            raise ValueError('inputs must name one simulation source')
    if mismatches and not args.allow_independent_thresholds:
        raise ValueError(f'mismatched thresholds/settings: {mismatches}')
    quota = min(r['rows'] for r in records) if args.rows_per_snapshot is None else args.rows_per_snapshot
    if quota <= 0 or any(quota > r['rows'] for r in records): raise ValueError('invalid snapshot quota')
    rows = quota * len(records)
    epoch_rows = args.rows_per_epoch or rows
    if epoch_rows < len(records) or epoch_rows > rows or epoch_rows % len(records):
        raise ValueError('rows per epoch must be <= pool and divisible by snapshot count')
    # Raw float32 + scaled float64 + three label vectors, distance scratch and margin.
    bytes_per_row = len(records[0]['names'])*(4+8*args.write_scaled)
    deferred = args.statistics == 'later'
    fits = len(args.lattices)*len(args.seeds)
    bytes_per_row += 4*fits + 12*(fits if deferred else args.seed_workers)
    if not (args.scratch or os.environ.get('SLURM_TMPDIR') or os.environ.get('TMPDIR')):
        threads = args.threads_per_seed or thread_budget(args)['threads_per_seed']
        bytes_per_row += 8 if deferred else max(16,8*min(threads,len(records[0]['names'])))*args.seed_workers
    need = max(args.min_free_gb*1e9, rows*bytes_per_row + 1e9)
    if on_tigris(args.out):  # no free-space read on RIT RC; nothing recorded
        return records, quota, dict(required_bytes=need), mismatches
    free = shutil.disk_usage(args.out.parent).free  # sysinfo-ok: GP-only branch, skipped on RIT RC
    if free < need: raise ValueError(f'Low disk: {free} bytes free; need {need}')
    return records, quota, dict(free_bytes=free, required_bytes=need), mismatches


def quantiles(dataset, start=0, stop=None, scratch=None):
    """Exact quartiles using one scratch-backed float64 column."""
    from aweSOM.pool_statistics import scratch_directory
    stop = len(dataset) if stop is None else stop
    result = []
    with scratch_directory(scratch, Path(dataset.file.filename).parent) as directory:
        column = np.lib.format.open_memmap(directory/'scaler-column.npy', mode='w+',
                                          dtype='f8', shape=(stop-start,))
        try:
            for j in range(dataset.shape[1]):
                for a in range(start,stop,2**18):
                    b = min(a+2**18,stop)
                    column[a-start:b-start] = dataset[a:b,j]
                result.append(np.percentile(column,[25,50,75],overwrite_input=True))
        finally:
            column._mmap.close()
    return np.stack(result, axis=1)



def pool(args, records, quota, out):
    strata, order_hash = [], hashlib.sha256()
    with h5py.File(out / 'pool.h5', 'x') as dst:
        dst['names'] = np.array(records[0]['names'], dtype='S')
        raw = dst.create_dataset('raw', (quota*len(records), len(records[0]['names'])), dtype='f4')
        offset = 0
        for rec in records:
            levels = sorted(rec['counts'])
            quotas = allocate([rec['counts'][l] for l in levels], quota)
            rec['pool_quotas'] = dict(zip(levels, map(int, quotas)))
            with h5py.File(rec['path'], 'r') as src:
                for level, q in zip(levels, quotas):
                    start, remaining, wanted = offset, rec['counts'][level], int(q)
                    rng = np.random.default_rng([args.sample_seed, int(rec['snapshot']), level])
                    wh = hashlib.sha256()
                    weight_summary = {}
                    for a in range(0, rec['rows'], args.chunk):
                        ix = np.flatnonzero(src['level'][a:a+args.chunk] == level)
                        take = int(rng.hypergeometric(len(ix), remaining-len(ix), wanted)) if remaining else 0
                        remaining -= len(ix); wanted -= take
                        chosen = np.sort(rng.choice(ix, take, replace=False))
                        if take:
                            block = src['features'][a:a+args.chunk][chosen]
                            if not np.isfinite(block).all(): raise ValueError('non-finite pool features')
                            raw[offset:offset+take] = block
                            order_hash.update(np.asarray(chosen+a, dtype='<i8').tobytes())
                            for key in ['weight', 'weight_volume']:
                                values = src[key][a:a+args.chunk][chosen]
                                wh.update(values.tobytes())
                                entry = weight_summary.setdefault(key, dict(sum=0., min=float(values.min()), max=float(values.max())))
                                entry['sum'] += float(values.sum(dtype=np.float64))
                                entry['min'] = min(entry['min'], float(values.min()))
                                entry['max'] = max(entry['max'], float(values.max()))
                            offset += take
                    assert wanted == 0 and offset-start == q
                    if q: strata.append(dict(snapshot=rec['snapshot'], level=level, start=start, rows=int(q)))
                    rec.setdefault('selected_weight_hashes', {})[str(level)] = wh.hexdigest()
                    rec.setdefault('per_level_sampling', {})[str(level)] = dict(
                        extracted_rows=rec['counts'][level], pooled_rows=int(q),
                        pooling_probability=float(q / rec['counts'][level]),
                        extraction_probability=1/weight_summary['weight']['min'] if q else UNAVAILABLE,
                        selected_weights=weight_summary,
                        effective_weight_multiplier=float(rec['counts'][level]/q) if q else UNAVAILABLE)
        q = quantiles(raw, scratch=args.scratch); center, unweighted = q[1], q[2]-q[0]
        from aweSOM.model import Scaler
        feature_weights = np.array([args.bz_weight if n == 'bz_frac' else
                                    args.pol_weight if n == 'pol_frac' else 1
                                    for n in records[0]['names']])
        scaler = Scaler('median-iqr', center, unweighted / feature_weights)
        drift = []
        for i, rec in enumerate(records):
            sq = quantiles(raw, i*quota, (i+1)*quota, scratch=args.scratch)
            # Median drift relative to |pooled median|, with pooled IQR as zero guard.
            md = np.abs(sq[1]-center)/np.maximum(np.abs(center), np.maximum(unweighted, 1e-30)*1e-12)
            iq = np.abs(sq[2]-sq[0]-unweighted)/np.maximum(unweighted, 1e-30)
            drift.append(dict(snapshot=rec['snapshot'], median_relative=md, iqr_relative=iq,
                              flagged_features=np.array(records[0]['names'])[np.maximum(md,iq)>.05].tolist(),
                              quantiles=sq))
        if args.write_scaled:
            scaled = dst.create_dataset('scaled', raw.shape, dtype='f8')
            for a in range(0, len(raw), args.chunk): scaled[a:a+args.chunk] = scaler.transform(raw[a:a+args.chunk])
    return dict(center=center, scale=scaler.scale, scale_unweighted=unweighted,
                feature_weights=feature_weights, kind='median-iqr', drift=drift,
                strata=strata, row_order_sha256=order_hash.hexdigest())


class EpochSource:
    """Fresh cyclic windows per physical stratum; exact equal snapshot budgets.

    Call k (counted by ``calls``) draws its windows from ``default_rng([seed, k])``.
    ``lazy_chunks()`` is one call that yields ``(rows, read)`` pairs and reads a window
    only when ``read()`` runs, so ranks of a multi-process run read only the windows
    they own while every rank advances ``calls`` identically. ``rows(ix)`` returns rows
    ``ix`` of call ``calls`` and counts as one call. ``train_batch_stream`` advances
    ``calls`` by one when it skips the counting pass, so the call numbering (and the
    weights) equal those of earlier drivers, which always made that pass.
    ``transform`` (per-chunk scaling) is applied to every block either path returns.
    """
    def __init__(self, data, strata, rows, seed, chunk, *, transform=None):
        self.transform = transform
        self.data, self.strata, self.seed, self.chunk = data, strata, seed, chunk
        self.calls = 0
        snapshots = list(dict.fromkeys(s['snapshot'] for s in strata))
        if rows > len(data) or rows < len(snapshots) or rows % len(snapshots):
            raise ValueError('rows per epoch must be <= pool and divisible by snapshot count')
        self.rows_per_call = rows
        self.groups = [[i for i,s in enumerate(strata) if s['snapshot'] == snap] for snap in snapshots]
        if any(rows//len(snapshots) < len(ix) for ix in self.groups):
            raise ValueError('epoch budget smaller than populated strata per snapshot')
        self.quotas = np.zeros(len(strata), dtype=int)

    @property
    def n_feat(self):
        shape = getattr(self.data, 'shape', ())
        return int(shape[1]) if len(shape) == 2 else None

    def _windows(self, call):
        """Pool windows ``(start, rows)`` of call ``call``, in yield order, and its quotas."""
        rng = np.random.default_rng([self.seed, call])
        quotas = np.zeros(len(self.strata), dtype=int)
        for ix in self.groups:
            quotas[ix] = randomized_allocate([self.strata[i]['rows'] for i in ix],
                                             self.rows_per_call//len(self.groups), rng)
        windows = []
        for s, q in zip(self.strata, quotas):
            pos = int(rng.integers(s['rows']))
            while q:
                n = min(int(q), self.chunk, s['rows']-pos)
                windows.append((s['start']+pos, n))
                q -= n; pos = (pos+n) % s['rows']
        return windows, quotas

    def _read(self, start, n):
        block = self.data[start:start+n]
        return self.transform(block) if self.transform is not None else block

    def lazy_chunks(self):
        windows, self.quotas = self._windows(self.calls); self.calls += 1
        for start, n in windows:
            yield n, functools.partial(self._read, start, n)

    def __call__(self):
        for _, read in self.lazy_chunks():
            yield read()

    def rows(self, ix):
        windows, _ = self._windows(self.calls); self.calls += 1
        ix = np.asarray(ix, dtype=np.int64)
        if len(ix) and (ix.min() < 0 or ix.max() >= self.rows_per_call):
            raise IndexError('row index outside the epoch')
        starts = np.array([a for a, _ in windows], dtype=np.int64)
        offsets = np.r_[0, np.cumsum([n for _, n in windows])].astype(np.int64)
        k = np.searchsorted(offsets, ix, side='right') - 1
        uniq, inv = np.unique(starts[k] + ix - offsets[k], return_inverse=True)
        got = self.data[uniq] if isinstance(self.data, np.ndarray) else self.data[uniq.tolist()]
        got = np.asarray(got)[inv]
        return self.transform(got) if self.transform is not None else got


if 'NUMBA_NUM_THREADS' not in os.environ and 'SLURM_CPUS_PER_TASK' in os.environ:
    os.environ['NUMBA_NUM_THREADS'] = os.environ['SLURM_CPUS_PER_TASK']
from numba import njit


@njit(cache=False)
def scatter_order(keys, positions, order):
    """Stable counting-sort permutation, with no comparison-sort workspace."""
    for i in range(len(keys)):
        k = keys[i]
        order[positions[k]] = i
        positions[k] += 1


def grouped_order(keys, n_groups):
    edges = np.r_[0, np.cumsum(np.bincount(keys, minlength=n_groups))]
    order = np.empty(len(keys), dtype=np.intp)
    scatter_order(keys, edges[:-1].copy(), order)
    return order, edges


def export_statistics(raw, nodes, distances, labels, n_nodes, out, tag, chunk,
                      names=None, strata=None, n_clusters=None, scratch=None, workers=1):
    """Exact quartiles and tied ranks using bounded external sorts.

    Feature runs hold 8-byte packed float32/label keys; distance runs use
    16-byte records. A single bounded-buffer k-way merge consumes each run.
    File-backed features run concurrently within the fit thread budget.
    Output keys match the in-memory implementation.
    """
    from aweSOM.pool_statistics import scratch_directory, summarize, read_slice
    n, D = raw.shape
    K = int(n_clusters or 0)
    for a in range(0,n,chunk): K = max(K, int(np.max(read_slice(labels,a,a+chunk)))+1)
    counts = np.zeros(K, 'i8'); node_counts = np.zeros(n_nodes, 'i8')
    for a in range(0,n,chunk):
        counts += np.bincount(read_slice(labels,a,a+chunk), minlength=K)
        node_counts += np.bincount(read_slice(nodes,a,a+chunk), minlength=n_nodes)
    snapshot_rows = {}
    for st in strata or ():
        rows = snapshot_rows.setdefault(str(st['snapshot']), np.zeros(K, 'i8'))
        for a in range(st['start'],st['start']+st['rows'],chunk):
            rows += np.bincount(read_slice(labels,a,min(a+chunk,st['start']+st['rows'])), minlength=K)
    effects = np.full((K,D),np.nan); quartiles = np.full((K,3,D),np.nan)
    pooled_medians = np.empty(D)
    medians = {int(c): [] for a in range(0,n,chunk) for c in np.unique(read_slice(labels,a,a+chunk))}
    with scratch_directory(scratch, out) as directory:
        blocks = ((read_slice(distances,a,a+chunk), read_slice(nodes,a,a+chunk)) for a in range(0,n,chunk))
        qs, _, _ = summarize(blocks, node_counts, np.asarray([99.,99.9])/100, directory, chunk)
        q, reject = qs[:-1,0], float(qs[-1,1])
        if workers > 1 and isinstance(raw,h5py.Dataset) and raw.file.mode == 'r' and isinstance(labels,np.memmap):
            from aweSOM.pool_statistics import parallel_features
            summaries = parallel_features(raw,labels,counts,chunk,directory,workers)
        else:
            summaries = []
            for j in range(D):
                blocks = ((raw[a:a+chunk][:,j], read_slice(labels,a,a+chunk)) for a in range(0,n,chunk))
                with scratch_directory(directory) as feature_directory:
                    summaries.append(summarize(blocks,counts,[.25,.5,.75],feature_directory,chunk))
        for j,(qs,mids,sums) in enumerate(summaries):
            quartiles[:,:,j] = qs[:-1]; quartiles[:,1,j] = mids[:-1,1]
            pooled_medians[j] = mids[-1,1]
            effects[:,j] = effects_from_rank_sums(sums,counts,n)
            for c in medians: medians[c].append(float(mids[c,1]))
    directions = dict(SIGNATURE) if names is not None and all(f in names for f,_ in SIGNATURE) else {}
    joint = None
    if directions:
        hits = np.zeros(K)
        for a in range(0,n,chunk):
            values = raw[a:a+chunk]; favourable = np.ones(len(values),bool)
            for feature, sign in directions.items():
                j = names.index(feature)
                favourable &= values[:,j] > pooled_medians[j] if sign > 0 else values[:,j] < pooled_medians[j]
            hits += np.bincount(read_slice(labels,a,a+chunk),weights=favourable,minlength=K)
        joint = hits/np.maximum(counts,1); joint[counts==0] = np.nan
    return q, reject, medians, dict(cluster_rows=dict(all=counts,**snapshot_rows),
        cluster_share=dict(all=counts/n,**{s:r/max(1,int(r.sum())) for s,r in snapshot_rows.items()}),
        cluster_effects=effects,cluster_quartiles=quartiles,pooled_medians=pooled_medians,
        cluster_joint_favorable_fraction=joint)


# Sheet-candidate signature (opus.md D1): the sign of each feature's rank effect, strict.
SIGNATURE = (('log_chi_B', 1), ('log_grad_ln_b2', 1), ('log_beta_inv', -1))
CANDIDATE_RULE = ('candidate: E(log_chi_B) > 0 and E(log_grad_ln_b2) > 0 and E(log_beta_inv) < 0, '
                  'E = 2 AUC - 1 one-vs-rest on the training rows, ordered by E(log_chi_B) descending, '
                  'then cluster ID; ')
SELECTION_RULE = (CANDIDATE_RULE +
                  'eligible: the first two candidates of each seed; first pick: among tuples of one eligible '
                  'candidate per seed that are mutual best Jaccard matches (over all clusters) for every seed '
                  'pair, the one with the largest minimum E(log_chi_B), the first enumerated in candidate order '
                  'on exact ties; second pick: the unused eligible candidate of every seed, if each seed has one '
                  'and they are mutual best for every seed pair; at most two; fewer than two seeds is '
                  'unavailable; no magnitude thresholds')
# The rule approved before 2026-11-06; its outcome is still recorded as approved_rule_outcome.
PREVIOUS_SELECTION_RULE = (CANDIDATE_RULE +
                  'selected sheet cluster: the first candidate of every seed when they are mutual best '
                  'Jaccard matches for every seed pair, plus the second candidates when those are too; '
                  'at most two; fewer than two seeds is unavailable; no magnitude thresholds')
# Not a tuning knob: the existing at-most-two picks fix how many candidates per seed are eligible.
ELIGIBLE_PER_SEED = 2
# The rejected median thresholds, kept only as a diagnostic label for comparison with older runs.
LEGACY_THRESHOLDS = dict(log_chi_B_gt=1., log_beta_inv_lt=.5)


def rank_sums(column, keys, n_groups):
    """Twice the per-group sum of global average ranks (1-based; ties share their mean rank),
    and the column's median.

    Twice the average rank of a tie block [start, end) is start+end+1, an integer, so the
    sums are exact in float64 below 2**53. float32 columns sort as one uint64 array, the
    order-preserving value bits above the label, about 3x faster than an argsort."""
    column = np.asarray(column); n = len(column)
    if column.dtype == np.float32:
        bits = (column + np.float32(0)).view(np.uint32)  # -0.0 becomes +0.0, as comparisons see it
        negative = (bits >> np.uint32(31)).astype(bool)
        packed = np.where(negative, ~bits, bits | np.uint32(0x80000000)).astype(np.uint64)
        del bits, negative
        packed <<= np.uint64(32); packed |= np.asarray(keys).astype(np.uint64)
        packed.sort()
        lab = (packed & np.uint64(0xFFFFFFFF)).astype(np.intp)
        packed >>= np.uint64(32); ordered = packed

        def value(i):
            u = np.uint32(ordered[i])
            return float((u & np.uint32(0x7FFFFFFF) if u >> np.uint32(31) else ~u).view(np.float32))
    else:
        idx = np.argsort(column)
        ordered = column[idx]; lab = np.asarray(keys)[idx]
        del idx
        value = lambda i: float(ordered[i])
    median = (value((n-1)//2) + value(n//2)) / 2
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    del ordered
    if len(starts) == n:
        twice = np.arange(2, 2*n+1, 2, dtype='f8')
    else:
        ends = np.r_[starts[1:], n]
        twice = np.repeat((starts+ends+1).astype('f8'), ends-starts)
    return np.bincount(lab, weights=twice, minlength=n_groups), median


def effects_from_rank_sums(twice_rank_sums, counts, n):
    """E = 2 AUC - 1 = (2 R - n_c (n + 1)) / (n_c (n - n_c)); NaN for empty or all-row clusters."""
    counts = np.asarray(counts, dtype='f8')
    E = np.full(len(counts), np.nan)
    ok = (counts > 0) & (counts < n)
    E[ok] = (twice_rank_sums[ok] - counts[ok]*(n+1)) / (counts[ok]*(n-counts[ok]))
    return E


def rank_effects(X, labels, n_groups):
    """K x D one-vs-rest rank effects of X's columns; the reference for export_statistics."""
    labels = np.asarray(labels); counts = np.bincount(labels, minlength=n_groups)
    return np.stack([effects_from_rank_sums(rank_sums(X[:, j], labels, n_groups)[0], counts, len(labels))
                     for j in range(X.shape[1])], axis=1)


def sheet_candidates(effects, names):
    """Clusters with the strict signature, ordered by E(log_chi_B) descending, then by ID."""
    E = np.asarray(effects, dtype='f8')
    col = {f: list(names).index(f) for f, _ in SIGNATURE}
    conditions = {f'E_{f}_{"positive" if s > 0 else "negative"}': (E[:, col[f]] > 0) if s > 0 else (E[:, col[f]] < 0)
                  for f, s in SIGNATURE}  # NaN (empty cluster) fails every comparison
    candidate = np.logical_and.reduce(list(conditions.values()))
    chi = E[:, col['log_chi_B']]
    order = sorted(map(int, np.flatnonzero(candidate)), key=lambda c: (-chi[c], c))
    ties = [[a, b] for a, b in zip(order, order[1:]) if chi[a] == chi[b]]
    return dict(order=order, ties=ties,
                clusters=[dict(cluster=c, candidate=bool(candidate[c]),
                               **{k: bool(v[c]) for k, v in conditions.items()},
                               **{f'E_{f}': E[c, col[f]] for f, _ in SIGNATURE}) for c in range(len(E))])


def jaccard_matrix(a, b, ka, kb, chunk=2**21):
    """Row-count Jaccard between every cluster of labelling a and of labelling b on the same rows."""
    inter = np.zeros(ka*kb, np.int64)
    for i in range(0, len(a), chunk):
        inter += np.bincount(np.asarray(a[i:i+chunk], np.int64)*kb + np.asarray(b[i:i+chunk]), minlength=ka*kb)
    inter = inter.reshape(ka, kb)
    union = inter.sum(1)[:, None] + inter.sum(0)[None, :] - inter
    return np.divide(inter, union, out=np.zeros((ka, kb)), where=union > 0)


def mutual_best(J):
    """Boolean matrix: (a, b) is each other's argmax Jaccard match."""
    flags = np.zeros(J.shape, bool)
    rows = np.argmax(J, axis=1)
    for a, b in enumerate(rows):
        flags[a, b] = int(np.argmax(J[:, b])) == a
    return flags


def mutual_pair(J, a, b):
    """Audit entry for clusters a (rows of J) and b (columns): mutual best over all clusters."""
    best_a, best_b = int(np.argmax(J[a])), int(np.argmax(J[:, b]))
    return dict(clusters=[a, b], jaccard=float(J[a, b]), best_of_first=best_a, best_of_second=best_b,
                mutual=best_a == b and best_b == a)


def check_tuple(seeds, members, jaccard):
    """Pairwise mutual-best check of one cluster per seed (members: seed -> cluster)."""
    pairs = {f'{i}-{j}': mutual_pair(jaccard[(i, j)], members[i], members[j]) for i, j in jaccard}
    return dict(reproduced=all(p['mutual'] for p in pairs.values()),
                candidates={s: members[s] for s in seeds}, pairs=pairs)


def approved_rule_outcome(orders, jaccard):
    """The previous rule (first candidates only), kept for the audit; picks and outcome."""
    seeds = list(orders)
    picks = {s: [] for s in seeds}
    outcome = dict(rule=PREVIOUS_SELECTION_RULE, status=None, reason=None, selected=picks)
    if len(seeds) < 2:
        return dict(outcome, status='unavailable',
                    reason=f'fewer than two seeds ({len(seeds)}): reproduction cannot be checked')
    empty = [s for s in seeds if not orders[s]]
    if empty:
        return dict(outcome, status='none', reason=f'empty signature set in seed(s) {empty}')
    first = check_tuple(seeds, {s: orders[s][0] for s in seeds}, jaccard)
    if not first['reproduced']:
        failed = [k for k, p in first['pairs'].items() if not p['mutual']]
        return dict(outcome, status='none',
                    reason=f'first candidates are not mutual best Jaccard matches for pair(s) {failed}')
    for s in seeds: picks[s].append(orders[s][0])
    missing = [s for s in seeds if len(orders[s]) < 2]
    if missing:
        reason = f'first candidates reproduced; no second candidate in seed(s) {missing}'
    else:
        second = check_tuple(seeds, {s: orders[s][1] for s in seeds}, jaccard)
        if second['reproduced']:
            for s in seeds: picks[s].append(orders[s][1])
            reason = 'first and second candidates reproduced across every seed pair'
        else:
            failed = [k for k, p in second['pairs'].items() if not p['mutual']]
            reason = f'first candidates reproduced; second candidates not mutual for pair(s) {failed}'
    return dict(outcome, status='selected', reason=reason)


def select_group(orders, jaccard, chi):
    """Decide the selected sheet clusters once for a lattice group with the top-two rule.

    orders: seed -> candidate IDs by E(log_chi_B) descending, then ID.
    jaccard: (seed_i, seed_j) -> Jaccard matrix for every pair in seed order.
    chi: seed -> {cluster: E(log_chi_B)} for at least every candidate.
    Each seed's first two candidates are eligible. The first pick is the tuple, one eligible
    candidate per seed, that is pairwise mutual best and has the largest minimum E(log_chi_B);
    tuples run in candidate order and only a strictly larger score replaces the winner. The
    second pick is the unused eligible candidate of every seed, if each seed has one and they
    are pairwise mutual best. Returns picks per seed and the audit; status is selected, none
    or unavailable. The audit also carries the previous rule's outcome."""
    import itertools
    seeds = list(orders)
    picks = {s: [] for s in seeds}
    eligible = {s: list(orders[s][:ELIGIBLE_PER_SEED]) for s in seeds}
    audit = dict(seeds=seeds, rule=SELECTION_RULE,
                 eligible={s: [dict(cluster=c, E_log_chi_B=float(chi[s][c])) for c in eligible[s]] for s in seeds},
                 tuples=[], first=None, second=None, status=None, reason=None,
                 approved_rule_outcome=approved_rule_outcome(orders, jaccard))
    if len(seeds) < 2:
        audit.update(status='unavailable',
                     reason=f'fewer than two seeds ({len(seeds)}): reproduction cannot be checked')
        return picks, audit
    empty = [s for s in seeds if not orders[s]]
    if empty:
        audit.update(status='none', reason=f'empty signature set in seed(s) {empty}')
        return picks, audit
    winner = None
    for members in itertools.product(*(eligible[s] for s in seeds)):
        members = dict(zip(seeds, members))
        score = min(float(chi[s][c]) for s, c in members.items())
        checked = check_tuple(seeds, members, jaccard)
        audit['tuples'].append(dict(candidates=members, min_E_log_chi_B=score,
                                    mutual={k: p['mutual'] for k, p in checked['pairs'].items()},
                                    all_mutual=checked['reproduced']))
        if checked['reproduced'] and (winner is None or score > winner[0]):
            winner = (score, checked)
    if winner is None:
        audit.update(status='none', reason='no tuple of eligible candidates is mutual best for every seed pair')
        return picks, audit
    first = dict(winner[1], min_E_log_chi_B=winner[0])
    audit['first'] = first
    for s in seeds: picks[s].append(first['candidates'][s])
    unused = {s: [c for c in eligible[s] if c != first['candidates'][s]] for s in seeds}
    missing = [s for s in seeds if len(unused[s]) != 1]
    if missing:
        audit['second'] = dict(reproduced=False, missing_in_seeds=missing, pairs={})
        reason = f'first pick made; no unused eligible candidate in seed(s) {missing}'
    else:
        second = check_tuple(seeds, {s: unused[s][0] for s in seeds}, jaccard)
        audit['second'] = second
        if second['reproduced']:
            for s in seeds: picks[s].append(unused[s][0])
            reason = 'first pick made; the unused eligible candidates are mutual best for every seed pair too'
        else:
            failed = [k for k, p in second['pairs'].items() if not p['mutual']]
            reason = f'first pick made; unused eligible candidates not mutual for pair(s) {failed}'
    audit.update(status='selected', reason=reason)
    return picks, audit


def seed_overlap(jaccard):
    return {f'{i}-{j}': dict(seeds=[i, j], jaccard=J, mutual_best=mutual_best(J))
            for (i, j), J in jaccard.items()}


def replace_json(path, data):
    """Atomically replace a JSON file that this run wrote earlier."""
    tmp = Path(str(path) + '.partial')
    with tmp.open('x') as f:
        json.dump(plain(data), f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def one_line(data):
    return json.dumps(plain(data), allow_nan=False, separators=(',', ':'))


def group_selection(out, recs, chunk):
    """Select once per lattice group, then record the result in every seed's h5 and JSON.

    recs are the fit records of one lattice, in seed order, on the same pool rows."""
    import itertools
    from aweSOM import SOMModel
    if any(r.get('statistics') == 'deferred' for r in recs):
        raise ValueError('statistics are deferred; run --statistics-only before selection')
    seeds = [r['seed'] for r in recs]
    orders = {r['seed']: list(r['sheet_candidates']['order']) for r in recs}
    chi = {r['seed']: {c['cluster']: c['E_log_chi_B'] for c in r['sheet_candidates']['clusters']} for r in recs}
    sizes = {r['seed']: len(r['cluster_effects']) for r in recs}
    labels = {r['seed']: np.load(out/f"{r['tag']}-labels.npy", mmap_mode='r') for r in recs}
    jaccard = {(i, j): jaccard_matrix(labels[i], labels[j], sizes[i], sizes[j], chunk)
               for i, j in itertools.combinations(seeds, 2)}
    picks, audit = select_group(orders, jaccard, chi)
    overlap = seed_overlap(jaccard)
    for r in recs:
        s = r['seed']
        own = dict(audit, seed=s, selected=picks[s])
        entries = dict(selected_sheet_clusters=','.join(map(str, picks[s])),
                       sheet_clusters=','.join(map(str, picks[s])),
                       sheet_selection_status=audit['status'],
                       selection_audit=one_line(own),
                       # h5 attributes stop at 64 KB; 6 decimals keep K~30, three seeds well inside it
                       seed_overlap=one_line({k: dict(v, jaccard=np.round(v['jaccard'], 6)) for k, v in overlap.items()}))
        with h5py.File(out/f"{r['tag']}.h5", 'r+') as f:
            for k, v in entries.items(): f['metadata'].attrs[k] = v
        saved = SOMModel.load(out/f"{r['tag']}.h5").metadata
        if any(saved[k] != v for k, v in entries.items()):
            raise RuntimeError('selection metadata did not round-trip')
        r.update(selected_sheet_clusters=picks[s], sheet_selection_status=audit['status'],
                 selection_audit=own, seed_overlap=overlap)
        replace_json(out/f"{r['tag']}.json", r)
    return plain(dict(picks={str(s): p for s, p in picks.items()}, audit=audit))


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
    # Thread values this process uses, not environment variables (user rule 2026-11-06: records
    # hold only SLURM_* and CUDA_VISIBLE_DEVICES from the environment). Native BLAS/OpenMP
    # libraries are pinned to 1 thread at import (top of this file).
    return dict(numba_threads=numba.get_num_threads(), numba_thread_ceiling=numba.config.NUMBA_NUM_THREADS,
                native_library_threads=1, pid=os.getpid(), hostname=node_name(),
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                slurm_cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK'),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'))



STATISTIC_FIELDS = ('cluster_medians','pooled_medians','cluster_effects','cluster_quartiles',
                    'cluster_rows','cluster_share','cluster_joint_favorable_fraction','sheet_candidates',
                    'legacy_threshold_rule_diagnostic','tube_rule','reject_distance')


def statistics_fields(args, raw, nodes, distances, labels, n_nodes, n_clusters, names,
                      strata, out, tag):
    """Build identical model and JSON fields for inline and deferred statistics."""
    q, reject, medians, described = export_statistics(
        raw,nodes,distances,labels,n_nodes,out,tag,args.chunk,names=names,strata=strata,
        n_clusters=n_clusters,scratch=args.scratch,workers=args.threads_per_seed or 1)
    candidates = sheet_candidates(described['cluster_effects'],names)
    chi, beta = names.index('log_chi_B'),names.index('log_beta_inv')
    legacy = [c for c,m in medians.items()
              if m[chi] > LEGACY_THRESHOLDS['log_chi_B_gt'] and m[beta] < LEGACY_THRESHOLDS['log_beta_inv_lt']]
    bz = names.index('bz_frac') if 'bz_frac' in names else None
    tubes = [c for c,m in medians.items()
             if bz is not None and m[bz] > args.tube_bz and m[beta] > args.tube_beta]
    exports = {key:described[key] for key in ('cluster_effects','cluster_quartiles','cluster_rows',
               'cluster_share','cluster_joint_favorable_fraction')}
    exports['sheet_candidates'] = candidates
    fields = dict(cluster_medians=medians,pooled_medians=described['pooled_medians'],**exports,
                  legacy_threshold_rule_diagnostic=dict(
                      role='diagnostic only; plays no part in selection',statistic='median',**LEGACY_THRESHOLDS,IDs=legacy),
                  tube_rule=dict(status='characterisation only',bz_frac_gt=args.tube_bz,log_beta_inv_gt=args.tube_beta,IDs=tubes),
                  reject_distance=reject)
    if set(fields) != set(STATISTIC_FIELDS):
        raise ValueError('statistics fields differ from STATISTIC_FIELDS')
    fields = {key: fields[key] for key in STATISTIC_FIELDS}
    metadata = dict(**{key:one_line(value) for key,value in exports.items()},
                    tube_clusters=','.join(map(str,tubes)),
                    tube_rule=f'characterisation only: median bz_frac > {args.tube_bz} and median log_beta_inv > {args.tube_beta}')
    return q,reject,fields,metadata



def fit(args, shape, seed, scaling, out, comm=None, restored=None, pool_file=None):
    """One fit. With ``comm`` every rank trains on the windows it owns; rank 0 alone does
    the rest and returns the record, the other ranks return None after training."""
    from aweSOM import Lattice, SOMModel, Scaler
    from aweSOM.stream import train_batch_stream, peak_rss_bytes
    from aweSOM.inference import best_match_nodes_patch
    t = time.perf_counter()
    tag = f'{shape[0]}x{shape[1]}-s{seed}'
    clock = PhaseClock(tag)
    execution = execution_info()
    print(json.dumps(dict(tag=tag, execution=execution)), flush=True)
    scaler = Scaler('median-iqr', np.array(scaling['center']), np.array(scaling['scale']))
    with h5py.File(pool_file or out / 'pool.h5', 'r') as h:
        raw = h['raw']
        transform = lambda block: scaler.transform(np.asarray(block,dtype='f8'))
        if args.check_scaling:
            if 'scaled' not in h: raise ValueError('--check-scaling requires a pool built with --write-scaled')
            for a in range(0,min(args.check_scaling,len(raw)),args.chunk):
                b = min(a+args.chunk,args.check_scaling,len(raw))
                if not np.array_equal(transform(raw[a:b]),h['scaled'][a:b]):
                    raise RuntimeError('per-chunk scaling differs from stored scaled rows')
        source = EpochSource(raw, scaling['strata'], args.rows_per_epoch or len(raw), args.sample_seed, args.chunk, transform=transform)
        import inspect as signature_inspect
        parameters = signature_inspect.signature(train_batch_stream).parameters
        checkpoint_args = {}
        if 'checkpoint' in parameters: checkpoint_args['checkpoint'] = out / f'{tag}-checkpoint.npz'
        if args.resume is not None:
            if 'resume' not in parameters: raise ValueError('installed trainer does not support resume')
            # The trainer resumes from its own checkpoint path: rank 0 stages the requested file
            # there (the source stays unchanged). The other ranks never write a checkpoint, so
            # they read the source and never look up a file rank 0 has just created.
            if comm is None or not comm.Get_rank():
                shutil.copyfile(args.resume, checkpoint_args['checkpoint'])
            else:
                checkpoint_args['checkpoint'] = args.resume.resolve()
            if comm is not None: comm.Barrier()
            checkpoint_args['resume'] = True
        w, info = train_batch_stream(source, *shape, epochs=args.epochs, chunk=args.chunk,
                                    bmu=args.bmu, seed=seed, groups=args.groups, r_max=args.r_max,
                                    r_min=args.r_min if args.r_min is not None else (8 if shape[0] >= 91 else 5),
                                    final_pass='none', n_samples=args.rows_per_epoch or len(raw), verbose=True,
                                    comm=comm, **checkpoint_args)
        if comm is not None:
            import numba
            per_rank = comm.gather(dict(hostname=node_name(), numba_threads=numba.get_num_threads(),
                                        affinity_restored=restored), root=0)
            if comm.Get_rank(): return None
        clock.finish('training')
        if w.dtype != np.float64 or not np.isfinite(w).all():
            raise RuntimeError('training returned invalid float64 weights')
        lat = Lattice(*shape); lat.lattice = w; lat.umat = lat.compute_umat()
        nc = lat.assign_cluster_to_lattice(merge_cost=args.merge_cost, target_clusters=args.target_clusters).ravel(order='F').astype(np.int32)
        np.save(out / f'{tag}-umat.npy', lat.umat)
        clock.finish('clustering')
        labels = np.lib.format.open_memmap(out / f'{tag}-labels.npy', mode='w+', dtype='i4', shape=(len(raw),))
        node_path, distance_path = out/f'{tag}-nodes.npy', out/f'{tag}-distances.npy'
        nodes = np.lib.format.open_memmap(node_path, mode='w+', dtype='i4', shape=(len(raw),))
        distances = np.lib.format.open_memmap(distance_path, mode='w+', dtype='f8', shape=(len(raw),))
        for a in range(0, len(raw), args.chunk):
            b = min(a+args.chunk, len(raw))
            nodes[a:b], d2 = best_match_nodes_patch(w, transform(raw[a:b]), *shape)
            labels[a:b] = nc[nodes[a:b]]; distances[a:b] = np.sqrt(d2)
            from aweSOM.pool_statistics import release_mapping
            for mapped in (labels,nodes,distances): release_mapping(mapped)
        clock.finish('labelling')
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in h['names'][:]]
        deferred = args.statistics == 'later'
        if deferred:
            q, reject = np.full(len(w),np.nan), float('nan')
            statistics = dict.fromkeys(STATISTIC_FIELDS)
            statistic_metadata = dict(statistics='deferred')
        else:
            q,reject,statistics,statistic_metadata = statistics_fields(
                args,raw,nodes,distances,labels,len(w),lat.n_clusters,names,scaling['strata'],out,tag)
            clock.finish('statistics')
        del nodes, distances
        merge_settings = dict(merge_cost_used=lat.merge_cost_used,
                              target_clusters=args.target_clusters, n_clusters=lat.n_clusters)
        # Selection is decided across the lattice's seeds in main(); until then nothing is selected.
        pending = dict(status='pending', reason='awaiting the other seeds of this lattice')
        meta = dict(**merge_settings, seed=str(seed), scale_unweighted=json.dumps(plain(scaling['scale_unweighted'])),
                    selection_rule=SELECTION_RULE, selected_sheet_clusters='', sheet_clusters='',
                    sheet_selection_status='pending', selection_audit=one_line(pending), seed_overlap='{}',
                    effect_feature_order=one_line(names), **statistic_metadata,
                    feature_settings=json.dumps(plain(args.anchor_attrs)),
                    feature_weights=json.dumps(plain(scaling['feature_weights'])),
                    b2_min=str(args.anchor_attrs['b2_min']), s_v=str(args.anchor_attrs['s_v']),
                    pipeline='v2', freeze_manifest='freeze-manifest.json', aweSOM_commit=args.code_commit)
        from aweSOM.features_grmhd import FeatureThresholds
        if deferred:
            meta.update(sheet_selection_status='deferred',
                        selection_audit=one_line(dict(status='deferred',reason='awaiting statistics')))
        thresholds = FeatureThresholds.from_h5_attrs(args.anchor_attrs)
        thresholds.feature_names = tuple(names)
        thresholds.__post_init__()
        meta.update(thresholds.to_metadata())
        model = SOMModel(*shape,w,nc,names,scaler,q,reject,metadata=meta)
        temp = out / f'{tag}.partial.h5'; model.save(temp); publish(temp,out / f'{tag}.h5')
        clock.finish('export')
        loaded = SOMModel.load(out / f'{tag}.h5')
        mismatch = parity = 0; scaler_error = 0.
        for a in range(0,len(raw) if args.verify_labels else min(len(raw),1_000_000),args.chunk):
            stop = min(a+args.chunk,len(raw) if args.verify_labels else 1_000_000)
            block = raw[a:stop]
            mismatch += int(np.count_nonzero(loaded.predict(block,backend='patch') != labels[a:a+len(block)]))
            scaler_error = max(scaler_error,float(np.max(np.abs(loaded.scaler.transform(block)-transform(block)))))
        probe = raw[np.unique(np.linspace(0,len(raw)-1,min(8192,len(raw)),dtype=int))]
        parity = int(np.count_nonzero(loaded.predict(probe,backend='patch') != loaded.predict(probe,backend='screen')))
        from aweSOM.inference import best_match_nodes_screen
        probe_scaled = loaded.scaler.transform(probe)
        pn, pd = best_match_nodes_patch(loaded.weights, probe_scaled, *shape)
        sn, sd = best_match_nodes_screen(loaded.weights, probe_scaled)
        bmu_mismatches = int(np.count_nonzero(pn != sn))
        distance_parity = bool(np.allclose(np.sqrt(pd), np.sqrt(sd), rtol=1e-6, atol=0))
        components = dict(weights=np.array_equal(w,loaded.weights),
                          scaler=loaded.scaler.kind == scaler.kind and np.array_equal(scaler.offset,loaded.scaler.offset) and np.array_equal(scaler.scale,loaded.scaler.scale),
                          node_map=np.array_equal(nc,loaded.node_cluster),
                          distance_quantiles=np.array_equal(q,loaded.node_dist_q99,equal_nan=True),
                          reject_distance=reject == loaded.reject_distance or (np.isnan(reject) and np.isnan(loaded.reject_distance)))
        if mismatch or parity or scaler_error > 1e-4 or bmu_mismatches or not distance_parity or not all(components.values()):
            raise RuntimeError('freeze verification failed')
        if not deferred:
            node_path.unlink(); distance_path.unlink()
        labels.flush()
        clock.finish('verification')
    rec = dict(tag=tag, git_commit=args.code_commit, ranks=1, scratch=None if args.scratch is None else str(args.scratch), lattice_shape=shape, seed=seed, init='sampling', sample_seed=args.sample_seed,
               epochs=args.epochs, rows_per_epoch=info['rows_per_epoch'], radius_schedule=info['radius'],
               chunk=args.chunk, groups=args.groups, bmu_backend=info['bmu_kind'], fallback_count=info['fallbacks'],
               execution=execution, phase_timings=clock.records, training_info=info, wall_seconds=time.perf_counter()-t, peak_rss_bytes=peak_rss_bytes(),
               weights_dtype=str(w.dtype), weights_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
               node_to_cluster=nc, merge_cost=args.merge_cost, **merge_settings, statistics='deferred' if deferred else 'complete',
               selection_rule=SELECTION_RULE, effect_feature_order=names, **statistics,
               selected_sheet_clusters=[], sheet_selection_status='pending', selection_audit=pending, seed_overlap={},
               umat_file=f'{tag}-umat.npy',
               verification=dict(reload_mismatches=mismatch,cluster_label_patch_screen_mismatches=parity,
                                 bmu_node_mismatches=bmu_mismatches, bmu_distance_rtol_1e6=distance_parity,
                                 saved_components_equal=components,
                                 parity_rows=len(probe),scaler_max_abs_error=scaler_error))
    if comm is not None:
        rec.update(ranks=comm.Get_size(), rank_hostnames=[r['hostname'] for r in per_rank],
                   rank_numba_threads=[r['numba_threads'] for r in per_rank],
                   affinity_restored=[r['affinity_restored'] for r in per_rank])
    if deferred:
        rec.update(selected_sheet_clusters=None,sheet_selection_status='deferred',selection_audit=None,seed_overlap=None)
        rec['verification']['saved_components_equal'].update(distance_quantiles=None,reject_distance=None)
    dump(out / f'{tag}.json',rec)
    return plain(rec)


# Set by MPI launchers (srun --mpi=pmix, mpirun, mpiexec) in every rank, never in a plain run.
LAUNCHER_VARIABLES = ('PMIX_RANK', 'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE')


def restore_affinity(saved, get=None, put=None):
    """Give this thread back the CPU set ``saved`` if MPI_Init shrank it; None if it did not.

    Copied from examples/slurm_scripts/scale_out/scale_out.py. On GP, libpsm2 (Omni-Path) pins
    a whole-node rank to one CPU in MPI_Init (jobs 8413, 8444, 8448); HFI_NO_CPUAFFINITY=1 in
    the launch turns that off and this stays as the guard. Call it before numba starts its
    threads (set_num_threads or the first parallel call). Returns {before, after_init,
    after_restore, cpus_after_init} and writes one line to stderr when it acts.
    """
    get = get or getattr(os, 'sched_getaffinity', None)
    put = put or getattr(os, 'sched_setaffinity', None)
    if saved is None or get is None or put is None:
        return None
    now = set(get(0))
    if len(now) >= len(saved):
        return None
    rec = dict(before=len(saved), after_init=len(now), cpus_after_init=sorted(now))
    try:
        put(0, set(saved))
    except OSError as e:
        rec['error'] = f'{type(e).__name__}: {e}'
    rec['after_restore'] = len(set(get(0)))
    print(f"{node_name()}: CPU set shrank in MPI_Init from {rec['before']} to {rec['after_init']} CPUs; "
          f"restored to {rec['after_restore']}" + (f" ({rec['error']})" if 'error' in rec else ''),
          file=sys.stderr, flush=True)
    return rec


def mpi_comm(args):
    """``(comm, restored)``: the communicator of a multi-rank launch (None otherwise) and the
    affinity restore record. MPI is initialised only under --mpi or an MPI launcher."""
    if not (args.mpi or any(k in os.environ for k in LAUNCHER_VARIABLES)):
        return None, None
    from aweSOM import distributed
    comm = distributed.get_comm()
    return comm, restore_affinity(AFFINITY_AT_START)


def mpi_threads(args):
    """Numba threads of one rank: --threads-per-seed, else NUMBA_NUM_THREADS, else the CPU set."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    threads = args.threads_per_seed or int(os.environ.get('NUMBA_NUM_THREADS') or cpus)
    if threads > cpus:
        raise RuntimeError(f'{node_name()}: {cpus} CPUs in this rank\'s affinity mask, '
                           f'fewer than {threads} threads; check the srun --cpu-bind and HFI_NO_CPUAFFINITY')
    return threads


def worker(conn, *args):
    try:
        config = args[0]
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
            os.environ[key] = '1'
        os.environ['NUMBA_CACHE_DIR'] = str(args[-1] / f'numba-cache-{os.getpid()}')
        import numba
        numba.set_num_threads(config.threads_per_seed)
        conn.send(('ok',fit(*args)))
    except BaseException: conn.send(('error',traceback.format_exc()))
    finally: conn.close()


def run_wave(jobs, target=worker, exit_grace=60):
    """Run one wave of fits in spawned workers; results come back in job order.

    One daemon reader thread per pipe posts exactly one terminal event; the
    supervisor only ever waits one second at a time, so the first error message,
    abnormal exit or dead reader stops the siblings and raises with whatever
    detail arrived. A worker that delivered its result but does not exit within
    exit_grace seconds is terminated and its result kept. Every wait is bounded;
    SIGTERM to the parent runs the same cleanup."""
    import queue, signal, threading
    jobs = list(jobs); ctx = mp.get_context('spawn'); running = []; readers = []; inbox = queue.Queue()
    results = [None]*len(jobs); waiting = set(range(len(jobs)))

    def read(i, receive):  # one recv, one event, whatever happens
        try:
            status, payload = receive.recv()
            if status not in ('ok', 'error'): raise ValueError(f'unknown status {status!r}')
        except (EOFError, OSError) as error: status, payload = 'error', f'worker {i} exited without result ({error!r})'
        except BaseException as error: status, payload = 'error', f'worker {i} result could not be read ({error!r})'
        inbox.put((i, status, payload))

    def pump(timeout):  # apply every queued event, waiting at most timeout for the first
        while True:
            try: i, status, payload = inbox.get(timeout=timeout)
            except queue.Empty: return
            timeout = 0
            if i not in waiting: continue
            if status != 'ok':
                process = running[i][1]; process.join(5)
                raise RuntimeError(payload if not process.exitcode else f'{payload}\nworker {i} exit code {process.exitcode}')
            results[i] = payload; waiting.discard(i)

    def on_term(signum, frame): raise SystemExit(128+signum)
    main = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, on_term) if main else None
    try:
        for job in jobs:
            receive, send = ctx.Pipe(duplex=False)
            with send:  # closed even if start() fails; receive is closed in finally
                process = ctx.Process(target=target, args=(send, *job)); running.append((receive, process))
                previous_threads = os.environ.get('NUMBA_NUM_THREADS')
                if target is worker: os.environ['NUMBA_NUM_THREADS'] = str(job[0].threads_per_seed)
                try: process.start()
                finally:
                    if previous_threads is None: os.environ.pop('NUMBA_NUM_THREADS',None)
                    else: os.environ['NUMBA_NUM_THREADS'] = previous_threads
            readers.append(threading.Thread(target=read, args=(len(readers), receive), daemon=True)); readers[-1].start()
        while waiting:
            dead = [j for j in waiting if not readers[j].is_alive()]  # checked before pump: a finished reader's event is already queued
            pump(1)
            for j in dead:
                if j in waiting: raise RuntimeError(f'worker {j} result reader died without reporting')
            for j, (_, process) in enumerate(running):
                if process.exitcode and j in waiting:  # pipe is closed now, so its message or EOF is at most a decode away
                    readers[j].join(5); pump(0)
                    if j in waiting: raise RuntimeError(f'worker {j} exited without result (exit code {process.exitcode})')
                if process.exitcode: raise RuntimeError(f'worker {j} exit code {process.exitcode} after its result')
        deadline = time.monotonic()+exit_grace
        for j, (_, process) in enumerate(running):
            process.join(max(0, deadline-time.monotonic()))
            if process.exitcode: raise RuntimeError(f'worker {j} exit code {process.exitcode} after its result')
            if process.exitcode is None:
                print(f'warning: worker {j} sent its result but did not exit within {exit_grace} s; terminating', flush=True)
        return results
    finally:
        if main: signal.signal(signal.SIGTERM, previous)
        for _, process in running:
            if process.is_alive(): process.terminate()
        for _, process in running:  # is_alive() is False for a process that never started
            if process.is_alive(): process.join(10)
            if process.is_alive(): process.kill(); process.join(10)
            if process.is_alive(): print(f'warning: worker {process.pid} survived SIGKILL (uninterruptible?)', flush=True)
        for reader in readers: reader.join(5)
        for receive, _ in running: receive.close()


def agreement(a,b,chunk=2**21):
    """Optimal one-to-one cluster relabelling, divided by all pool rows."""
    from scipy.optimize import linear_sum_assignment
    matrix = np.zeros((int(a.max())+1,int(b.max())+1),np.int64)
    for i in range(0,len(a),chunk): np.add.at(matrix,(a[i:i+chunk],b[i:i+chunk]),1)
    x,y = linear_sum_assignment(matrix,maximize=True)
    return float(matrix[x,y].sum()/len(a))


def medoid(paths,chunk):
    arrays = [np.load(p,mmap_mode='r') for p in paths]
    matrix = np.eye(len(arrays))
    for i in range(len(arrays)):
        for j in range(i): matrix[i,j] = matrix[j,i] = agreement(arrays[i],arrays[j],chunk)
    means = (matrix.sum(axis=1)-1)/max(1,len(arrays)-1)
    return int(means.argmax()), dict(pairwise_agreement=matrix,mean_agreement=means,
                                    band=[float(matrix[np.triu_indices(len(arrays),1)].min()),
                                          float(matrix[np.triu_indices(len(arrays),1)].max())] if len(arrays)>1 else [1,1])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,nargs='+')
    p.add_argument('--statistics',choices=['inline','later'],default='inline')
    p.add_argument('--statistics-only',type=Path,metavar='FIT_OUT')
    p.add_argument('--pool-only',action='store_true')
    p.add_argument('--write-scaled',action='store_true')
    p.add_argument('--check-scaling',type=int,default=0)
    p.add_argument('--scratch',type=Path)
    p.add_argument('--bmu',default='auto')
    p.add_argument('--verify-labels',action='store_true')
    p.add_argument('--resume',type=Path,help='--fit only: continue from this checkpoint (copied into OUT, left unchanged); same rank count')
    p.add_argument('--reuse-pool',type=Path)
    p.add_argument('--fit',nargs=2,metavar=('LATTICE','SEED'))
    p.add_argument('--mpi',action='store_true',help='require a multi-rank MPI launch (--fit with --reuse-pool only); rank 0 writes every output')
    p.add_argument('--collect',type=Path,nargs='+')
    p.add_argument('--anchor'); p.add_argument('--held-out',nargs='+')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rows-per-snapshot',type=int); p.add_argument('--rows-per-epoch',type=int,default=100_000_000)
    p.add_argument('--lattices',nargs='+',default=['91x64','48x34']); p.add_argument('--seeds',nargs='+',type=int,default=[42,43],help='two or more seeds per lattice; the sheet-candidate selection needs at least two')
    p.add_argument('--seed-workers',type=int,default=1,help='concurrent spawned seeds within each lattice')
    p.add_argument('--threads-per-seed',type=int,help='Numba threads for the entire worker, including export and verification')
    p.add_argument('--epochs',type=int,default=40); p.add_argument('--chunk',type=int,default=2**21)
    p.add_argument('--sample-seed',type=int,default=42); p.add_argument('--groups',type=int,default=64)
    p.add_argument('--target-clusters',type=int,default=10,help='merge to at most this count; ties may yield fewer')
    p.add_argument('--merge-cost',type=float,default=.05,help='merge cost floor; keep the selected cost for subsequent training'); p.add_argument('--r-max',type=float); p.add_argument('--r-min',type=float)
    p.add_argument('--bz-weight',type=float,default=2.); p.add_argument('--pol-weight',type=float,default=2.)
    p.add_argument('--tube-bz',type=float,default=.7); p.add_argument('--tube-beta',type=float,default=0.)
    p.add_argument('--min-free-gb',type=float,default=10)
    p.add_argument('--allow-independent-thresholds',action='store_true',help='diagnostic only; manifest cannot be production accepted')
    p.add_argument('--allow-six-features',action='store_true',help='legacy local rehearsal only')
    return p


# The approved production training settings (production_pool.sbatch defaults). A run that departs
# from any of them is a rehearsal: its manifest is marked diagnostic_only with the reasons.
# r_min=8 on every lattice and target_clusters=10 approved by the user on 2026-10-29.
# Two seeds per lattice (2026-10-29): the consensus sheet-candidate selection needs at least two,
# and three are too slow. The post-clustering sheet thresholds were rejected and removed.
APPROVED = dict(rows_per_snapshot=10_000_000, rows_per_epoch=20_000_000, lattices=['91x64','48x34'],
                seeds=[42,43], epochs=40, chunk=2**21, sample_seed=42, groups=64, merge_cost=.05, target_clusters=10,
                r_max=None, r_min=8., bz_weight=2., pol_weight=2.,
                tube_bz=.7, tube_beta=0.)
# The approved training snapshots, in pool order, and the reserved snapshots. Relocating these
# files is operational; a different selection, count or order is a different training set.
APPROVED_SNAPSHOTS = dict(snapshots=['94208','98304'], anchor='94208', held_out=['405504'])
# SHA256 of each approved training file. The iteration alone does not identify the data: another
# extraction or simulation can reuse it.
APPROVED_INPUT_SHA256 = {
    # HHR_flux region7 extractions, seven features, pipeline v2; 98304 uses the 94208 thresholds
    '94208': '24ed3e72e428fc6874d334e6227c2a4dee0a3423f7e3dc34d364be31567b5fc5',  # job 8004
    '98304': '8570765e401d3a1796c182d670d88d15cf9b6e63d2c6a18704c992df044ee680',  # job 8010
}


def diagnostic_reasons(args, records=()):
    """Departures from the approved production run; ``records`` are inspect()'s, with snapshot and sha256."""
    reasons=[f'{k}={getattr(args,k)!r}, approved {v!r}' for k,v in APPROVED.items() if getattr(args,k)!=v]
    used=dict(snapshots=[r['snapshot'] for r in records], anchor=args.anchor, held_out=list(args.held_out))
    reasons+=[f'{k}={used[k]!r}, approved {v!r}' for k,v in APPROVED_SNAPSHOTS.items() if used[k]!=v]
    reasons+=[f"input {r['snapshot']} sha256={r['sha256']}, approved {APPROVED_INPUT_SHA256.get(r['snapshot'])}"
              for r in records if APPROVED_INPUT_SHA256.get(r['snapshot'])!=r['sha256']]
    if args.allow_independent_thresholds: reasons.append('allow_independent_thresholds')
    if args.allow_six_features: reasons.append('allow_six_features')
    return reasons


def fit_disk_budget(args, rows, features, itemsize=4, *, full_run=False, pool_new=True, output_per_row=16):
    """Check scratch and output capacity before pooling or training starts."""
    from aweSOM.pool_statistics import scratch_root
    scratch = scratch_root(args.scratch,args.out.parent)
    output = args.out.parent.resolve()
    # Distance runs are removed before concurrent feature runs are formed.
    workers = min(args.threads_per_seed,features)
    concurrent = min(args.seed_workers,len(args.seeds)) if full_run else 1
    deferred = getattr(args,'statistics','inline') == 'later'
    scratch_bytes = rows*max(16,(8 if itemsize == 4 else 16)*workers)*concurrent
    if deferred: scratch_bytes = rows*8 if full_run and pool_new else 0
    output_bytes = rows*output_per_row
    if full_run:
        pool_bytes = features*(4+8*args.write_scaled) if pool_new else 0
        fits = len(args.lattices)*len(args.seeds)
        output_bytes = rows*(pool_bytes+4*fits+12*(fits if deferred else concurrent))
    if on_tigris(args.out, scratch):  # no free-space reads on RIT RC; nothing recorded
        return dict(scratch_bytes=scratch_bytes,output_bytes=output_bytes,shared_filesystem=None)
    scratch_free = shutil.disk_usage(scratch).free  # sysinfo-ok: GP-only branch, skipped on RIT RC
    output_free = shutil.disk_usage(output).free  # sysinfo-ok: GP-only branch, skipped on RIT RC
    shared = scratch.stat().st_dev == output.stat().st_dev
    floor = args.min_free_gb*1e9
    scratch_need = max(floor,scratch_bytes+output_bytes if shared else scratch_bytes)
    output_need = max(floor,scratch_bytes+output_bytes if shared else output_bytes)
    if scratch_free < scratch_need:
        raise ValueError(f'Low scratch disk: {scratch_free} bytes free; need {scratch_need}')
    if output_free < output_need:
        raise ValueError(f'Low output disk: {output_free} bytes free; need {output_need}')
    return dict(scratch_bytes=scratch_bytes,output_bytes=output_bytes,shared_filesystem=shared)



def statistics_only(args):
    """Complete deferred fits into a fresh OUT without changing the source."""
    from aweSOM import SOMModel
    from aweSOM.stream import peak_rss_bytes
    started = time.perf_counter()
    if args.out.exists(): raise ValueError('OUT exists; use a fresh output directory')
    source = args.statistics_only
    source = source/'results' if (source/'results').is_dir() else source
    bundle = json.loads((source/'fit-results.json').read_text())
    state = json.loads((source/'pool-state.json').read_text())
    records = bundle['models']
    if not records or any(r.get('statistics') != 'deferred' for r in records):
        raise ValueError('--statistics-only requires deferred fits')
    args.seed_workers = 1
    args.seeds = [r['seed'] for r in records]
    args.threads_per_seed = thread_budget(args)['threads_per_seed']
    for key in ('tube_bz','tube_beta'):
        setattr(args,key,bundle['settings'][key])
    with h5py.File(source/'pool.h5','r') as h:
        rows,features = h['raw'].shape
        fit_disk_budget(args,rows,features,h['raw'].dtype.itemsize,output_per_row=4*len(records))
    for record in records:
        tag = record['tag']
        model = SOMModel.load(source/f'{tag}.h5')
        if hashlib.sha256(model.weights.tobytes()).hexdigest() != record['weights_sha256']:
            raise ValueError(f'saved weights differ for {tag}')
        if json.loads((source/f'{tag}.json').read_text()) != record:
            raise ValueError(f'fit JSON differs from bundle for {tag}')
        for suffix in ('labels','nodes','distances'):
            array = np.load(source/f'{tag}-{suffix}.npy',mmap_mode='r')
            if array.shape != (rows,): raise ValueError(f'invalid {suffix} shape for {tag}')
            del array
    args.out.mkdir()
    os.environ['NUMBA_CACHE_DIR'] = str(args.out/'numba-cache')
    # Budget checks imported these dispatchers before OUT existed. Bind their caches
    # to this job's directory before the parent performs the distance scan.
    import numba
    from aweSOM import pool_statistics
    numba.config.reload_config()
    for function in (pool_statistics._sift,pool_statistics._drain,pool_statistics._scan):
        function.enable_caching()
    out = args.out/'results.partial'; out.mkdir()
    os.symlink((source/'pool.h5').resolve(),out/'pool.h5')
    dump(out/'pool-state.json',state)
    completed = []
    with h5py.File(out/'pool.h5','r') as h:
        for original in records:
            record = copy.deepcopy(original); tag = record['tag']
            model = SOMModel.load(source/f'{tag}.h5')
            args.chunk = record['chunk']
            labels = np.load(source/f'{tag}-labels.npy',mmap_mode='r')
            nodes = np.load(source/f'{tag}-nodes.npy',mmap_mode='r')
            distances = np.load(source/f'{tag}-distances.npy',mmap_mode='r')
            clock = PhaseClock(tag)
            q,reject,fields,metadata = statistics_fields(args,h['raw'],nodes,distances,labels,
                len(model.weights),record['n_clusters'],model.feature_names,state['scaling']['strata'],out,tag)
            clock.finish('statistics')
            pending = dict(status='pending',reason='awaiting the other seeds of this lattice')
            model.node_dist_q99,model.reject_distance = q,reject
            model.metadata.pop('statistics',None)
            model.metadata.update(metadata,selected_sheet_clusters='',sheet_clusters='',
                sheet_selection_status='pending',selection_audit=one_line(pending),seed_overlap='{}')
            temporary = out/f'{tag}.partial.h5'; model.save(temporary); publish(temporary,out/f'{tag}.h5')
            loaded = SOMModel.load(out/f'{tag}.h5')
            if (not np.array_equal(q,loaded.node_dist_q99,equal_nan=True)
                    or reject != loaded.reject_distance or loaded.metadata != model.metadata):
                raise RuntimeError('statistics model verification failed')
            for suffix in ('-labels.npy','-umat.npy'):
                shutil.copyfile(source/f'{tag}{suffix}',out/f'{tag}{suffix}')
            record.update(fields,statistics='complete',selected_sheet_clusters=[],
                          sheet_selection_status='pending',selection_audit=pending,seed_overlap={})
            record['phase_timings']['statistics'] = clock.records['statistics']
            record['wall_seconds'] += time.perf_counter()-clock.wall+clock.records['statistics']['wall_seconds']
            record['verification']['saved_components_equal'].update(distance_quantiles=True,reject_distance=True)
            dump(out/f'{tag}.json',record); completed.append(record)
            del labels,nodes,distances
    dump(out/'fit-results.json',dict(bundle,models=completed))
    dump(out/'statistics-job.json',dict(git_commit=provenance(),rows=rows,threads=args.threads_per_seed,
         ranks=1,hostname=node_name(),peak_rss_bytes=peak_rss_bytes(),
         wall_seconds=time.perf_counter()-started,
         models={r['tag']:r['phase_timings']['statistics'] for r in completed}))
    os.rename(out,args.out/'results')
    return completed


def main(argv=None):
    started = time.perf_counter()
    cli = parser()
    args = cli.parse_args(argv)
    comm, restored = mpi_comm(args)
    if args.mpi and comm is None:
        cli.error('--mpi: no MPI communicator with more than one rank; launch with srun --mpi=pmix '
                  '(or mpirun) and python -m mpi4py, with mpi4py importable')
    if comm is not None and not (args.fit and args.reuse_pool):
        cli.error('an MPI launch runs only --fit with --reuse-pool')
    rank = 0 if comm is None else comm.Get_rank()
    if args.statistics_only:
        if args.fit or args.collect or args.pool_only or args.reuse_pool or args.statistics != 'inline':
            cli.error('--statistics-only cannot be combined with fit, collect, pool, or later modes')
        return statistics_only(args)
    if not args.inputs or not args.anchor or not args.held_out:
        cli.error('--inputs, --anchor and --held-out are required for pool and fit modes')
    if args.collect and not args.reuse_pool:
        cli.error('--collect requires --reuse-pool')
    if args.collect and args.statistics == 'later':
        cli.error('--collect requires completed statistics')
    if args.out.exists(): raise ValueError('OUT exists; use a fresh output directory')
    if args.chunk < 1 or args.epochs < 1 or not args.seeds: raise ValueError('invalid training settings')
    if not all(np.isfinite(v) and v > 0 for v in [args.bz_weight,args.pol_weight]):
        raise ValueError('feature weights must be finite and positive')
    if args.fit:
        lattice, seed = args.fit
        args.lattices, args.seeds, args.seed_workers = [lattice], [int(seed)], 1
    if sum([bool(args.fit),bool(args.collect),args.pool_only]) > 1:
        raise ValueError('--fit, --collect and --pool-only are exclusive')
    if args.fit and not args.reuse_pool: raise ValueError('--fit requires --reuse-pool')
    if args.resume is not None and not args.fit: raise ValueError('--resume requires --fit (one lattice, one seed)')
    if args.check_scaling < 0: raise ValueError('--check-scaling must be nonnegative')
    shapes = [tuple(map(int,s.split('x'))) for s in args.lattices]
    if any(len(s)!=2 or min(s)<2 for s in shapes): raise ValueError('invalid lattice')
    if len(set(args.seeds)) != len(args.seeds): raise ValueError('duplicate seeds')
    if comm is not None: args.threads_per_seed = mpi_threads(args)
    auto_threads = args.threads_per_seed is None
    execution_plan = thread_budget(args)
    args.threads_per_seed = execution_plan['threads_per_seed']
    if comm is not None:
        import numba
        numba.set_num_threads(args.threads_per_seed)
    args.code_commit = provenance()
    state = None
    if args.reuse_pool:
        directory = args.reuse_pool if args.reuse_pool.is_dir() else args.reuse_pool.parent
        state = json.loads((directory/'pool-state.json').read_text())
        for key in ('rows_per_snapshot','sample_seed','bz_weight','pol_weight'):
            if state['settings'][key] != getattr(args,key): raise ValueError(f'reused pool setting differs: {key}')
        if state['anchor'] != args.anchor or state['held_out'] != args.held_out:
            raise ValueError('reused pool snapshot selection differs')
        if [str(p.resolve()) for p in args.inputs] != [r['path'] for r in state['records']]:
            raise ValueError('reused pool input paths differ')
        records,quota,disk,mismatches = state['records'],state['quota'],state['disk'],state['mismatches']
        if not (args.collect or args.pool_only):
            with h5py.File(directory/'pool.h5','r') as h:
                fit_disk_budget(args,len(h['raw']),h['raw'].shape[1],h['raw'].dtype.itemsize,
                                full_run=not bool(args.fit),pool_new=False)
    else:
        records,quota,disk,mismatches = inspect(args)
        if not args.pool_only:
            fit_disk_budget(args,quota*len(records),len(records[0]['names']),full_run=True)
    if comm is not None: comm.Barrier()  # every rank has checked that OUT is fresh
    if not rank: args.out.mkdir()
    os.environ.setdefault('NUMBA_CACHE_DIR',str(args.out/'numba-cache'))
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    args.anchor_attrs=next(r['attrs'] for r in records if r['snapshot']==args.anchor)
    out=args.out/'results.partial'
    if not rank: out.mkdir()
    if state is None:
        scaling=pool(args,records,quota,out)
        state = dict(records=records,quota=quota,disk=disk,mismatches=mismatches,scaling=scaling,
                     settings={k:getattr(args,k) for k in APPROVED},anchor=args.anchor,held_out=args.held_out)
    else:
        scaling=state['scaling']
        if not rank: os.symlink((directory/'pool.h5').resolve(),out/'pool.h5')
    if not rank: dump(out/'pool-state.json',state)
    if args.pool_only:
        os.rename(out,args.out/'results'); return
    collected=[]
    if args.collect:
        identity = one_line(state)
        for location in args.collect:
            location = location/'results' if (location/'results').is_dir() else location
            if one_line(json.loads((location/'pool-state.json').read_text())) != identity:
                raise ValueError('collection pools differ')
            bundle = json.loads((location/'fit-results.json').read_text())
            if bundle['bmu'] != args.bmu: raise ValueError('collection BMU backends differ')
            for key,value in bundle['settings'].items():
                if key not in ('lattices','seeds') and value != getattr(args,key):
                    raise ValueError(f'collection fit settings differ: {key}')
            for record in bundle['models']:
                if record.get('statistics') == 'deferred':
                    raise ValueError(f"statistics are deferred for {record['tag']}; run --statistics-only first")
                if record['tag'] in {r['tag'] for r in collected}: raise ValueError('duplicate collected fit')
                collected.append(record)
                for suffix in ('.h5','.json','-labels.npy','-umat.npy'):
                    shutil.copyfile(location/(record['tag']+suffix),out/(record['tag']+suffix))
    fits=[]; choices=[]
    for shape in shapes:
        trio=[r for r in collected if tuple(r['lattice_shape']) == shape]
        trio.sort(key=lambda r: args.seeds.index(r['seed']) if r['seed'] in args.seeds else len(args.seeds))
        if args.collect and {r['seed'] for r in trio} != set(args.seeds):
            raise ValueError(f'incomplete collected seeds for {shape}')
        width=execution_plan['seed_workers']
        for start in range(0,0 if args.collect else len(args.seeds),width):
            seeds=args.seeds[start:start+width]
            wave_args=copy.copy(args)
            if auto_threads:
                import numba
                wave_args.threads_per_seed=min(execution_plan['allocated_cpus']//len(seeds),
                                               numba.config.NUMBA_NUM_THREADS)
            jobs=[(wave_args,shape,seed,scaling,out) for seed in seeds]
            # MPI: every rank trains in this process and reads the pool where it was prepared.
            trio.extend(run_wave(jobs) if comm is None else
                        [fit(*job,comm=comm,restored=restored,pool_file=(directory/'pool.h5').resolve()) for job in jobs])
        if args.fit or args.statistics == 'later':
            fits.extend(trio); continue
        # Decided once for the lattice and written into every seed's model before the frozen link.
        sheet_selection=group_selection(out,trio,args.chunk)
        idx,band=medoid([out/f"{r['tag']}-labels.npy" for r in trio],args.chunk)
        chosen=trio[idx]
        os.link(out/f"{chosen['tag']}.h5",out/f'frozen-{shape[0]}x{shape[1]}.h5')
        choices.append(dict(shape=shape,medoid_seed=chosen['seed'],model=f'frozen-{shape[0]}x{shape[1]}.h5',**band,
                            sheet_selection=sheet_selection))
        fits.extend(trio)
    if args.fit or args.statistics == 'later':
        if not rank:
            dump(out/'fit-results.json',dict(models=fits,settings={k:getattr(args,k) for k in APPROVED},bmu=args.bmu))
            os.rename(out,args.out/'results')
        if comm is not None: comm.Barrier()  # the other ranks exit once rank 0 has published
        return
    if args.collect and len(fits) != len(collected): raise ValueError('unexpected collected lattice')
    attrs=args.anchor_attrs
    physics={k:attrs.get(k,UNAVAILABLE) for k in ['b2_min','s_v','accuracy','epsilon','exponent','gamma_K','C_B_box',
                                               'direction_norm','horizon_mask','horizon_centres','horizon_radius','horizon_mass','horizon_spin']}
    physics['winsor']='none'
    extraction_commit=attrs.get('git_commit','unknown')
    from aweSOM.stream import peak_rss_bytes
    manifest=dict(schema_version=1,execution_plan=dict(**execution_plan,auto_threads=auto_threads),feature_names=records[0]['names'],pipeline='v2',
                  end_to_end_wall_seconds=time.perf_counter()-started,
                  parent_peak_rss_bytes=peak_rss_bytes(),
                  concurrent_peak_rss_upper_bound_bytes=peak_rss_bytes()+execution_plan['seed_workers']*max(r['peak_rss_bytes'] for r in fits),
                  implementation_sha256=digest(Path(__file__)),
                  etk_analysis_commit=extraction_commit if extraction_commit!='unknown' else UNAVAILABLE,
                  current_local_etk_checkout_commit=commit('/Volumes/ExternalNVMe/etk-analysis') if Path('/Volumes/ExternalNVMe/etk-analysis/.git').exists() else UNAVAILABLE,
                  physics=physics,scaler=scaling,models=fits,selection=choices,
                  sample_manifest=dict(snapshots=records,anchor_ID=args.anchor,held_out_IDs=args.held_out,
                                       held_out_evaluation={'status':'not available','reason':'named snapshots reserved; no held-out feature inputs supplied'},
                                       equal_snapshot_quota=quota,levels='extractor p_l=min(1,c*dx_l**2), then proportional quotas',
                                       epoch_sampling='exact equal snapshot quotas; floor level quotas with seeded dependent fractional-seat rounding each epoch; fresh uniform cyclic window per stratum; callable setup consumes first two draws',
                                       extraction_seeds=[r['attrs'].get('seed',UNAVAILABLE) for r in records],
                                       volume_weights='input weight_volume; selected weight hashes recorded; effective inverse probability includes pool quota / extracted count',
                                       row_order_sha256=scaling['row_order_sha256']),
                  aweSOM_commit=args.code_commit,input_hashes={r['path']:r['sha256'] for r in records},disk=disk,
                  diagnostic_only=bool(diagnostic_reasons(args,records)),diagnostic_reasons=diagnostic_reasons(args,records),
                  threshold_override_mismatches=mismatches,
                  verification={r['tag']:r['verification'] for r in fits})
    dump(out/'freeze-manifest.json',manifest)
    os.rename(out,args.out/'results')
    print(json.dumps(plain(choices)),flush=True)


if __name__=='__main__':
    if sys.argv[1:] == ['--verify-source-only']: print(provenance())
    else: main()
