"""Train and verify frozen SOMs from equal, level-stratified snapshot quotas.

Inputs are read only. Outputs require a fresh directory. See --help for overrides.
The orchestration runs each fit in a fresh worker, as in cluster/kit/check7.py.
"""
from __future__ import annotations

import copy
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
# Bound native libraries before NumPy is imported, including in spawned workers.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import h5py
import numpy as np

NAMES = ['log_beta_inv', 'log_theta', 'log_chi_B', 'log_grad_ln_b2',
         'asinh_div_v', 'bz_frac', 'pol_frac']
UNAVAILABLE = {'status': 'not available', 'reason': 'not recorded by the extraction'}


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
    need = max(args.min_free_gb*1e9, rows*(len(NAMES)*12 + 40) + 1e9)
    free = shutil.disk_usage(args.out.parent).free
    if free < need: raise ValueError(f'Low disk: {free} bytes free; need {need}')
    return records, quota, dict(free_bytes=free, required_bytes=need), mismatches


def quantiles(dataset, start=0, stop=None):
    """Exact quartiles using one disk-backed column and chunked HDF5 reads."""
    stop = len(dataset) if stop is None else stop
    path = Path(dataset.file.filename).with_name('scaler-column.npy')
    column = np.lib.format.open_memmap(path, mode='w+', dtype='f8', shape=(stop-start,))
    result = []
    for j in range(dataset.shape[1]):
        for a in range(start,stop,2**18):
            column[a-start:min(a+2**18,stop)-start] = dataset[a:min(a+2**18,stop),j]
        result.append(np.percentile(column, [25,50,75], overwrite_input=True))
    del column
    path.unlink()
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
        q = quantiles(raw); center, unweighted = q[1], q[2]-q[0]
        from aweSOM.model import Scaler
        feature_weights = np.array([args.bz_weight if n == 'bz_frac' else
                                    args.pol_weight if n == 'pol_frac' else 1
                                    for n in records[0]['names']])
        scaler = Scaler('median-iqr', center, unweighted / feature_weights)
        drift = []
        for i, rec in enumerate(records):
            sq = quantiles(raw, i*quota, (i+1)*quota)
            # Median drift relative to |pooled median|, with pooled IQR as zero guard.
            md = np.abs(sq[1]-center)/np.maximum(np.abs(center), np.maximum(unweighted, 1e-30)*1e-12)
            iq = np.abs(sq[2]-sq[0]-unweighted)/np.maximum(unweighted, 1e-30)
            drift.append(dict(snapshot=rec['snapshot'], median_relative=md, iqr_relative=iq,
                              flagged_features=np.array(records[0]['names'])[np.maximum(md,iq)>.05].tolist(),
                              quantiles=sq))
        scaled = dst.create_dataset('scaled', raw.shape, dtype='f8')
        for a in range(0, len(raw), args.chunk): scaled[a:a+args.chunk] = scaler.transform(raw[a:a+args.chunk])
    return dict(center=center, scale=scaler.scale, scale_unweighted=unweighted,
                feature_weights=feature_weights, kind='median-iqr', drift=drift,
                strata=strata, row_order_sha256=order_hash.hexdigest())


class EpochSource:
    """Fresh cyclic windows per physical stratum; exact equal snapshot budgets."""
    def __init__(self, data, strata, rows, seed, chunk):
        self.data, self.strata, self.seed, self.chunk = data, strata, seed, chunk
        self.calls = 0
        snapshots = list(dict.fromkeys(s['snapshot'] for s in strata))
        if rows > len(data) or rows < len(snapshots) or rows % len(snapshots):
            raise ValueError('rows per epoch must be <= pool and divisible by snapshot count')
        self.rows = rows
        self.groups = [[i for i,s in enumerate(strata) if s['snapshot'] == snap] for snap in snapshots]
        if any(rows//len(snapshots) < len(ix) for ix in self.groups):
            raise ValueError('epoch budget smaller than populated strata per snapshot')
        self.quotas = np.zeros(len(strata), dtype=int)

    def __call__(self):
        rng = np.random.default_rng([self.seed, self.calls]); self.calls += 1
        for ix in self.groups:
            self.quotas[ix] = randomized_allocate([self.strata[i]['rows'] for i in ix],
                                                  self.rows//len(self.groups), rng)

        for s, q in zip(self.strata, self.quotas):
            pos = int(rng.integers(s['rows']))
            while q:
                n = min(int(q), self.chunk, s['rows']-pos)
                yield self.data[s['start']+pos:s['start']+pos+n]
                q -= n; pos = (pos+n) % s['rows']


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


def export_statistics(raw, nodes, distances, labels, n_nodes, out, tag, chunk):
    """Exact float64 statistics with one sequential pass over the raw rows.

    Stable grouping reproduces the old counting scatter's input order, including
    its per-node partitioning before the global rejection percentile. Keep the
    float64 column buffer separate from raw so median partitioning cannot change
    another column. No scratch files are created.

    RAM is O(N*D + N + chunk*D + groups): raw retains its input dtype, one intp
    permutation and one float64 column are live during feature statistics.
    See statistics-performance.md for conservative production memory estimates.
    """
    order, edges = grouped_order(nodes, n_nodes)
    buf = np.asarray(distances, dtype='f8')[order]
    q = np.full(n_nodes, np.nan)
    for i in range(n_nodes):
        if edges[i+1] > edges[i]:
            q[i] = np.percentile(buf[edges[i]:edges[i+1]], 99., overwrite_input=True)
    reject = float(np.percentile(buf, 99.9, overwrite_input=True))
    del buf, order

    # Copy mapped labels once, then release them before loading the raw matrix.
    keys = np.array(labels)
    medians = {int(c): [] for a in range(0,len(keys),chunk) for c in np.unique(keys[a:a+chunk])}
    order, edges = grouped_order(keys, int(max(medians))+1)
    del keys
    values = np.empty(raw.shape, dtype=raw.dtype)
    for a in range(0, len(raw), chunk):
        values[a:a+chunk] = raw[a:a+chunk]
    buf = np.empty(len(raw), dtype='f8')
    for j in range(raw.shape[1]):
        for a in range(0, len(raw), chunk):
            buf[a:a+chunk] = values[order[a:a+chunk], j]
        for c in medians:
            medians[c].append(float(np.median(buf[edges[c]:edges[c+1]], overwrite_input=True)))
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


def fit(args, shape, seed, scaling, out):
    from aweSOM import Lattice, SOMModel, Scaler
    from aweSOM.stream import train_batch_stream, peak_rss_bytes
    from aweSOM.inference import best_match_nodes_patch
    t = time.perf_counter()
    tag = f'{shape[0]}x{shape[1]}-s{seed}'
    clock = PhaseClock(tag)
    execution = execution_info()
    print(json.dumps(dict(tag=tag, execution=execution)), flush=True)
    scaler = Scaler('median-iqr', np.array(scaling['center']), np.array(scaling['scale']))
    with h5py.File(out / 'pool.h5', 'r') as h:
        raw, scaled = h['raw'], h['scaled']
        source = EpochSource(scaled, scaling['strata'], args.rows_per_epoch or len(raw), args.sample_seed, args.chunk)
        w, info = train_batch_stream(source, *shape, epochs=args.epochs, chunk=args.chunk,
                                    bmu='patch', seed=seed, groups=args.groups, r_max=args.r_max,
                                    r_min=args.r_min if args.r_min is not None else (8 if shape[0] >= 91 else 5),
                                    final_pass='none', n_samples=args.rows_per_epoch or len(raw), verbose=True)
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
            nodes[a:b], d2 = best_match_nodes_patch(w, scaled[a:b], *shape)
            labels[a:b] = nc[nodes[a:b]]; distances[a:b] = np.sqrt(d2)
        clock.finish('labelling')
        q, reject, medians = export_statistics(raw, nodes, distances, labels, len(w), out, tag, args.chunk)
        clock.finish('statistics')
        del nodes, distances
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in h['names'][:]]
        chi, beta = names.index('log_chi_B'), names.index('log_beta_inv')
        sheets = [c for c,m in medians.items() if m[chi] > args.sheet_chi and m[beta] < args.sheet_beta]
        bz = names.index('bz_frac') if 'bz_frac' in names else None
        tubes = [c for c,m in medians.items() if bz is not None and m[bz] > args.tube_bz and m[beta] > args.tube_beta]
        merge_settings = dict(merge_cost_used=lat.merge_cost_used,
                              target_clusters=args.target_clusters, n_clusters=lat.n_clusters)
        meta = dict(**merge_settings, seed=str(seed), scale_unweighted=json.dumps(plain(scaling['scale_unweighted'])),
                    sheet_clusters=','.join(map(str,sheets)), sheet_rule_statistic='median',
                    sheet_rule=f'median log_chi_B > {args.sheet_chi} and median log_beta_inv < {args.sheet_beta}',
                    tube_clusters=','.join(map(str,tubes)), tube_rule=f'characterisation only: median bz_frac > {args.tube_bz} and median log_beta_inv > {args.tube_beta}',
                    feature_settings=json.dumps(plain(args.anchor_attrs)),
                    feature_weights=json.dumps(plain(scaling['feature_weights'])),
                    b2_min=str(args.anchor_attrs['b2_min']), s_v=str(args.anchor_attrs['s_v']),
                    pipeline='v2', freeze_manifest='freeze-manifest.json', aweSOM_commit=args.code_commit)
        from aweSOM.features_grmhd import FeatureThresholds
        thresholds = FeatureThresholds.from_h5_attrs(args.anchor_attrs)
        thresholds.feature_names = tuple(names)
        thresholds.__post_init__()
        meta.update(thresholds.to_metadata())
        model = SOMModel(*shape,w,nc,names,scaler,q,reject,metadata=meta)
        temp = out / f'{tag}.partial.h5'; model.save(temp); publish(temp,out / f'{tag}.h5')
        clock.finish('export')
        loaded = SOMModel.load(out / f'{tag}.h5')
        mismatch = parity = 0; scaler_error = 0.
        for a in range(0,len(raw),args.chunk):
            block = raw[a:a+args.chunk]
            mismatch += int(np.count_nonzero(loaded.predict(block,backend='patch') != labels[a:a+len(block)]))
            scaler_error = max(scaler_error,float(np.max(np.abs(loaded.scaler.transform(block)-scaled[a:a+len(block)]))))
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
                          reject_distance=reject == loaded.reject_distance)
        if mismatch or parity or scaler_error > 1e-4 or bmu_mismatches or not distance_parity or not all(components.values()):
            raise RuntimeError('freeze verification failed')
        node_path.unlink(); distance_path.unlink()
        labels.flush()
        clock.finish('verification')
    rec = dict(tag=tag, lattice_shape=shape, seed=seed, init='sampling', sample_seed=args.sample_seed,
               epochs=args.epochs, rows_per_epoch=info['rows_per_epoch'], radius_schedule=info['radius'],
               chunk=args.chunk, groups=args.groups, bmu_backend=info['bmu_kind'], fallback_count=info['fallbacks'],
               execution=execution, phase_timings=clock.records, training_info=info, wall_seconds=time.perf_counter()-t, peak_rss_bytes=peak_rss_bytes(),
               weights_dtype=str(w.dtype), weights_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
               node_to_cluster=nc, merge_cost=args.merge_cost, **merge_settings, cluster_medians=medians,
               sheet_rule=dict(statistic='median',log_chi_B_gt=args.sheet_chi,log_beta_inv_lt=args.sheet_beta,IDs=sheets),
               tube_rule=dict(status='characterisation only',bz_frac_gt=args.tube_bz,log_beta_inv_gt=args.tube_beta,IDs=tubes),
               reject_distance=reject, umat_file=f'{tag}-umat.npy',
               verification=dict(reload_mismatches=mismatch,cluster_label_patch_screen_mismatches=parity,
                                 bmu_node_mismatches=bmu_mismatches, bmu_distance_rtol_1e6=distance_parity,
                                 saved_components_equal=components,
                                 parity_rows=len(probe),scaler_max_abs_error=scaler_error))
    dump(out / f'{tag}.json',rec)
    return plain(rec)


def worker(conn, *args):
    try:
        import numba
        numba.set_num_threads(args[0].threads_per_seed)
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
                process = ctx.Process(target=target, args=(send, *job)); running.append((receive, process)); process.start()
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
    p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--anchor',required=True); p.add_argument('--held-out',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rows-per-snapshot',type=int); p.add_argument('--rows-per-epoch',type=int,default=100_000_000)
    p.add_argument('--lattices',nargs='+',default=['91x64','48x34']); p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--seed-workers',type=int,default=1,help='concurrent spawned seeds within each lattice')
    p.add_argument('--threads-per-seed',type=int,help='Numba threads for the entire worker, including export and verification')
    p.add_argument('--epochs',type=int,default=40); p.add_argument('--chunk',type=int,default=2**21)
    p.add_argument('--sample-seed',type=int,default=42); p.add_argument('--groups',type=int,default=64)
    p.add_argument('--target-clusters',type=int,default=10,help='merge to at most this count; ties may yield fewer')
    p.add_argument('--merge-cost',type=float,default=.05,help='merge cost floor; keep the selected cost for subsequent training'); p.add_argument('--r-max',type=float); p.add_argument('--r-min',type=float)
    p.add_argument('--bz-weight',type=float,default=2.); p.add_argument('--pol-weight',type=float,default=2.)
    p.add_argument('--sheet-chi',type=float,default=1.); p.add_argument('--sheet-beta',type=float,default=.5)
    p.add_argument('--tube-bz',type=float,default=.7); p.add_argument('--tube-beta',type=float,default=0.)
    p.add_argument('--min-free-gb',type=float,default=10)
    p.add_argument('--allow-independent-thresholds',action='store_true',help='diagnostic only; manifest cannot be production accepted')
    p.add_argument('--allow-six-features',action='store_true',help='legacy local rehearsal only')
    return p


# The approved production training settings (production_pool.sbatch defaults). A run that departs
# from any of them is a rehearsal: its manifest is marked diagnostic_only with the reasons.
# r_min=8 on every lattice and target_clusters=10 approved by the user on 2026-10-29.
APPROVED = dict(rows_per_snapshot=10_000_000, rows_per_epoch=20_000_000, lattices=['91x64','48x34'],
                seeds=[42,43,44], epochs=40, chunk=2**21, sample_seed=42, groups=64, merge_cost=.05, target_clusters=10,
                r_max=None, r_min=8., bz_weight=2., pol_weight=2., sheet_chi=1., sheet_beta=.5,
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


def main(argv=None):
    started = time.perf_counter()
    args = parser().parse_args(argv)
    if args.out.exists(): raise ValueError('OUT exists; use a fresh output directory')
    if args.chunk < 1 or args.epochs < 1 or not args.seeds: raise ValueError('invalid training settings')
    if not all(np.isfinite(v) and v > 0 for v in [args.bz_weight,args.pol_weight]):
        raise ValueError('feature weights must be finite and positive')
    shapes = [tuple(map(int,s.split('x'))) for s in args.lattices]
    if any(len(s)!=2 or min(s)<2 for s in shapes): raise ValueError('invalid lattice')
    if len(set(args.seeds)) != len(args.seeds): raise ValueError('duplicate seeds')
    auto_threads = args.threads_per_seed is None
    execution_plan = thread_budget(args)
    args.threads_per_seed = execution_plan['threads_per_seed']
    args.code_commit = provenance()
    records,quota,disk,mismatches = inspect(args)
    args.out.mkdir()
    os.environ.setdefault('NUMBA_CACHE_DIR',str(args.out/'numba-cache'))
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    args.anchor_attrs=next(r['attrs'] for r in records if r['snapshot']==args.anchor)
    out=args.out/'results.partial'; out.mkdir()
    scaling=pool(args,records,quota,out)
    fits=[]; choices=[]
    for shape in shapes:
        trio=[]
        width=execution_plan['seed_workers']
        for start in range(0,len(args.seeds),width):
            seeds=args.seeds[start:start+width]
            wave_args=copy.copy(args)
            if auto_threads:
                import numba
                wave_args.threads_per_seed=min(execution_plan['allocated_cpus']//len(seeds),
                                               numba.config.NUMBA_NUM_THREADS)
            trio.extend(run_wave([(wave_args,shape,seed,scaling,out) for seed in seeds]))
        idx,band=medoid([out/f"{r['tag']}-labels.npy" for r in trio],args.chunk)
        chosen=trio[idx]
        os.link(out/f"{chosen['tag']}.h5",out/f'frozen-{shape[0]}x{shape[1]}.h5')
        choices.append(dict(shape=shape,medoid_seed=chosen['seed'],model=f'frozen-{shape[0]}x{shape[1]}.h5',**band))
        fits.extend(trio)
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
