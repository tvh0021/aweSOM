"""Bucketed cuda_screen search: same nodes and distances as the dense kernel, bit for bit.

Three layers:

* CPU models (:mod:`aweSOM._gpu_bucket`): seed, bucketed pass with the float32
  exclusion test, fb_screen and fb_exact, emulated with the kernels' rounding.
  They must equal :func:`aweSOM.inference.best_match_nodes` for every row on
  random, trained-like, tied, duplicate-node, overflow and non-finite inputs,
  for any candidate-list lengths (pilot-tuned, random or none).
* The stream-ordered fake CuPy of ``gpu_pipeline_test``, with every bucket
  kernel run by its CPU model when its stream is synchronized. Training and
  labelling through ``cuda_screen`` must equal the dense path (and numba),
  including table rebuilds; this checks the hook, the buffers, the row order
  and the stream ordering.
* ``test_device*``: the real kernels against the dense kernel (skipped
  without CuPy and a CUDA device).

The kernel source itself runs on the CPU in ``gpu_bucket_host_test``.
"""
import gc
import hashlib
import sys
import weakref

import numpy as np
import pytest
from gpu_pipeline_test import Device, NullStream, Stream, _fake_cupy, _run

from aweSOM import _gpu_bucket as B
from aweSOM import _gpu_screen, stream
from aweSOM.inference import best_match_nodes
from aweSOM.stream import predict_stream, train_batch_stream


def _bits(a):
    return np.asarray(a, np.float64).view(np.uint64)


def _same(got, want):
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(_bits(got[1]), _bits(want[1]))


# ----------------------------------------------------------------------------- inputs
def _grid(side, F, rng, noise=0.0):
    """A trained-like lattice: a smooth sheet through F dimensions."""
    a, b = np.meshgrid(np.linspace(-1, 1, side), np.linspace(-1, 1, side))
    a, b = a.ravel(), b.ravel()
    cols = [np.sin((f + 1) * a + 0.3 * f) + np.cos((f % 3 + 1) * b) * 0.7 + 0.2 * f * a * b for f in range(F)]
    w = np.column_stack(cols)
    return w + rng.normal(size=w.shape) * noise


def lattice(kind, F, seed=0):
    rng = np.random.default_rng(seed)
    if kind == 'random':
        return rng.normal(size=(397, F))
    if kind == 'trained':
        return _grid(20, F, rng, 1e-3)
    if kind == 'tied':  # integer nodes: many exact ties between distinct nodes, and duplicates
        return rng.integers(-2, 3, size=(300, F)).astype(np.float64)
    if kind == 'dup':  # copies of node 0 in other lanes, warps and fallback tiles
        w = _grid(20, F, rng, 1e-3)
        for j in (1, 31, 33, 257, 399):
            w[j] = w[0]
        w[64] = w[63]
        return w
    if kind == 'big':  # |w| ~ 1e154: unsafe for the float32 screen, all rows take fb_exact
        return rng.normal(size=(150, F)) * 1e154
    if kind == 'f32max':  # beyond float32 squares: unsafe
        return rng.uniform(-3e38, 3e38, size=(150, F))
    if kind == 'nonfinite':
        w = rng.normal(size=(150, F))
        w[7, 0] = np.inf
        w[9, -1] = np.nan
        return w
    if kind == 'edge':  # largest safe weights: float32 squares near 1e30
        return rng.normal(size=(200, F)) * 1e14
    raise ValueError(kind)


def rows(w, n, seed=1):
    """Rows near nodes, on midpoints (ties), at group centroids, far away, and overflow/non-finite."""
    rng = np.random.default_rng(seed)
    nodes, F = w.shape
    finite = np.isfinite(w).all(axis=1)
    base = w[finite] if finite.any() else np.zeros((1, F))
    scale = np.nan_to_num(base.std(axis=0), nan=1.0, posinf=1.0) + 1e-12
    out = [base[rng.integers(len(base), size=n)] + rng.normal(size=(n, F)) * scale * 0.02]
    i, j = rng.integers(len(base), size=(2, n // 4))
    out.append((base[i] + base[j]) / 2)  # exact midpoints: equal distances in exact arithmetic
    out.append(base[rng.integers(len(base), size=n // 8)])  # exact node copies, d2 == 0
    out.append(rng.normal(size=(n // 8, F)) * scale * 10)  # far rows
    special = np.zeros((10, F))
    special[0] = np.nan
    special[1] = np.inf
    special[2] = -np.inf
    special[3] = 1e200
    special[4] = -1e200
    special[5] = 1e160
    special[6] = 3e38
    special[7] = -3e38
    special[8, 0] = np.nan  # one NaN feature
    special[9] = 1e18
    out.append(special)
    return np.concatenate(out)


def reference(w, x):
    return best_match_nodes(w, x)


def bucket(w, x, pilot=None, K=None, rare=None):
    """CPU model of the device path with tables tuned on ``pilot`` (default: ``x``), or with
    explicit list lengths ``K`` and rare-bucket mask."""
    w = np.ascontiguousarray(w, np.float64)
    with np.errstate(over='ignore'):  # float32 tables of unsafe lattices (modelled only)
        if K is None:
            T = B.build_tables(w, x if pilot is None else pilot)
        else:
            grp = B.build_groups(w)
            sc = B.screen_scalars(w, grp)
            T = B.pack(w, grp, sc, np.asarray(K, np.int64), np.asarray(rare, bool))
            T.update(sc=sc, grp=grp)
        idx, d2, info = B._bucket_reference(w, x, T)
    return (idx, d2), info, T


# ----------------------------------------------------------------------------- CPU models
@pytest.mark.parametrize('F', [1, 2, 3, 5, 8, 10])
@pytest.mark.parametrize('kind', ['random', 'trained', 'tied', 'dup', 'big', 'f32max', 'nonfinite', 'edge'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_model_matches_reference(kind, F, dtype):
    w = lattice(kind, F)
    with np.errstate(over='ignore', invalid='ignore'):
        x = rows(w, 400).astype(dtype)
    got, info, T = bucket(w, x)
    _same(got, reference(w, x))
    certified = int((info['code'] == 0).sum())
    if kind == 'trained' and F >= 2:  # the bucketed pass, not only the fallback, decides most rows
        assert certified > 0.6 * len(x)
    if kind in ('big', 'f32max', 'nonfinite'):
        assert T['sc']['safe'] == 0 and certified == 0


@pytest.mark.parametrize('kind', ['random', 'trained', 'tied', 'dup'])
@pytest.mark.parametrize('pilot', ['shifted', 'tiny', 'random-K', 'no-lists', 'all-rare'])
def test_model_any_lists(kind, pilot):
    """Results never depend on the pilot rows or the list lengths."""
    F = 4
    w = lattice(kind, F)
    x = rows(w, 300)
    rng = np.random.default_rng(7)
    grp = B.build_groups(w)
    P = grp['P']
    if pilot == 'shifted':  # pilot from elsewhere: the lists fit other rows
        got, info, _ = bucket(w, x, pilot=x[:50] + 5.0)
    elif pilot == 'tiny':  # most buckets rare
        got, info, _ = bucket(w, x, pilot=x[:9])
    elif pilot == 'random-K':
        got, info, _ = bucket(w, x, K=rng.integers(1, P + 1, size=P), rare=rng.random(P) < 0.3)
    elif pilot == 'no-lists':  # every bucket screens only its own group
        got, info, _ = bucket(w, x, K=np.ones(P, np.int64), rare=np.zeros(P, bool))
    else:
        got, info, _ = bucket(w, x, K=np.full(P, P), rare=np.ones(P, bool))
    _same(got, reference(w, x))


def test_pack_covers_every_node_once():
    w = lattice('trained', 5)
    grp = B.build_groups(w)
    P = grp['P']
    rng = np.random.default_rng(3)
    K = rng.integers(1, P + 1, size=P)
    rare = rng.random(P) < 0.25
    T = B.pack(w, grp, B.screen_scalars(w, grp), K, rare)
    sizes = grp['sizes']
    for p in range(P):
        cand = T['cid'][T['cand_lo'][p]:T['cand_hi'][p]]
        ex = np.arange(T['ex_lo'][p], T['ex_hi'][p])
        if rare[p]:
            np.testing.assert_array_equal(cand, np.arange(len(w)))
            assert len(ex) == 0
            continue
        groups = grp['order'][p, :K[p]]
        assert groups[0] == p
        assert len(cand) == sizes[groups].sum() and len(np.unique(cand)) == len(cand)
        assert len(cand) + sizes[grp['order'][p, K[p]:]].sum() == len(w)
        np.testing.assert_array_equal(T['ex32'][ex, :5], grp['cent'][grp['order'][p, K[p]:]].astype(np.float32))
        assert np.all(T['ex32'][ex, 5].astype(np.float64) >= grp['rad'][grp['order'][p, K[p]:]])
        np.testing.assert_array_equal(T['cw'][T['cand_lo'][p]:T['cand_hi'][p], :5], w[cand].astype(np.float32))


def test_directed_rounding_models():
    """The models of __fadd_ru, __fmul_ru and __double2float_ru round upward, checked exactly."""
    from fractions import Fraction

    def up(r, e):
        assert np.float64(np.float32(r)) == np.float64(r) or not np.isfinite(r)  # a float32 value
        r = np.float32(r)
        with np.errstate(over='ignore'):
            below = np.nextafter(r, np.float32(-np.inf))
        return Fraction(float(r)) >= e and (not np.isfinite(below) or Fraction(float(below)) < e)

    rng = np.random.default_rng(0)
    a = (rng.normal(size=2000) * 10.0 ** rng.integers(-30, 30, 2000)).astype(np.float32)
    b = (rng.normal(size=2000) * 10.0 ** rng.integers(-30, 30, 2000)).astype(np.float32)
    b[:200] = a[:200] * np.float32(-1.0000001)  # cancellation
    for x, y in zip(a, b):
        s = B._fadd_ru(x, y)
        if np.isfinite(s):
            assert up(s, Fraction(float(x)) + Fraction(float(y)))
        p = B._fmul_ru(x, y)
        if np.isfinite(p):
            assert up(p, Fraction(float(x)) * Fraction(float(y)))
    for v in rng.normal(size=500) * 10.0 ** rng.integers(-40, 39, 500):
        r = B._ru32(v)
        if np.isfinite(r):
            assert up(r, Fraction(float(v)))


def test_tune_k_prefers_short_lists():
    w = _grid(60, 5, np.random.default_rng(0), 1e-3)
    x = rows(w, 4000)[:4000]
    T = B.build_tables(w, x)
    assert T['K'].mean() < T['P'] / 2 and T['cand_len'].mean() < len(w) / 2


# ----------------------------------------------------------------------------- fake CuPy
class _Pointer:
    """A device argument as a kernel holds it: the memory at launch (``.a``), and a weak reference
    to its allocation, which must still be alive when the kernel runs."""

    def __init__(self, device):
        self.a = device.a
        while device.base is not None:
            device = device.base
        self.alive = weakref.ref(device)


def _fake_bucket_kernels(model_launches):
    """CPU models of the bucket kernels, run when their stream is synchronized (fake CuPy)."""

    def seed(args, stream):
        c32, x, out, n, P, zero, nzero = args

        def op():
            zero.a[:int(nzero)] = 0
            s = np.empty(int(n), np.int64)
            F = x.a.shape[1]
            B._seed_rows(np.ascontiguousarray(c32.a[:, :F]), np.ascontiguousarray(x.a[:int(n)]), s)
            out.a[:int(n)] = s
        return op

    def hist(args, stream):
        seed_, n, P, counts = args
        return lambda: counts.a.__iadd__(np.bincount(seed_.a[:int(n)], minlength=int(P)).astype(np.int32))

    def scan(args, stream):
        counts, P, rb, boff, cursor, toff = args

        def op():
            c = counts.a[:int(P)].astype(np.int64)
            boff.a[:] = np.concatenate([[0], np.cumsum(c)])
            toff.a[:] = np.concatenate([[0], np.cumsum(-(-c // int(rb)))])
            cursor.a[:] = boff.a[:int(P)]
        return op

    def scatter(args, stream):
        seed_, n, P, cursor, perm = args

        def op():
            s = seed_.a[:int(n)]
            for b in range(int(P)):
                rows_ = np.nonzero(s == b)[0][::-1]  # any order inside a bucket: here reversed
                perm.a[cursor.a[b]:cursor.a[b] + len(rows_)] = rows_
                cursor.a[b] += len(rows_)
        return op

    def bpass(args, stream):
        (w, x, perm, boff, toff, cw, cid, clo, chi, ex32, elo, ehi, idx, d2, fb_rows, counters, rare,
         P, ct, wmax, cmax, g1, safe, eta, one_m_eps) = args

        def op():
            n = int(boff.a[int(P)])
            r = perm.a[:n].astype(np.int64)
            bk = np.repeat(np.arange(int(P)), np.diff(boff.a[:int(P) + 1]))
            code = np.empty(n, np.int8)
            j1 = np.empty(n, np.int64)
            a = np.empty(n)
            B._pass2_rows(w.a, cw.a, cid.a, clo.a, chi.a, ex32.a, elo.a, ehi.a, x.a, r, bk, float(wmax),
                          float(cmax), np.float32(g1), int(safe), code, j1, a)
            ok = code == 0
            idx.a[r[ok]] = j1[ok]
            d2.a[r[ok]] = a[ok]
            bad = r[~ok]
            c0 = counters.a[0]
            fb_rows.a[c0:c0 + len(bad)] = bad[::-1]  # list order is arbitrary on the device too
            counters.a[0] += len(bad)
            counters.a[1] += int((code == 1).sum())
            counters.a[3] += int((~ok & (rare.a[bk] != 0)).sum())
        return op

    def fb_screen(args, stream):
        w, x, lst, cnt, idx, d2, list2, cnt2, nodes, wmax, safe = args

        def op():
            m = int(cnt.a[0])
            r = lst.a[:m].astype(np.int64)
            cert = np.empty(m, np.int8)
            j1 = np.empty(m, np.int64)
            a = np.empty(m)
            B._fb_screen_rows(w.a, x.a, r, float(wmax), int(safe), cert, j1, a)
            ok = cert == 1
            idx.a[r[ok]] = j1[ok]
            d2.a[r[ok]] = a[ok]
            c = int(cnt2.a[0])
            list2.a[c:c + int((~ok).sum())] = r[~ok]
            cnt2.a[0] += int((~ok).sum())
        return op

    def fb_exact(args, stream):
        w, x, lst, cnt, idx, d2, nodes = args

        def op():
            m = int(cnt.a[0])
            r = lst.a[:m].astype(np.int64)
            i = np.empty(m, np.int64)
            d = np.empty(m)
            B._fb_exact_rows(w.a, x.a, r, i, d)
            idx.a[r] = i
            d2.a[r] = d
        return op

    table = {'seed': seed, 'bucket_hist': hist, 'bucket_scan': scan, 'bucket_scatter': scatter, 'bpass': bpass,
             'fb_screen': fb_screen, 'fb_exact': fb_exact}

    def make(name):
        key = name if name in table else name.rsplit('_', 1)[0]
        build = table[key]

        def launch(grid, block, args, shared_mem, stream):
            model_launches.append(name)
            args = tuple(_Pointer(a) if isinstance(a, Device) else a for a in args)
            op = build(args, stream)

            def checked():
                freed = [i for i, a in enumerate(args) if isinstance(a, _Pointer) and a.alive() is None]
                assert not freed, f'{name}: arguments {freed} were freed before the kernel ran'
                op()
            _run(stream, checked)  # on a slot or pilot stream, never the null stream
        return launch
    return make


class _Launches(list):
    """Names of the bucket kernels launched; ``attrs``: device attributes queried; ``cp``: the fake."""


@pytest.fixture
def fake_bucket(monkeypatch):
    """The fake CuPy with the bucket kernels, eligible from tiny lattices (groups of 4 nodes)."""

    def install(chunk=2**20, overlap=True, major=9, shared=49152):
        cp, launches, busy, pinned_fail = _fake_cupy()
        attrs = []
        cp.cuda.runtime.deviceGetAttribute = lambda attr, device: attrs.append(attr) or {75: major}[attr]
        cp.cuda.runtime.getDeviceProperties = lambda _: dict(sharedMemPerBlock=shared, maxThreadsPerBlock=1024)
        names = _Launches()
        names.attrs = attrs
        dense_raw, bucket_raw = cp.RawKernel, _fake_bucket_kernels(names)
        cp.RawKernel = lambda source, name, options: (bucket_raw(name) if name in B.KERNEL_NAMES
                                                      else dense_raw(source, name, options))
        Stream.created.clear()
        NullStream.queue.clear()
        monkeypatch.setitem(sys.modules, 'cupy', cp)
        names.cp = cp
        _gpu_screen._kernels.cache_clear()
        B._kernels.cache_clear()
        monkeypatch.setattr(_gpu_screen, '_BUCKET_MIN_NODES', 0)
        monkeypatch.setattr(B, 'LEAF', 4)
        base = _gpu_screen.GpuScreenSearcher
        small = type('Small', (base,), {'__init__': lambda self, *a, **k:
                     base.__init__(self, *a, chunk=chunk, overlap=overlap, **k)})
        monkeypatch.setattr(_gpu_screen, 'GpuScreenSearcher', small)
        B.reset_totals()
        return names
    yield install
    _gpu_screen._kernels.cache_clear()
    B._kernels.cache_clear()


def _sha(w):
    return hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest()


def _train_data(n=1200, F=5, dtype=np.float64, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, F)) * rng.uniform(0.5, 2.0, F)
    x[::7] = x[3]  # repeated rows: ties between nodes once the lattice holds them
    return x.astype(dtype)


TRAIN = dict(xdim=6, ydim=5, epochs=3, phases=(1, 1, 1), r_min=1, n_threads=2, bmu='cuda_screen')


def _train(monkeypatch, x, bucket_on, pipeline, **kw):
    monkeypatch.setattr(_gpu_screen, '_BUCKET', bucket_on)
    monkeypatch.setattr(stream, '_GPU_PIPELINE', pipeline)
    return train_batch_stream(x, **{**TRAIN, **kw})


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('pipeline', [True, False])
@pytest.mark.parametrize('chunk,sub', [(400, 2**20), (128, 50), (1200, 300)])
def test_fake_training_equals_dense(fake_bucket, monkeypatch, dtype, pipeline, chunk, sub):
    names = fake_bucket(sub)
    x = _train_data(dtype=dtype)
    kw = dict(chunk=chunk, prefetch=False, sample_fraction=1.0)
    w0, info0 = _train(monkeypatch, x, False, pipeline, **kw)
    assert not names
    w1, info1 = _train(monkeypatch, x, True, pipeline, **kw)
    assert names.count('bpass_' + ('f64' if dtype == np.float64 else 'f32')) >= 3  # one or more per epoch
    assert _sha(w1) == _sha(w0)
    np.testing.assert_array_equal(info1['hits'], info0['hits'])
    np.testing.assert_array_equal(_bits(info1['qe']), _bits(info0['qe']))
    w2, _ = train_batch_stream(x, **{**TRAIN, 'bmu': 'numba'}, **kw)
    assert _sha(w1) == _sha(w2)
    # one searcher per epoch plus the final QE pass, each over every row
    assert B.TOTALS['engines'] >= 3 and B.TOTALS['rows'] == B.TOTALS['engines'] * len(x)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('chunk,sub', [(97, 40), (1000, 2**20)])
def test_fake_labels_equal_reference(fake_bucket, monkeypatch, dtype, chunk, sub):
    names = fake_bucket(sub)
    rng = np.random.default_rng(2)
    w = _grid(6, 5, rng, 1e-3)
    with np.errstate(over='ignore'):
        x = rows(w, 600).astype(dtype)
    clusters = np.arange(len(w)) % 7
    d = np.empty(len(x))
    labels, _ = predict_stream(w, clusters, x, out_distance=d, chunk=chunk, backend='cuda_screen',
                               prefetch=False)
    assert 'bpass_' + ('f64' if dtype == np.float64 else 'f32') in names
    ri, rd = reference(w, x)
    np.testing.assert_array_equal(labels, clusters[ri])
    np.testing.assert_array_equal(_bits(d), _bits(np.sqrt(rd)))  # out_distance holds distances


def test_fake_rebuild_on_moving_input(fake_bucket, monkeypatch):
    """Rows sorted along the sheet: the first chunk's pilot misses later regions, the counters
    trigger rebuilds, and the results stay exact. The other slot still has a chunk queued on the old
    tables at each rebuild; the fake kernels fail if those tables are freed before it runs."""
    names = fake_bucket(64)
    busy = []
    build = B.BucketEngine._build

    def spy(self, first=None):
        if first is None:  # a rebuild: count the other streams with queued work on the old tables
            busy.append(sum(bool(s.stream.queue) for s in self.scratches if s.stream is not None))
        return build(self, first)
    monkeypatch.setattr(B.BucketEngine, '_build', spy)
    monkeypatch.setattr(B, 'RETUNE_MIN_ROWS', 1)
    monkeypatch.setattr(B, 'RETUNE_GAP', 1)
    monkeypatch.setattr(B, 'RETUNE_FRACTION', 0.01)
    monkeypatch.setattr(B, 'RARE', 2)
    rng = np.random.default_rng(4)
    w = _grid(8, 4, rng, 1e-3)
    x = rows(w, 900)[:900]
    x = x[np.argsort(x[:, 0])]  # moving distribution
    clusters = np.arange(len(w))
    d = np.empty(len(x))
    labels, _ = predict_stream(w, clusters, x, out_distance=d, chunk=128, backend='cuda_screen',
                               prefetch=False)
    ri, rd = reference(w, x)
    np.testing.assert_array_equal(labels, ri)
    np.testing.assert_array_equal(_bits(d), _bits(np.sqrt(rd)))
    assert B.TOTALS['builds'] > 1
    assert B.TOTALS['rows'] == len(x)
    assert names.count('bpass_f64') == -(-len(x) // 64)
    assert busy and max(busy) >= 1


def test_overlap_counted_once(fake_bucket, monkeypatch):
    """60 rows that both fall back (tied duplicate nodes) and land in a whole-lattice bucket are
    under 8 % of a 1024-row chunk: no rebuild. Counted twice they were 11.7 %."""
    fake_bucket(1024)
    monkeypatch.setattr(B, 'RETUNE_GAP', 1)
    rng = np.random.default_rng(3)
    w = _grid(32, 2, rng, 1e-3)
    w[1000] = w[1001]  # a tie for rows on this node
    near = w[rng.integers(0, 20, size=4)]  # ordinary rows near four nodes: busy, listed buckets
    ordinary = lambda n: near[rng.integers(0, 4, size=n)] + rng.normal(size=(n, 2)) * 1e-4  # noqa: E731
    # The second chunk holds the 60 rows; its counters are read two chunks later (two slots).
    x = np.concatenate([ordinary(1024), ordinary(964), np.repeat(w[1001:1002], 60, axis=0)]
                       + [ordinary(1024) for _ in range(3)])
    clusters = np.arange(len(w))
    d = np.empty(len(x))
    labels, _ = predict_stream(w, clusters, x, out_distance=d, chunk=1024, backend='cuda_screen',
                               prefetch=False)
    ri, rd = reference(w, x)
    np.testing.assert_array_equal(labels, ri)
    np.testing.assert_array_equal(_bits(d), _bits(np.sqrt(rd)))
    T = B.TOTALS
    assert T['rare_fallback_rows'] == 60 and T['rare_rows'] >= 60 and T['fallback_rows'] >= 60
    assert T['fallback_rows'] + T['rare_rows'] > B.RETUNE_FRACTION * 1024  # double counting would rebuild
    assert T['fallback_rows'] + T['rare_rows'] - T['rare_fallback_rows'] <= B.RETUNE_FRACTION * 1024
    assert T['builds'] == 1


def test_overlap_of_chunk_on_old_tables(fake_bucket, monkeypatch):
    """A chunk still in flight on the old tables at a rebuild is counted with those tables: its
    rare rows include its rare fallback rows (Sol's round-2 repro)."""
    fake_bucket(1024)
    monkeypatch.setattr(B, 'RETUNE_GAP', 1)
    rng = np.random.default_rng(3)
    w = _grid(32, 2, rng, 1e-3)
    w[1000] = w[1001]
    near = w[rng.integers(0, 20, size=4)]
    ordinary = lambda n: near[rng.integers(0, 4, size=n)] + rng.normal(size=(n, 2)) * 1e-4  # noqa: E731
    tied = lambda: np.concatenate([ordinary(924), np.repeat(w[1001:1002], 100, axis=0)])  # noqa: E731
    # Chunk 1's counters schedule a rebuild before chunk 3; chunk 2 is then queued on the old tables.
    x = np.concatenate([ordinary(1024), tied(), tied(), ordinary(1024), ordinary(1024)])
    labels, _ = predict_stream(w, np.arange(len(w)), x, chunk=1024, backend='cuda_screen', prefetch=False)
    np.testing.assert_array_equal(labels, reference(w, x)[0])
    T = B.TOTALS
    assert T['builds'] == 2
    assert T['rare_fallback_rows'] == 200 and T['rare_rows'] >= 200


def test_weights_snapshot(fake_bucket):
    """A reader that changes the caller's weights after the searcher is built (Sol's repro): the
    bucket tables, the screen and the exact distances all use the weights of the call."""
    names = fake_bucket()
    w = np.arange(2048, dtype=np.float64)[:, None]
    w[[0, 1]] = w[[1, 0]]  # node 1 sits at 0
    clusters = np.arange(len(w))

    def reader():
        w[[0, 1]] = w[[1, 0]]  # now node 0 sits at 0
        yield np.zeros((1, 1), np.float32)
    reader.n_features = 1
    d = np.empty(1)
    labels, _ = predict_stream(w, clusters, reader, out_distance=d, n_samples=1, backend='cuda_screen',
                               prefetch=False)
    assert 'bpass_f32' in names
    assert w[0, 0] == 0  # the reader ran after the searcher was built
    assert labels[0] == 1 and d[0] == 0


def test_searcher_freed_without_cycle_collector(fake_bucket):
    fake_bucket(50)
    rng = np.random.default_rng(1)
    w = _grid(6, 5, rng, 1e-3)
    x = rows(w, 300)
    gc.collect()
    gc.disable()
    try:
        s = _gpu_screen.GpuScreenSearcher(w, 36, 1)
        idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
        s(x, idx, d2)
        ring = s._train_ring(2)
        for k, part in enumerate(np.array_split(x, 2)):
            ring.rows(k, len(part), x.dtype)[...] = part
            ring.launch(k, len(part))
        ring.drain()
        refs = [weakref.ref(s), weakref.ref(s.bucket), weakref.ref(s.slots[0]['scratch'])]
        del s, ring
        assert [r() is None for r in refs] == [True, True, True]
    finally:
        gc.enable()


@pytest.mark.parametrize('major', [6, 7])
def test_compute_capability_gate(fake_bucket, major):
    names = fake_bucket(major=major)
    rng = np.random.default_rng(0)
    s = _gpu_screen.GpuScreenSearcher(_grid(6, 3, rng, 1e-3), 36, 1)
    assert (s.bucket is not None) == (major >= 7)
    assert names.attrs == [75]  # cudaDevAttrComputeCapabilityMajor, once
    x = rows(s.bucket.w if s.bucket else _grid(6, 3, np.random.default_rng(0), 1e-3), 50)
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
    s(x, idx, d2)
    assert ('bpass_f64' in names) == (major >= 7)


def test_group_cap_before_ordering(fake_bucket, monkeypatch):
    """Lattices with too many groups for one scan block are rejected before the P x P ordering."""
    fake_bucket()
    splits, orders = [], []
    split, order = B.split_groups, B.order_groups
    monkeypatch.setattr(B, 'split_groups', lambda w: splits.append(len(w)) or split(w))
    monkeypatch.setattr(B, 'order_groups', lambda parts: orders.append(parts['P']) or order(parts))
    monkeypatch.setattr(B, 'MAX_GROUPS', 3)  # LEAF = 4: at most 12 nodes
    rng = np.random.default_rng(0)
    assert _gpu_screen.GpuScreenSearcher(rng.normal(size=(13, 3)), 13, 1).bucket is None  # > LEAF * MAX_GROUPS
    assert splits == [] and orders == []
    w = rng.normal(size=(12, 3))
    assert B.split_groups(w)['P'] > 3
    splits.clear()
    assert _gpu_screen.GpuScreenSearcher(w, 12, 1).bucket is None  # P > MAX_GROUPS after splitting
    assert splits == [12] and orders == []
    monkeypatch.setattr(B, 'MAX_GROUPS', 1024)
    splits.clear()
    assert _gpu_screen.GpuScreenSearcher(w, 12, 1).bucket is not None
    assert splits == [12] and len(orders) == 1


def test_shared_memory_checked_before_grouping(fake_bucket, monkeypatch):
    fake_bucket(shared=4000)  # the dense kernel fits (3072 bytes), the bucketed pass (5120) does not
    splits = []
    split = B.split_groups
    monkeypatch.setattr(B, 'split_groups', lambda w: splits.append(len(w)) or split(w))
    rng = np.random.default_rng(0)
    assert _gpu_screen.GpuScreenSearcher(_grid(6, 3, rng, 1e-3), 36, 1).bucket is None
    assert splits == []


def test_fake_ring_and_call(fake_bucket):
    fake_bucket(50)
    rng = np.random.default_rng(1)
    w = _grid(6, 5, rng, 1e-3)
    x = rows(w, 300)
    s = _gpu_screen.GpuScreenSearcher(w, 36, 1)
    assert s.bucket is not None
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
    s(x, idx, d2)
    _same((idx, d2), reference(w, x))
    ring = s._train_ring(2)
    for k, part in enumerate(np.array_split(x, 2)):
        ring.rows(k, len(part), x.dtype)[...] = part
        ring.launch(k, len(part))
    for k, part in enumerate(np.array_split(x, 2)):
        _, i, d = ring.result(k, len(part))
        _same((i, d), reference(w, part))
    ring.drain()


def test_dense_for_ineligible_lattices(fake_bucket, monkeypatch):
    fake_bucket()
    rng = np.random.default_rng(0)
    assert _gpu_screen.GpuScreenSearcher(rng.normal(size=(20, 11)), 20, 1).bucket is None  # F > 10
    assert _gpu_screen.GpuScreenSearcher(rng.normal(size=(20, 3)) * 1e16, 20, 1).bucket is None  # unsafe
    monkeypatch.setattr(_gpu_screen, '_BUCKET_MIN_NODES', 100)
    assert _gpu_screen.GpuScreenSearcher(rng.normal(size=(20, 3)), 20, 1).bucket is None  # small
    monkeypatch.setattr(_gpu_screen, '_BUCKET', False)
    monkeypatch.setattr(_gpu_screen, '_BUCKET_MIN_NODES', 0)
    assert _gpu_screen.GpuScreenSearcher(rng.normal(size=(20, 3)), 20, 1).bucket is None  # switched off


# ----------------------------------------------------------------------------- real device
@pytest.fixture
def cuda():
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('no CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('no CUDA device')
    return cp


def _device_lattice(F, seed=0):
    return _grid(100, F, np.random.default_rng(seed), 1e-3)


def _search(w, x, bucket_on, monkeypatch, chunk=2**20):
    monkeypatch.setattr(_gpu_screen, '_BUCKET', bucket_on)
    s = _gpu_screen.GpuScreenSearcher(w, len(w), 1, chunk=chunk)
    assert (s.bucket is not None) == bucket_on
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
    s(x, idx, d2)
    return idx, d2, s


@pytest.mark.parametrize('F', [3, 5, 7, 10])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_device_bucket_equals_dense(cuda, monkeypatch, F, dtype):
    w = _device_lattice(F)
    rng = np.random.default_rng(F)
    with np.errstate(over='ignore', invalid='ignore'):
        x = np.concatenate([rows(w, 2**19, seed=F), rows(w + 0.3, 2**16, seed=F + 1)]).astype(dtype)
    x = x[rng.permutation(len(x))]
    dense = _search(w, x, False, monkeypatch)
    got = _search(w, x, True, monkeypatch)
    _same(got[:2], dense[:2])
    stats = got[2].bucket.stats
    assert stats['rows'] == len(x) and stats['fallback_rows'] < 0.2 * len(x)
    sub = slice(0, 20000)
    _same((got[0][sub], got[1][sub]), reference(w, x[sub]))


@pytest.mark.parametrize('kind', ['tied', 'dup', 'edge'])
def test_device_adversarial_lattices(cuda, monkeypatch, kind):
    monkeypatch.setattr(_gpu_screen, '_BUCKET_MIN_NODES', 0)
    w = lattice(kind, 5)
    with np.errstate(over='ignore', invalid='ignore'):
        x = rows(w, 50000)
    for dtype in (np.float32, np.float64):
        xx = x.astype(dtype)
        got = _search(w, xx, True, monkeypatch)
        _same(got[:2], _search(w, xx, False, monkeypatch)[:2])
        _same(got[:2], reference(w, xx))


def test_device_rebuild_on_sorted_rows(cuda, monkeypatch):
    """Rows sorted along one feature: the first chunk's pilot misses later regions, the device
    counters trigger rebuilds, and every row still equals the dense kernel."""
    w = _device_lattice(5)
    x = rows(w, 2**20)
    x = x[np.argsort(x[:, 0])]
    B.reset_totals()
    got = _search(w, x, True, monkeypatch, chunk=2**16)
    _same(got[:2], _search(w, x, False, monkeypatch)[:2])
    assert B.TOTALS['builds'] > 1 and B.TOTALS['rows'] == len(x)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('pipeline', [True, False])
def test_device_training_equals_dense(cuda, monkeypatch, dtype, pipeline):
    rng = np.random.default_rng(9)
    w = _device_lattice(5)
    x = (w[rng.integers(len(w), size=300000)] + rng.normal(size=(300000, 5)) * 0.02).astype(dtype)
    kw = dict(xdim=100, ydim=100, epochs=2, phases=(1, 0, 1), chunk=2**17, prefetch=False,
              bmu='cuda_screen', seed=3)
    monkeypatch.setattr(stream, '_GPU_PIPELINE', pipeline)
    monkeypatch.setattr(_gpu_screen, '_BUCKET', False)
    w0, i0 = train_batch_stream(x, **kw)
    monkeypatch.setattr(_gpu_screen, '_BUCKET', True)
    B.reset_totals()
    w1, i1 = train_batch_stream(x, **kw)
    # one searcher per epoch plus the final QE pass (final_pass="full"), each over every row
    assert B.TOTALS['engines'] == 3 and B.TOTALS['rows'] == 3 * len(x)
    assert _sha(w1) == _sha(w0)
    np.testing.assert_array_equal(i1['hits'], i0['hits'])
    np.testing.assert_array_equal(_bits(i1['qe']), _bits(i0['qe']))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_device_labels_equal_dense(cuda, monkeypatch, dtype):
    w = _device_lattice(5)
    x = rows(w, 2**19).astype(dtype)
    clusters = (np.arange(len(w)) * 7) % 11
    out = {}
    for on in (False, True):
        monkeypatch.setattr(_gpu_screen, '_BUCKET', on)
        d = np.empty(len(x))
        labels, _ = predict_stream(w, clusters, x, out_distance=d, chunk=2**18, backend='cuda_screen',
                                   prefetch=False)
        out[on] = (labels.copy(), d)
    np.testing.assert_array_equal(out[True][0], out[False][0])
    np.testing.assert_array_equal(_bits(out[True][1]), _bits(out[False][1]))
    ri, _ = reference(w, x[:20000])
    np.testing.assert_array_equal(out[True][0][:20000], clusters[ri])
