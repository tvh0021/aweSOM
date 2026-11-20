"""The queued cuda_screen training pass equals the synchronous one, bit for bit.

Runs on CPUs with the stream-ordered fake CuPy of ``gpu_pipeline_test``:
device work (uploads, the float64 reference search, downloads) runs only when
its stream is synchronized, so staging into a slot that is still queued, or
reading a result before its synchronize, gives wrong sums here. The real
device checks at the end skip without CuPy and a CUDA device.
"""
import hashlib

import numpy as np
import pytest
from gpu_pipeline_test import Stream, fake  # noqa: F401  (fake is a fixture)

from aweSOM import _gpu_screen, stream
from aweSOM.inference import _bmu_kernel
from aweSOM.model import Scaler
from aweSOM.stream import predict_stream, train_batch_stream


def _sha(w):
    return hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest()


def _data(n=900, features=5, dtype=np.float64, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, features)) * rng.uniform(0.5, 2.0, features)
    x[::7] = x[min(3, n - 1)]  # repeated rows: ties between nodes once the lattice holds them
    return x.astype(dtype)


OPTIONS = dict(xdim=5, ydim=4, epochs=3, phases=(1, 1, 1), r_min=1, n_threads=2)


def _train(monkeypatch, data, pipeline, **kw):
    monkeypatch.setattr(stream, '_GPU_PIPELINE', pipeline)
    return train_batch_stream(data, **{**OPTIONS, **kw})


def _same_info(a, b):
    np.testing.assert_array_equal(a['qe'].view(np.uint64), b['qe'].view(np.uint64))
    np.testing.assert_array_equal(a['hits'], b['hits'])
    np.testing.assert_array_equal(a['rows_per_epoch'], b['rows_per_epoch'])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('chunk,sub', [(1000, 2**20), (128, 2**20), (128, 50), (37, 8), (1, 2**20)])
@pytest.mark.parametrize('prefetch', [False, True])
@pytest.mark.parametrize('fraction', [1.0, 0.45])
@pytest.mark.parametrize('threads', ['default', 'many'])
def test_array_source(fake, monkeypatch, dtype, chunk, sub, prefetch, fraction, threads):
    """Weights, QE and hits of the queued pass equal the synchronous pass and numba."""
    f = fake(sub)
    if threads == 'many':  # ordered workers even on small chunks
        monkeypatch.setattr(stream, '_STAGE_MIN', 1)
        monkeypatch.setattr(stream, '_STAGE_THREADS', 3)
    x = _data(dtype=dtype)
    kw = dict(chunk=chunk, prefetch=prefetch, sample_fraction=fraction, sample_block=16,
              bmu='cuda_screen')
    old, old_info = _train(monkeypatch, x, False, **kw)
    if chunk <= sub:  # one synchronous launch per chunk: nothing queued at a launch
        assert not any(f.busy)
    del f.busy[:]
    new, new_info = _train(monkeypatch, x, True, **kw)
    ref, ref_info = _train(monkeypatch, x, True, **{**kw, 'bmu': 'numba'})
    assert _sha(new) == _sha(old) == _sha(ref)
    _same_info(new_info, old_info)
    _same_info(new_info, ref_info)
    if sub >= chunk < 900 * fraction:  # several queued chunks: launches find earlier ones queued
        assert any(f.busy)


class Recycling:
    """Callable reader that writes every chunk into the same ``buffers`` arrays in turn
    and returns them, so a chunk's storage is overwritten ``len(buffers)`` reads later."""

    def __init__(self, raw, chunk, buffers, scaler=None, own_rows=False):
        self.raw, self.chunk = raw, chunk
        self.buffers = [np.empty((chunk, raw.shape[1]), raw.dtype) for _ in range(buffers)]
        self.n_features = raw.shape[1]
        self.reads = 0
        if scaler is not None:
            self._awesom_raw_scaler = scaler
        if own_rows:
            self.rows = lambda ix: self.raw[ix]

    def lazy_chunks(self):
        for a in range(0, len(self.raw), self.chunk):
            m = min(self.chunk, len(self.raw) - a)
            yield m, (lambda a=a, m=m: self._read(a, m))

    def _read(self, a, m):
        buf = self.buffers[self.reads % len(self.buffers)]
        self.reads += 1
        buf[:] = -1e30  # stale contents never look like data
        buf[:m] = self.raw[a:a + m]
        return buf[:m]

    def __call__(self):
        for _, read in self.lazy_chunks():
            yield read()


@pytest.mark.parametrize('prefetch,buffers', [(False, 1), (True, 2)])
@pytest.mark.parametrize('chunk', [64, 300])
@pytest.mark.parametrize('fraction', [1.0, 0.6])
def test_recycling_reader(fake, monkeypatch, prefetch, buffers, chunk, fraction):
    """A reader that reuses its arrays: the queued pass accumulates from its own pinned copy.

    Without prefetch one buffer is enough for the synchronous pass; with prefetch
    the next read runs while a chunk is searched, so the reader keeps two."""
    fake(2**20)
    x = _data()
    kw = dict(chunk=chunk, prefetch=prefetch, sample_fraction=fraction, n_samples=len(x),
              bmu='cuda_screen')
    old, old_info = _train(monkeypatch, Recycling(x, chunk, buffers), False, **kw)
    new, new_info = _train(monkeypatch, Recycling(x, chunk, buffers), True, **kw)
    ref, ref_info = _train(monkeypatch, x.copy(), True, **{**kw, 'bmu': 'numba', 'n_samples': None})
    assert _sha(new) == _sha(old)
    _same_info(new_info, old_info)
    if fraction == 1.0:  # an array source and a callable sample alike only for full passes
        assert _sha(new) == _sha(ref)
        _same_info(new_info, ref_info)


@pytest.mark.parametrize('bmu,pipeline', [('cuda_screen', True), ('cuda_screen', False),
                                          ('screen', True), ('patch', True)])
@pytest.mark.parametrize('fraction', [1.0, 0.6])
@pytest.mark.parametrize('own_rows', [False, True])
@pytest.mark.parametrize('raw_dtype', [np.float32, np.float64])
def test_raw_scaler(fake, monkeypatch, bmu, pipeline, fraction, own_rows, raw_dtype):
    """A callable with the private ``_awesom_raw_scaler`` yields raw rows; training equals a
    callable yielding ``Scaler.transform`` of the same rows, on every backend."""
    fake(50)
    monkeypatch.setattr(stream, '_STAGE_MIN', 1)
    monkeypatch.setattr(stream, '_STAGE_THREADS', 4)
    raw = (_data() * 3.0 + 7.0).astype(raw_dtype)
    scaler = Scaler('median-iqr', np.median(raw, axis=0), np.std(raw, axis=0) * 1.3)
    kw = dict(chunk=128, prefetch=True, sample_fraction=fraction, bmu=bmu)
    scaled = Recycling(raw, 128, 2, own_rows=own_rows)
    plain = scaled._read
    scaled._read = lambda a, m: scaler.transform(plain(a, m))
    if own_rows:
        scaled.rows = lambda ix: scaler.transform(raw[ix])
    expected, expected_info = _train(monkeypatch, scaled, False, **kw)
    got, got_info = _train(monkeypatch, Recycling(raw, 128, 2, scaler, own_rows), pipeline, **kw)
    assert _sha(got) == _sha(expected)
    _same_info(got_info, expected_info)


def test_raw_scaler_predict(fake):
    """Other readers of the callable (labelling) also see scaled rows."""
    fake(50)
    raw = _data() * 3.0 + 7.0
    scaler = Scaler('median-iqr', np.median(raw, axis=0), np.std(raw, axis=0))
    w = _data(n=20, seed=9)
    expected, _ = predict_stream(w, np.arange(20), scaler.transform(raw), chunk=64, backend='numba')
    got, _ = predict_stream(w, np.arange(20), Recycling(raw, 64, 1, scaler), n_samples=len(raw),
                            chunk=64, backend='cuda_screen', prefetch=False)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('m,sub', [(1, 4), (37, 4), (64, 64), (100, 2**20)])
def test_ring_matches_call(fake, dtype, m, sub):
    """Ring slots return the nodes and distances of ``__call__``, also when one slot is
    reused, grown and searched in several pieces."""
    fake(sub)
    w = _data(n=13, features=4, seed=1)
    search = _gpu_screen.GpuScreenSearcher(w, 13, 1)
    ring = search._train_ring(2)
    for rep, n in enumerate([m, max(1, m // 2), m + 5]):
        x = _data(n=n, features=4, dtype=dtype, seed=rep)
        k = rep % 2
        ring.rows(k, n, x.dtype)[:] = x
        ring.launch(k, n)
        xs, idx, d2 = ring.result(k, n)
        ri, rd = np.empty(n, np.int32), np.empty(n)
        search(x, ri, rd)
        np.testing.assert_array_equal(xs, x)
        np.testing.assert_array_equal(idx, ri)
        np.testing.assert_array_equal(d2.view(np.uint64), rd.view(np.uint64))
        _bmu_kernel(w, x, ri, rd)
        np.testing.assert_array_equal(idx, ri)


def test_error_drains(fake, monkeypatch):
    """A failed launch ends the pass with no device work left queued."""
    fake(64, fail_after=5)
    monkeypatch.setattr(stream, '_GPU_PIPELINE', True)
    with pytest.raises(RuntimeError, match='launch failed'):
        train_batch_stream(_data(), chunk=50, bmu='cuda_screen', **OPTIONS)
    assert all(not s.queue for s in Stream.created)


def test_failed_stream_drains_the_others(fake, monkeypatch):
    """After a failed launch, a stream that also fails to synchronize neither leaves the
    other streams' work queued nor hides the launch error; the ring alone raises it."""
    f = fake(64, fail_after=5)
    monkeypatch.setattr(stream, '_GPU_PIPELINE', True)
    rings = []
    make = _gpu_screen.GpuScreenSearcher._train_ring

    def train_ring(self, slots):
        ring = make(self, slots)
        first = ring.streams[0]

        def sync():
            if len(f.launches) > 5:  # after the failed launch
                raise RuntimeError('sync failed')
            Stream.synchronize(first)
        first.synchronize = sync
        rings.append(ring)
        return ring

    monkeypatch.setattr(_gpu_screen.GpuScreenSearcher, '_train_ring', train_ring)
    with pytest.raises(RuntimeError, match='launch failed'):
        train_batch_stream(_data(), chunk=50, bmu='cuda_screen', **OPTIONS)
    (ring,) = rings
    assert ring.streams[0].queue and ring.streams[1].queue == ring.streams[2].queue == []
    ring.streams[1].queue.append(lambda: None)
    with pytest.raises(RuntimeError, match='sync failed'):
        ring.drain()
    assert ring.streams[1].queue == []


class Sizes:
    """Callable yielding consecutive slices of ``x`` with the given row counts."""

    def __init__(self, x, sizes, scaler=None):
        self.x, self.sizes = x, sizes
        if scaler is not None:
            self._awesom_raw_scaler = scaler

    def __call__(self):
        a = 0
        for m in self.sizes:
            yield self.x[a:a + m]
            a += m


@pytest.mark.parametrize('scaled', [False, True])
@pytest.mark.parametrize('prefetch', [False, True])
def test_chunks_above_searcher_chunk(fake, monkeypatch, scaled, prefetch):
    """Chunks of more rows than the searcher's run synchronously between queued ones,
    with bitwise-equal results and no pinned slot above the searcher's chunk."""
    f = fake(50)
    pinned = []
    plain = _gpu_screen._pinned
    monkeypatch.setattr(_gpu_screen, '_pinned', lambda cp, shape, dt: pinned.append(shape[0]) or
                        plain(cp, shape, dt))
    sizes = [30, 80, 0, 20, 50, 51, 200, 10, 10, 449]
    raw = _data() * 3.0 + 7.0
    scaler = Scaler('median-iqr', np.median(raw, axis=0), np.std(raw, axis=0)) if scaled else None
    rows = scaler.transform(raw) if scaled else raw
    kw = dict(chunk=64, prefetch=prefetch, n_samples=len(raw), bmu='cuda_screen')
    old, old_info = _train(monkeypatch, Sizes(rows, sizes), False, **kw)
    new, new_info = _train(monkeypatch, Sizes(raw, sizes, scaler), True, **kw)
    ref, ref_info = _train(monkeypatch, Sizes(rows, sizes), True, **{**kw, 'bmu': 'numba'})
    assert _sha(new) == _sha(old) == _sha(ref)
    _same_info(new_info, old_info)
    _same_info(new_info, ref_info)
    assert pinned and max(pinned) <= 50 and any(f.busy)


# --------------------------------------------------------------------- real device


@pytest.fixture
def cuda():
    cp = pytest.importorskip('cupy', reason='CuPy is not installed')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('No CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA driver unavailable')
    return cp


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('chunk', [997, 4096])
@pytest.mark.parametrize('prefetch', [False, True])
def test_device_training(cuda, monkeypatch, dtype, chunk, prefetch):
    x = _data(n=20000, dtype=dtype)
    kw = dict(chunk=chunk, prefetch=prefetch, bmu='cuda_screen', sample_fraction=0.7, sample_block=64)
    old, old_info = _train(monkeypatch, x, False, **kw)
    new, new_info = _train(monkeypatch, x, True, **kw)
    ref, _ = _train(monkeypatch, x, True, **{**kw, 'bmu': 'patch'})
    assert _sha(new) == _sha(old) == _sha(ref)
    _same_info(new_info, old_info)
    rec = Recycling(x, chunk, 2 if prefetch else 1)
    got, _ = _train(monkeypatch, rec, True, **{**kw, 'n_samples': len(x), 'sample_fraction': 1.0})
    want, _ = _train(monkeypatch, x, False, **{**kw, 'sample_fraction': 1.0})
    assert _sha(got) == _sha(want)


def test_device_ring_matches_call(cuda):
    w = _data(n=100, features=5, seed=2)
    search = _gpu_screen.GpuScreenSearcher(w, 10, 10, chunk=257, block=64, tile=16)
    ring = search._train_ring(3)
    for k, n in enumerate([1000, 3, 257, 1001]):
        x = _data(n=n, seed=k)
        ring.rows(k % 3, n, x.dtype)[:] = x
        ring.launch(k % 3, n)
        _, idx, d2 = ring.result(k % 3, n)
        ri, rd = np.empty(n, np.int32), np.empty(n)
        search(x, ri, rd)
        np.testing.assert_array_equal(idx, ri)
        np.testing.assert_array_equal(d2.view(np.uint64), rd.view(np.uint64))
