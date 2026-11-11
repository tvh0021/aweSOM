"""CPU checks of the queued CUDA labelling path with a stream-ordered fake CuPy.

Device work runs only when its stream is synchronized, so reading a result
or reusing a staging buffer before the matching synchronize gives wrong
labels here. The fake kernel is the float64 reference search for both
searchers (the screen kernel's own parity is checked on a real device).
"""
import sys
import types

import numpy as np
import pytest

from aweSOM import _gpu, _gpu_screen
from aweSOM.inference import _bmu_kernel
from aweSOM.stream import predict_stream


class Stream:
    current = None

    def __init__(self, **kwargs):
        self.queue = []

    def __enter__(self):
        self.previous, Stream.current = Stream.current, self
        return self

    def __exit__(self, *args):
        Stream.current = self.previous

    def synchronize(self):
        queue, self.queue = self.queue, []
        for op in queue:
            op()


class NullStream(Stream):
    def synchronize(self):
        pass


def _run(stream, op):
    if stream is None or isinstance(stream, NullStream):
        op()
    else:
        stream.queue.append(op)


class Device:
    gets = []

    def __init__(self, array):
        self.a = array

    @property
    def dtype(self):
        return self.a.dtype

    def __getitem__(self, key):
        return Device(self.a[key])

    def set(self, a, stream):
        _run(stream, lambda: self.a.__setitem__(Ellipsis, a))

    def get(self, out, stream, blocking):
        assert not blocking
        Device.gets.append(self.a.dtype)
        _run(stream, lambda: out.__setitem__(Ellipsis, self.a))


def _fake_cupy(fail_after=None):
    launches = []

    def raw(source, name, options):
        def launch(grid, block, args, shared_mem, stream):
            w, x, idx, d2, n = args[:5]
            launches.append(n)
            if fail_after is not None and len(launches) > fail_after:
                raise RuntimeError('launch failed')
            _run(stream, lambda: _bmu_kernel(w.a, x.a[:n], idx.a[:n], d2.a[:n]))
        return launch

    def take(table, idx, out):
        _run(Stream.current, lambda: out.a.__setitem__(Ellipsis, np.take(table.a, idx.a)))

    runtime = types.SimpleNamespace(getDeviceCount=lambda: 1, getDevice=lambda: 0,
                                    getDeviceProperties=lambda _: dict(sharedMemPerBlock=49152,
                                                                       maxThreadsPerBlock=1024))
    cp = types.SimpleNamespace(
        asarray=lambda a: Device(np.array(a)),
        empty=lambda shape, dtype: Device(np.full(shape, 99, dtype)),
        take=take, RawKernel=raw,
        cuda=types.SimpleNamespace(runtime=runtime, Stream=Stream,
                                   alloc_pinned_memory=bytearray,
                                   get_current_stream=NullStream))
    return cp, launches


@pytest.fixture
def fake(monkeypatch):
    def install(chunk, overlap=True, fail_after=None):
        cp, launches = _fake_cupy(fail_after)
        monkeypatch.setitem(sys.modules, 'cupy', cp)
        _gpu._kernels.cache_clear()
        _gpu_screen._kernels.cache_clear()
        for module, name in [(_gpu, 'GpuSearcher'), (_gpu_screen, 'GpuScreenSearcher')]:
            base = getattr(module, name)
            small = type('Small' + name, (base,), {'__init__': lambda self, *a, base=base, **k:
                         base.__init__(self, *a, chunk=chunk, overlap=overlap, **k)})
            monkeypatch.setattr(module, name, small)
        Device.gets.clear()
        return launches
    yield install
    _gpu._kernels.cache_clear()
    _gpu_screen._kernels.cache_clear()


class H5Like:
    """Non-ndarray output (as h5py.Dataset): written through host buffers."""

    def __init__(self, n, dtype):
        self.a = np.zeros(n, dtype)
        self.dtype = self.a.dtype

    def __setitem__(self, key, value):
        self.a[key] = value


def _data(n=47, nodes=13, features=4, dtype=np.float32):
    rng = np.random.default_rng(3)
    w = rng.normal(size=(nodes, features))
    x = rng.normal(size=(n, features)).astype(dtype)
    x[:3] = w[2]  # exact hits, d2 == 0
    return w, x


@pytest.mark.parametrize('backend', ['cuda', 'cuda_screen'])
@pytest.mark.parametrize('overlap', [True, False])
@pytest.mark.parametrize('chunk,sub', [(10, 3), (47, 4), (5, 8), (100, 2**20)])
@pytest.mark.parametrize('distance', [False, True])
@pytest.mark.parametrize('out_dtype,offset', [(None, 0), (np.uint8, 0), (np.int64, 0),
                                              (np.uint8, 300), ('h5', 0)])
def test_predict_stream_queued(fake, backend, overlap, chunk, sub, distance, out_dtype, offset):
    launches = fake(sub, overlap)
    w, x = _data()
    clusters = np.arange(len(w)) % 5 + offset
    expected_out = None if out_dtype is None else (
        H5Like(len(x), np.uint8) if out_dtype == 'h5' else np.zeros(len(x), out_dtype))
    actual_out = None if out_dtype is None else (
        H5Like(len(x), np.uint8) if out_dtype == 'h5' else np.zeros(len(x), out_dtype))
    ed = np.empty(len(x)) if distance else None
    ad = np.empty(len(x)) if distance else None
    expected, _ = predict_stream(w, clusters, x, out=expected_out, out_distance=ed,
                                 chunk=chunk, backend='numba', prefetch=False)
    actual, stats = predict_stream(w, clusters, x, out=actual_out, out_distance=ad,
                                   chunk=chunk, backend=backend, prefetch=chunk == 10)
    if out_dtype == 'h5':
        expected, actual = expected.a, actual.a
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    if distance:
        np.testing.assert_array_equal(ad.view(np.uint64), ed.view(np.uint64))
        assert np.float64 in Device.gets
    else:
        assert np.float64 not in Device.gets  # d2 stays on the device
    assert stats['rows'] == len(x)
    assert sum(launches) == len(x)
    assert len(launches) == sum(-(-min(chunk, len(x) - s) // sub) for s in range(0, len(x), chunk))


def test_label_dtype_and_direct_write(fake):
    fake(4)
    w, x = _data()
    out = np.zeros(len(x), np.uint8)
    predict_stream(w, np.arange(len(w)) % 3, x, out=out, chunk=9, backend='cuda', prefetch=False)
    assert Device.gets and set(Device.gets) == {np.dtype(np.uint8)}  # gathered, 1 B per row
    Device.gets.clear()
    predict_stream(w, np.arange(len(w)) - 1, x, out=out, chunk=9, backend='cuda', prefetch=False)
    assert set(Device.gets) == {np.dtype(np.int32)}  # -1 does not fit uint8


def test_searcher_queue_and_error(fake):
    fake(4, fail_after=5)
    w, x = _data(n=30)
    search = _gpu.GpuSearcher(w, len(w), 1)
    ri, rd = np.empty(len(x), np.int32), np.empty(len(x))
    _bmu_kernel(w, x, ri, rd)
    idx, d2 = np.full(len(x), -1, np.int32), np.full(len(x), -1.)
    token = search.submit(x[:10], idx[:10], d2[:10])
    assert search.submit(x[:0], idx[:0]) == token
    assert any(s['pending'] for s in search.slots)  # results not written yet
    search.wait(token)
    np.testing.assert_array_equal(idx[:10], ri[:10])
    np.testing.assert_array_equal(d2[:10].view(np.uint64), rd[:10].view(np.uint64))
    assert all(s['pending'] is None for s in search.slots)
    with pytest.raises(RuntimeError, match='launch failed'):
        search.submit(x[10:], idx[10:], d2[10:])
    assert all(s['pending'] is None for s in search.slots)
    assert all(not s.queue for s in search.streams)
