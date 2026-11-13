"""CPU checks of the queued CUDA labelling path with a stream-ordered fake CuPy.

Device work runs only when its stream is synchronized, so reading a result
or reusing a staging buffer before the matching synchronize gives wrong
labels here. The fake kernel is the float64 reference search (the screen
kernel's own parity is checked on a real device).
"""
import sys
import types

import numpy as np
import pytest

from aweSOM import _gpu_screen
from aweSOM.inference import _bmu_kernel
from aweSOM.stream import predict_stream


class Stream:
    """Work queued on a stream runs only at its synchronize, as late as CUDA may run it."""
    current = None
    created = []

    def __init__(self, **kwargs):
        self.queue = []
        Stream.created.append(self)

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
    """The current (legacy) stream: its uploads run only at its own synchronize."""
    queue = []

    def __init__(self):
        pass

    def synchronize(self):
        queue, NullStream.queue[:] = list(NullStream.queue), []
        for op in queue:
            op()


def _run(stream, op):
    assert stream is not None and not isinstance(stream, NullStream)
    stream.queue.append(op)


POISON = 77  # what a freed or not yet uploaded device buffer holds here


class Device:
    gets = []

    def __init__(self, array, base=None):
        self.a, self.base = array, base  # a view keeps its allocation alive, as in CuPy
        assert array.dtype.isnative

    def __del__(self):
        if self.base is None:  # freed: the pool may hand the memory to the next allocation
            self.a.fill(POISON)

    @property
    def dtype(self):
        return self.a.dtype

    def __getitem__(self, key):
        return Device(self.a[key], self)

    def set(self, a, stream):
        assert a.shape == self.a.shape and a.dtype == self.a.dtype
        _run(stream, lambda: self.a.__setitem__(Ellipsis, a))

    def get(self, out, stream, blocking):
        assert not blocking
        assert out.shape == self.a.shape and out.dtype == self.a.dtype and out.flags.c_contiguous
        Device.gets.append(self.a.dtype)
        _run(stream, lambda: out.__setitem__(Ellipsis, self.a))


def _asarray(a):
    a = np.asarray(a)
    device = Device(np.full(a.shape, POISON, a.dtype))
    raw = device.a  # the copy runs at the null stream's synchronize
    NullStream.queue.append(lambda: raw.__setitem__(Ellipsis, a.copy()))
    return device


def _fake_cupy(fail_after=None):
    launches = []
    busy = []  # per launch: is another stream's work still unsynchronized?
    pinned_fail = []

    def raw(source, name, options):
        def launch(grid, block, args, shared_mem, stream):
            w, x, idx, d2, n = args[:5]
            launches.append(n)
            busy.append(any(s.queue for s in Stream.created if s is not stream))
            if fail_after is not None and len(launches) > fail_after:
                raise RuntimeError('launch failed')
            _run(stream, lambda: _bmu_kernel(w.a, x.a[:n], idx.a[:n], d2.a[:n]))
        return launch

    def take(table, idx, out):
        assert idx.dtype == np.int32 and out.dtype == table.dtype and out.a.shape == idx.a.shape
        memory = table.a  # the kernel holds a pointer, not a reference to the table
        _run(Stream.current, lambda: out.a.__setitem__(Ellipsis, np.take(memory, idx.a)))

    def pinned(n):
        if pinned_fail:
            raise MemoryError('pinned allocation failed')
        return bytearray(n)

    runtime = types.SimpleNamespace(getDeviceCount=lambda: 1, getDevice=lambda: 0,
                                    getDeviceProperties=lambda _: dict(sharedMemPerBlock=49152,
                                                                       maxThreadsPerBlock=1024))
    cp = types.SimpleNamespace(
        asarray=_asarray,
        empty=lambda shape, dtype: Device(np.full(shape, 99, dtype)),
        take=take, RawKernel=raw,
        cuda=types.SimpleNamespace(runtime=runtime, Stream=Stream,
                                   alloc_pinned_memory=pinned,
                                   get_current_stream=NullStream))
    return cp, launches, busy, pinned_fail


@pytest.fixture
def fake(monkeypatch):
    def install(chunk, overlap=True, fail_after=None):
        cp, launches, busy, pinned_fail = _fake_cupy(fail_after)
        Stream.created.clear()
        NullStream.queue.clear()
        monkeypatch.setitem(sys.modules, 'cupy', cp)
        _gpu_screen._kernels.cache_clear()
        base = _gpu_screen.GpuScreenSearcher
        small = type('Small', (base,), {'__init__': lambda self, *a, **k:
                     base.__init__(self, *a, chunk=chunk, overlap=overlap, **k)})
        monkeypatch.setattr(_gpu_screen, 'GpuScreenSearcher', small)
        Device.gets.clear()
        return types.SimpleNamespace(launches=launches, busy=busy, pinned_fail=pinned_fail)
    yield install
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


@pytest.mark.parametrize('overlap', [True, False])
@pytest.mark.parametrize('chunk,sub', [(10, 3), (47, 4), (5, 8), (100, 2**20)])
@pytest.mark.parametrize('distance', [False, True])
@pytest.mark.parametrize('out_dtype,offset', [(None, 0), (np.uint8, 0), (np.int64, 0),
                                              (np.uint8, 300), ('h5', 0)])
def test_predict_stream_queued(fake, overlap, chunk, sub, distance, out_dtype, offset):
    f = fake(sub, overlap)
    launches = f.launches
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
                                   chunk=chunk, backend='cuda_screen', prefetch=chunk == 10)
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
    if overlap:  # every launch after the first finds earlier work still queued, also across chunks
        assert all(f.busy[1:])


def test_label_dtype_and_direct_write(fake, monkeypatch, tmp_path):
    fake(4)
    w, x = _data()
    shared = []
    submit = _gpu_screen.GpuScreenSearcher.submit

    def spy(self, block, out, *args):
        shared.append(np.shares_memory(out, target))
        return submit(self, block, out, *args)
    monkeypatch.setattr(_gpu_screen.GpuScreenSearcher, 'submit', spy)
    target = np.zeros(len(x), np.uint8)
    predict_stream(w, np.arange(len(w)) % 3, x, out=target, chunk=9, backend='cuda_screen', prefetch=False)
    assert set(Device.gets) == {np.dtype(np.uint8)}  # gathered, 1 B per row
    assert shared and all(shared)  # written from pinned memory straight into out
    Device.gets.clear()
    shared.clear()
    predict_stream(w, np.arange(len(w)) - 1, x, out=target, chunk=9, backend='cuda_screen', prefetch=False)
    assert set(Device.gets) == {np.dtype(np.int32)}  # -1 does not fit uint8
    assert not any(shared)  # int32 labels go through host buffers into the uint8 out
    for target in [np.zeros(len(x), '>i4'),  # non-native byte order (h5py, FITS)
                   np.lib.format.open_memmap(tmp_path / 'l.npy', 'w+', np.int16, (len(x),))]:
        expected = predict_stream(w, np.arange(len(w)) % 3, x, chunk=9, backend='numba')[0]
        predict_stream(w, np.arange(len(w)) % 3, x, out=target, chunk=9, backend='cuda_screen',
                       prefetch=False)
        np.testing.assert_array_equal(target, expected)


def test_table_outlives_submit(fake):
    """A queued gather keeps its cluster table alive until the result is written."""
    fake(4)
    w, x = _data(n=30)
    search = _gpu_screen.GpuScreenSearcher(w, len(w), 1)
    ri, rd = np.empty(len(x), np.int32), np.empty(len(x))
    _bmu_kernel(w, x, ri, rd)
    out = np.zeros(len(x), np.int32)
    search.submit(x, out, clusters=search.cluster_map(np.arange(len(w)) + 1000))
    other = search.cluster_map(np.arange(len(w)))  # may reuse a freed table's memory
    search.wait()
    np.testing.assert_array_equal(out, ri + 1000)
    assert other is not None


def test_wait_oldest_first(fake):
    """Two queued results for the same rows: the newer one is written last."""
    fake(4)
    w, x = _data(n=12)
    search = _gpu_screen.GpuScreenSearcher(w, len(w), 1)
    idx, d2 = np.empty(4, np.int32), np.empty(4)
    search.submit(x[:4], idx, d2)  # slot 0
    search.submit(x[4:8], idx, d2)  # slot 1
    search.submit(x[8:12], idx, d2)  # slot 0: writes the first; slot 1 is now the older
    search.wait()
    ri, rd = np.empty(4, np.int32), np.empty(4)
    _bmu_kernel(w, x[8:12], ri, rd)
    np.testing.assert_array_equal(idx, ri)
    np.testing.assert_array_equal(d2.view(np.uint64), rd.view(np.uint64))


def test_allocation_failure_recovers(fake):
    f = fake(4)
    w, x = _data(n=12)
    search = _gpu_screen.GpuScreenSearcher(w, len(w), 1)
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
    search(x, idx, d2)
    f.pinned_fail.append(True)
    with pytest.raises(MemoryError):
        search(x.astype(np.float64), idx, d2)
    f.pinned_fail.clear()
    ri, rd = np.empty(3, np.int32), np.empty(3)
    _bmu_kernel(w, x[:3], ri, rd)
    search(x[:3], idx[:3], d2[:3])
    np.testing.assert_array_equal(idx[:3], ri)


def test_searcher_queue_and_error(fake):
    fake(4, fail_after=5)
    w, x = _data(n=30)
    search = _gpu_screen.GpuScreenSearcher(w, len(w), 1)
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
