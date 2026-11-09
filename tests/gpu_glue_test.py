"""CPU-only CUDA plumbing tests, including the real staging code."""
import subprocess
import sys
import types

import numpy as np
import pytest

from aweSOM import _gpu, _gpu_screen
from aweSOM.inference import _bmu_kernel, predict
from aweSOM.stream import _BMU, predict_stream


class FakeSearcher:
    calls = []

    def __init__(self, weights, *args, **kwargs):
        self.w = weights
        self.calls.append(self)

    def __call__(self, block, idx, d2):
        _bmu_kernel(self.w, block, idx, d2)
        return len(block)  # Exercise fallback propagation independently of A.


def test_dispatch(monkeypatch):
    monkeypatch.setattr(_gpu, 'GpuSearcher', FakeSearcher)
    w, x = np.arange(12.).reshape(3, 4), np.arange(44.).reshape(11, 4)
    idx, d2 = np.empty(11, np.int32), np.empty(11)
    search = _BMU(w, 'cuda', 3, 1, first_epoch=True)
    assert search(x, idx, d2) == 11
    expected = predict(w, np.arange(3), x, return_distance=True, backend='numba')
    actual = predict(w, np.arange(3), x, chunk=3, return_distance=True, backend='cuda')
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    before = len(FakeSearcher.calls)
    distance = np.empty(11)
    result, stats = predict_stream(w, np.arange(3), x, out_distance=distance,
                                   chunk=3, backend='cuda', prefetch=False)
    assert len(FakeSearcher.calls) == before + 1
    assert stats['rows'] == 11
    np.testing.assert_array_equal(result, expected[0])
    np.testing.assert_array_equal(distance, expected[1])


def test_lazy_import():
    subprocess.run([sys.executable, '-c',
                    'import sys; import aweSOM.stream; assert "cupy" not in sys.modules'], check=True)


def test_missing_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', None)
    with pytest.raises(ImportError, match=r'aweSOM\[cuda\]'):
        _gpu.GpuSearcher(np.zeros((2, 4)), 2, 1)


class Device:
    def __init__(self, array):
        self.a = array

    def __getitem__(self, key):
        return Device(self.a[key])

    def set(self, a, stream):
        self.a[...] = a

    def get(self, out, stream, blocking):
        assert not blocking
        out[...] = self.a


class Stream:
    def __init__(self, **kwargs):
        self.syncs = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def synchronize(self):
        self.syncs += 1


@pytest.mark.parametrize('screen', [False, True])
@pytest.mark.parametrize('overlap', [True, False])
def test_staging(monkeypatch, overlap, screen):
    allocations, sources = [], []

    def pinned(n):
        allocations.append(n)
        return bytearray(n)

    def raw(source, name, options):
        sources.append(source)
        assert '--fmad=false' in options
        assert '#include' not in source
        assert '__longlong_as_double(0x7ff0000000000000LL)' in source
        def launch(grid, block, args, **kwargs):
            w, x, idx, d2, n, nodes, tile = args
            _bmu_kernel(w.a, x.a[:n], idx.a[:n], d2.a[:n])
        return launch

    runtime = types.SimpleNamespace(getDeviceCount=lambda: 1, getDevice=lambda: 0,
                                    getDeviceProperties=lambda _: dict(sharedMemPerBlock=49152,
                                                                       maxThreadsPerBlock=1024))
    cp = types.SimpleNamespace(asarray=lambda a: Device(a.copy()),
                               empty=lambda shape, dtype: Device(np.empty(shape, dtype)),
                               RawKernel=raw,
                               cuda=types.SimpleNamespace(runtime=runtime, Stream=Stream,
                                   alloc_pinned_memory=pinned, get_current_stream=Stream))
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    _gpu._kernels.cache_clear()
    _gpu_screen._kernels.cache_clear()
    w = np.arange(28.).reshape(7, 4)
    cls = _gpu_screen.GpuScreenSearcher if screen else _gpu.GpuSearcher
    search = cls(w, 7, 1, chunk=3, overlap=overlap)
    for dtype in [np.float32, np.float64]:
        x = np.arange(88., dtype=dtype).reshape(22, 4)[::2]  # noncontiguous
        idx, d2 = np.empty(11, np.int32), np.empty(11)
        assert search(x, idx, d2) == 0
        ri, rd = np.empty_like(idx), np.empty_like(d2)
        _bmu_kernel(w, x, ri, rd)
        np.testing.assert_array_equal(idx, ri)
        np.testing.assert_array_equal(d2.view(np.uint64), rd.view(np.uint64))
        assert all(s['pending'] is None for s in search.slots)
        count = len(allocations)
        search(x[:2], idx[:2], d2[:2])
        assert len(allocations) == count
        assert len(search.slots) == (2 if overlap else 1)
        assert all(len(s['owners']) == 3 for s in search.slots)
    _gpu._kernels.cache_clear()
    _gpu_screen._kernels.cache_clear()


@pytest.mark.parametrize('bmu', ['numba', 'auto', 'screen', 'patch'])
def test_cpu_reference(bmu):
    """Every CPU search kind reproduces the weights stored from the base commit."""
    import json
    from pathlib import Path
    from aweSOM.stream import train_batch_stream

    root = Path(__file__).parent / 'fixtures'
    ref = np.load(root / 'scale_out_cpu.npz')
    meta = json.loads((root / 'scale_out_cpu.json').read_text())
    for threads in [1, 4]:
        options = dict(meta['options'], n_threads=threads, bmu=bmu)
        w, _ = train_batch_stream(ref['data'], initial_weights=ref['initial'], **options)
        np.testing.assert_array_equal(w.view(np.uint64), ref['expected'].view(np.uint64))


def test_cpu_reference_sampled_init():
    """The default sampled initialisation (seed from the fixture) is pinned too."""
    import json
    from pathlib import Path
    from aweSOM.stream import train_batch_stream

    root = Path(__file__).parent / 'fixtures'
    ref = np.load(root / 'scale_out_cpu.npz')
    meta = json.loads((root / 'scale_out_cpu.json').read_text())
    options = dict(meta['options'], bmu='auto', seed=meta['sampled_seed'])
    w, _ = train_batch_stream(ref['data'], **options)
    np.testing.assert_array_equal(w.view(np.uint64), ref['expected_sampled'].view(np.uint64))


def test_training_dispatch(monkeypatch):
    from aweSOM.stream import train_batch_stream
    monkeypatch.setattr(_gpu, 'GpuSearcher', FakeSearcher)
    x = np.arange(88.).reshape(22, 4)
    options = dict(xdim=3, ydim=1, epochs=2, chunk=5, groups=2,
                   n_threads=1, final_pass='none', prefetch=False)
    cpu, _ = train_batch_stream(x, bmu='numba', **options)
    gpu, stats = train_batch_stream(x, bmu='cuda', **options)
    np.testing.assert_array_equal(gpu.view(np.uint64), cpu.view(np.uint64))
    np.testing.assert_array_equal(stats['fallbacks'], [22, 22])
    assert stats['bmu_kind'] == ['cuda', 'cuda']
