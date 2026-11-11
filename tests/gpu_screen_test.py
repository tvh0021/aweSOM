"""CUDA screen certification on CPUs, with optional real-device checks."""
import sys

import numpy as np
import pytest

from aweSOM import _gpu_screen
from aweSOM.inference import best_match_nodes, best_match_nodes_screen, predict
from aweSOM.stream import _BMU, predict_stream, train_batch_stream


def _case(case, features, dtype):
    rng = np.random.default_rng(42)
    w = rng.normal(size=(73, features))
    x = rng.normal(size=(131, features))
    if case == 'clustered':
        w *= 1e-4
        x = w[rng.integers(len(w), size=len(x))] + rng.normal(size=x.shape) * 1e-7
    elif case == 'smooth':
        a, b = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 8))
        w = np.column_stack([np.sin(a.ravel() * (f + 1)) + b.ravel() for f in range(features)])
    elif case == 'ties':
        w[:] = 1
        x[:] = 0
    elif case == 'near':
        w[:] = 1
        w[0, 0] = np.nextafter(1., 2.)
        x[:] = 0
    elif case == 'duplicate':
        w[::2] = w[0]
        x[:5] = w[0]
    elif case == 'large':
        w *= 1e14
        x *= 1e14
    elif case == 'unsafe':
        w *= 1e160
        x *= 1e160
    elif case == 'tiny':
        w *= 1e-160
        x *= 1e-160
    elif case == 'nonfinite':
        w[0] = np.nan
        w[1] = np.inf
        x[0] = np.nan
        x[1] = np.inf
    elif case == 'extreme_rows':
        x[:4] = np.array([np.nan, np.inf, 1e38, 1e-40])[:, None]
    return w, x.astype(dtype)


def _same(actual, expected):
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1].view(np.uint64), expected[1].view(np.uint64))


CASES = ['random', 'clustered', 'smooth', 'ties', 'near', 'duplicate', 'large',
         'unsafe', 'tiny', 'nonfinite', 'extreme_rows']


@pytest.mark.parametrize('features', [1, 7, 32])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('case', CASES)
def test_reference(case, features, dtype):
    w, x = _case(case, features, dtype)
    actual = _gpu_screen._cuda_screen_reference(w, x, chunk=37, return_fallbacks=True)
    _same(actual, best_match_nodes(w, x))
    _same(actual, best_match_nodes_screen(w, x))
    if case in ('ties', 'near', 'tiny', 'unsafe', 'nonfinite'):
        assert actual[2] == len(x)
    if case == 'random':
        assert actual[2] < len(x)  # Exercise the singleton candidate path too.


def test_reference_edges():
    w = np.array([[0.], [2.]])
    _same(_gpu_screen._cuda_screen_reference(w, np.array([1.])), best_match_nodes(w, [1.]))
    idx, d2 = _gpu_screen._cuda_screen_reference(w, np.empty((0, 1)))
    assert idx.dtype == np.int32 and d2.dtype == np.float64 and len(idx) == 0
    x = np.arange(10.)[:, None][::2]
    _same(_gpu_screen._cuda_screen_reference(w, x), best_match_nodes(w, x))
    with pytest.raises(ValueError, match='positive'):
        _gpu_screen._cuda_screen_reference(w, x, chunk=0)
    with pytest.raises(ValueError, match='32 features'):
        _gpu_screen._cuda_screen_reference(np.zeros((2, 33)), np.zeros((1, 33)))


def test_missing_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', None)
    with pytest.raises(ImportError, match=r'aweSOM\[cuda\]'):
        _gpu_screen.best_match_nodes_cuda_screen(np.zeros((2, 1)), np.zeros((1, 1)))


class ReferenceSearcher:
    calls = 0

    def __init__(self, weights, *args, **kwargs):
        self.w = weights
        type(self).calls += 1

    def __call__(self, x, idx, d2):
        idx[:], d2[:], fb = _gpu_screen._cuda_screen_reference(self.w, x, return_fallbacks=True)
        return fb


def test_dispatch(monkeypatch):
    monkeypatch.setattr(_gpu_screen, 'GpuScreenSearcher', ReferenceSearcher)
    w, x = _case('random', 7, np.float32)
    clusters = np.arange(len(w))
    expected = predict(w, clusters, x, backend='numba', return_distance=True)
    actual = predict(w, clusters, x, backend='cuda_screen', chunk=37, return_distance=True)
    _same(actual, expected)
    before = ReferenceSearcher.calls
    distance = np.empty(len(x))
    actual, stats = predict_stream(w, clusters, x, backend='cuda_screen', chunk=37,
                                   out_distance=distance, prefetch=False)
    assert ReferenceSearcher.calls == before + 1
    _same((actual, distance), expected)
    assert stats['rows'] == len(x)
    bmu = _BMU(w, 'cuda_screen', len(w), 1, first_epoch=True)
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
    bmu(x, idx, d2)
    _same((idx, d2), best_match_nodes(w, x))
    options = dict(xdim=3, ydim=2, epochs=2, chunk=37, groups=2, n_threads=1,
                   final_pass='none', prefetch=False)
    cpu, _ = train_batch_stream(x, bmu='numba', **options)
    gpu, stats = train_batch_stream(x, bmu='cuda_screen', **options)
    np.testing.assert_array_equal(cpu.view(np.uint64), gpu.view(np.uint64))
    assert stats['bmu_kind'] == ['cuda_screen', 'cuda_screen']


@pytest.fixture
def cuda():
    cp = pytest.importorskip('cupy', reason='CuPy is not installed')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('No CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA driver unavailable')
    return cp


@pytest.mark.parametrize('features', [1, 7, 32])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('case', CASES)
def test_device(cuda, case, features, dtype):
    from aweSOM._gpu import best_match_nodes_cuda

    w, x = _case(case, features, dtype)
    actual = _gpu_screen.best_match_nodes_cuda_screen(w, x, chunk=37)
    _same(actual, best_match_nodes_cuda(w, x, chunk=37))
    _same(actual, _gpu_screen._cuda_screen_reference(w, x))


def test_device_staging_and_training(cuda):
    from aweSOM._gpu import best_match_nodes_cuda

    rng = np.random.default_rng(7)
    w = rng.normal(size=(73, 7))
    w[1] = w[0]
    x = rng.normal(size=(1038, 7))[::2]  # Multiple blocks and noncontiguous input.
    x[:20] = w[0]  # Certified and fallback rows share a block.
    search = _gpu_screen.GpuScreenSearcher(w, 73, 1, chunk=257, block=64, tile=16)
    for dtype in (np.float32, np.float64):
        points = x.astype(dtype)
        idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
        search(points, idx, d2)
        _same((idx, d2), best_match_nodes_cuda(w, points))
    options = dict(xdim=4, ydim=3, epochs=2, chunk=257, groups=2, n_threads=1,
                   seed=42, final_pass='none', prefetch=False)
    cpu, _ = train_batch_stream(x, bmu='numba', **options)
    gpu, _ = train_batch_stream(x, bmu='cuda_screen', **options)
    np.testing.assert_array_equal(cpu.view(np.uint64), gpu.view(np.uint64))


def test_device_index_matches_patch(cuda):
    """An explicit device (the per-rank choice of aweSOM.distributed.gpu_device) through the
    searcher, BMUSearcher and predict_stream: nodes and float64 distances bitwise equal to patch."""
    from aweSOM import distributed
    from aweSOM.inference import best_match_nodes_patch
    from aweSOM.stream import BMUSearcher, cuda_available

    assert cuda_available()
    device = distributed.gpu_device(None)
    assert device == 0
    a, b = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-1, 1, 9), indexing='ij')
    w = np.column_stack([a.ravel(order='F'), b.ravel(order='F')]
                        + [np.sin((f + 2) * a.ravel(order='F')) for f in range(5)])
    w[40] = w[13]  # equal nodes: the lowest index must win
    rng = np.random.default_rng(11)
    x = w[rng.integers(len(w), size=3001)] + rng.normal(scale=0.02, size=(3001, 7))
    x[::7] = w[13]
    for dtype in (np.float32, np.float64):
        rows = x.astype(dtype)
        ref = best_match_nodes_patch(w, rows, 12, 9)
        assert (ref[0][::7] == 13).all()
        idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
        _gpu_screen.GpuScreenSearcher(w, 12, 9, chunk=509, device=device)(rows, idx, d2)
        _same((idx, d2), ref)
        _same(BMUSearcher(w, 'cuda_screen', device=device)(rows), ref)
        nodes, dist = np.empty(len(x), np.int32), np.empty(len(x))
        _, st = predict_stream(w, np.arange(len(w)), rows, out_node=nodes, out_distance=dist,
                               chunk=509, backend='cuda_screen', device=device, prefetch=False)
        assert st['device'] == device
        _same((nodes, dist), (ref[0], np.sqrt(ref[1])))
