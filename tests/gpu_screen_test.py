"""CUDA screen certification on CPUs, with optional real-device checks."""
import sys
from fractions import Fraction

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

    # The queued interface of predict_stream, run synchronously.
    def cluster_map(self, clusters, dtype):
        return np.asarray(clusters).astype(dtype)

    def submit(self, x, out, d2=None, clusters=None):
        idx, dist = _gpu_screen._cuda_screen_reference(self.w, x)
        out[...] = clusters[idx]
        if d2 is not None:
            d2[...] = dist

    def wait(self, token=None):
        pass

    def cancel(self):
        pass


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


def _stream_case(n, nodes, seed=11):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(nodes, 7))
    w[1] = w[0]  # duplicate nodes: tie rule
    x = (w[rng.integers(nodes, size=n)] + rng.normal(size=(n, 7)) * 0.05).astype(np.float32)
    x[:20] = w[0]  # certified and fallback rows in one block
    x[20] = np.nan
    return w, x


def _stream_parity(backend, w, x, clusters, chunk, distance, out_dtype):
    """Labels (and distances) of the queued CUDA path equal the float64 CPU reference bitwise."""
    def run(kind):
        out = None if out_dtype is None else np.zeros(len(x), out_dtype)
        d = np.empty(len(x)) if distance else None
        labels, stats = predict_stream(w, clusters, x, out=out, out_distance=d, chunk=chunk,
                                       backend=kind, prefetch=True)
        assert stats['rows'] == len(x)
        return labels, d
    expected, actual = run('numba'), run(backend)
    assert actual[0].dtype == expected[0].dtype
    np.testing.assert_array_equal(actual[0], expected[0])
    if distance:
        np.testing.assert_array_equal(actual[1].view(np.uint64), expected[1].view(np.uint64))


@pytest.mark.parametrize('backend', ['cuda', 'cuda_screen'])
@pytest.mark.parametrize('distance', [False, True])
@pytest.mark.parametrize('out_dtype', [None, np.uint8, np.int64])
def test_device_predict_stream_subchunks(cuda, monkeypatch, backend, distance, out_dtype):
    """Queued labelling with several subchunks per chunk and chunks not aligned to them."""
    from aweSOM import _gpu

    for module, name in [(_gpu, 'GpuSearcher'), (_gpu_screen, 'GpuScreenSearcher')]:
        base = getattr(module, name)
        small = type('Small' + name, (base,), {'__init__': lambda self, *a, base=base, **k:
                     base.__init__(self, *a, chunk=1000, **k)})
        monkeypatch.setattr(module, name, small)
    w, x = _stream_case(10_007, 73)
    _stream_parity(backend, w, x, np.arange(len(w)) % 7, 2500, distance, out_dtype)


@pytest.mark.parametrize('backend', ['cuda', 'cuda_screen'])
@pytest.mark.parametrize('distance', [False, True])
def test_device_predict_stream_large(cuda, backend, distance):
    """Default 2**20-row subchunks; chunks not aligned to them; N a multiple of neither."""
    w, x = _stream_case(3 * 2**20 + 12_345, 1600)
    _stream_parity(backend, w, x, np.arange(len(w)), 2**20 + 777, distance, None)
    _stream_parity(backend, w, x, np.arange(len(w)) % 200, 2**20, distance, np.uint8)


def _round32(value):
    """Correctly round an exact Fraction to float32, ties to even."""
    near = np.float32(float(value))
    options = [near, np.nextafter(near, np.float32(-np.inf)), np.nextafter(near, np.float32(np.inf))]
    options = [o for o in options if np.isfinite(o)]
    key = lambda o: (abs(Fraction(float(o)) - value), int(np.array(o).view(np.uint32)) & 1)
    return min(options, key=key)


def test_fma32_single_rounding():
    f32 = np.float32
    a, b, c = f32(-(1 + 2**-15) * 2**-12), f32((1 - 2**-15) * 2**-12), f32(1 + 2**-23)
    assert np.float32(np.float64(a) * np.float64(b) + np.float64(c)) == f32(1.0)  # double rounding
    assert _gpu_screen._fma32(a, b, c) == f32(1 + 2**-23)
    rng = np.random.default_rng(3)
    triples = rng.normal(size=(4000, 3)).astype(f32)
    triples[:2000, 2] *= f32(2.0**-20)  # products dominate; many sums near float32 midpoints
    near = rng.integers(-2**12, 2**12, size=(2000, 3)).astype(f32) * f32(2**-12)
    near[:, 2] += f32(2**-23)
    for a, b, c in np.concatenate([triples, near]):
        exact = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
        assert _gpu_screen._fma32(a, b, c) == _round32(exact), (a, b, c)
    # d * d + acc as in the kernel, at the overflow edge
    big = f32(2e19)  # 4e38 > FLT_MAX
    assert _gpu_screen._fma32(big, big, f32(0)) == np.inf
    assert _gpu_screen._fma32(big, big, f32(-np.inf)) == -np.inf


def _blocked_case(case, features, n, seed=9):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(301, features))  # more than two node tiles of 128
    x = rng.normal(size=(n, features))
    if case == 'smooth':
        x = w[rng.integers(len(w), size=n)] + rng.normal(size=x.shape) * 0.05
    elif case == 'mixed':
        w[1] = w[0]
        w[200] = w[0]  # duplicate in a later tile
        w[5] = w[4]
        w[5, 0] = np.nextafter(w[4, 0], np.inf)
        x[:7] = w[0]
        x[7:11] = w[4]
        x[11], x[12], x[13] = np.nan, np.inf, 1e30
        x[14:17] = (w[2] + w[3]) / 2
        x[300:310] = w[0] + 1e-9
    elif case == 'large':
        w *= 1e12
        x *= 1e12
    return w, x


@pytest.mark.parametrize('features', [1, 3, 7, 8, 10])
@pytest.mark.parametrize('case', ['random', 'smooth', 'mixed', 'large'])
def test_device_row_blocked(cuda, case, features):
    """Rows not a multiple of R or of the block, several tiles, both input dtypes."""
    from aweSOM._gpu import best_match_nodes_cuda

    w, x = _blocked_case(case, features, 4 * 128 * 3 + 37)
    for dtype in (np.float32, np.float64):
        points = x.astype(dtype)
        expected = best_match_nodes(w, points)
        _same(_gpu_screen._cuda_screen_reference(w, points), expected)
        _same(best_match_nodes_cuda(w, points), expected)
        _same(_gpu_screen.best_match_nodes_cuda_screen(w, points, chunk=1001), expected)
        for block, tile in ((128, 128), (64, 16), (32, 128)):
            search = _gpu_screen.GpuScreenSearcher(w, len(w), 1, chunk=999, block=block, tile=tile)
            idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
            search(points, idx, d2)
            _same((idx, d2), expected)


OVERFLOW = [  # an overflowed float32 runner-up must not certify (reviewer inputs)
    (np.array([[18373320559154.32, -19118858046266.434], [14490963125571.676, -14263481110883.94]]),
     np.array([[1.3043817436596711e19, 1.3043817436596711e19]])),
    (np.array([[1e10, 0, -1e9], [-1e10, 0, 1e9]]),
     np.array([[np.float32(2**64 - 2**41), np.float32(np.sqrt(3) * 2**52),
                np.float32(np.sqrt(0.5) * 2**52)]], np.float64)),
]


@pytest.mark.parametrize('k', range(len(OVERFLOW)))
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_overflowed_runner_up(k, dtype):
    w, x = OVERFLOW[k]
    x = x.astype(dtype)
    expected = best_match_nodes(w, x)
    for actual in (best_match_nodes_screen(w, x, return_fallbacks=True),
                   _gpu_screen._cuda_screen_reference(w, x, return_fallbacks=True)):
        _same(actual, expected)
        assert actual[2] == 1


@pytest.mark.parametrize('k', range(len(OVERFLOW)))
def test_device_overflowed_runner_up(cuda, k):
    w, x = OVERFLOW[k]
    for dtype in (np.float32, np.float64):
        _same(_gpu_screen.best_match_nodes_cuda_screen(w, x.astype(dtype)),
              best_match_nodes(w, x.astype(dtype)))


def test_fma32_overflow_midpoint():
    """Exact a*b + c just inside or outside +-(2**128 - 2**103), the float32 overflow midpoint."""
    f32 = np.float32
    a, b = f32(18631), f32(1801 * 2.0**103)  # a * b = 2**128 - 2**103 exactly
    tiny = f32(2**-149)
    big = np.finfo(np.float32).max
    assert _gpu_screen._fma32(a, b, -tiny) == big
    assert _gpu_screen._fma32(-a, b, tiny) == -big
    assert _gpu_screen._fma32(a, b, tiny) == np.inf
    assert _gpu_screen._fma32(-a, b, -tiny) == -np.inf


def test_reference_unsafe_without_overflow():
    w, x = _case('unsafe', 7, np.float64)
    with np.errstate(over='raise'):
        _same(_gpu_screen._cuda_screen_reference(w, x), best_match_nodes(w, x))
