"""CPU emulation of CUDA pruning, plus optional real-device parity checks."""
import numpy as np
import pytest

from aweSOM._gpu_prune import _cuda_prune_reference, _prepare, best_match_nodes_cuda_prune
from aweSOM.inference import best_match_nodes, predict


def smooth_lattice(n=32, features=7):
    u, v = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    return np.column_stack([u.ravel(), v.ravel()] + [
        (0.2 * np.sin((f + 1) * u + v)).ravel() for f in range(features - 2)])


def equal(got, ref):
    np.testing.assert_array_equal(got[0], ref[0])
    np.testing.assert_array_equal(got[1].view(np.uint64), ref[1].view(np.uint64))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('smooth', [False, True])
def test_random_and_clustered(dtype, smooth):
    rng = np.random.default_rng(20)
    w = smooth_lattice() if smooth else rng.normal(size=(1024, 7))
    x = np.concatenate([rng.normal(size=(200, 7)),
                        w[rng.integers(len(w), size=200)] + rng.normal(scale=0.01, size=(200, 7)),
                        np.full((2, 7), 1e30)]).astype(dtype)
    ref = best_match_nodes(w, x)
    for hint in (None, ref[0], np.full(len(x), len(w) - 1, np.int32),
                 rng.integers(len(w), size=len(x), dtype=np.int32)):
        equal(_cuda_prune_reference(w, x, (32, 32), hint, chunk=113), ref)


@pytest.mark.parametrize('case', ['ties', 'near', 'tiny', 'huge', 'nonfinite'])
def test_rounding_and_ties(case):
    rng = np.random.default_rng(9)
    w, x = rng.normal(size=(256, 7)), rng.normal(size=(40, 7))
    if case == 'ties':
        w[:] = 10
        w[0], w[255] = -1, 1
        w[128] = w[0]
        x[:] = 0  # seed the high-index equal-distance winner
    elif case == 'near':
        w[:] = 1
        w[0, 0] = np.nextafter(1., 2.)
        x[:] = 0
    elif case == 'tiny':
        w *= 1e-160
        x *= 1e-160
        x[0] = w[255]
    elif case == 'huge':
        w *= 1e200
        x *= 1e200
        x[0] = w[255]
    else:
        w[0, 0] = np.nan
        w[1, 0] = np.inf
        x[0, 0] = np.nan
        x[1, 0] = -np.inf
    ref = best_match_nodes(w, x)
    for hint in (None, np.full(len(x), 255, np.int32)):
        equal(_cuda_prune_reference(w, x, (16, 16), hint), ref)


def test_underflow_zero_tie_across_boxes():
    w = np.full((1024, 2), 10.)
    w[0] = 1e-170
    w[-1] = 0
    x = np.zeros((1, 2))
    equal(_cuda_prune_reference(w, x, (32, 32), np.array([1023], np.int32)),
          best_match_nodes(w, x))


def test_partition_and_validation():
    w = smooth_lattice()
    perm, offsets, boxes, finite = _prepare(w, (32, 32))
    assert sorted(perm) == list(range(len(w)))
    for b in range(len(boxes)):
        members = w[perm[offsets[b]:offsets[b + 1]]]
        assert (members >= boxes[b, 0]).all() and (members <= boxes[b, 1]).all()
    x = w[:3]
    for hint in (np.zeros(3), np.zeros(2, np.int32), np.array([0, 1, 1024], np.int32),
                 np.array([-1, 0, 0], np.int32)):
        with pytest.raises(ValueError):
            _cuda_prune_reference(w, x, hint=hint)
    with pytest.raises(ValueError):
        _cuda_prune_reference(w, x, (32, 31))
    with pytest.raises(ValueError):
        _cuda_prune_reference(w, x, chunk=0)
    equal(_cuda_prune_reference(w, x[:0]), best_match_nodes(w, x[:0]))


def test_trained_lattice_and_work_fraction():
    from aweSOM.stream import train_batch_stream
    rng = np.random.default_rng(41)
    cloud = smooth_lattice()
    x = cloud[rng.integers(len(cloud), size=2000)] + rng.normal(scale=0.01, size=(2000, 7))
    w, _ = train_batch_stream(x, 16, 16, epochs=4, chunk=1000, bmu='numba',
                              n_threads=2, prefetch=False, final_pass='none')
    ref = best_match_nodes(w, x[:200])
    for hint in (None, ref[0]):
        got = _cuda_prune_reference(w, x[:200], (16, 16), hint, return_evaluations=True)
        equal(got, ref)
        print('trained nodes checked, hint=%s: %.6f' % (hint is not None, got[2].mean() / len(w)))
    # Smooth large lattice is the latency proxy, independent of training noise.
    previous = rng.integers(len(cloud), size=1000, dtype=np.int32)
    rows = cloud[previous] + rng.normal(scale=0.002, size=(1000, 7))
    for hint in (None, previous):
        got = _cuda_prune_reference(cloud, rows, (32, 32), hint, return_evaluations=True)
        equal(got, best_match_nodes(cloud, rows))
        assert got[2].mean() / len(cloud) < 0.3
        print('smooth nodes checked, hint=%s: %.6f' % (hint is not None, got[2].mean() / len(cloud)))


def device():
    cp = pytest.importorskip('cupy', reason='CuPy is not installed')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('No CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA driver unavailable')


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('case', ['smooth', 'random', 'ties', 'near', 'tiny', 'huge', 'nonfinite'])
def test_device(dtype, case):
    device()
    from aweSOM._gpu import best_match_nodes_cuda
    rng = np.random.default_rng(28)
    w = smooth_lattice(32)
    x = w[rng.integers(len(w), size=259)] + rng.normal(scale=0.01, size=(259, 7))
    if case == 'random':
        w = rng.normal(size=w.shape)
    elif case == 'ties':
        w[:] = w[-1]
    elif case == 'near':
        w[:] = 1
        w[0, 0] = np.nextafter(1., 2.)
        x[:] = 0
    elif case == 'tiny':
        w *= 1e-160
        x *= 1e-160
    elif case == 'huge':
        w *= 1e200
        x *= 1e200
    elif case == 'nonfinite':
        w[0, 0] = np.nan
        x[1, 0] = np.inf
    x = x.astype(dtype)
    ref = best_match_nodes_cuda(w, x, chunk=113)
    for hint in (None, ref[0], np.full(len(x), 1023, np.int32)):
        got = best_match_nodes_cuda_prune(w, x, (32, 32), hint, chunk=113)
        equal(got, ref)
        equal(got, _cuda_prune_reference(w, x, (32, 32), hint))
    labels, distances = predict(w, np.arange(len(w)), x, backend='cuda_prune',
                                lattice_shape=(32, 32), hint=ref[0], return_distance=True)
    np.testing.assert_array_equal(labels, ref[0])
    np.testing.assert_array_equal(distances.view(np.uint64), np.sqrt(ref[1]).view(np.uint64))


def test_device_stream():
    device()
    from aweSOM.stream import predict_stream, _BMU
    w = smooth_lattice()
    x = w[::3].copy()
    hint = np.full(len(x), 1023, np.int32)
    ref = best_match_nodes(w, x)
    distances = np.empty(len(x))
    out, _ = predict_stream(w, np.arange(len(w)), x, out_distance=distances,
                            backend='cuda_prune', lattice_shape=(32, 32), hint=hint,
                            chunk=113, prefetch=False)
    np.testing.assert_array_equal(out, ref[0])
    np.testing.assert_array_equal(distances.view(np.uint64), np.sqrt(ref[1]).view(np.uint64))
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
    _BMU(w, 'cuda_prune', 32, 32)(x, idx, d2)
    equal((idx, d2), ref)


@pytest.mark.parametrize('features', [1, 2, 32])
def test_feature_extremes_and_no_shape(features):
    rng = np.random.default_rng(features)
    w = rng.normal(size=(193, features))
    x = rng.normal(size=(19, features))
    for hint in (None, np.full(len(x), 192, np.int32)):
        equal(_cuda_prune_reference(w, x, hint=hint), best_match_nodes(w, x))


def test_tied_bound_must_not_skip_lower_index():
    w = np.full((1024, 2), 100.)
    w[0], w[-1] = -1, 1
    x = np.zeros((1, 2))
    got = _cuda_prune_reference(w, x, (32, 32), np.array([1023], np.int32),
                               return_evaluations=True)
    equal(got, best_match_nodes(w, x))
    assert got[0][0] == 0
    assert got[2][0] < len(w) // 2  # exercise pruning, not full-scan fallback


def test_predict_dispatch_and_stream_hints(monkeypatch):
    from aweSOM import _gpu_prune
    from aweSOM.stream import predict_stream
    class EmulatedSearcher:
        def __init__(self, weights, lattice_shape=None, **kwargs):
            self.w, self.shape = weights, lattice_shape

        def __call__(self, data, idx, d2, hint=None):
            idx[:], d2[:] = _cuda_prune_reference(self.w, data, self.shape, hint)
            return 0

    monkeypatch.setattr(_gpu_prune, 'PruneGpuSearcher', EmulatedSearcher)
    w = smooth_lattice()
    x = w[::5].copy()
    hint = np.full(len(x), 1023, np.int32)
    clusters = np.arange(len(w)) % 7
    ref = predict(w, clusters, x, backend='numba', return_distance=True)
    got = predict(w, clusters, x, backend='cuda_prune', lattice_shape=(32, 32),
                  hint=hint, return_distance=True)
    equal(got, ref)
    distances = np.empty(len(x))
    labels, _ = predict_stream(w, clusters, x, backend='cuda_prune',
                              lattice_shape=(32, 32), hint=hint, chunk=37,
                              out_distance=distances, prefetch=False)
    equal((labels, distances), ref)


def test_device_global_metadata():
    device()
    from aweSOM._gpu_prune import PruneGpuSearcher
    from aweSOM._gpu import best_match_nodes_cuda
    rng = np.random.default_rng(1)
    w = rng.normal(size=(70 * 70, 32))
    x = w[[0, 2500, 4899]]
    search = PruneGpuSearcher(w, (70, 70))
    assert search.shared == 0
    i, d = np.empty(len(x), np.int32), np.empty(len(x))
    search(x, i, d)
    equal((i, d), best_match_nodes_cuda(w, x))


@pytest.mark.parametrize('features', [1, 7, 32])
@pytest.mark.parametrize('exponent', [-1070, -550, -520, -500, -100, 0, 100, 500, 520, 1000])
def test_exponent_sweep(features, exponent):
    rng = np.random.default_rng(246)
    w = np.ldexp(rng.uniform(-1, 1, (193, features)), exponent)
    x = np.ldexp(rng.uniform(-1, 1, (12, features)), exponent)
    x[0] = w[-1]
    ref = best_match_nodes(w, x)
    for hint in (None, np.full(12, 192, np.int32)):
        equal(_cuda_prune_reference(w, x, hint=hint), ref)
