"""Tests for the streaming CPU batch trainer (aweSOM.stream). Small and fast; no JAX needed
except for the comparison with aweSOM.batch.train_batch, which is skipped without it."""

import importlib.util
import math
import sys

import numpy as np
import pytest

import aweSOM.stream as stream
from aweSOM._batch_core import neighbourhood_matrix, quantization_error, radius_schedule
from aweSOM.stream import radius_phases
from aweSOM.inference import best_match_nodes, predict
from aweSOM.stream import (
    NpyReader,
    _stratified_block_starts,
    neighbourhood_update,
    predict_stream,
    stratified_sample,
    train_batch_stream,
)

CENTERS = np.array([[0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0], [0.0, 3.0, 3.0, 3.0]])


def three_gaussians(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return CENTERS[rng.integers(0, 3, n)] + rng.normal(0.0, 0.5, (n, 4))


@pytest.fixture(scope="module")
def data():
    return three_gaussians(6000, seed=0)


def numpy_reference(x, xdim, ydim, epochs, seed=42, r_min=8):
    """Plain NumPy batch map with exact float64 BMUs and the dense float64 H."""
    nodes = xdim * ydim
    w = x[np.random.default_rng(seed).integers(0, len(x), nodes)].astype(np.float64)
    for r in radius_schedule(epochs, max(xdim, ydim) + 1, min(r_min, max(xdim, ydim) + 1)):
        bmu, _ = best_match_nodes(w, x)
        s = np.zeros_like(w)
        np.add.at(s, bmu, x.astype(np.float64))
        n = np.bincount(bmu, minlength=nodes).astype(np.float64)
        h = _h64(xdim, ydim, r)
        num, den = h.T @ s, h.T @ n
        w = np.where((den > 0)[:, None], num / np.where(den > 0, den, 1.0)[:, None], w)
    return w


def _h64(xdim, ydim, r):
    """Dense float64 neighbourhood matrix, as used by aweSOM.batch.train_batch."""
    from aweSOM._batch_core import _neighbourhood64

    return _neighbourhood64(xdim, ydim, r)


def test_dense_matrix_helper_matches_public_float32():
    np.testing.assert_allclose(_h64(7, 5, 3.0), neighbourhood_matrix(7, 5, 3.0), rtol=1e-7)


def test_matches_numpy_reference(data):
    w, info = train_batch_stream(data, 9, 6, epochs=8, chunk=1000)
    ref = numpy_reference(data, 9, 6, 8)
    np.testing.assert_allclose(w, ref, rtol=1e-12, atol=1e-12)
    assert info["hits"].sum() == len(data)
    assert info["qe"][-1] == pytest.approx(quantization_error(w, data), rel=1e-12)


def test_matches_jax_train_batch_float64(data):
    pytest.importorskip("jax")
    from aweSOM.batch import train_batch

    w_ref, _ = train_batch(data, 9, 6, epochs=8, dtype=np.float64)
    w, _ = train_batch_stream(data, 9, 6, epochs=8)
    np.testing.assert_allclose(w, w_ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("radius", [0.0, 1.0, 2.5, 7.0, 40.0])
def test_neighbourhood_update_equals_dense(radius):
    rng = np.random.default_rng(1)
    xdim, ydim = 7, 5
    s = rng.normal(size=(35, 3))
    n = rng.integers(0, 4, 35).astype(float)
    s[n == 0] = 0.0
    w = rng.normal(size=(35, 3))
    h = _h64(xdim, ydim, radius)
    num, den = h.T @ s, h.T @ n
    ref = np.where((den > 0)[:, None], num / np.where(den > 0, den, 1)[:, None], w)
    np.testing.assert_allclose(neighbourhood_update(s, n, w, xdim, ydim, radius), ref, rtol=1e-13, atol=1e-14)


def test_bmu_backends_and_threads_bitwise(data):
    import numba

    x32 = data.astype(np.float32)
    w_s, i_s = train_batch_stream(x32, 9, 6, epochs=6, chunk=1000, bmu="screen")
    w_n, i_n = train_batch_stream(x32, 9, 6, epochs=6, chunk=1000, bmu="numba")
    w_p, i_p = train_batch_stream(x32, 9, 6, epochs=6, chunk=1000, bmu="patch")
    np.testing.assert_array_equal(w_s, w_n)
    np.testing.assert_array_equal(w_s, w_p)
    if numba.config.NUMBA_NUM_THREADS > 1:
        w_1, _ = train_batch_stream(x32, 9, 6, epochs=6, chunk=1000, n_threads=1)
        np.testing.assert_array_equal(w_s, w_1)


def test_sources_memmap_h5py_callable(tmp_path, data):
    x32 = data.astype(np.float32)
    w_ref, i_ref = train_batch_stream(x32, 6, 4, epochs=5, chunk=1024)
    mm = np.lib.format.open_memmap(tmp_path / "x.npy", mode="w+", dtype=np.float32, shape=x32.shape)
    mm[:] = x32
    mm.flush()
    w_mm, _ = train_batch_stream(np.load(tmp_path / "x.npy", mmap_mode="r"), 6, 4, epochs=5, chunk=1024)
    np.testing.assert_array_equal(w_mm, w_ref)
    w_rd, _ = train_batch_stream(NpyReader(tmp_path / "x.npy"), 6, 4, epochs=5, chunk=1024)
    np.testing.assert_array_equal(w_rd, w_ref)
    np.testing.assert_array_equal(NpyReader(tmp_path / "x.npy")[17:4000], x32[17:4000])
    w_np, _ = train_batch_stream(x32, 6, 4, epochs=5, chunk=1024, prefetch=False)
    np.testing.assert_array_equal(w_np, w_ref)

    def chunks():
        for s in range(0, len(x32), 1024):
            yield x32[s : s + 1024]

    w_cb, i_cb = train_batch_stream(chunks, 6, 4, epochs=5)
    np.testing.assert_array_equal(w_cb, w_ref)
    assert i_cb["n_samples"] == len(x32)

    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "x.h5", "w") as f:
        f.create_dataset("features", data=x32, chunks=(512, 4))
    with h5py.File(tmp_path / "x.h5", "r") as f:
        w_h5, _ = train_batch_stream(f["features"], 6, 4, epochs=5, chunk=1024)
    np.testing.assert_array_equal(w_h5, w_ref)


def test_degenerate_nodes_keep_weights(data):
    initial = np.zeros((24, 4))
    initial[:, 0] = 10.0 + np.arange(24)
    w, info = train_batch_stream(data, 6, 4, epochs=1, r_max=0, r_min=0, initial_weights=initial)
    bmu, _ = best_match_nodes(initial, data)
    hit = np.bincount(bmu, minlength=24) > 0
    assert 0 < hit.sum() < 24
    np.testing.assert_array_equal(w[~hit], initial[~hit])
    for c in np.flatnonzero(hit):
        np.testing.assert_allclose(w[c], data[bmu == c].mean(axis=0), rtol=1e-12)


def test_stratified_blocks_cover_every_stratum():
    rng = np.random.default_rng(0)
    starts = _stratified_block_starts(100_000, 100, 0.05, rng)
    assert len(starts) == 50 and np.all(np.diff(starts) > 0) and np.all(starts % 100 == 0)
    strata = (starts // 100) // 20  # 1000 blocks, 50 strata of 20
    np.testing.assert_array_equal(strata, np.arange(50))


def test_sample_fraction(data):
    x = three_gaussians(40_000, seed=5)
    w_f, i_f = train_batch_stream(x, 8, 6, epochs=10, sample_fraction=0.25, sample_block=50)
    assert np.all(np.abs(i_f["rows_per_epoch"] - 10_000) <= 50)
    w_f2, _ = train_batch_stream(x, 8, 6, epochs=10, sample_fraction=0.25, sample_block=50)
    np.testing.assert_array_equal(w_f, w_f2)  # reproducible
    w_full, i_full = train_batch_stream(x, 8, 6, epochs=10)
    assert i_f["hits"].sum() == len(x)  # final pass over all rows by default
    assert quantization_error(w_f, x) < 1.03 * quantization_error(w_full, x)
    # callable sources use a Bernoulli row sample
    w_cb, i_cb = train_batch_stream(lambda: iter([x[:20_000], x[20_000:]]), 8, 6, epochs=3,
                                    sample_fraction=0.5, final_pass="sample")
    assert 19_000 < i_cb["rows_per_epoch"][0] < 21_000


def test_predict_stream_matches_predict(tmp_path):
    rng = np.random.default_rng(3)
    w = rng.normal(size=(30, 5))
    x = rng.normal(size=(5000, 5)).astype(np.float32)
    clusters = rng.integers(-1, 4, 30)
    mm_path = tmp_path / "x.npy"
    np.save(mm_path, x)
    out = np.lib.format.open_memmap(tmp_path / "labels.npy", mode="w+", dtype=np.int32, shape=(5000,))
    dist = np.empty(5000)
    _, stats = predict_stream(w, clusters, np.load(mm_path, mmap_mode="r"), out=out, out_distance=dist, chunk=999)
    ref, ref_d = predict(w, clusters, x, return_distance=True)
    np.testing.assert_array_equal(np.asarray(out), ref)
    np.testing.assert_array_equal(dist, ref_d)
    assert stats["rows"] == 5000


@pytest.mark.parametrize("clusters,dtype", [
    (np.arange(10_000) % 10, np.uint8),  # 10 clusters
    (np.arange(10_000) % 300, np.uint16),  # 300 clusters
    (np.arange(10_000), np.uint16),  # raw node IDs of a 100x100 lattice
    (np.arange(10_000) % 10 - 1, np.int32),  # -1 = unassigned
    (np.arange(10_000) * 7, np.int32),  # ids above 65535
])
def test_predict_stream_default_dtype(clusters, dtype):
    """Without ``out`` the labels come in the narrowest dtype and equal the int32 labels."""
    rng = np.random.default_rng(5)
    w = rng.normal(size=(10_000, 3))
    x = rng.normal(size=(2_000, 3)).astype(np.float32)
    labels, _ = predict_stream(w, clusters, x, chunk=777, prefetch=False)
    wide = np.empty(len(x), np.int32)
    predict_stream(w, clusters, x, out=wide, chunk=777, prefetch=False)
    assert labels.dtype == dtype
    np.testing.assert_array_equal(labels.astype(np.int64), wide)
    np.testing.assert_array_equal(wide, predict(w, clusters, x))


def test_stratified_sample_size_and_dtype():
    x = np.arange(100_000 * 2, dtype=np.float32).reshape(-1, 2)
    s = stratified_sample(x, 10_000, block=100, seed=1)
    assert s.shape == (10_000, 2) and s.dtype == np.float32
    # whole blocks of consecutive rows
    assert np.all(np.diff(s[:100, 0]) == 2)


def scalar_bmu(w, x):
    """Independent ascending scan in Python floats: acc = 0; acc += d * d; strict <."""
    idx, dist2 = np.empty(len(x), np.int32), np.empty(len(x))
    nodes = np.asarray(w, np.float64).tolist()
    for p, row in enumerate(np.asarray(x).astype(np.float64).tolist()):
        best, winner = float('inf'), 0
        for i, node in enumerate(nodes):
            acc = 0.0
            for wf, xf in zip(node, row):
                d = wf - xf
                acc += d * d
            if acc < best:
                best, winner = acc, i
        idx[p], dist2[p] = winner, best
    return idx, dist2


def first_epoch_bmu(w, x, xd, yd):
    from aweSOM.stream import _BMU
    bmu = _BMU(w, 'patch', xd, yd, first_epoch=True)
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
    bmu(x, idx, d2)
    return idx, d2


def assert_same_bmu(actual, expected):
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(19, 13), (7, 3), (1, 1)])
def test_feature_grouped_bmu_exact(shape, dtype):
    """Repartitioning unordered nodes preserves reference arithmetic and ties."""
    rng = np.random.default_rng(184)
    xd, yd = shape
    w = rng.normal(size=(xd * yd, 7))
    w[-1] = w[0]
    x = rng.normal(size=(600, 7)).astype(dtype)
    x[:len(w)] = w
    x[-4, 0] = np.nan
    x[-3, 0] = np.inf
    x[-2] = 1e30
    x[-1] = 0.5 * (w[0] + w[len(w)//2])
    assert_same_bmu(first_epoch_bmu(w, x, xd, yd), best_match_nodes(w, x))


def test_unordered_patch_pruning():
    """Unordered initialization must not force nearly a full node scan."""
    from aweSOM.stream import _BMU
    from aweSOM.inference import _feature_patch_kernel
    rng = np.random.default_rng(184)
    # A thin, correlated distribution, with no lattice ordering.
    t = rng.uniform(-3, 3, (4096, 1))
    w = t + 0.05 * rng.normal(size=(4096, 7))
    x = rng.uniform(-3, 3, (512, 1)) + 0.05 * rng.normal(size=(512, 7))
    bmu = _BMU(w, 'patch', 64, 64, first_epoch=True)
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
    evaluations = _feature_patch_kernel(*bmu.patches, x, idx, d2)
    assert evaluations / len(x) < len(w) / 2
    assert_same_bmu((idx, d2), best_match_nodes(w, x))


@pytest.mark.parametrize('scale', [1.0, 1e-162, 1e-300, 5e-324, 1e15])
def test_feature_patches_partition_and_bounds(scale):
    """Exact rational check that every ball contains its members, at any scale."""
    from fractions import Fraction
    from aweSOM.inference import _feature_patches
    rng = np.random.default_rng(2)
    if scale == 5e-324:
        w = rng.integers(-40, 40, (257, 7)) * scale  # subnormal multiples, exact
    else:
        w = rng.normal(size=(257, 7)) * scale
    wp, perm, offs, cent, rad, _ = _feature_patches(w)
    np.testing.assert_array_equal(np.sort(perm), np.arange(len(w)))
    np.testing.assert_array_equal(wp.T, w[perm])
    assert offs[0] == 0 and offs[-1] == len(w)
    assert np.all((np.diff(offs) > 0) & (np.diff(offs) <= 16))
    for b in range(len(rad)):
        c = [Fraction(float(v)) for v in cent[:, b]]
        r2 = Fraction(float(rad[b])) ** 2
        for j in perm[offs[b]:offs[b+1]]:
            assert sum((Fraction(float(v)) - cf) ** 2 for v, cf in zip(w[j], c)) <= r2


def test_feature_search_underflow_regression():
    """Review case: squared deviations near 1e-162 underflow to zero."""
    from aweSOM.inference import best_match_nodes_patch
    rng = np.random.default_rng(1001)
    w = rng.normal(size=(35, 1)) * 1e-162
    x = rng.normal(size=(100, 1)) * 1e-162
    assert x[14, 0] == -3.950961781815714e-163
    actual = first_epoch_bmu(w, x, 7, 5)
    assert_same_bmu(actual, scalar_bmu(w, x))
    assert_same_bmu(actual, best_match_nodes(w, x))
    assert actual[0][14] == 0 and actual[1][14] == 0.0
    # The lattice search also retains the review's row-14 result.
    lattice = best_match_nodes_patch(w, x[14:15], 7, 5)
    assert_same_bmu((actual[0][14:15], actual[1][14:15]), lattice)


def test_feature_search_underflow_sweep():
    """The review's search loop, every trial, against the reference kernel."""
    rng = np.random.default_rng(1001)
    for scale in [1e-162, 2e-162, 1e-161, 1e-160]:
        for _ in range(100):
            w = rng.normal(size=(35, 1)) * scale
            x = rng.normal(size=(100, 1)) * scale
            assert_same_bmu(first_epoch_bmu(w, x, 7, 5), best_match_nodes(w, x))


def test_feature_search_underflow_training_weights():
    """One epoch on the review's query; final weights match the lattice baseline."""
    rng = np.random.default_rng(1001)
    w = rng.normal(size=(35, 1)) * 1e-162
    x = (rng.normal(size=(100, 1)) * 1e-162)[14:15]
    options = dict(xdim=7, ydim=5, epochs=1, initial_weights=w, r_max=0, r_min=0, final_pass='none')
    actual, info = train_batch_stream(x, bmu='patch', **options)
    expected = w.copy()
    expected[0] = x[0]  # radius 0: only the BMU, node 0, moves to its single row
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, train_batch_stream(x, bmu='numba', **options)[0])


def adversarial_cases():
    rng = np.random.default_rng(77)
    cases = {}
    # Subnormal grid: many exactly zero and tied rounded distances.
    w = rng.integers(-3, 4, (35, 2)) * 5e-324
    cases['subnormal'] = (w, np.concatenate([rng.integers(-4, 5, (80, 2)) * 5e-324, w]))
    # Clusters at very different scales; queries near the tiny cluster and zero.
    w = rng.normal(size=(35, 3))
    w[::2] *= 1e-162
    x = rng.normal(size=(120, 3)) * 1e-162
    x[:10] = 0.0
    cases['mixed'] = (w, x)
    # Exact ties across groups: duplicated nodes at mirrored positions, queries on the
    # mirror plane, at small, ordinary and large magnitudes.
    for scale in (1e-170, 1.0, 1e100):
        base = rng.normal(size=(7, 2))
        w = np.concatenate([base, -base, base, -base, base])[:35] * scale
        x = np.concatenate([np.zeros((5, 2)), base * scale, rng.normal(size=(40, 2)) * scale])
        cases[f'ties-{scale:g}'] = (w, x)
    # Overflow end: weights at the 1e15 limit, queries whose distances overflow to
    # inf for some nodes or all, or for centroids but not for nodes.
    w = rng.normal(size=(35, 2)) * 1e15
    w[0] = 1e15
    x = rng.normal(size=(60, 2)) * 1e15
    x[:20] *= np.logspace(130, 160, 20)[:, None]
    x[20] = [1.3407807929942596e154, 0.0]
    x[21] = [-1.34e154, 1.34e154]
    x[22] = [1.79e308, -1.79e308]
    x[23] = [np.inf, 0.0]
    x[24] = [-np.inf, np.nan]
    cases['overflow'] = (w, x)
    return cases


@pytest.mark.parametrize('name', list(adversarial_cases()))
def test_feature_search_adversarial(name):
    from aweSOM.inference import best_match_nodes_patch
    w, x = adversarial_cases()[name]
    actual = first_epoch_bmu(w, x, 7, 5)
    assert_same_bmu(actual, scalar_bmu(w, x))
    assert_same_bmu(actual, best_match_nodes(w, x))
    if np.all(np.isfinite(x)):
        assert_same_bmu(actual, best_match_nodes_patch(w, x, 7, 5))


@pytest.mark.parametrize('value', [np.nan, np.inf, 1e16])
def test_feature_search_preserves_weight_fallback(value):
    from aweSOM.stream import _BMU
    w = np.random.default_rng(2).normal(size=(35, 3))
    w[0, 0] = value
    x = np.zeros((4, 3))
    bmu = _BMU(w, 'patch', 7, 5, first_epoch=True)
    assert bmu.kind == 'numba'
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
    bmu(x, idx, d2)
    assert_same_bmu((idx, d2), best_match_nodes(w, x))


@pytest.mark.parametrize('bmu', ['patch', 'auto'])
def test_feature_groups_only_in_first_epoch(monkeypatch, bmu):
    """Later epochs and the final pass keep lattice groups and build no feature groups."""
    from aweSOM import stream
    built = []
    real = stream._feature_patches
    monkeypatch.setattr(stream, '_feature_patches', lambda w: built.append(1) or real(w))
    x = np.random.default_rng(3).normal(size=(3000, 5))
    train_batch_stream(x, 11, 7, epochs=3, chunk=700, bmu=bmu)
    assert len(built) == 1
    later = stream._BMU(x[:77], 'patch', 11, 7)
    assert not later.feature and len(later.patches) == 5


def test_feature_search_training_bitwise_against_reference():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4000, 7))
    options = dict(xdim=19, ydim=13, epochs=4, chunk=700)
    actual, info = train_batch_stream(x, bmu='patch', **options)
    expected, old_info = train_batch_stream(x, bmu='numba', **options)
    np.testing.assert_array_equal(actual, expected)
    for key in ('hits', 'qe', 'qe_initial'):
        np.testing.assert_array_equal(info[key], old_info[key])


@pytest.fixture
def lattice_underflow_case():
    # Reproduces review-work/underflow-case.npz exactly, without an external fixture.
    rng = np.random.default_rng(1001)
    return rng.normal(size=(35, 1)) * 1e-162, rng.normal(size=(100, 1)) * 1e-162


@pytest.mark.parametrize('row', [0, 82, 83, 85, 95])
def test_lattice_underflow_saved_rows(lattice_underflow_case, row):
    from aweSOM.inference import best_match_nodes_patch
    w, x = lattice_underflow_case
    expected = scalar_bmu(w, x[row:row+1])
    actual = best_match_nodes_patch(w, x[row:row+1], 7, 5)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1].view(np.uint64), expected[1].view(np.uint64))


@pytest.mark.parametrize('streamed', [False, True])
def test_lattice_underflow_frozen_prediction(lattice_underflow_case, streamed):
    w, x = lattice_underflow_case
    idx, d2 = scalar_bmu(w, x)
    clusters = np.arange(len(w), dtype=np.int32)
    if streamed:
        distances = np.empty(len(x))
        labels, _ = predict_stream(w, clusters, x, out_distance=distances,
                                  backend='patch', lattice_shape=(7, 5), chunk=17)
    else:
        labels, distances = predict(w, clusters, x, return_distance=True,
                                    backend='patch', lattice_shape=(7, 5))
    np.testing.assert_array_equal(labels, idx)
    np.testing.assert_array_equal(distances.view(np.uint64), np.sqrt(d2).view(np.uint64))


def test_lattice_underflow_radius_zero_training(lattice_underflow_case, monkeypatch):
    import aweSOM.stream as stream
    w, x = lattice_underflow_case
    x = x[:1]
    # Exercise the later-epoch lattice policy in a single isolated training epoch.
    real_bmu = stream._BMU
    def lattice_bmu(*args, **kwargs):
        kwargs['first_epoch'] = False
        return real_bmu(*args, **kwargs)
    monkeypatch.setattr(stream, '_BMU', lattice_bmu)
    expected = w.copy()
    idx, _ = scalar_bmu(w, x)
    assert idx[0] == 2
    expected[idx[0]] = x[0]
    actual, _ = train_batch_stream(x, 7, 5, epochs=1, initial_weights=w,
                                  r_max=0, r_min=0, bmu='patch', final_pass='none')
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize('scale', [2.0**-501, 2.0**-500, 2.0**-499, 1e-162])
def test_lattice_underflow_threshold_and_ties(scale):
    from aweSOM.inference import best_match_nodes_patch
    rng = np.random.default_rng(821)
    w = rng.normal(size=(35, 7)) * scale
    w[33] = w[1]
    x = np.concatenate([w, rng.normal(size=(100, 7)) * scale])
    expected = scalar_bmu(w, x)
    actual = best_match_nodes_patch(w, x, 7, 5, chunk=17)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1].view(np.uint64), expected[1].view(np.uint64))


def test_lattice_unsafe_centroid_rows():
    from aweSOM.inference import best_match_nodes_patch
    w = np.random.default_rng(13).normal(size=(35, 2))
    x = np.array([[np.nan, 0], [np.inf, 0], [1.79e308, 0], [1.34e154, 0]])
    expected = scalar_bmu(w, x)
    actual = best_match_nodes_patch(w, x, 7, 5)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1].view(np.uint64), expected[1].view(np.uint64))


def test_peak_rss_bytes_without_resource(monkeypatch, data):
    """No ``resource`` module (Windows): peak_rss_bytes returns None, never 0."""
    monkeypatch.setattr(stream, "resource", None)
    assert stream.peak_rss_bytes() is None
    _, info = train_batch_stream(data[:500], 3, 2, epochs=1, final_pass="none")
    assert info["peak_rss"] is None


def test_stream_imports_without_resource(monkeypatch):
    """A fresh import with ``resource`` blocked takes the ImportError branch."""
    monkeypatch.setitem(sys.modules, "resource", None)
    spec = importlib.util.spec_from_file_location("aweSOM._stream_without_resource", stream.__file__)
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, fresh)  # dataclasses look up their module
    spec.loader.exec_module(fresh)
    assert fresh.resource is None
    assert fresh.peak_rss_bytes() is None


# ------------------------------------------------------------------ radius phases


def _old_radius_schedule(epochs, r_max, r_min):
    """radius_schedule at 00e8551, before the phases argument (the phases=None reference)."""
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    n_max = math.ceil(epochs / 2)
    n_lin = min(math.ceil(epochs / 4), epochs - n_max)
    n_min = epochs - n_max - n_lin
    linear = np.linspace(r_max, r_min, n_lin + 2)[1:-1]
    return np.concatenate([np.full(n_max, float(r_max)), linear, np.full(n_min, float(r_min))])


def test_radius_schedule_default_60_is_20_10_30():
    assert radius_phases(60) == (20, 10, 30)
    r = radius_schedule(60, 101, 8)
    ref = np.concatenate([np.full(20, 101.0), np.linspace(101, 8, 12)[1:-1], np.full(30, 8.0)])
    assert r.dtype == np.float64 and r.tobytes() == ref.tobytes()
    assert radius_schedule(60, 101, 8, phases=(20, 10, 30)).tobytes() == ref.tobytes()
    assert ((r[20:30] < 101) & (r[20:30] > 8)).all()


@pytest.mark.parametrize("epochs", [1, 2, 3, 4, 5, 7, 9, 11, 13, 40, 41, 59, 61, 99])
@pytest.mark.parametrize("r_max, r_min", [(101, 8), (92.0, 8.0), (6, 2), (7.5, 7.5), (65, 8.0)])
def test_radius_schedule_without_phases_matches_old(epochs, r_max, r_min):
    r = radius_schedule(epochs, r_max, r_min)
    ref = _old_radius_schedule(epochs, r_max, r_min)
    assert r.dtype == ref.dtype and r.tobytes() == ref.tobytes()
    n_max = math.ceil(epochs / 2)
    n_lin = min(math.ceil(epochs / 4), epochs - n_max)
    assert radius_phases(epochs) == (n_max, n_lin, epochs - n_max - n_lin)


def test_radius_schedule_explicit_phases():
    assert radius_schedule(5, 6, 2, phases=(1, 2, 2)).tolist() == [6, 6 - 4 / 3, 6 - 8 / 3, 2, 2]
    assert radius_schedule(3, 6, 2, phases=[0, 0, 3]).tolist() == [2, 2, 2]
    assert radius_schedule(3, 6, 2, phases=[3, 0, 0]).tolist() == [6, 6, 6]
    assert radius_schedule(3, 6, 2, phases=np.array([0, 3, 0])).tolist() == [5, 4, 3]
    # explicit phases equal to the default split give the default radii
    for e in (40, 61):
        assert radius_schedule(e, 50, 8, phases=radius_phases(e)).tobytes() == radius_schedule(e, 50, 8).tobytes()
    assert radius_schedule(60, 50, 8, phases=(30, 15, 15)).tobytes() == _old_radius_schedule(60, 50, 8).tobytes()
    p = radius_phases(4, np.array([1, 2, 1]))
    assert p == (1, 2, 1) and all(type(v) is int for v in p)


@pytest.mark.parametrize(
    "epochs, phases",
    [(60, (20, 10, 31)), (40, (20, 10, 30)), (60, (20, 40)), (60, (20, 10, 10, 20)), (60, (-1, 31, 30)),
     (60, (20.0, 10, 30)), (60, (20.5, 9.5, 30)), (60, "abc"), (60, 60), (60, ()), (0, (0, 0, 0))],
)
def test_radius_phases_rejects_bad_phases(epochs, phases):
    with pytest.raises(ValueError):
        radius_phases(epochs, phases)
    with pytest.raises(ValueError):
        radius_schedule(epochs, 10, 2, phases=phases)


def test_train_batch_stream_default_is_60_epochs_20_10_30():
    x = three_gaussians(300, seed=1)
    _, info = train_batch_stream(x, 6, 5, chunk=128, r_min=2, final_pass="none")
    r = info["radius"]
    assert info["radius_phases"] == (20, 10, 30) and len(r) == 60 == len(info["qe"])
    assert (r[:20] == 7).all() and (r[30:] == 2).all() and ((r[20:30] < 7) & (r[20:30] > 2)).all()
    assert r.tobytes() == radius_schedule(60, 7, 2, phases=(20, 10, 30)).tobytes()
    _, info = train_batch_stream(x, 6, 5, epochs=40, chunk=128, r_min=2, final_pass="none")
    assert info["radius_phases"] == (20, 10, 10)
    assert info["radius"].tobytes() == _old_radius_schedule(40, 7, 2).tobytes()


def test_train_batch_stream_resume_checks_phases(tmp_path):
    class Stop(Exception):
        pass

    def stop_after(epoch):
        def cb(e, w):
            if e == epoch:
                raise Stop

        return cb

    x = three_gaussians(400, seed=2)
    kw = dict(epochs=5, phases=(1, 2, 2), chunk=64, r_max=6, r_min=2, final_pass="none")
    w_ref, i_ref = train_batch_stream(x, 6, 5, **kw)
    ck = str(tmp_path / "ck.npz")
    with pytest.raises(Stop):
        train_batch_stream(x, 6, 5, checkpoint=ck, resume=True, callback=stop_after(1), **kw)
    w, info = train_batch_stream(x, 6, 5, checkpoint=ck, resume=True, **kw)
    assert info["resumed_from"] == 2 and info["radius_phases"] == (1, 2, 2)
    assert w.tobytes() == w_ref.tobytes()
    assert info["radius"].tobytes() == i_ref["radius"].tobytes()
    # a checkpoint written with other phases is refused, also by the default split
    for phases in ((2, 1, 2), None):
        with pytest.raises(ValueError, match="radius"):
            train_batch_stream(x, 6, 5, checkpoint=ck, resume=True, **dict(kw, phases=phases))
