"""Tests for the streaming CPU batch trainer (aweSOM.stream). Small and fast; no JAX needed
except for the comparison with aweSOM.batch.train_batch, which is skipped without it."""

import numpy as np
import pytest

from aweSOM._batch_core import neighbourhood_matrix, quantization_error, radius_schedule
from aweSOM.inference import best_match_nodes, predict
from aweSOM.stream import (
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


def test_matches_train_batch_kdtree_numba(data):
    pytest.importorskip("jax")
    from aweSOM.batch import train_batch_kdtree

    w_ref, i_ref = train_batch_kdtree(data, 9, 6, epochs=8, bmu="numba")
    w, info = train_batch_stream(data, 9, 6, epochs=8, chunk=777)
    np.testing.assert_allclose(w, w_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(info["hits"], i_ref["hits"])
    np.testing.assert_allclose(info["qe"], i_ref["qe"], rtol=1e-12)


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


def test_stratified_sample_size_and_dtype():
    x = np.arange(100_000 * 2, dtype=np.float32).reshape(-1, 2)
    s = stratified_sample(x, 10_000, block=100, seed=1)
    assert s.shape == (10_000, 2) and s.dtype == np.float32
    # whole blocks of consecutive rows
    assert np.all(np.diff(s[:100, 0]) == 2)
