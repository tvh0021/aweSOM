"""Tests for the Lattice wrappers added on research/grmhd-scale:
best_match delegation, predict, and train_lattice_batch."""

import contextlib
import io

import numpy as np
import pytest

from aweSOM import Lattice


@pytest.fixture(scope="module")
def trained():
    data = np.random.default_rng(3).normal(size=(3000, 4))
    som = Lattice(10, 8, 0.3, 600)
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(data, list("abcd"))
    return som, data


def test_best_match_matches_legacy_kernel_and_shape(trained):
    som, data = trained
    with contextlib.redirect_stdout(io.StringIO()):
        new = som.best_match(som.lattice, data)
        old = Lattice._best_match_legacy(som.lattice, data)
    assert new.shape == (len(data), 1)
    assert new.dtype == np.float64
    assert np.array_equal(new, old)


def test_best_match_full_still_returns_two_columns(trained):
    som, data = trained
    with contextlib.redirect_stdout(io.StringIO()):
        both = som.best_match(som.lattice, data[:50], full=True)
    assert both.shape == (50, 2)
    assert np.all(both[:, 0] != both[:, 1])


def test_best_match_handles_fewer_than_ten_points(trained):
    # The legacy kernel divided by obs.shape[0] // 10 == 0 for small inputs.
    som, data = trained
    out = som.best_match(som.lattice, data[:3])
    assert out.shape == (3, 1)


def test_predict_equals_map_then_assign(trained):
    som, data = trained
    with contextlib.redirect_stdout(io.StringIO()):
        clusters = som.assign_cluster_to_lattice(merge_cost=0.2)
        projection_2d = som.map_data_to_lattice()
        expected = som.assign_cluster_to_data(projection_2d, clusters)
        labels, dist = som.predict(data, return_distance=True, chunk=700)
    assert labels.dtype == np.int32
    assert np.array_equal(labels, expected.astype(np.int32))
    assert dist.shape == (len(data),) and np.all(dist >= 0)


def test_train_lattice_batch_fills_lattice_and_umat():
    jax = pytest.importorskip("jax")
    data = np.random.default_rng(5).normal(size=(4000, 3))
    som = Lattice(8, 6, 0.3, 100)
    info = som.train_lattice_batch(data, list("xyz"), epochs=5, r_min=2)
    assert som.lattice.shape == (48, 3) and np.all(np.isfinite(som.lattice))
    assert som.umat.shape == (8, 6)
    assert som.epoch == 5
    assert len(info["radius"]) == 5 and som.batch_info is info
    with contextlib.redirect_stdout(io.StringIO()):
        labels = som.predict(data, merge_cost=0.2)
    assert labels.shape == (4000,)
