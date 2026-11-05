"""Small inputs to the legacy best_match kernel and restart-lattice training.

The expected arrays in fixtures/restart_training_v1.1.1.npz were produced by
the released aweSOM 1.1.1 on the stored inputs with the stored sample order:
one 500-step batch from a restart array passed as a fresh copy (1.1.1 trained
it in place). Since 1.5.0 a restart lattice is copied to float64 before
training, so only the float64 reference still applies; the float32 and
two-batch entries of the fixture describe 1.1.1 behaviour and are not used.

The comparisons allow rtol=1e-5, atol=1e-6, because exp() in the
neighbourhood function can round differently across platforms and library
builds. Set AWESOM_EXACT_FIXTURES=1 to require byte equality as well; the
release check does this on the platform that generated the fixture.
"""

import contextlib
import io
import os
from pathlib import Path

import numpy as np
import pytest

from aweSOM import Lattice

FIXTURE = Path(__file__).parent / "fixtures" / "restart_training_v1.1.1.npz"


class FixedOrder:
    """Stand-in for numpy.random.default_rng(): shuffle() applies a stored permutation."""

    def __init__(self, perm):
        self.perm = perm

    def shuffle(self, x):
        x[:] = x[self.perm]


@pytest.fixture(scope="module")
def ref():
    with np.load(FIXTURE) as f:
        return {k: f[k] for k in f.files}


def _train(ref, restart, monkeypatch):
    xdim, ydim, train = (int(v) for v in ref["params"])
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: FixedOrder(ref["perm"]))
    som = Lattice(xdim, ydim, float(ref["alpha"]), train)
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(ref["data"], list("abc"), restart_lattice=restart)
    return som


def _assert_matches_reference(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    if os.environ.get("AWESOM_EXACT_FIXTURES") == "1":
        assert actual.tobytes() == expected.tobytes()


def _two_batches(ref, seed=None):
    # two short batches; the second restarts from som.lattice
    xdim, ydim, train, steps = (int(v) for v in ref["two_batch_params"])
    som = Lattice(xdim, ydim, float(ref["alpha"]), train)
    if seed is not None:
        som.seed = seed
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(ref["data"], list("abc"), number_of_steps=steps)
        first_batch = som.lattice
        som.train_lattice(
            ref["data"], list("abc"), number_of_steps=steps, restart_lattice=first_batch
        )
    return som, first_batch


@pytest.mark.parametrize("n", [1, 5, 9])
def test_best_match_with_fewer_than_ten_points(n):
    # the progress print used i % (N // 10), which divided by zero for N < 10
    rng = np.random.default_rng(0)
    lattice = rng.random((20, 4))
    obs = rng.random((n, 4))
    expected = np.argmin(((lattice[None, :, :] - obs[:, None, :]) ** 2).sum(axis=2), axis=1)

    with contextlib.redirect_stdout(io.StringIO()):
        bmu = Lattice.best_match(lattice, obs)
        bmu_full = Lattice.best_match(lattice, obs, full=True)

    assert bmu.shape == (n, 1)
    assert bmu_full.shape == (n, 2)
    assert np.array_equal(bmu[:, 0], expected)
    assert np.array_equal(bmu_full[:, 0], expected)
    assert np.all(bmu_full[:, 1] != bmu_full[:, 0])


@pytest.mark.parametrize("xdim, ydim, train", [(7, 5, 1000), (10, 8, 3)])
def test_lattice_side_seven_and_batch_below_four_steps(xdim, ydim, train):
    # both made nsize_max == nsize_min, and the neighbourhood period divided by zero
    data = np.random.default_rng(1).normal(size=(300, 3))
    som = Lattice(xdim, ydim, 0.3, train)
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(data, list("abc"))
    assert som.epoch == train
    assert som.lattice.shape == (xdim * ydim, 3)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_restart_lattice_is_not_modified(ref, dtype, monkeypatch):
    restart = ref["restart64"].astype(dtype)
    before = restart.copy()
    som = _train(ref, restart, monkeypatch)
    assert restart.dtype == before.dtype
    assert restart.tobytes() == before.tobytes()
    assert not np.shares_memory(restart, som.lattice)


def test_float64_restart_training_matches_v1_1_1(ref, monkeypatch):
    som = _train(ref, ref["restart64"], monkeypatch)
    _assert_matches_reference(som.lattice, ref["expected64"])


def test_float32_restart_trains_in_float64(ref, monkeypatch):
    # a float32 restart array is copied to float64: training from it equals
    # training from its float64 cast, bit for bit
    restart = ref["restart32"]
    som32 = _train(ref, restart, monkeypatch)
    som64 = _train(ref, restart.astype(np.float64), monkeypatch)
    assert restart.dtype == np.float32
    assert som32.lattice.dtype == np.float64
    assert som32.lattice.tobytes() == som64.lattice.tobytes()


def test_restart_from_own_lattice_uses_final_weights(ref):
    # the second batch has no snapshot after its first step, so the U-matrix
    # must come from the final weights, not from a stored snapshot
    som, first_batch = _two_batches(ref)
    assert not np.shares_memory(first_batch, som.lattice)
    assert som.lattice.dtype == np.float64
    np.testing.assert_array_equal(som.umat, som.compute_umat())
    with contextlib.redirect_stdout(io.StringIO()):
        clusters = som.assign_cluster_to_lattice(merge_cost=0.0).copy()
        som.umat = som.compute_umat()
        recomputed = som.assign_cluster_to_lattice(merge_cost=0.0)
    assert np.array_equal(clusters, recomputed)


def test_same_seed_and_data_reproduce_the_weights(ref):
    som_a, _ = _two_batches(ref, seed=7)
    som_b, _ = _two_batches(ref, seed=7)
    som_c, _ = _two_batches(ref, seed=8)
    assert som_a.seed == som_b.seed == 8  # the restart moved the seed on by one
    assert som_a.lattice.tobytes() == som_b.lattice.tobytes()
    assert som_a.umat.tobytes() == som_b.umat.tobytes()
    assert som_a.lattice.tobytes() != som_c.lattice.tobytes()
