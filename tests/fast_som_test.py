"""Bitwise check of the compiled online trainer against the original NumPy loop.

reference_train is a frozen copy of the Lattice.fast_som loop as it was before the
numba kernel (git HEAD of research/grmhd-scale at the time of the change), reduced
to the arithmetic: same generator use, same schedule, same per-step operations.
"""

import contextlib
import io

import numpy as np
import pytest

from aweSOM import Lattice


def reference_train(data, xdim, ydim, alpha_0, train, alpha_type, sampling_type, seed):
    """Original fast_som for a fresh lattice (epoch 0, no restart).

    Returns the final lattice, the snapshot list and the final learning rate."""
    number_input_vectors, number_features = data.shape
    number_nodes = xdim * ydim
    this_batch_train = train
    save_frequency = train // 200 if train > 200 else 5
    alpha_type = 1 if alpha_type == "decay" else 0

    rng = np.random.default_rng(seed)
    if sampling_type == "uniform":
        lattice = rng.uniform(0.0, 1.0, (number_nodes, number_features))
    else:
        ix = rng.integers(0, number_input_vectors, number_nodes)
        lattice = data[ix, :].astype(np.float64, copy=True)
    history = [lattice.copy()]

    alpha = alpha_0
    alpha_freq = max(1, train // 25) if alpha_type == 1 else 1
    nsize_max = max(xdim, ydim) + 1
    nsize_min = (
        8 if this_batch_train // 4 > nsize_max - 8 else nsize_max - this_batch_train // 4
    )
    nsize_step = this_batch_train // 4
    nsize_freq = max(1, nsize_step // max(1, nsize_max - nsize_min))
    nsize = nsize_max

    m = np.reshape(list(range(number_nodes)), (number_nodes, 1))
    m2Ds = Lattice.coordinate(m, xdim)

    indices = np.arange(number_input_vectors)
    rng.shuffle(indices)
    if this_batch_train > number_input_vectors:
        indices = np.tile(indices, this_batch_train // number_input_vectors + 1)
    xk = data[indices[:this_batch_train], :]

    for epoch in range(this_batch_train):
        xk_m = xk[epoch, :]
        diff = lattice - xk_m
        squ = diff**2
        s = np.sum(squ, axis=1)
        c = np.argmin(s)
        gamma_m = np.outer(Lattice.Gamma(c, m2Ds, alpha, nsize), np.ones(number_features))
        lattice -= diff * gamma_m

        if epoch > 2 * nsize_step and epoch % nsize_freq == 0 and nsize > nsize_min:
            nsize = nsize_max - (epoch - 2 * nsize_step) // nsize_freq
        if epoch % alpha_freq == 0 and alpha_type == 1 and epoch != 0:
            alpha *= 0.75
        if epoch % save_frequency == 0:
            history.append(lattice.copy())

    return lattice, history, alpha


def _train_new(data, xdim, ydim, alpha_0, train, alpha_type, sampling_type, seed, history=True):
    som = Lattice(xdim, ydim, alpha_0, train, alpha_type=alpha_type, sampling_type=sampling_type)
    som.seed = seed
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(data, [f"f{i}" for i in range(data.shape[1])], history=history)
    return som


DATA = np.random.default_rng(7).normal(size=(3000, 4)) * np.array([1.0, 3.0, 0.2, 10.0])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("sampling_type", ["sampling", "uniform"])
@pytest.mark.parametrize("alpha_type", ["decay", "static"])
@pytest.mark.parametrize("train", [300, 1000])
def test_kernel_is_bitwise_equal_to_reference(seed, sampling_type, alpha_type, train):
    ref_lattice, ref_history, ref_alpha = reference_train(
        DATA, 12, 9, 0.3, train, alpha_type, sampling_type, seed
    )
    som = _train_new(DATA, 12, 9, 0.3, train, alpha_type, sampling_type, seed)
    assert np.array_equal(som.lattice, ref_lattice)
    assert som.alpha == ref_alpha
    # the first history entry aliases the training array, as in the original code
    assert len(som.lattice_history) == len(ref_history)
    for got, want in zip(som.lattice_history[1:], ref_history[1:]):
        assert np.array_equal(got, want)
    assert len(som.umat_history) == len(ref_history)


@pytest.mark.parametrize("n_features", [8, 10, 17])
def test_kernel_matches_numpy_pairwise_sum_for_many_features(n_features):
    data = np.random.default_rng(n_features).normal(size=(1500, n_features))
    ref_lattice, _, _ = reference_train(data, 10, 8, 0.3, 600, "decay", "sampling", 3)
    som = _train_new(data, 10, 8, 0.3, 600, "decay", "sampling", 3)
    assert np.array_equal(som.lattice, ref_lattice)


def test_history_off_gives_same_lattice_and_no_snapshots():
    on = _train_new(DATA, 12, 9, 0.3, 1000, "decay", "sampling", 0, history=True)
    off = _train_new(DATA, 12, 9, 0.3, 1000, "decay", "sampling", 0, history=False)
    assert np.array_equal(on.lattice, off.lattice)
    assert on.alpha == off.alpha and on.epoch == off.epoch
    assert off.lattice_history == [] and off.umat_history == []
    assert off.umat.shape == (12, 9)
    # without snapshots the U-matrix is computed from the final lattice
    assert np.array_equal(off.umat, off.compute_umat())


def test_training_restart_continues_from_given_lattice():
    som = _train_new(DATA, 12, 9, 0.3, 1000, "decay", "sampling", 0, history=False)
    weights = som.lattice.copy()
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(
            DATA, list("abcd"), number_of_steps=200, restart_lattice=weights, history=False
        )
    assert som.epoch == 1200
    assert not np.shares_memory(weights, som.lattice)
    assert not np.array_equal(weights, som.lattice)


@pytest.mark.parametrize("xdim,ydim", [(12, 9), (64, 37), (5, 30)])
def test_inline_neighborhood_weight_is_bitwise_equal_to_gamma(xdim, ydim):
    from aweSOM.som import _neighborhood_weight

    n = xdim * ydim
    m2Ds = Lattice.coordinate(np.arange(n).reshape(n, 1), xdim)
    alphas = [0.3, 0.1 * 0.75**7, 1, 0.5 * 0.75**24]
    for nsize in sorted({1, 3, 8, 9, max(xdim, ydim), max(xdim, ydim) + 1}):
        for alpha in alphas:
            for c in range(0, n, 7):
                want = Lattice.Gamma(c, m2Ds, float(alpha), nsize)
                cheb = np.max(np.abs(m2Ds[c] - m2Ds), axis=1)
                got = np.zeros(n)
                for i in np.flatnonzero(cheb <= nsize):
                    got[i] = _neighborhood_weight(float(cheb[i]), float(alpha), nsize)
                assert np.array_equal(got, want)
