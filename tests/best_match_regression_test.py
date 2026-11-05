"""Regression cases for the Lattice.best_match backends.

Each case runs ``backend="screen"`` (the default), ``backend="numba"`` and the
original kernel ``Lattice._best_match_legacy``. Screen and numba always agree.
Where the legacy kernel can differ (float32 arithmetic near ties, non-finite
lattice nodes), the tests check the documented contract instead of equality.
"""

import contextlib
import io

import numpy as np
import pytest

from aweSOM import Lattice


def _three(w, x):
    with contextlib.redirect_stdout(io.StringIO()):
        screen = Lattice.best_match(w, x)
        numba = Lattice.best_match(w, x, backend="numba")
        legacy = Lattice._best_match_legacy(w, x)
    for out in (screen, numba, legacy):
        assert out.shape == (len(x), 1) and out.dtype == np.float64
    return (a.ravel().astype(np.int64) for a in (screen, numba, legacy))


def _dist2(w, x, idx):
    w64, x64 = np.asarray(w, np.float64), np.asarray(x, np.float64)
    return ((w64[idx] - x64) ** 2).sum(axis=1)


@pytest.mark.parametrize("F", [2, 4, 6, 10])
def test_float32_lattice_and_observations_near_ties(F):
    # points near the midpoint of two nodes: float32 sums can order such
    # nodes differently from the float64 accumulation of screen and numba
    rng = np.random.default_rng(11)
    w = rng.normal(size=(300, F)).astype(np.float32)
    a, b = rng.integers(0, 300, 5000), rng.integers(0, 300, 5000)
    a, b = a[a != b], b[a != b]
    mid = (w[a].astype(np.float64) + w[b].astype(np.float64)) / 2
    x = np.concatenate(
        [
            (mid + rng.normal(scale=1e-6, size=mid.shape)).astype(np.float32),
            rng.normal(size=(5000, F)).astype(np.float32),
        ]
    )
    screen, numba, legacy = _three(w, x)
    assert np.array_equal(screen, numba)
    differ = screen != legacy
    if differ.any():
        # the default is the float64 nearest node of the promoted values; the
        # legacy choice is only a float32 rounding away from it
        ds, dl = _dist2(w, x[differ], screen[differ]), _dist2(w, x[differ], legacy[differ])
        assert np.all(ds <= dl)
        assert np.all(dl - ds <= 4 * (F + 2) * 2.0**-24 * dl)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_exact_ties_go_to_lowest_node_on_every_backend(dtype):
    g = np.stack(np.meshgrid(np.arange(6.0), np.arange(5.0)), -1).reshape(-1, 2)
    w = np.concatenate([g, g[:7]]).astype(dtype)  # duplicated nodes
    x = np.concatenate([g + 0.5, g, g + 0.25, g + np.array([0.5, 0.0])]).astype(dtype)
    screen, numba, legacy = _three(w, x)
    assert np.array_equal(screen, numba) and np.array_equal(screen, legacy)
    d2 = ((x[:, None, :].astype(np.float64) - w[None].astype(np.float64)) ** 2).sum(-1)
    assert np.array_equal(screen, np.argmin(d2, axis=1))


def test_non_finite_observation_rows():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(50, 3))
    x = rng.normal(size=(200, 3))
    x[0, 1] = np.nan
    x[1, 0] = np.inf
    x[2] = -np.inf
    x[3] = np.nan
    x[4, 0], x[4, 1] = np.inf, -np.inf
    screen, numba, legacy = _three(w, x)
    assert np.array_equal(screen, numba) and np.array_equal(screen, legacy)
    # no finite distance: node 0 on every backend
    assert screen[:5].tolist() == [0, 0, 0, 0, 0]


def test_non_finite_lattice_nodes_are_never_selected():
    rng = np.random.default_rng(2)
    w = rng.normal(size=(50, 3))
    w[4, 0] = np.nan
    w[9, 1] = np.inf
    x = rng.normal(size=(200, 3))
    screen, numba, legacy = _three(w, x)
    assert np.array_equal(screen, numba)
    assert not np.isin(screen, [4, 9]).any()
    finite = np.isfinite(w).all(axis=1)
    d2 = ((x[:, None, :] - w[None, finite]) ** 2).sum(-1)
    assert np.array_equal(screen, np.flatnonzero(finite)[np.argmin(d2, axis=1)])
    # the legacy kernel returns the first NaN distance like np.argmin; with a
    # NaN node every legacy label may point at it, so no equality is asserted


@pytest.mark.parametrize("N", [0, 1, 3, 9])
def test_fewer_than_ten_observations(N):
    rng = np.random.default_rng(N)
    w = rng.normal(size=(80, 4))
    x = rng.normal(size=(N, 4))
    screen, numba, legacy = _three(w, x)
    assert np.array_equal(screen, numba) and np.array_equal(screen, legacy)
