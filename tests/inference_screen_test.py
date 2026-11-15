import numpy as np
import pytest

from aweSOM.inference import best_match_nodes, best_match_nodes_screen, predict


def _same(a, b):
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1].view(np.int64), b[1].view(np.int64))


@pytest.mark.parametrize("F", [1, 2, 4, 6, 10])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_screen_bitwise_random(F, dtype):
    rng = np.random.default_rng(F)
    w = rng.normal(size=(300, F))
    x = rng.normal(size=(5000, F)).astype(dtype)
    assert _same(best_match_nodes_screen(w, x), best_match_nodes(w, x))


def test_screen_ties_duplicates_and_grid():
    # integer grid: many exact ties; duplicated nodes; points on nodes
    g = np.stack(np.meshgrid(np.arange(6.0), np.arange(5.0)), -1).reshape(-1, 2)
    w = np.concatenate([g, g[:7]])
    x = np.concatenate([g + 0.5, g, g + 0.25, np.random.default_rng(0).uniform(-1, 7, (3000, 2))])
    i, d, fb = best_match_nodes_screen(w, x, return_fallbacks=True)
    assert _same((i, d), best_match_nodes(w, x))
    assert fb > 0


def test_screen_nonfinite_and_extreme_points():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(50, 3))
    x = rng.normal(size=(200, 3))
    x[0, 1] = np.nan
    x[1, 0] = np.inf
    x[2] = -np.inf
    x[3] = 1e30  # float32 overflow
    x[4] = 1e20  # float32 square overflow
    x[5] = 1e-300
    x[6] = 5e-320
    x[7] = w[3]
    assert _same(best_match_nodes_screen(w, x), best_match_nodes(w, x))


def test_screen_nonfinite_weights_fall_back():
    rng = np.random.default_rng(2)
    w = rng.normal(size=(20, 2))
    w[4, 0] = np.nan
    x = rng.normal(size=(100, 2))
    assert _same(best_match_nodes_screen(w, x), best_match_nodes(w, x))


def test_screen_chunks_and_predict():
    rng = np.random.default_rng(3)
    w = rng.normal(size=(64, 4))
    x = rng.normal(size=(1001, 4))
    ref = best_match_nodes(w, x)
    assert _same(best_match_nodes_screen(w, x, chunk=97), ref)
    cl = np.arange(64) % 5
    assert np.array_equal(predict(w, cl, x, backend="screen"), predict(w, cl, x, backend="numba"))


OVERFLOW = [
    (
        np.array([[18373320559154.32, -19118858046266.434], [14490963125571.676, -14263481110883.94]]),
        np.array([[1.3043817436596711e19, 1.3043817436596711e19]]),
    ),
    (
        np.array([[1e10, 0, -1e9], [-1e10, 0, 1e9]]),
        np.array(
            [[np.float32(2**64 - 2**41), np.float32(np.sqrt(3) * 2**52), np.float32(np.sqrt(0.5) * 2**52)]],
            np.float64,
        ),
    ),
]


@pytest.mark.parametrize("k", range(len(OVERFLOW)))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_screen_overflowed_runner_up_falls_back(k, dtype):
    w, x = OVERFLOW[k]
    x = x.astype(dtype)
    idx, d2, fb = best_match_nodes_screen(w, x, return_fallbacks=True)
    assert _same((idx, d2), best_match_nodes(w, x))
    assert fb == 1


@pytest.mark.parametrize("F", [1, 4])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_screen_single_node_lattice_falls_back(F, dtype):
    rng = np.random.default_rng(10 + F)
    w = rng.normal(size=(1, F))
    x = rng.normal(size=(20, F))
    x[0] = w[0]
    x[1, 0] = np.nan
    x[2] = np.nan
    x[3, 0] = np.inf
    x[4, -1] = -np.inf
    x = x.astype(dtype)
    idx, d2, fb = best_match_nodes_screen(w, x, return_fallbacks=True)
    assert fb == len(x)
    assert _same((idx, d2), best_match_nodes(w, x))
