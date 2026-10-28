"""The lattice-patch pruned BMU search is bitwise equal to the float64 reference scan."""

import numpy as np
import pytest

from aweSOM.inference import (
    best_match_nodes,
    best_match_nodes_patch,
    lattice_patches,
    predict,
)


def smooth_lattice(xdim, ydim, n_feat, seed):
    """A trained-looking lattice: a smooth curved sheet plus small noise."""
    rng = np.random.default_rng(seed)
    i = np.arange(xdim * ydim)
    u = (i % xdim) / max(1, xdim - 1)
    v = (i // xdim) / max(1, ydim - 1)
    w = np.zeros((xdim * ydim, n_feat))
    w[:, 0] = 2 * u - 1
    w[:, 1 % n_feat] += 2 * v - 1
    for f in range(2, n_feat):
        w[:, f] = 0.3 * np.sin(np.pi * (f * u + v))
    return w + 0.01 * rng.normal(size=w.shape)


@pytest.mark.parametrize("xdim,ydim,n_feat,side", [
    (12, 9, 5, None), (12, 9, 5, 1), (12, 9, 5, 5), (7, 3, 2, 4), (20, 15, 10, 3), (5, 4, 3, 50),
])
@pytest.mark.parametrize("smooth", [True, False])
def test_patch_equals_reference(xdim, ydim, n_feat, side, smooth):
    rng = np.random.default_rng(xdim * 100 + n_feat)
    w = smooth_lattice(xdim, ydim, n_feat, 1) if smooth else rng.normal(size=(xdim * ydim, n_feat))
    x = rng.normal(scale=0.7, size=(3000, n_feat))
    x[:20] = w[rng.integers(0, len(w), 20)]  # exact hits
    x[20:30] = 0.5 * (w[0] + w[1])  # possible exact ties
    for data in (x, x.astype(np.float32)):
        ref_i, ref_d = best_match_nodes(w, data)
        i, d = best_match_nodes_patch(w, data, xdim, ydim, side=side, chunk=700)
        np.testing.assert_array_equal(i, ref_i)
        np.testing.assert_array_equal(d, ref_d)


def test_exact_ties_go_to_lowest_index():
    # duplicate weights in different patches: the reference picks the lowest index
    w = smooth_lattice(8, 8, 3, 0)
    w[63] = w[0]
    w[36] = w[0]
    x = np.repeat(w[:1], 5, axis=0) + np.array([[0.0], [1e-3], [-1e-3], [2e-3], [0.0]])
    ref_i, ref_d = best_match_nodes(w, x)
    i, d = best_match_nodes_patch(w, x, 8, 8, side=2)
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(d, ref_d)


def test_non_finite_rows_and_weights():
    w = smooth_lattice(6, 5, 4, 2)
    x = np.random.default_rng(0).normal(size=(50, 4))
    x[3, 1] = np.nan
    x[4, 0] = np.inf
    x[5] = 1e200
    ref = best_match_nodes(w, x)
    got = best_match_nodes_patch(w, x, 6, 5)
    np.testing.assert_array_equal(got[0], ref[0])
    np.testing.assert_array_equal(got[1], ref[1])
    w_bad = w.copy()
    w_bad[7, 2] = np.nan
    ref = best_match_nodes(w_bad, x)
    got = best_match_nodes_patch(w_bad, x, 6, 5)
    np.testing.assert_array_equal(got[0], ref[0])


def test_patches_partition_the_lattice():
    w = smooth_lattice(11, 7, 3, 0)
    _, perm, offs, cent_t, rad = lattice_patches(w, 11, 7, side=4)
    assert sorted(perm.tolist()) == list(range(77))
    assert offs[-1] == 77 and cent_t.shape == (3, len(offs) - 1)
    for b in range(len(offs) - 1):
        members = w[perm[offs[b] : offs[b + 1]]]
        assert np.all(np.sqrt(((members - cent_t[:, b]) ** 2).sum(1)) <= rad[b])


def test_predict_patch_backend():
    w = smooth_lattice(10, 6, 4, 3)
    x = np.random.default_rng(1).normal(size=(500, 4)).astype(np.float32)
    clusters = np.arange(60) % 7
    np.testing.assert_array_equal(
        predict(w, clusters, x, backend="patch", lattice_shape=(10, 6)), predict(w, clusters, x)
    )
    with pytest.raises(ValueError):
        predict(w, clusters, x, backend="patch")
