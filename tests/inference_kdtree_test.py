"""The KD-tree backend must give the numba labels (no ties) and bitwise-equal distances."""

import numpy as np
import pytest

from aweSOM.inference import best_match_nodes, best_match_nodes_kdtree, predict


@pytest.mark.parametrize("n_features, dtype", [(2, np.float64), (4, np.float64), (6, np.float32), (9, np.float64)])
def test_matches_numba(n_features, dtype):
    rng = np.random.default_rng(n_features)
    weights = rng.random((64 * 37, n_features))
    data = rng.random((20_001, n_features)).astype(dtype)
    idx_ref, d2_ref = best_match_nodes(weights, data)
    idx, d2 = best_match_nodes_kdtree(weights, data, chunk=4096)
    assert idx.dtype == np.int32 and d2.dtype == np.float64
    assert np.array_equal(idx, idx_ref)
    assert np.array_equal(d2, d2_ref)


def test_predict_backend_kdtree():
    rng = np.random.default_rng(1)
    weights = rng.random((30, 3))
    node_cluster = rng.integers(-1, 5, size=30)
    data = rng.random((1000, 3))
    ref_lab, ref_d = predict(weights, node_cluster, data, return_distance=True, backend="numba")
    lab, d = predict(weights, node_cluster, data, return_distance=True, backend="kdtree", chunk=300)
    assert np.array_equal(ref_lab, lab)
    assert np.array_equal(ref_d, d)
    assert np.array_equal(predict(weights, node_cluster, data[0], backend="numba"), predict(weights, node_cluster, data[0]))


def test_rejects_non_finite_weights_and_bad_chunk():
    w = np.zeros((3, 2))
    w[1, 0] = np.nan
    with pytest.raises(ValueError):
        best_match_nodes_kdtree(w, np.zeros((2, 2)))
    with pytest.raises(ValueError):
        best_match_nodes_kdtree(np.zeros((3, 2)), np.zeros((2, 2)), chunk=0)


def test_overflowing_distances_fall_back_to_the_scan():
    # squared distances overflow float64: the tree reports "no neighbour" (index == n_nodes)
    weights = np.array([[1e200], [2e200]])
    data = np.array([[0.0], [1.5e200]])
    idx_ref, d2_ref = best_match_nodes(weights, data)
    idx, d2 = best_match_nodes_kdtree(weights, data)
    assert np.array_equal(idx, idx_ref)
    assert np.array_equal(d2, d2_ref)


def test_kdtree_is_the_default_backend():
    import inspect

    from aweSOM.model import SOMModel
    from aweSOM.som import Lattice

    for fn in (predict, SOMModel.predict, Lattice.predict):
        assert inspect.signature(fn).parameters["backend"].default == "kdtree"


def test_lattice_predict_passes_backend():
    from aweSOM.som import Lattice

    rng = np.random.default_rng(2)
    som = Lattice(6, 5, 0.3, 10)
    som.lattice = rng.random((30, 3))
    som.lattice_assigned_clusters = rng.integers(-1, 4, size=(6, 5))
    data = rng.random((500, 3))
    lab, d = som.predict(data, return_distance=True)
    ref_lab, ref_d = som.predict(data, return_distance=True, backend="numba")
    assert np.array_equal(lab, ref_lab) and np.array_equal(d, ref_d)
    with pytest.raises(ValueError):
        som.predict(data, backend="cupy")
