"""Tests for aweSOM.inference and SOMModel.from_lattice on a trained Lattice."""

import contextlib
import io

import numpy as np
import pytest

from aweSOM import Lattice, SOMModel
from aweSOM.inference import best_match_nodes, predict


def _legacy_best_match(weights, data):
    with contextlib.redirect_stdout(io.StringIO()):
        return np.asarray(Lattice._best_match_legacy(weights, data)).ravel().astype(np.int32)


@pytest.fixture(scope="module")
def trained():
    data = np.random.default_rng(0).normal(size=(2000, 4))
    som = Lattice(10, 8, 0.3, 500)
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(data, list("abcd"))
        som.assign_cluster_to_lattice(merge_cost=0.2)
    return som, data


def test_best_match_nodes_equals_lattice_best_match():
    rng = np.random.default_rng(1)
    weights = rng.normal(size=(80, 6))
    data = rng.normal(size=(1000, 6))
    idx, dist2 = best_match_nodes(weights, data)
    assert idx.dtype == np.int32 and dist2.dtype == np.float64
    assert np.array_equal(idx, _legacy_best_match(weights, data))


def test_best_match_nodes_matches_numpy_argmin_up_to_ties():
    rng = np.random.default_rng(2)
    weights = rng.normal(size=(80, 6))
    data = rng.normal(size=(1000, 6))
    idx, dist2 = best_match_nodes(weights, data)
    d2_np = ((data[:, None, :] - weights[None, :, :]) ** 2).sum(-1)
    ref = np.argmin(d2_np, axis=1)
    differ = np.flatnonzero(idx != ref)
    # numpy's pairwise sum may round differently; any disagreement must be a
    # near-tie between the two candidate nodes
    for p in differ:
        a, b = d2_np[p, idx[p]], d2_np[p, ref[p]]
        assert abs(a - b) <= 1e-12 * max(a, b), f"point {p}: not a tie ({a} vs {b})"
    np.testing.assert_allclose(dist2, d2_np[np.arange(len(data)), idx], rtol=1e-14)


def test_exact_ties_go_to_lowest_node_index():
    weights = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    idx, dist2 = best_match_nodes(weights, data)
    assert idx.tolist() == [0, 0, 1, 0]
    assert idx.tolist() == np.argmin(((data[:, None] - weights[None]) ** 2).sum(-1), axis=1).tolist()
    assert dist2.tolist() == [1.0, 0.0, 0.0, 0.5]


@pytest.mark.parametrize("chunk", [1, 7, 333, 2**20])
def test_chunking_does_not_change_results(chunk):
    rng = np.random.default_rng(3)
    weights = rng.normal(size=(30, 3))
    data = rng.normal(size=(1000, 3))
    ref_idx, ref_d2 = best_match_nodes(weights, data)
    idx, d2 = best_match_nodes(weights, data, chunk=chunk)
    assert np.array_equal(idx, ref_idx) and np.array_equal(d2, ref_d2)


def test_non_contiguous_and_float32_input():
    rng = np.random.default_rng(4)
    weights = rng.normal(size=(30, 3))
    data = np.asfortranarray(rng.normal(size=(500, 3)))
    ref_idx, ref_d2 = best_match_nodes(weights, np.ascontiguousarray(data))
    idx, d2 = best_match_nodes(weights, data, chunk=64)
    assert np.array_equal(idx, ref_idx) and np.array_equal(d2, ref_d2)

    data32 = data.astype(np.float32)
    idx32, d2_32 = best_match_nodes(weights, data32)
    # float32 input is promoted exactly, so it equals the float64 run on the promoted values
    idx64, d2_64 = best_match_nodes(weights, data32.astype(np.float64))
    assert np.array_equal(idx32, idx64) and np.array_equal(d2_32, d2_64)


def test_predict_maps_nodes_to_clusters_and_distances():
    rng = np.random.default_rng(5)
    weights = rng.normal(size=(12, 2))
    node_cluster = np.arange(12, dtype=np.int32) % 3 - 1
    data = rng.normal(size=(200, 2))
    idx, d2 = best_match_nodes(weights, data)
    labels, dist = predict(weights, node_cluster, data, return_distance=True)
    assert labels.dtype == np.int32
    assert np.array_equal(labels, node_cluster[idx])
    assert np.array_equal(dist, np.sqrt(d2))
    assert np.array_equal(predict(weights, node_cluster, data), labels)


def test_shape_errors():
    with pytest.raises(ValueError):
        best_match_nodes(np.zeros((4, 3)), np.zeros((5, 2)))
    with pytest.raises(ValueError):
        predict(np.zeros((4, 3)), np.zeros(3), np.zeros((5, 3)))


def test_from_lattice_node_cluster_follows_rowix(trained):
    som, data = trained
    model = SOMModel.from_lattice(som, data, merge_cost=0.2, metadata={"dataset": "unit test"})
    assert model.weights.shape == (80, 4)
    assert np.array_equal(model.weights, som.lattice)
    for x in range(som.xdim):
        for y in range(som.ydim):
            assert model.node_cluster[som.rowix(x, y)] == som.lattice_assigned_clusters[x, y]
    assert model.feature_names == list("abcd")
    assert model.metadata["dataset"] == "unit test"
    assert model.metadata["merge_cost"] == "0.2"
    assert model.metadata["train"] == "500"
    assert model.metadata["format_version"] == "1"


def test_predict_reproduces_lattice_cluster_assignment(trained):
    som, data = trained
    model = SOMModel.from_lattice(som, data)
    with contextlib.redirect_stdout(io.StringIO()):
        expected = som.assign_cluster_to_data(
            som.map_data_to_lattice(), som.lattice_assigned_clusters
        )
    assert np.array_equal(model.predict(data), expected)
    assert np.array_equal(model.predict(data, chunk=97), expected)


def test_from_lattice_assigns_clusters_when_missing(trained):
    som, data = trained
    fresh = Lattice(10, 8, 0.3, 500)
    fresh.xdim, fresh.ydim = som.xdim, som.ydim
    fresh.lattice, fresh.umat = som.lattice.copy(), som.umat.copy()
    fresh.features_names, fresh.data_array = som.features_names, data
    model = SOMModel.from_lattice(fresh, data, merge_cost=0.2)
    assert np.array_equal(model.node_cluster, som.lattice_assigned_clusters.ravel(order="F"))


def test_distance_thresholds(trained):
    som, data = trained
    model = SOMModel.from_lattice(som, data)
    idx, d2 = best_match_nodes(som.lattice, data)
    dist = np.sqrt(d2)
    assert model.reject_distance == pytest.approx(np.percentile(dist, 99.9))
    for node in range(model.n_nodes):
        mine = dist[idx == node]
        if mine.size == 0:
            assert np.isnan(model.node_dist_q99[node])
        else:
            assert model.node_dist_q99[node] == np.percentile(mine, 99.0)
            assert model.node_dist_q99[node] <= mine.max()
