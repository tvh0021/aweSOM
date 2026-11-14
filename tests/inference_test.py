"""Tests for aweSOM.inference and SOMModel.from_lattice on a trained Lattice."""

import contextlib
import io

import numpy as np
import pytest

from aweSOM import Lattice, SOMModel
from aweSOM.inference import (
    best_match_nodes,
    best_match_nodes_patch,
    best_match_nodes_screen,
    predict,
)


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
    labels, dist = predict(weights, node_cluster, data, return_distance=True, backend="numba")
    assert labels.dtype == np.int32
    assert np.array_equal(labels, node_cluster[idx])
    assert np.array_equal(dist, np.sqrt(d2))
    assert np.array_equal(predict(weights, node_cluster, data, backend="numba"), labels)


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
    with contextlib.redirect_stdout(io.StringIO()), pytest.warns(DeprecationWarning):
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


def _assert_d3_all_backends(weights, node_cluster, data, xdim, ydim, chunk=7):
    idx_n, d2_n = best_match_nodes(weights, data, chunk=chunk)
    idx_s, d2_s = best_match_nodes_screen(weights, data, chunk=chunk)
    idx_p, d2_p = best_match_nodes_patch(weights, data, xdim, ydim, chunk=chunk, side=2)
    idx_p_def, d2_p_def = best_match_nodes_patch(weights, data, xdim, ydim, chunk=chunk)

    for idx, d2 in ((idx_n, d2_n), (idx_s, d2_s), (idx_p, d2_p), (idx_p_def, d2_p_def)):
        assert idx.dtype == np.int32 and d2.dtype == np.float64
        assert np.array_equal(idx, idx_n)
        assert np.array_equal(d2.view(np.uint64), d2_n.view(np.uint64))

    lab_n, dist_n = predict(weights, node_cluster, data, chunk=chunk, return_distance=True, backend="numba")
    lab_s, dist_s = predict(weights, node_cluster, data, chunk=chunk, return_distance=True, backend="screen")
    lab_p, dist_p = predict(
        weights,
        node_cluster,
        data,
        chunk=chunk,
        return_distance=True,
        backend="patch",
        lattice_shape=(xdim, ydim),
    )

    ref_dist = np.sqrt(d2_n)
    for lab, dist in ((lab_n, dist_n), (lab_s, dist_s), (lab_p, dist_p)):
        assert lab.dtype == np.int32 and dist.dtype == np.float64
        assert np.array_equal(lab, node_cluster[idx_n])
        assert np.array_equal(dist.view(np.uint64), ref_dist.view(np.uint64))
    return idx_n, d2_n, lab_n, dist_n


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_d3_exact_ties_across_backends(dtype):
    xdim, ydim = 6, 4
    weights = np.full((xdim * ydim, 3), 100.0, dtype=np.float64)
    for i in range(xdim * ydim):
        weights[i] += np.array([10.0 * i, -5.0 * i, 2.0 * i])
    # Distinct cluster ID per node so tie-breaking on node index determines label
    node_cluster = np.arange(xdim * ydim, dtype=np.int32) + 10

    # Duplicate weights across separate 2x2 patches: lowest index is 2
    weights[2] = np.array([-4.0, 2.0, 1.0])
    weights[19] = weights[2]
    weights[23] = weights[2]

    # Second duplicate group across patches: lowest index is 5
    weights[5] = np.array([8.0, -4.0, 0.5])
    weights[14] = weights[5]

    # Exact dyadic midpoint across patches: nodes 3 and 21 are both at squared distance 2.0
    weights[3] = np.array([0.0, 2.0, -1.0])
    weights[21] = np.array([2.0, 0.0, -1.0])

    data = np.array(
        [
            [-4.0, 2.0, 1.0],      # exact zero-distance tie -> node 2
            [-3.5, 2.0, 1.0],      # non-zero tie on duplicate nodes -> node 2
            [8.0, -3.75, 0.5],     # non-zero tie on duplicate nodes -> node 5
            [1.0, 1.0, -1.0],      # equidistant midpoint across patches -> node 3
        ],
        dtype=dtype,
    )
    idx, d2, lab, dist = _assert_d3_all_backends(weights, node_cluster, data, xdim, ydim)
    assert idx.tolist() == [2, 2, 5, 3]
    assert d2.tolist() == [0.0, 0.25, 0.0625, 2.0]
    assert lab.tolist() == [node_cluster[2], node_cluster[2], node_cluster[5], node_cluster[3]]
    assert np.array_equal(dist.view(np.uint64), np.sqrt(d2).view(np.uint64))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_d3_nan_rows_and_nan_weights_across_backends(dtype):
    xdim, ydim = 5, 4
    rng = np.random.default_rng(11)
    weights = rng.normal(size=(xdim * ydim, 3))
    node_cluster = (np.arange(xdim * ydim, dtype=np.int32) + 7) % 9 - 1
    node_cluster[0] = 42

    data = np.array(
        [
            [0.25, np.nan, -0.5],
            [0.25, 0.5, np.nan],
            [np.nan, np.nan, np.nan],
            weights[6],
        ],
        dtype=dtype,
    )
    idx, d2, lab, dist = _assert_d3_all_backends(weights, node_cluster, data, xdim, ydim)
    assert idx[:3].tolist() == [0, 0, 0]
    assert np.all(np.isposinf(d2[:3])) and np.all(np.isposinf(dist[:3]))
    assert lab[:3].tolist() == [42, 42, 42]
    assert idx[3] == 6

    # Single NaN node in weights is skipped in favor of finite nodes
    w_nan = weights.copy()
    target_point = weights[0].copy()
    w_nan[0, 1] = np.nan
    query = np.array([target_point, [np.nan, 0.0, 0.0]], dtype=dtype)
    idx_w, d2_w, lab_w, dist_w = _assert_d3_all_backends(w_nan, node_cluster, query, xdim, ydim)
    assert idx_w[0] != 0 and np.isfinite(d2_w[0]) and np.isfinite(dist_w[0])
    assert idx_w[1] == 0 and np.isposinf(d2_w[1]) and lab_w[1] == 42


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_d3_infinite_rows_and_single_node_lattice_across_backends(dtype):
    xdim, ydim = 5, 4
    rng = np.random.default_rng(12)
    weights = rng.normal(size=(xdim * ydim, 3))
    node_cluster = np.arange(xdim * ydim, dtype=np.int32) + 100
    node_cluster[0] = -1

    data = np.array(
        [
            [np.inf, 0.0, 0.0],
            [0.0, -np.inf, 0.0],
            [np.inf, -np.inf, 0.5],
            [np.inf, np.inf, np.inf],
            [-np.inf, -np.inf, -np.inf],
            [np.inf, np.nan, -np.inf],
            weights[9],
        ],
        dtype=dtype,
    )
    idx, d2, lab, dist = _assert_d3_all_backends(weights, node_cluster, data, xdim, ydim)
    assert idx[:6].tolist() == [0, 0, 0, 0, 0, 0]
    assert np.all(np.isposinf(d2[:6])) and np.all(np.isposinf(dist[:6]))
    assert lab[:6].tolist() == [-1] * 6
    assert idx[6] == 9

    # 1x1 single-node lattice across numba, screen, and patch
    w1 = np.array([[1.5, -2.0, 0.5]], dtype=np.float64)
    c1 = np.array([4], dtype=np.int32)
    x1 = np.array(
        [
            [1.5, -2.0, 0.5],
            [0.5, -1.0, 0.5],
            [np.nan, 0.0, 0.0],
            [np.inf, 0.0, 0.0],
            [0.0, -np.inf, 0.0],
        ],
        dtype=dtype,
    )
    idx1, d2_1, lab1, dist1 = _assert_d3_all_backends(w1, c1, x1, 1, 1)
    assert idx1.tolist() == [0, 0, 0, 0, 0]
    assert lab1.tolist() == [4, 4, 4, 4, 4]
    assert d2_1[0] == 0.0 and d2_1[1] == 2.0
    assert np.all(np.isposinf(d2_1[2:])) and np.all(np.isposinf(dist1[2:]))

    with pytest.raises(ValueError, match='backend must be "numba", "screen" or "patch"'):
        predict(weights, node_cluster, data, backend="unknown")
