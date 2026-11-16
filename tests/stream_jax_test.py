"""Tests for aweSOM.stream.predict_stream(..., backend="jax")."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from aweSOM.inference import best_match_nodes_jax, predict  # noqa: E402
from aweSOM.stream import predict_stream  # noqa: E402


def _max_rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.maximum(np.abs(a64), np.abs(b64)), 1e-12)
    return float(np.max(np.abs(a64 - b64) / denom))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("prefetch", [False, True])
def test_predict_stream_jax_matches_numba_and_inference_predict(dtype, prefetch):
    rng = np.random.default_rng(20261115)
    n_nodes, n_feat = 48, 5
    n_rows, chunk = 257, 100  # uneven split: 100 + 100 + 57

    weights = rng.uniform(-1.0, 1.0, size=(n_nodes, n_feat))
    node_cluster = (np.arange(n_nodes, dtype=np.int32) * 5) % 9 - 1
    data = rng.normal(scale=0.6, size=(n_rows, n_feat)).astype(dtype)

    dist_jax = np.empty(n_rows, dtype=np.float64)
    node_jax = np.empty(n_rows, dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        labels_jax, stats_jax = predict_stream(
            weights,
            node_cluster,
            data,
            out_distance=dist_jax,
            out_node=node_jax,
            chunk=chunk,
            backend="jax",
            prefetch=prefetch,
        )

    dist_numba = np.empty(n_rows, dtype=np.float64)
    node_numba = np.empty(n_rows, dtype=np.int32)
    labels_numba, stats_numba = predict_stream(
        weights,
        node_cluster,
        data,
        out_distance=dist_numba,
        out_node=node_numba,
        chunk=chunk,
        backend="numba",
        prefetch=prefetch,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        labels_inf, dist_inf = predict(
            weights,
            node_cluster,
            data,
            chunk=chunk,
            return_distance=True,
            backend="jax",
        )
        node_inf, _ = best_match_nodes_jax(weights, data, chunk=chunk)

    assert stats_jax["rows"] == stats_numba["rows"] == n_rows
    assert stats_jax["start"] == 0 and stats_jax["stop"] == n_rows
    assert stats_jax["device"] is None
    assert labels_jax.dtype == labels_numba.dtype == np.int32

    np.testing.assert_array_equal(labels_jax, labels_numba)
    np.testing.assert_array_equal(labels_jax, labels_inf)
    np.testing.assert_array_equal(node_jax, node_numba)
    np.testing.assert_array_equal(node_jax, node_inf)

    rel_vs_numba = _max_rel_diff(dist_jax, dist_numba)
    rel_vs_inf = _max_rel_diff(dist_jax, dist_inf)
    print(
        f"dtype={np.dtype(dtype).name} prefetch={prefetch} "
        f"max_rel_vs_numba={rel_vs_numba:.6e} max_rel_vs_inf={rel_vs_inf:.6e}"
    )
    assert rel_vs_numba < 1e-4
    assert rel_vs_inf < 1e-4


def test_predict_stream_jax_narrow_uint8_and_uneven_callable_chunks():
    rng = np.random.default_rng(73)
    weights = rng.uniform(-1.0, 1.0, size=(36, 4))
    node_cluster = np.arange(36, dtype=np.int32) % 6  # non-negative -> uint8 default out
    data = rng.normal(scale=0.5, size=(193, 4)).astype(np.float32)

    edges = [0, 70, 121, 193]

    def uneven_source():
        for a, b in zip(edges[:-1], edges[1:]):
            yield data[a:b]

    dist_jax = np.empty(len(data), dtype=np.float64)
    node_jax = np.empty(len(data), dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        labels_jax, stats = predict_stream(
            weights,
            node_cluster,
            uneven_source,
            out_distance=dist_jax,
            out_node=node_jax,
            chunk=64,
            backend="jax",
            prefetch=False,
            n_samples=len(data),
        )

    dist_numba = np.empty(len(data), dtype=np.float64)
    node_numba = np.empty(len(data), dtype=np.int32)
    labels_numba, _ = predict_stream(
        weights,
        node_cluster,
        data,
        out_distance=dist_numba,
        out_node=node_numba,
        chunk=64,
        backend="numba",
        prefetch=False,
    )

    assert stats["rows"] == len(data)
    assert labels_jax.dtype == labels_numba.dtype == np.uint8
    np.testing.assert_array_equal(labels_jax, labels_numba)
    np.testing.assert_array_equal(node_jax, node_numba)
    assert _max_rel_diff(dist_jax, dist_numba) < 1e-4
