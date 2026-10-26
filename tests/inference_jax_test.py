"""Tests for the JAX inference path (aweSOM.inference.best_match_nodes_jax).

The float64 path must reproduce the numba kernel bitwise. The float32 path is
approximate: a point whose two nearest nodes are within float32 rounding of
each other may get the other node, so only the agreement rate is checked.
On a backend without float64 (Apple Metal) the "highest" tests fall back to
float32 with a warning and are skipped.
"""

import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from aweSOM.inference import (  # noqa: E402
    _jax_float64_scope,
    best_match_nodes,
    best_match_nodes_jax,
    predict,
)

HAS_X64 = _jax_float64_scope(jax.local_devices()) is not None
needs_x64 = pytest.mark.skipif(not HAS_X64, reason="JAX backend has no float64")

TIE_WEIGHTS = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
TIE_DATA = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])


@pytest.fixture(scope="module")
def lattice_data():
    rng = np.random.default_rng(7)
    weights = rng.uniform(-1.0, 1.0, size=(64 * 37, 4))
    data = rng.normal(scale=0.5, size=(2000, 4))
    return weights, data


@needs_x64
@pytest.mark.parametrize("chunk", [None, 333, 2000])
def test_float64_equals_numba_bitwise(lattice_data, chunk):
    weights, data = lattice_data
    ref_idx, ref_d2 = best_match_nodes(weights, data)
    idx, d2 = best_match_nodes_jax(weights, data, chunk=chunk)
    assert idx.dtype == np.int32 and d2.dtype == np.float64
    assert np.array_equal(idx, ref_idx)
    np.testing.assert_allclose(d2, ref_d2, rtol=1e-12, atol=0)
    # the runtime-zero guard against FMA contraction makes them bitwise equal
    assert np.array_equal(d2, ref_d2)


@needs_x64
def test_float64_exact_ties_go_to_lowest_node():
    idx, d2 = best_match_nodes_jax(TIE_WEIGHTS, TIE_DATA)
    assert idx.tolist() == [0, 0, 1, 0]
    assert idx.tolist() == best_match_nodes(TIE_WEIGHTS, TIE_DATA)[0].tolist()
    assert d2.tolist() == [1.0, 0.0, 0.0, 0.5]


@needs_x64
def test_float32_input_is_promoted_like_numba(lattice_data):
    weights, data = lattice_data
    data32 = data.astype(np.float32)
    ref_idx, ref_d2 = best_match_nodes(weights, data32)
    idx, d2 = best_match_nodes_jax(weights, data32, chunk=512)
    assert np.array_equal(idx, ref_idx) and np.array_equal(d2, ref_d2)


@needs_x64
def test_nan_distances_are_skipped_like_numba():
    weights = np.array([[np.nan, 0.0], [1.0, 1.0], [2.0, 2.0]])
    data = np.array([[0.0, 0.0], [2.1, 2.0], [np.nan, 0.0]])
    ref_idx, ref_d2 = best_match_nodes(weights, data)
    idx, d2 = best_match_nodes_jax(weights, data)
    assert idx.tolist() == ref_idx.tolist() == [1, 2, 0]
    assert np.array_equal(d2, ref_d2) and np.isinf(d2[2])


@needs_x64
def test_global_x64_flag_is_restored(lattice_data):
    weights, data = lattice_data
    before = bool(jax.config.read("jax_enable_x64"))
    best_match_nodes_jax(weights, data[:10])
    assert bool(jax.config.read("jax_enable_x64")) == before


def test_float32_agrees_with_numba(lattice_data):
    weights, data = lattice_data
    ref_idx, ref_d2 = best_match_nodes(weights, data)
    idx, d2 = best_match_nodes_jax(weights, data, precision="float32")
    assert idx.dtype == np.int32 and d2.dtype == np.float32
    # near-ties (relative gap below ~(F + 2) * 6e-8) may flip; require >= 99.99 %
    assert np.mean(idx == ref_idx) >= 0.9999
    same = idx == ref_idx
    np.testing.assert_allclose(d2[same], ref_d2[same], rtol=1e-5, atol=1e-6)
    tie_idx, _ = best_match_nodes_jax(TIE_WEIGHTS, TIE_DATA, precision="float32")
    assert tie_idx.tolist() == [0, 0, 1, 0]  # these ties are exact in float32 too


def test_round_robin_over_devices_matches_one_device(lattice_data):
    weights, data = lattice_data
    dev = jax.local_devices()[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = best_match_nodes_jax(weights, data, chunk=300, devices=[dev])
        out = best_match_nodes_jax(weights, data, chunk=300, devices=[dev, dev, dev])
    assert np.array_equal(out[0], ref[0]) and np.array_equal(out[1], ref[1])


@needs_x64
def test_two_cpu_devices_in_subprocess():
    code = textwrap.dedent(
        """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax
        import numpy as np
        from aweSOM.inference import best_match_nodes, best_match_nodes_jax
        assert jax.device_count() == 2, jax.devices()
        rng = np.random.default_rng(3)
        w = rng.normal(size=(200, 4))
        x = rng.normal(size=(1001, 4))
        a = best_match_nodes(w, x)
        b = best_match_nodes_jax(w, x, chunk=128)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        print("ok")
        """
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().endswith("ok")


def test_predict_backend_jax(lattice_data):
    weights, data = lattice_data
    node_cluster = np.arange(weights.shape[0], dtype=np.int32) % 5 - 1
    ref_labels, ref_dist = predict(weights, node_cluster, data, return_distance=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        labels, dist = predict(weights, node_cluster, data, return_distance=True, backend="jax")
    assert labels.dtype == np.int32
    if HAS_X64:
        assert np.array_equal(labels, ref_labels) and np.array_equal(dist, ref_dist)
    else:
        assert np.mean(labels == ref_labels) >= 0.9999


def test_argument_errors():
    w = np.zeros((4, 3))
    with pytest.raises(ValueError):
        best_match_nodes_jax(w, np.zeros((5, 2)))
    with pytest.raises(ValueError):
        best_match_nodes_jax(w, np.zeros((5, 3)), precision="float16")
    with pytest.raises(ValueError):
        best_match_nodes_jax(w, np.zeros((5, 3)), chunk=0)
    with pytest.raises(ValueError):
        predict(w, np.zeros(4), np.zeros((5, 3)), backend="cupy")
    with pytest.raises(ValueError):
        best_match_nodes_jax(w, np.zeros((5, 3)), devices=[])
    with pytest.raises(ValueError):  # zero features are rejected on both backends
        best_match_nodes_jax(np.zeros((2, 0)), np.zeros((3, 0)))
    with pytest.raises(ValueError):
        best_match_nodes(np.zeros((2, 0)), np.zeros((3, 0)))


def test_empty_input():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        idx, d2 = best_match_nodes_jax(np.zeros((4, 3)), np.zeros((0, 3)))
    assert idx.shape == (0,) and d2.shape == (0,)


@needs_x64
def test_sommodel_predict_backend_jax():
    from aweSOM.model import Scaler, SOMModel

    rng = np.random.default_rng(3)
    raw = rng.normal(loc=[1.0, -20.0, 300.0], scale=[0.1, 5.0, 40.0], size=(500, 3))
    model = SOMModel(
        xdim=4,
        ydim=3,
        weights=rng.normal(size=(12, 3)),
        node_cluster=np.arange(12) % 4,
        feature_names=["a", "b", "c"],
        scaler=Scaler.fit(raw, kind="manual"),
        node_dist_q99=np.full(12, np.inf),
        reject_distance=1.0,
    )
    lab_numba, d_numba = model.predict(raw, return_distance=True)
    lab_jax, d_jax = model.predict(raw, return_distance=True, backend="jax", chunk=128)
    assert np.array_equal(lab_numba, lab_jax)
    assert np.array_equal(d_numba, d_jax)
