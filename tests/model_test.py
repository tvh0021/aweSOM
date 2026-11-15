"""Tests for aweSOM.model: Scaler, SOMModel construction, prediction and file round-trips."""

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

from aweSOM import Lattice, SOMModel, Scaler
from aweSOM.inference import predict
from aweSOM.model import distance_quantiles, distance_quantiles_fast
from aweSOM.run_som import manual_scaling

FIXTURES = Path(__file__).parent / "fixtures"


def _random_model(seed=0, xdim=6, ydim=5, n_features=3, kind="manual"):
    rng = np.random.default_rng(seed)
    raw = rng.normal(loc=[1.0, -20.0, 300.0], scale=[0.1, 5.0, 40.0], size=(500, n_features))
    scaler = Scaler.fit(raw, kind=kind)
    weights = rng.normal(size=(xdim * ydim, n_features))
    node_q = rng.uniform(size=xdim * ydim)
    node_q[3] = np.nan
    return SOMModel(
        xdim=xdim,
        ydim=ydim,
        weights=weights,
        node_cluster=rng.integers(-1, 4, size=xdim * ydim),
        feature_names=["b", "j_par", "rho"],
        scaler=scaler,
        node_dist_q99=node_q,
        reject_distance=1.0 / 3.0,
        metadata={"dataset": "random model with spaces in the value", "seed": 7},
    ), raw


@pytest.mark.parametrize("bulk_range", [1.0, 2.0, 0.5])
def test_manual_scaler_matches_manual_scaling_exactly(bulk_range):
    data = np.random.default_rng(0).normal(loc=3.0, scale=[0.2, 7.0, 1e4], size=(1000, 3))
    scaler = Scaler.fit(data, kind="manual", bulk_range=bulk_range)
    assert np.array_equal(scaler.transform(data), manual_scaling(data, bulk_range))


def test_manual_scaler_close_for_other_bulk_range():
    # (x - m) / (2s / b) and (x - m) / (2s) * b differ by at most a rounding
    data = np.random.default_rng(1).normal(size=(1000, 3))
    scaler = Scaler.fit(data, kind="manual", bulk_range=0.7)
    np.testing.assert_allclose(scaler.transform(data), manual_scaling(data, 0.7), rtol=4e-16, atol=0)


def test_minmax_and_none_scalers():
    data = np.random.default_rng(2).uniform(-3, 9, size=(200, 2))
    mm = Scaler.fit(data, kind="minmax")
    scaled = mm.transform(data)
    assert scaled.min(axis=0) == pytest.approx([0, 0]) and scaled.max(axis=0) == pytest.approx([1, 1])
    none = Scaler.fit(data, kind="none")
    assert np.array_equal(none.transform(data), data)
    for s in (mm, none, Scaler.fit(data)):
        np.testing.assert_allclose(s.inverse_transform(s.transform(data)), data, rtol=1e-14, atol=1e-14)


def test_zero_scale_is_guarded():
    data = np.ones((10, 2))
    data[:, 1] = np.arange(10)
    with pytest.warns(RuntimeWarning, match="zero"):
        scaler = Scaler.fit(data)
    assert scaler.scale[0] == 1.0
    assert np.all(np.isfinite(scaler.transform(data)))


def test_invalid_scaler_kind():
    with pytest.raises(ValueError):
        Scaler.fit(np.zeros((3, 2)), kind="robust")


def test_model_predict_applies_scaler():
    model, raw = _random_model()
    labels, dist = model.predict(raw, return_distance=True)
    ref_labels, ref_dist = predict(
        model.weights, model.node_cluster, model.scaler.transform(raw), return_distance=True
    )
    assert np.array_equal(labels, ref_labels) and np.array_equal(dist, ref_dist)
    labels_c, dist_c = model.predict(raw, chunk=13, return_distance=True)
    assert np.array_equal(labels_c, labels) and np.array_equal(dist_c, dist)
    assert np.array_equal(model.predict(raw[0]), labels[:1])
    labels_n, dist_n = model.predict(raw, return_distance=True, backend="numba")
    assert np.array_equal(labels_n, labels) and np.array_equal(dist_n, dist)


def _assert_models_equal(a: SOMModel, b: SOMModel):
    assert (a.xdim, a.ydim) == (b.xdim, b.ydim)
    assert a.weights.tobytes() == b.weights.tobytes()
    assert np.array_equal(a.node_cluster, b.node_cluster) and b.node_cluster.dtype == np.int32
    assert a.node_dist_q99.tobytes() == b.node_dist_q99.tobytes()
    assert a.feature_names == b.feature_names
    assert a.scaler.kind == b.scaler.kind and a.scaler.bulk_range == b.scaler.bulk_range
    assert a.scaler.offset.tobytes() == b.scaler.offset.tobytes()
    assert a.scaler.scale.tobytes() == b.scaler.scale.tobytes()
    assert a.reject_distance == b.reject_distance
    assert a.metadata == b.metadata


@pytest.mark.parametrize("suffix", [".h5", ".hdf5", ".txt"])
def test_round_trip_is_bitwise(tmp_path, suffix):
    model, raw = _random_model(kind="minmax")
    path = tmp_path / f"model{suffix}"
    if suffix == ".txt":
        model.save_text(path)
    else:
        model.save(path)
    loaded = SOMModel.load(path)
    _assert_models_equal(model, loaded)
    assert np.array_equal(loaded.predict(raw), model.predict(raw))


def test_text_format_layout(tmp_path):
    model, _ = _random_model()
    path = tmp_path / "model.txt"
    model.save_text(path)
    lines = path.read_text().splitlines()
    header = [l for l in lines if l.startswith("#")]
    body = [l for l in lines if not l.startswith("#")]
    assert "# feature_names b,j_par,rho" in header
    assert "# dataset random model with spaces in the value" in header
    assert len(body) == model.n_nodes
    first = body[0].split()
    assert len(first) == model.n_features + 1
    assert float(first[0]) == model.weights[0, 0] and int(first[-1]) == model.node_cluster[0]


def test_default_metadata_keys():
    from aweSOM import __version__
    from aweSOM.model import _default_metadata

    meta = _default_metadata()
    assert meta["format_version"] == "1"
    assert meta["awesom_version"] == __version__
    assert meta["created"].endswith("+00:00")
    assert meta["git_commit"] == "unknown" or len(meta["git_commit"]) == 40


def test_invalid_metadata_and_names():
    model, _ = _random_model()
    kwargs = dict(
        xdim=model.xdim, ydim=model.ydim, weights=model.weights, node_cluster=model.node_cluster,
        scaler=model.scaler, node_dist_q99=model.node_dist_q99, reject_distance=1.0,
    )
    with pytest.raises(ValueError, match="reserved"):
        SOMModel(feature_names=model.feature_names, metadata={"xdim": "3"}, **kwargs)
    with pytest.raises(ValueError):
        SOMModel(feature_names=model.feature_names, metadata={"bad key": "x"}, **kwargs)
    with pytest.raises(ValueError):
        SOMModel(feature_names=model.feature_names, metadata={"k": "two\nlines"}, **kwargs)
    with pytest.raises(ValueError):
        SOMModel(feature_names=["a,b", "c", "d"], **kwargs)
    with pytest.raises(ValueError):
        SOMModel(feature_names=["a", "b"], **kwargs)


@pytest.mark.parametrize("name", ["iris_like_model.txt", "iris_like_model.h5"])
def test_committed_fixture_predictions(name):
    model = SOMModel.load(FIXTURES / name)
    points = np.load(FIXTURES / "iris_like_points.npy")
    labels, dist = model.predict(points, return_distance=True)
    assert np.array_equal(labels, np.load(FIXTURES / "iris_like_expected_labels.npy"))
    assert np.array_equal(dist, np.load(FIXTURES / "iris_like_expected_dist.npy"))


def test_committed_fixture_formats_agree():
    _assert_models_equal(
        SOMModel.load(FIXTURES / "iris_like_model.txt"), SOMModel.load(FIXTURES / "iris_like_model.h5")
    )


def test_median_iqr_scaler_and_fast_quantiles(tmp_path):
    rng = np.random.default_rng(6)
    data = rng.normal(loc=[-5.0, 12.0, 100.0], scale=[0.5, 2.0, 15.0], size=(300, 3))
    miqr = Scaler.fit(data, kind="median-iqr")
    q25, q50, q75 = np.percentile(data, [25, 50, 75], axis=0)
    assert np.array_equal(miqr.offset, q50)
    assert np.array_equal(miqr.scale, q75 - q25)
    np.testing.assert_allclose(miqr.inverse_transform(miqr.transform(data)), data, rtol=1e-14, atol=1e-14)

    model, _ = _random_model(kind="median-iqr")
    for suffix in (".h5", ".txt"):
        path = tmp_path / f"miqr{suffix}"
        model.save_text(path) if suffix == ".txt" else model.save(path)
        _assert_models_equal(model, SOMModel.load(path))

    nodes_idx = rng.integers(0, 12, size=500, dtype=np.int32)
    nodes_idx[nodes_idx == 4] = 0  # leave node 4 empty
    dist = rng.uniform(0.0, 3.0, size=500)
    q_slow, rej_slow = distance_quantiles(nodes_idx, dist, 12)
    q_fast, rej_fast = distance_quantiles_fast(nodes_idx, dist, 12)
    assert q_slow.tobytes() == q_fast.tobytes()
    assert rej_slow == rej_fast


@pytest.mark.parametrize("suffix", [".h5", ".txt"])
def test_format_version_is_fixed_and_newer_files_are_refused(tmp_path, suffix):
    model, _ = _random_model(kind="manual")
    assert next(iter(model.metadata.items())) == ("format_version", "1")
    override = SOMModel(**{**model.__dict__, "metadata": {"format_version": "2", "seed": 7}})
    assert override.metadata == {"format_version": "1", "seed": "7"}
    path = tmp_path / f"m{suffix}"
    model.save(path) if suffix == ".h5" else model.save_text(path)
    if suffix == ".h5":
        import h5py

        with h5py.File(path, "r+") as f:
            f.attrs["format_version"] = 2
    else:
        path.write_text(path.read_text().replace("# format_version 1\n", "# format_version 2\n"))
    with pytest.raises(ValueError, match="format_version 2"):
        SOMModel.load(path)


def test_float32_scaled_training_changes_few_labels():
    """Scaler works in float64 while manual_scaling on float32 inputs works in float32, so a lattice trained on float32-scaled data changes 59 of 10^6 labels and 11,544 of 10^7 labels (0.12%) on the BBH validation sets when frozen with Scaler.fit."""
    rng = np.random.default_rng(20261112)
    centers = np.array(
        [
            [1.5, -2.8, -0.6, -0.7, 800.0],
            [2.8, -1.2, 0.4, -0.1, 801.5],
            [0.4, -3.9, -1.4, -1.3, 798.5],
        ],
        dtype=np.float64,
    )
    stds = np.array([0.7, 0.85, 0.6, 0.45, 0.95], dtype=np.float64)
    idx = rng.integers(0, 3, size=20_000)
    raw = (centers[idx] + rng.normal(size=(20_000, 5)) * stds).astype(np.float32)

    scaled32 = manual_scaling(raw)
    assert scaled32.dtype == np.float32

    som = Lattice(12, 9, 0.5, 2000, alpha_type="decay", sampling_type="uniform")
    som.seed = 42
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(scaled32, [f"f{i}" for i in range(5)], history=False)
        som.assign_cluster_to_lattice(smoothing=None, merge_cost=0.2)

    ref_labels = som.predict(scaled32.astype(np.float64))
    assert np.array_equal(ref_labels, som.predict(scaled32))

    model64 = SOMModel.from_lattice(som, scaled32, scaler=Scaler.fit(raw, kind="manual"), merge_cost=0.2)
    mismatches = int(np.count_nonzero(model64.predict(raw) != ref_labels))
    assert mismatches == 3
    assert mismatches < 0.01 * len(raw)

    model_id = SOMModel.from_lattice(som, scaled32, scaler=Scaler.identity(5), merge_cost=0.2)
    assert np.array_equal(model_id.predict(scaled32), ref_labels)


def test_predict_patch_rectangular_lattice_roundtrip(tmp_path):
    """Frozen patch inference uses the saved rectangular topology in every chunk."""
    rng = np.random.default_rng(7)
    model = SOMModel(
        5, 3, rng.normal(size=(15, 4)), np.arange(15) % 4,
        ["a", "b", "c", "d"],
        Scaler("median-iqr", np.arange(4), np.arange(4) + 1),
        np.ones(15), 10.,
    )
    path = tmp_path / "model.h5"
    model.save(path)
    loaded = SOMModel.load(path)
    data = rng.normal(size=(53, 4))
    expected, distance = model.predict(data, chunk=11, return_distance=True, backend="screen")
    actual, patch_distance = loaded.predict(data, chunk=7, return_distance=True, backend="patch")
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(patch_distance, distance, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(patch_distance.view(np.uint64), distance.view(np.uint64))
    np.testing.assert_array_equal(loaded.predict(data[0], backend="patch"), expected[:1])
    assert loaded.predict(data[:0], backend="patch").shape == (0,)


@pytest.mark.parametrize("name", ["iris_like_model.txt", "iris_like_model.h5"])
def test_committed_fixture_patch_backend(name):
    model = SOMModel.load(FIXTURES / name)
    points = np.load(FIXTURES / "iris_like_points.npy")
    expected_labels = np.load(FIXTURES / "iris_like_expected_labels.npy")
    expected_dist = np.load(FIXTURES / "iris_like_expected_dist.npy")
    labels, dist = model.predict(points, return_distance=True, backend="patch")
    numba_labels, numba_dist = model.predict(points, return_distance=True, backend="numba")
    assert labels.dtype == np.int32 and dist.dtype == np.float64
    assert np.array_equal(labels, expected_labels)
    assert np.array_equal(dist, expected_dist)
    assert np.array_equal(dist.view(np.uint64), expected_dist.view(np.uint64))
    assert np.array_equal(labels, numba_labels)
    assert np.array_equal(dist.view(np.uint64), numba_dist.view(np.uint64))
