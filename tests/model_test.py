"""Tests for aweSOM.model: Scaler, SOMModel construction, prediction and file round-trips."""

from pathlib import Path

import numpy as np
import pytest

from aweSOM import SOMModel, Scaler
from aweSOM.inference import predict
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
