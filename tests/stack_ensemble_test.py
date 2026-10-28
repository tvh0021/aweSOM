"""Synthetic end-to-end test of examples/grmhd-bbh/stack_ensemble.py (runs in seconds)."""

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "grmhd-bbh" / "stack_ensemble.py"


@pytest.fixture(scope="module")
def se():
    spec = importlib.util.spec_from_file_location("stack_ensemble", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def blobs(tmp_path_factory):
    """Three separated Gaussian blobs in 3 features, shuffled, in the feature-file layout."""
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 1.0], [0.0, 4.0, -1.0]])
    sizes = [1500, 900, 600]
    x = np.concatenate([rng.normal(c, 0.4, (n, 3)) for c, n in zip(centres, sizes)]).astype(np.float32)
    truth = np.repeat(np.arange(3), sizes)
    order = rng.permutation(len(x))
    path = tmp_path_factory.mktemp("feat") / "features.h5"
    with h5py.File(path, "w") as f:
        f["features"] = x[order]
        f["names"] = np.array([b"a", b"b", b"c"])
        f["weight_volume"] = np.ones(len(x))
    return path, truth[order]


def test_lattice_dims_match_train_som(se):
    assert se.lattice_dims(1_000_000, 5) == (51, 36)
    assert se.lattice_dims(10_000_000, 5) == (91, 64)
    x2, y2 = se.lattice_dims(1_000_000, 5, scale=2.0)
    assert abs(x2 * y2 / (51 * 36) - 2.0) < 0.05 and abs(y2 / x2 - 0.7) < 0.03


def test_compact_labels(se):
    lab, ids = se.compact_labels(np.array([5, 0, 5, 9], dtype=np.int32))
    assert lab.dtype == np.uint8 and lab.tolist() == [1, 0, 1, 2] and ids == [0, 5, 9]


def test_job_list(se):
    assert se._job_list([1, 2], [0.7, 1.0], None) == [(1, 0.7), (1, 1.0), (2, 0.7), (2, 1.0)]
    assert se._job_list([1, 2], [0.7, 1.0], 3) == [(1, 0.7), (1, 1.0), (2, 0.7)]
    with pytest.raises(ValueError):
        se._job_list([1], [1.0], 2)


def test_stack_ensemble_end_to_end(se, blobs, tmp_path):
    path, truth = blobs
    out = tmp_path / "out"
    summary = se.main(
        ["--input", str(path), "--N", "3000", "--seeds", "1", "2", "3", "--lattice-scales", "1.0", "1.4",
         "--output-dir", str(out), "--name", "blobs"]
    )
    params = sorted((out / "realizations").glob("params.*.json"))
    assert len(params) == 6
    for p in params:
        info = json.loads(p.read_text())
        lab = np.load(info["labels_file"])
        assert lab.dtype == np.uint8 and lab.shape == (3000,)
        assert lab.max() + 1 == info["n_clusters"]
        assert (out / "realizations" / f"som_object.{info['tag']}.pkl").exists()
    stacked = np.load(out / "stacked_labels.npy")
    signals = np.load(out / "sce_signals.npy")
    assert stacked.shape == (3000,) and signals.shape[1] == 3000
    assert np.array_equal(stacked, np.argmax(signals, axis=0))
    # six realizations, one vote each: the signals of every cell add up to 6
    total = signals.sum(axis=0)
    assert np.array_equal(total, np.full(3000, 6.0, dtype=np.float32))
    # the anchored stack recovers the three blobs
    assert adjusted_rand_score(truth, stacked) > 0.95
    assert len(summary["anchored"]["clusters"]) == 3
    assert sum(r["count"] for r in summary["anchored"]["clusters"]) == 3000
    assert abs(sum(r["volume_fraction"] for r in summary["anchored"]["clusters"]) - 1.0) < 1e-9
    ranged = np.load(out / "stacked_labels_gsum_ranges.npy")
    ranged_signals = np.load(out / "sce_signals_gsum_ranges.npy")
    assert np.array_equal(ranged, np.argmax(ranged_signals, axis=0))
    assert ranged_signals.shape[0] == len(summary["gsum_ranges"]["cluster_ranges"])

    # a second call reuses the saved realizations instead of retraining
    before = {p.name: p.stat().st_mtime_ns for p in (out / "realizations").glob("labels.*.npy")}
    se.main(
        ["--input", str(path), "--N", "3000", "--seeds", "1", "2", "3", "--lattice-scales", "1.0", "1.4",
         "--output-dir", str(out), "--name", "blobs", "--train-only"]
    )
    after = {p.name: p.stat().st_mtime_ns for p in (out / "realizations").glob("labels.*.npy")}
    assert before == after
