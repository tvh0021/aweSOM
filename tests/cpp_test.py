"""Build the C++ reference classifier and check it against SOMModel.predict."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aweSOM import SOMModel, Scaler

ROOT = Path(__file__).resolve().parents[1]
CPP_DIR = ROOT / "cpp"
FIXTURES = Path(__file__).parent / "fixtures"


def _compiler():
    for name in ("clang++", "g++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


@pytest.fixture(scope="module")
def classifier(tmp_path_factory):
    cxx = _compiler()
    if cxx is None:
        pytest.skip("no C++ compiler found")
    exe = tmp_path_factory.mktemp("cpp") / "test_classify"
    cmd = [cxx, "-O2", "-std=c++17", "-ffp-contract=off", "-o", str(exe),
           str(CPP_DIR / "test_classify.cpp")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"compilation failed:\n{proc.stderr}")
    return exe


def _run(exe, model_path, points):
    csv = Path(model_path).with_suffix(".csv")
    np.savetxt(csv, points, fmt="%.17g", delimiter=",")
    proc = subprocess.run([str(exe), str(model_path), str(csv)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = np.loadtxt(proc.stdout.splitlines(), ndmin=2)
    return out[:, 0].astype(np.int32), out[:, 1]


def test_cpp_matches_python_on_random_model(classifier, tmp_path):
    rng = np.random.default_rng(20261026)
    n_features, xdim, ydim = 6, 12, 9
    raw_train = rng.normal(loc=np.arange(n_features), scale=np.geomspace(1e-2, 1e3, n_features),
                           size=(5000, n_features))
    scaler = Scaler.fit(raw_train, kind="manual")
    model = SOMModel(
        xdim=xdim, ydim=ydim,
        weights=rng.normal(scale=0.5, size=(xdim * ydim, n_features)),
        node_cluster=rng.integers(-1, 5, size=xdim * ydim),
        feature_names=[f"f{i}" for i in range(n_features)],
        scaler=scaler,
        node_dist_q99=rng.uniform(size=xdim * ydim),
        reject_distance=0.9,
        metadata={"dataset": "cpp_test random model"},
    )
    path = tmp_path / "model.txt"
    model.save_text(path)
    points = rng.normal(loc=np.arange(n_features), scale=np.geomspace(1e-2, 1e3, n_features),
                        size=(2000, n_features))
    labels_py, dist_py = model.predict(points, return_distance=True, backend="numba")
    labels_cpp, dist_cpp = _run(classifier, path, points)
    assert np.array_equal(labels_cpp, labels_py)
    np.testing.assert_allclose(dist_cpp, dist_py, rtol=1e-12, atol=0)


def test_cpp_matches_committed_fixture(classifier, tmp_path):
    points = np.load(FIXTURES / "iris_like_points.npy")
    model_path = tmp_path / "iris_like_model.txt"
    shutil.copy(FIXTURES / "iris_like_model.txt", model_path)
    labels_cpp, dist_cpp = _run(classifier, model_path, points)
    assert np.array_equal(labels_cpp, np.load(FIXTURES / "iris_like_expected_labels.npy"))
    np.testing.assert_allclose(dist_cpp, np.load(FIXTURES / "iris_like_expected_dist.npy"),
                               rtol=1e-12, atol=0)
    assert np.array_equal(labels_cpp, SOMModel.load(model_path).predict(points, backend="numba"))
