import os
import pickle
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from aweSOM.run_som import (
    batch_separator,
    number_of_nodes,
    initialize_lattice,
    manual_scaling,
    save_som_object,
    save_cluster_labels,
)


# Mock class for Lattice to use in tests
class MockLattice:
    def __init__(
        self, xdim, ydim, alpha_0, train, alpha_type="decay", sampling_type="uniform"
    ):
        self.xdim = xdim
        self.ydim = ydim
        self.alpha_0 = alpha_0
        self.train = train
        self.alpha_type = alpha_type
        self.sampling_type = sampling_type
        self.lattice = np.zeros((xdim, ydim))
        self.seed = 42


@pytest.fixture
def sample_data():
    return np.random.rand(100, 10)


def test_batch_separator(sample_data):
    batches = batch_separator(sample_data, 5)
    assert batches.shape == (5, 20, 10)


def test_number_of_nodes():
    assert number_of_nodes(100, 10) == int(5 * np.sqrt(100 * 10) / 6)


def test_initialize_lattice(sample_data):
    xdim, ydim = initialize_lattice(sample_data, 0.5)
    assert xdim * ydim == pytest.approx(number_of_nodes(100, 10), 5)


def test_manual_scaling(sample_data):
    scaled_data = manual_scaling(sample_data)
    assert scaled_data.shape == sample_data.shape


def test_save_som_object(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the helpers write into the current directory
    som = MockLattice(10, 10, 0.5, 100)
    save_som_object(som, 10, 10, 0.5, 100, name_of_dataset="test")
    file_name = "som_object.test-10-10-0.5-100-1s.pkl"
    assert os.path.isfile(
        file_name
    ), f"{file_name} does not exist in the current working directory."
    os.remove(file_name)


@pytest.mark.parametrize(
    "top, dtype", [(5, np.uint8), (255, np.uint8), (300, np.uint16), (70000, np.int32)]
)
def test_save_cluster_labels_smallest_unsigned_dtype(tmp_path, monkeypatch, top, dtype):
    monkeypatch.chdir(tmp_path)
    labels = (np.arange(1000, dtype=np.int32) * 7919) % (top + 1)
    labels[0] = top
    save_cluster_labels(labels, 10, 10, 0.5, 100, name_of_dataset="test")
    loaded = np.load("labels.test-10-10-0.5-100-1s.npy")
    assert loaded.dtype == dtype
    assert loaded.max() == top
    np.testing.assert_array_equal(loaded.astype(np.int64), labels.astype(np.int64))


def test_run_som_cli_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "src" / "aweSOM" / "run_som.py"
    src_dir = repo_root / "src"

    h5_name = "features_test_smoke.h5"
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 4)).astype(np.float64)
    names = np.array([b"f0", b"f1", b"f2", b"f3"], dtype="S2")
    with h5py.File(tmp_path / h5_name, "w") as f5:
        f5.create_dataset("features", data=features)
        f5.create_dataset("names", data=names)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = (
        f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(src_dir)
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--features_path",
            f"{tmp_path}{os.sep}",
            "--file",
            h5_name,
            "--xdim",
            "5",
            "--ydim",
            "4",
            "--train",
            "50",
            "--threshold",
            "0.5",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "Data scaled with manual" in proc.stdout

    label_files = list(tmp_path.glob("labels.*.npy"))
    assert len(label_files) == 1
    labels = np.load(label_files[0])
    assert labels.shape == (60,)
    assert np.issubdtype(labels.dtype, np.unsignedinteger)

    som_files = list(tmp_path.glob("som_object.*.pkl"))
    assert len(som_files) == 1
    with open(som_files[0], "rb") as f:
        som = pickle.load(f)
    assert som.xdim == 5
    assert som.ydim == 4
    assert som.lattice.shape == (20, 4)


# Run the tests
if __name__ == "__main__":
    pytest.main()
