"""Tests for aweSOM.inference.narrow_label_dtype and v1.4.0 label-file byte parity."""

from __future__ import annotations

import numpy as np
import pytest

import aweSOM.inference as inference
from aweSOM.inference import narrow_label_dtype
from aweSOM.run_som import save_cluster_labels


def _save_v140_rule(som_labels, path) -> None:
    """Reference v1.4.0 save_cluster_labels rule."""
    arr = np.asarray(som_labels)
    if np.issubdtype(arr.dtype, np.integer) and arr.size:
        if arr.min() >= 0:
            top = arr.max()
            for dtype in (np.uint8, np.uint16):
                if top <= np.iinfo(dtype).max:
                    arr = arr.astype(dtype)
                    break
    np.save(path, arr)


def test_narrow_label_dtype_in_all():
    assert "narrow_label_dtype" in inference.__all__


@pytest.mark.parametrize(
    "ids, expected",
    [
        (np.array([], dtype=np.int32), np.int32),
        (np.array([], dtype=np.int64), np.int32),
        (np.array([], dtype=np.uint8), np.int32),
        (np.array([-1], dtype=np.int32), np.int32),
        (np.array([0, 255, -1], dtype=np.int32), np.int32),
        (np.array([-5, 10], dtype=np.int64), np.int32),
        (np.array([0], dtype=np.int32), np.uint8),
        (np.array([0, 255], dtype=np.int32), np.uint8),
        (np.array([255], dtype=np.int64), np.uint8),
        (np.array([0, 256], dtype=np.int32), np.uint16),
        (np.array([256, 65535], dtype=np.int32), np.uint16),
        (np.array([65535], dtype=np.int64), np.uint16),
        (np.array([0, 65536], dtype=np.int32), np.int32),
        (np.array([65536, 100000], dtype=np.int64), np.int32),
        ([0, 1, 255], np.uint8),
        ([0, 256], np.uint16),
        ([[0, 10], [20, 255]], np.uint8),
    ],
)
def test_narrow_label_dtype_boundaries(ids, expected):
    got = narrow_label_dtype(ids)
    assert isinstance(got, np.dtype)
    assert got == np.dtype(expected)


@pytest.mark.parametrize(
    "case_name, labels, expected_dtype",
    [
        ("int32_0_to_255", np.array([0, 1, 127, 255], dtype=np.int32), np.uint8),
        ("int64_0_to_255", np.array([0, 42, 255], dtype=np.int64), np.uint8),
        ("int32_256_to_65535", np.array([0, 256, 65535], dtype=np.int32), np.uint16),
        ("int64_256_to_65535", np.array([256, 10000, 65535], dtype=np.int64), np.uint16),
        ("int32_above_65535", np.array([0, 65536, 70000], dtype=np.int32), np.int32),
        ("int64_above_65535", np.array([0, 65536, 100000], dtype=np.int64), np.int64),
        ("int32_negative", np.array([-1, 0, 5], dtype=np.int32), np.int32),
        ("int64_negative", np.array([-2, 0, 255], dtype=np.int64), np.int64),
        ("int32_empty", np.array([], dtype=np.int32), np.int32),
        ("int64_empty", np.array([], dtype=np.int64), np.int64),
        ("uint8_empty", np.array([], dtype=np.uint8), np.uint8),
        ("float32_input", np.array([0.0, 1.0, 2.5], dtype=np.float32), np.float32),
        ("float64_input", np.array([0.0, 1.0, 255.0], dtype=np.float64), np.float64),
        ("bool_input", np.array([False, True, False], dtype=bool), bool),
    ],
)
def test_save_cluster_labels_matches_v140_bytes(tmp_path, monkeypatch, case_name, labels, expected_dtype):
    monkeypatch.chdir(tmp_path)
    save_cluster_labels(labels, 10, 10, 0.5, 100, name_of_dataset=case_name)
    actual_path = tmp_path / f"labels.{case_name}-10-10-0.5-100-1s.npy"
    ref_path = tmp_path / f"ref.{case_name}.npy"
    _save_v140_rule(labels, ref_path)

    assert actual_path.read_bytes() == ref_path.read_bytes()
    loaded = np.load(actual_path)
    assert loaded.dtype == np.dtype(expected_dtype)
    np.testing.assert_array_equal(loaded, np.load(ref_path))
