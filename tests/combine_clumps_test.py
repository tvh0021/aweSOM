"""Edge cases of the fast clump helpers in examples/plasma-turbulence/combine_clumps.py."""

import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest

pytest.importorskip("tqdm")
_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "plasma-turbulence" / "combine_clumps.py"
_spec = importlib.util.spec_from_file_location("combine_clumps", _PATH)
cc = importlib.util.module_from_spec(_spec)
sys.modules["combine_clumps"] = cc  # numba's on-disk cache resolves functions by module name
_spec.loader.exec_module(cc)


@pytest.mark.parametrize(
    "dicts",
    [({}, {}, {}), ({"a": []}, {}, {}), ({"a": []}, {"p": []}, {"n": []})],
)
def test_merge_double_sheets_fast_without_coordinates_matches_original(dicts):
    copies = [{k: list(v) for k, v in d.items()} for d in dicts]
    assert cc.merge_double_sheets_fast(*dicts) == cc.merge_double_sheets(*copies, fast=False)
    assert list(dicts) == copies


def test_label_cc3d_declines_ranges_that_wrap_uint32(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("cc3d must not see wrapped labels")

    monkeypatch.setitem(sys.modules, "cc3d", types.SimpleNamespace(connected_components=fail))
    data = np.array([0, 2**32], dtype=np.int64).reshape(1, 1, 2)
    structure = cc._structure_from_deltas(cc.generate_neighbors(6), 3)
    assert cc._label_cc3d(data, structure, periodic=False) is None
    _, n = cc.label_clumps(data, cc.generate_neighbors(6))
    assert n == 2


def _sheet_dicts():
    # one double sheet touching a positive and a negative clump, one isolated double sheet
    D = {0: [(2, 2, 2), (2, 3, 2)], 1: [(8, 8, 8)]}
    P = {5: [(1, 2, 2)], 6: [(5, 5, 5)]}
    N = {7: [(3, 3, 2)]}
    return D, P, N


def test_merge_double_sheets_defaults_to_fast_with_same_result():
    a, b = _sheet_dicts(), _sheet_dicts()
    out_default = cc.merge_double_sheets(*a)
    out_loop = cc.merge_double_sheets(*b, fast=False)
    assert list(out_default) == list(out_loop)
    assert {k: set(v) for k, v in out_default.items()} == {k: set(v) for k, v in out_loop.items()}
    assert [type(v) for v in out_default.values()] == [type(v) for v in out_loop.values()]
    assert list(a[1]) == list(b[1]) == [6] and list(a[2]) == list(b[2]) == []


def test_merge_double_sheets_falls_back_on_overlapping_clumps():
    D = {0: [(2, 2, 2)]}
    P = {5: [(1, 2, 2)], 6: [(1, 2, 2)]}  # overlap: fast path refuses, loop runs
    N = {7: [(3, 2, 2)]}
    copies = [{k: list(v) for k, v in d.items()} for d in (D, P, N)]
    with pytest.raises(ValueError):
        cc.merge_double_sheets_fast(*[{k: list(v) for k, v in d.items()} for d in (D, P, N)])
    out = cc.merge_double_sheets(D, P, N)
    ref = cc.merge_double_sheets(*copies, fast=False)
    assert out == ref and list(P) == list(copies[1]) and list(N) == list(copies[2])


def _label_clumps_per_value(data_array, connectivity, periodic):
    """The previous scipy path of label_clumps: one full-size mask per value."""
    from scipy import ndimage

    structure = cc._structure_from_deltas(connectivity, data_array.ndim)
    out = np.zeros(data_array.shape, dtype=np.int64)
    offset = 0
    for value in np.unique(data_array):
        lab, n = ndimage.label(data_array == value, structure=structure)
        out[lab > 0] = lab[lab > 0] + offset
        offset += n
    out -= 1
    if periodic:
        out, offset = cc._merge_periodic(data_array, out, offset, connectivity)
    return out, offset


def _clumps_to_ids(clumps, shape):
    ids = np.full(shape, -1, dtype=np.int32)
    for k, cl in enumerate(clumps):
        ids[tuple(np.array(cl).T)] = k
    return ids


@pytest.mark.parametrize("shape,conn", [((9, 11, 7), 6), ((9, 11, 7), 10), ((9, 11, 7), 26), ((13, 10), 4), ((13, 10), 8)])
@pytest.mark.parametrize("n_values", [2, 5, 40])
def test_label_clumps_bounding_box_path_matches_reference(shape, conn, n_values):
    rng = np.random.default_rng(conn * 100 + n_values)
    data = rng.integers(-1, n_values - 1, size=shape)
    deltas = cc.generate_neighbors(conn)
    ids, n = cc.label_clumps(data, deltas, use_cc3d=False)
    clumps = cc.get_clumps(data, deltas)
    assert n == len(clumps) and np.array_equal(ids, _clumps_to_ids(clumps, shape))
    for periodic in (False, True):
        raw, n_raw = cc.label_clumps(data, deltas, periodic=periodic, use_cc3d=False, renumber=False)
        ref, n_ref = _label_clumps_per_value(data, deltas, periodic)
        assert n_raw == n_ref and np.array_equal(raw, ref)
