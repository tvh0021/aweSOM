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
    assert cc.merge_double_sheets_fast(*dicts) == cc.merge_double_sheets(*copies)
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
