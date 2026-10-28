"""Tests for aweSOM.io.carpet on a synthetic two-level Carpet run (tests/grmhd_fixtures.py)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from aweSOM.io import CarpetHDF5Reader, iter_level_blocks, open_reader
from aweSOM.io.carpet import parse_active

sys.path.insert(0, str(Path(__file__).parent))
from grmhd_fixtures import CARPET_VARS, field, write_carpet  # noqa: E402

ETKPLOT = Path(os.environ.get("AWESOM_ETKPLOT_PATH", "/Volumes/ExternalNVMe/etk-analysis"))
VARS = list(CARPET_VARS)
# rl0: active points 1..15 on each axis (15^3), less the rl1 active region
# x in [-1.5, 2.5], y, z in [-2, 2] (9^3 points of rl0); rl1: 17^3 active points.
OWNED = {0: 15 ** 3 - 9 ** 3, 1: 17 ** 3}
SPHERE = [((-3.0, -3.0, -3.0), 0.8)]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("carpet")
    write_carpet(d, it=0)
    return d


def _collect(blocks):
    out = {}
    for b in blocks:
        o = out.setdefault(b.level, {"xyz": [], "keys": [], **{v: [] for v in VARS}})
        o["xyz"].append(b.xyz())
        o["keys"].append(b.key)
        for v in VARS:
            o[v].append(b.data[v][b.interior][b.owned_mask])
    return {rl: {k: (np.concatenate(v) if k != "keys" else v) for k, v in o.items()} for rl, o in out.items()}


def test_parse_active():
    s = "bboxset<CCTK_INT4,3>(set<bbox>:{([2,0,0]:[9,4,4]:[1,1,1]),([0,0,0]:[1,4,4]:[1,1,1])},stride:[1,1,1])"
    assert parse_active(s, (0, 0, 0)) == (((2, 0, 0), (10, 5, 5)), ((0, 0, 0), (2, 5, 5)))
    assert parse_active("garbage", (0, 0, 0)) is None


def test_native_blocks_ownership_values_and_coordinates(run):
    r = CarpetHDF5Reader(run, backend="native")
    assert r.backend == "native" and r.iterations() == [0] and r.levels(0) == [0, 1]
    got = _collect(r.iter_level_blocks(VARS, 0, slab=4))
    for rl, n in OWNED.items():
        xyz = got[rl]["xyz"]
        assert len(xyz) == n
        assert len(np.unique(xyz, axis=0)) == n
        for v in VARS:
            assert np.allclose(got[rl][v], field(v, *xyz.T), rtol=1e-13, atol=1e-14), (rl, v)
    # rl0 owns nothing inside the finer level's active region
    x, y, z = got[0]["xyz"].T
    inside = (x >= -1.5) & (x <= 2.5) & (np.abs(y) <= 2) & (np.abs(z) <= 2)
    assert not inside.any()
    # the outer boundary points (bbox faces) are not active
    assert got[0]["xyz"].min() == -3.5 and got[0]["xyz"].max() == 3.5


def test_block_layout_and_halo(run):
    r = CarpetHDF5Reader(run, backend="native")
    blocks = list(r.iter_level_blocks(["rho_b"], 0, levels=[0], slab=4, halo=2))
    # x point 16 is the outer boundary: that slab owns nothing and is not yielded
    assert [b.key for b in blocks] == [f"rl0/p0/x{a:05d}" for a in (0, 4, 8, 12)]
    b = blocks[1]  # x points 4..7 of the 17-point patch, halo 2..9
    level, origin, spacing, data = b
    assert level == 0 and np.allclose(spacing, 0.5) and b.shape == (8, 17, 17)
    assert np.allclose(origin, [-4.0 + 0.5 * 2, -4.0, -4.0])
    ax = [origin[d] + spacing[d] * np.arange(b.shape[d]) for d in range(3)]
    want = field("rho_b", *np.meshgrid(*ax, indexing="ij"))
    assert np.allclose(data["rho_b"], want, rtol=1e-13)  # the halo comes from the neighbouring component
    first = blocks[0].data["rho_b"]
    assert np.all(np.isnan(first[:2])) and np.all(np.isfinite(first[2:]))  # no data left of the patch


def test_slab_width_does_not_change_the_points(run):
    r = CarpetHDF5Reader(run, backend="native")
    a = _collect(r.iter_level_blocks(VARS, 0, slab=3))
    b = _collect(r.iter_level_blocks(VARS, 0, slab=64))
    for rl in OWNED:
        for k in ["xyz"] + VARS:
            assert np.array_equal(a[rl][k], b[rl][k]), (rl, k)


def test_spheres_and_skip(run):
    r = CarpetHDF5Reader(run, backend="native", spheres=SPHERE)
    blocks = list(r.iter_level_blocks(["P"], 0, levels=[0], slab=4))
    assert blocks[0].info["level_masked"] == 19  # centre, 6 faces and 12 edges of the 0.5 grid
    xyz = np.concatenate([b.xyz() for b in blocks])
    assert len(xyz) == OWNED[0] - 19
    assert np.all(np.sum((xyz + 3.0) ** 2, axis=1) >= 0.8 ** 2)
    rest = list(r.iter_level_blocks(["P"], 0, levels=[0], slab=4, skip={"rl0/p0/x00000"}))
    assert [b.key for b in rest] == [b.key for b in blocks[1:]]


def test_generic_entry_points(run):
    assert isinstance(open_reader(run, backend="native"), CarpetHDF5Reader)
    keys = [b.key for b in iter_level_blocks(run, ["P"], 0, levels=[1], backend="native")]
    assert keys == ["rl1/p0/x00000"]
    with pytest.raises(ValueError):
        next(CarpetHDF5Reader(run, backend="native").iter_level_blocks(["P"], 0, levels=[4]))


@pytest.mark.skipif(not (ETKPLOT / "etkplot").is_dir(), reason="etkplot not available")
def test_native_equals_etkplot_backend(run):
    try:
        e = CarpetHDF5Reader(run, backend="etkplot", etkplot_path=ETKPLOT, spheres=SPHERE)
    except ImportError:
        pytest.skip("etkplot not importable here")
    n = CarpetHDF5Reader(run, backend="native", spheres=SPHERE)
    be = list(e.iter_level_blocks(VARS, 0, slab=4))
    bn = list(n.iter_level_blocks(VARS, 0, slab=4))
    assert [b.key for b in be] == [b.key for b in bn]
    for x, y in zip(be, bn):
        assert np.array_equal(x.owned, y.owned) and np.array_equal(x.component, y.component)
        assert np.array_equal(x.origin, y.origin) and np.array_equal(x.xyz(), y.xyz())
        for v in VARS:
            assert np.array_equal(x.data[v], y.data[v], equal_nan=True), (x.key, v)
