"""Tests for aweSOM.io.carpetx: synthetic CarpetX Silo and openPMD-HDF5 runs (tests/grmhd_fixtures.py),
TSV parsing, and, when the FM torus run is on disk, its 2D Silo output against its TSV lines."""

import sys
from pathlib import Path

import numpy as np
import pytest

from aweSOM.io import ASTERX_NAMES, CarpetXReader, iter_level_blocks, open_reader, read_tsv

sys.path.insert(0, str(Path(__file__).parent))
from grmhd_fixtures import CX_GHOSTS, CX_X0, carpetx_levels, field, write_openpmd, write_silo, write_tsv  # noqa: E402

VARS = list(ASTERX_NAMES)
# rl0: 16^3 cells less the 6^3 that rl1 (cells [10, 22) at ratio 2 = rl0 cells [5, 11)) covers
OWNED = {0: 16 ** 3 - 6 ** 3, 1: 12 ** 3}
FM = Path("/Volumes/ExternalNVMe/simulation-data/eg-liwei-fm/FM_4lvl_Noble/FM_4lvl_Noble_r0000/data")


@pytest.fixture(scope="module")
def silo(tmp_path_factory):
    d = tmp_path_factory.mktemp("silo")
    write_silo(d, it=0)
    return d


@pytest.fixture(scope="module")
def opmd(tmp_path_factory):
    d = tmp_path_factory.mktemp("opmd")
    write_openpmd(d, it=0)
    return d


def _centres(level, lo, hi):
    dx, _ = carpetx_levels()[level]
    return np.meshgrid(*[CX_X0[d] + (np.arange(lo[d], hi[d]) + 0.5) * dx[d] for d in range(3)], indexing="ij")


def _collect(blocks):
    out = {}
    for b in blocks:
        o = out.setdefault(b.level, {"xyz": [], **{v: [] for v in VARS}})
        o["xyz"].append(b.xyz())
        for v in VARS:
            o[v].append(b.data[v][b.interior][b.owned_mask])
    return {rl: {k: np.concatenate(v) for k, v in o.items()} for rl, o in out.items()}


def test_silo_layout_and_boxes(silo):
    r = CarpetXReader(silo)
    assert r.format == "silo" and r.iterations() == [0] and r.levels(0) == [0, 1]
    snap = r.snapshot(0)
    assert np.allclose(snap.x0, CX_X0) and snap.time == 2.0
    for rl, (dx, boxes) in carpetx_levels().items():
        L = snap.levels[rl]
        assert np.allclose(L.dx, dx)
        assert [(tuple(b.lo), tuple(b.hi)) for b in L.boxes] == [(tuple(lo), tuple(hi)) for lo, hi in boxes]
        for b in L.boxes:
            a, alo = r.read_box("rho_b", 0, rl, b.c)
            assert np.array_equal(alo, b.lo - CX_GHOSTS)
            assert np.allclose(a, field("rho_b", *_centres(rl, alo, alo + a.shape)), rtol=1e-13)


@pytest.mark.parametrize("kind", ["silo", "opmd"])
def test_blocks_own_valid_cells_minus_finer_levels(kind, silo, opmd):
    r = CarpetXReader(silo if kind == "silo" else opmd)
    assert r.format == ("silo" if kind == "silo" else "openpmd")
    got = _collect(r.iter_level_blocks(VARS, 0, slab=5))
    for rl, n in OWNED.items():
        xyz = got[rl]["xyz"]
        assert len(xyz) == n and len(np.unique(xyz, axis=0)) == n
        for v in VARS:
            assert np.allclose(got[rl][v], field(v, *xyz.T), rtol=1e-12, atol=1e-14), (kind, rl, v)
    x, y, z = got[0]["xyz"].T
    assert not np.any((x > -3) & (x < 3) & (y > -3) & (y < 3) & (z > -3) & (z < 3))  # rl1 covers [-3, 3]^3


def test_silo_and_openpmd_give_the_same_points(silo, opmd):
    a = _collect(CarpetXReader(silo).iter_level_blocks(VARS, 0))
    b = _collect(CarpetXReader(opmd).iter_level_blocks(VARS, 0))
    for rl in OWNED:
        assert np.array_equal(a[rl]["xyz"], b[rl]["xyz"])
        for v in VARS:
            assert np.allclose(a[rl][v], b[rl][v], rtol=1e-14, atol=0)


def test_halo_sources(silo, opmd):
    """rl1's halo: Silo has the box's own ghosts; openPMD has no ghosts, so
    the halo is the coarser level injected ("coarse") or NaN ("nan"), never
    the zeros stored outside the boxes."""
    halo = 2
    (blk_s,) = [b for b in CarpetXReader(silo).iter_level_blocks(["P"], 0, levels=[1], halo=halo)]
    lo = np.array([10, 10, 10]) - halo
    want = field("P", *_centres(1, lo, lo + blk_s.shape))
    assert blk_s.shape == (16, 16, 16) and np.allclose(blk_s.data["P"], want, rtol=1e-13)
    assert np.allclose(blk_s.origin, CX_X0 + (lo + 0.5) * 0.5)

    (blk_c,) = list(CarpetXReader(opmd, fill="coarse").iter_level_blocks(["P"], 0, levels=[1], halo=halo))
    (blk_n,) = list(CarpetXReader(opmd, fill="nan").iter_level_blocks(["P"], 0, levels=[1], halo=halo))
    inner = blk_c.interior
    assert np.array_equal(blk_c.data["P"][inner], blk_n.data["P"][inner])
    edge = np.ones(blk_c.shape, bool)
    edge[inner] = False
    assert np.all(np.isnan(blk_n.data["P"][edge]))
    # injection: halo fine cell (i, j, k) takes coarse cell (i//2, j//2, k//2)
    ci = (lo + np.indices(blk_c.shape).reshape(3, -1).T) // 2
    coarse = field("P", *(CX_X0 + (ci + 0.5) * 1.0).T).reshape(blk_c.shape)
    assert np.allclose(blk_c.data["P"][edge], coarse[edge], rtol=1e-13)


def test_spheres_skip_and_entry_points(silo, opmd):
    r = CarpetXReader(silo, spheres=[((0.0, 0.0, 0.0), 1.0)])
    xyz = np.concatenate([b.xyz() for b in r.iter_level_blocks(["P"], 0, levels=[1])])
    assert len(xyz) == OWNED[1] - 32  # rl1 cell centres +-0.25, +-0.75 within r < 1: 8 + 24
    assert np.all(np.linalg.norm(xyz, axis=1) >= 1.0)
    keys = [b.key for b in CarpetXReader(silo).iter_level_blocks(["P"], 0, levels=[0], slab=4)]
    assert keys == [f"rl0/b{c}/x{a:05d}" for c in (0, 1) for a in (0, 4)]
    rest = [b.key for b in CarpetXReader(silo).iter_level_blocks(["P"], 0, levels=[0], slab=4, skip={keys[0]})]
    assert rest == keys[1:]
    assert isinstance(open_reader(silo), CarpetXReader) and isinstance(open_reader(opmd), CarpetXReader)
    assert [b.key for b in iter_level_blocks(opmd, ["P"], 0, levels=[1])] == ["rl1/b0/x00000"]
    with pytest.raises(KeyError):
        next(CarpetXReader(silo).iter_level_blocks(["not_a_var"], 0, levels=[1]))
    with pytest.raises(ValueError):
        CarpetXReader(silo, fill="zero")


def test_read_tsv(tmp_path):
    rows = write_tsv(tmp_path / "hydrobasex-rho.it001024.x.tsv")
    t = read_tsv(tmp_path / "hydrobasex-rho.it001024.x.tsv")
    assert list(t)[:10] == ["iteration", "time", "patch", "level", "i", "j", "k", "x", "y", "z"]
    assert t["level"].dtype == np.int64 and t["i"].min() == -3
    assert np.array_equal(t["rho"], rows[:, 10])
    bad = tmp_path / "bad.tsv"
    bad.write_text("1 2 3\n")
    with pytest.raises(ValueError):
        read_tsv(bad)


@pytest.mark.skipif(not (FM / "silo").is_dir(), reason="FM_4lvl_Noble run not on disk")
def test_real_silo_plane_matches_tsv_lines():
    """The FM torus run's xy-plane Silo output against its TSV x lines (it 76800):
    each level's plane is the cell layer below the z = 0.02 face, and the TSV
    line of that level runs through it, so every TSV cell in a Silo box must
    hold the same value."""
    r = CarpetXReader(FM / "silo", plane="xy")
    t = read_tsv(FM / "tsv/hydrobasex-vel.it076800.x.tsv")
    snap = r.snapshot(76800)
    n_cmp = 0
    for L, lv in snap.levels.items():
        sel = t["level"] == L
        assert np.allclose(t["z"][sel], 0.02 - 0.5 * lv.dx[0])  # the plane's cell centre
        for b in lv.boxes:
            a, alo = r.read_box("hydrobasex_velx", 76800, L, b.c)
            i, j = t["i"][sel] - alo[0], t["j"][sel] - alo[1]
            ok = (i >= 0) & (i < a.shape[0]) & (j >= 0) & (j < a.shape[1])
            assert np.array_equal(a[i[ok], j[ok], 0], t["velx"][sel][ok])
            x = snap.x0[0] + (t["i"][sel][ok] + 0.5) * lv.dx[0]
            assert np.allclose(x, t["x"][sel][ok], atol=1e-9)
            n_cmp += int(ok.sum())
    assert n_cmp > 500
