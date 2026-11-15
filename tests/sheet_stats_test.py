"""Tests for aweSOM.sheet_stats on synthetic nested grids."""

import csv
import json

import h5py
import numpy as np
import pytest

from aweSOM.sheet_stats import (
    LOG10_2,
    ArraySource,
    GridFileSource,
    SheetConfig,
    sheet_statistics,
    weighted_percentile,
    write_outputs,
)

SHEET = 3


def _box(lo, n, dx):
    """Points of a lattice box, z fastest (as in grid files)."""
    axes = [lo[k] + dx * np.arange(n[k]) for k in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def _single(n=(40, 40, 20), dx=0.5, lo=(0.0, 0.0, 0.0)):
    return _box(lo, n, dx)


def _run(xyz, labels, level=None, features=None, holes=None, cfg=None, spacing=None):
    level = np.full(len(xyz), 5, dtype=np.int8) if level is None else level
    src = ArraySource(xyz, level, labels, features=features, spacing=spacing)
    return sheet_statistics(src, [SHEET], holes=holes, config=cfg or SheetConfig(plane_z=None), log=None)


def test_flat_slab_volume_thickness_aspect():
    dx = 0.5
    xyz = _single(dx=dx)
    iz = np.rint(xyz[:, 2] / dx).astype(int)
    labels = np.where((iz >= 8) & (iz < 12), SHEET, 0)  # 4 cells thick
    res = _run(xyz, labels)
    s, t = res.summary, res.components
    assert s["volume"]["total_owned"] == pytest.approx(len(xyz) * dx**3)
    assert s["volume"]["sheet_fraction"] == pytest.approx(4 / 20)
    assert s["components"]["n_components"] == 1
    assert t["volume"][0] == pytest.approx(40 * 40 * 4 * dx**3)
    assert t["thickness_inertia"][0] == pytest.approx(4 * dx, rel=1e-9)  # exact for a slab
    assert t["extent_1"][0] == pytest.approx(40 * dx, rel=1e-9)
    assert t["aspect_13"][0] == pytest.approx(10.0, rel=1e-9)
    assert t["thickness_edt"][0] == pytest.approx(3.5 * dx)  # n - 1/2 cells for even n
    assert s["thickness"]["edt"]["per_level"]["5"]["median_in_cells"] == pytest.approx(3.5)


@pytest.mark.parametrize("n_cells, expect", [(1, 1.5), (3, 3.5), (5, 5.5)])
def test_edt_thickness_odd_slabs(n_cells, expect):
    dx = 0.25
    xyz = _single(n=(24, 24, 16), dx=dx)
    iz = np.rint(xyz[:, 2] / dx).astype(int)
    labels = np.where((iz >= 5) & (iz < 5 + n_cells), SHEET, 0)
    t = _run(xyz, labels).components
    assert t["thickness_edt"][0] == pytest.approx(expect * dx)
    assert t["thickness_inertia"][0] == pytest.approx(n_cells * dx)


def test_tilted_slab_thickness():
    dx = 0.125
    xyz = _single(n=(64, 64, 24), dx=dx, lo=(-4.0, -4.0, -1.5))
    nrm = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    T = 1.0
    labels = np.where(np.abs(xyz @ nrm) <= T / 2, SHEET, 0)
    t = _run(xyz, labels).components
    assert t["thickness_edt"][0] == pytest.approx(T, abs=dx)
    assert t["thickness_inertia"][0] == pytest.approx(T, rel=0.1)


def test_components_and_connectivity():
    xyz = _single(n=(20, 20, 20), dx=1.0)
    i = np.rint(xyz).astype(int)
    a = np.all((i >= 2) & (i < 5), axis=1)  # 3^3 cube
    b = np.all((i >= 5) & (i < 7), axis=1)  # 2^3 cube touching a at one corner
    labels = np.where(a | b, SHEET, 0)
    t6 = _run(xyz, labels).components
    assert len(t6["volume"]) == 2
    assert t6["volume"].tolist() == [27.0, 8.0]
    assert t6["n_cells"].tolist() == [27, 8]
    t26 = _run(xyz, labels, cfg=SheetConfig(plane_z=None, connectivity=26)).components
    assert t26["volume"].tolist() == [35.0]


def _two_level():
    """Coarse dx=1 box [0,31]^3 owning everything outside the fine box
    [8,24]^3 (dx=0.5)."""
    coarse = _box((0.0, 0.0, 0.0), (32, 32, 32), 1.0)
    inside = np.all((coarse >= 8) & (coarse <= 24), axis=1)
    coarse = coarse[~inside]
    fine = _box((8.0, 8.0, 8.0), (33, 33, 33), 0.5)
    xyz = np.concatenate([coarse, fine])
    level = np.concatenate([np.full(len(coarse), 5), np.full(len(fine), 6)]).astype(np.int8)
    return xyz, level, len(coarse), len(fine)


def test_components_join_across_levels():
    xyz, level, nc, nf = _two_level()
    z = xyz[:, 2]
    labels = np.where((z >= 14) & (z < 16), SHEET, 0)  # slab crossing the fine box
    res = _run(xyz, labels, level=level)
    s, t = res.summary, res.components
    assert s["volume"]["per_level"]["5"]["n_components_on_level"] == 1
    assert s["volume"]["per_level"]["6"]["n_components_on_level"] == 1
    assert s["components"]["n_components"] == 1
    n5 = int(((z >= 14) & (z < 16) & (level == 5)).sum())
    n6 = int(((z >= 14) & (z < 16) & (level == 6)).sum())
    assert t["n_cells_l5"][0] == n5 and t["n_cells_l6"][0] == n6
    assert t["volume"][0] == pytest.approx(n5 * 1.0 + n6 * 0.125)
    assert s["volume"]["total_owned"] == pytest.approx(nc * 1.0 + nf * 0.125)
    # the two levels' slabs are offset by a quarter cell (vertex-centred cells)
    assert 2.0 <= t["thickness_inertia"][0] <= 2.2
    # 2 cells thick on either level: 1.5 coarse cells (1.5 M) or 3.5 fine cells (1.75 M)
    assert t["thickness_edt"][0] == pytest.approx(2.0, abs=0.5)


def test_separate_sheets_on_two_levels_stay_separate():
    xyz, level, _, _ = _two_level()
    z = xyz[:, 2]
    x = xyz[:, 0]
    # a slab only inside the fine box, away from its faces, and one only in the coarse region
    fine_sheet = (level == 6) & (z >= 15) & (z < 16) & (x > 10) & (x < 22) & (xyz[:, 1] > 10) & (xyz[:, 1] < 22)
    coarse_sheet = (level == 5) & (z >= 2) & (z < 4)
    labels = np.where(fine_sheet | coarse_sheet, SHEET, 0)
    res = _run(xyz, labels, level=level)
    assert res.summary["components"]["n_components"] == 2


def test_between_holes_radial_and_features():
    dx = 0.5
    xyz = _single(n=(41, 41, 41), dx=dx, lo=(-10.0, -10.0, -10.0))
    holes = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    mid = np.all(np.abs(xyz) <= 1.0, axis=1)
    off = np.all(np.abs(xyz - [0.0, 8.0, 0.0]) <= 1.0, axis=1)
    labels = np.where(mid | off, SHEET, 0)
    feats = {"log_chi_B": xyz[:, 0].astype(np.float32), "log_beta_inv": xyz[:, 1].astype(np.float32),
             "log_theta": np.full(len(xyz), -2.0, np.float32)}
    cfg = SheetConfig(plane_z=0.0, between_widths=(2.0, 4.0, 10.0))
    res = _run(xyz, labels, features=feats, holes=holes, cfg=cfg)
    s, t = res.summary, res.components
    b = s["between"]["widths"]
    assert b["2"]["fraction_of_sheet_volume"] == pytest.approx(0.5)
    assert b["4"]["fraction_of_sheet_volume"] == pytest.approx(0.5)
    assert b["10"]["fraction_of_sheet_volume"] == pytest.approx(1.0)
    assert b["2"]["region_volume"] == pytest.approx(np.sum((xyz[:, 0] >= -5) & (xyz[:, 0] <= 5)
                                                           & (np.abs(xyz[:, 1]) <= 2) & (np.abs(xyz[:, 2]) <= 2)) * dx**3)
    assert len(t["volume"]) == 2
    assert sorted(t["between_fraction"].tolist()) == [0.0, 1.0]
    sheet_vol = s["volume"]["sheet"]
    for key in ("hole1", "hole2", "com"):
        assert sum(s["radial"][key]["sheet_volume"]) == pytest.approx(sheet_vol)
        assert sum(s["radial"][key]["total_volume"]) == pytest.approx(s["volume"]["total_owned"])
    assert s["holes"]["com"] == [0.0, 0.0, 0.0]
    f = s["features"]
    ins = labels == SHEET
    assert f["log_beta_inv"]["inside_mean"] == pytest.approx(xyz[ins, 1].mean())
    assert f["log_beta_inv"]["outside_mean"] == pytest.approx(xyz[~ins, 1].mean())
    # log_sigma derived from log_beta_inv and log_theta
    assert f["log_sigma"]["inside_mean"] == pytest.approx(LOG10_2 + xyz[ins, 1].mean() - 2.0, rel=1e-6)
    assert sum(f["log_chi_B"]["inside_volume"]) == pytest.approx(sheet_vol)
    assert sorted(t["mean_log_beta_inv"].tolist()) == pytest.approx([0.0, 8.0], abs=1e-6)
    # slice at z = 0: the plane holds both squares
    p = res.planes[5]
    assert p["sheet"].shape == (41, 41)
    assert int((p["sheet"] == 1).sum()) == 2 * 5 * 5
    assert np.isfinite(p["background"]).all()


def _write_grid(path, xyz, level, feats, names, attrs):
    with h5py.File(path, "w") as h:
        h["xyz"] = xyz.astype(np.float32)
        h["level"] = level.astype(np.int8)
        h["features"] = np.stack([feats[n] for n in names], axis=1).astype(np.float32)
        h["names"] = np.array([n.encode() for n in names])
        h["component"] = np.zeros(len(xyz), dtype=np.int16)
        for k, v in attrs.items():
            h.attrs[k] = v


def test_grid_file_source_matches_array_source(tmp_path):
    xyz, level, _, _ = _two_level()
    rng = np.random.default_rng(1)
    names = ["log_beta_inv", "log_theta", "log_chi_B"]
    feats = {n: rng.normal(size=len(xyz)).astype(np.float32) for n in names}
    z = xyz[:, 2]
    labels = np.where(((z >= 14) & (z < 16)) | (feats["log_chi_B"] > 2.5), SHEET, 1).astype(np.uint8)
    attrs = {"horizon_centres": np.array([[10.0, 16.0, 16.0], [22.0, 16.0, 16.0]]),
             "horizon_mass": np.array([0.5, 0.5]), "iteration": 7, "time": 1.5}
    _write_grid(tmp_path / "grid.h5", xyz, level, feats, names, attrs)
    np.save(tmp_path / "labels.npy", labels)
    cfg = SheetConfig(plane_z=16.0, chunk_rows=5000)
    src = GridFileSource(tmp_path / "grid.h5", tmp_path / "labels.npy")
    a = sheet_statistics(src, [SHEET], config=cfg, log=None)
    xyz32 = xyz.astype(np.float32)
    b = sheet_statistics(ArraySource(xyz32, level, labels, feats, attrs={k: np.asarray(v).tolist() for k, v in attrs.items()}),
                         [SHEET], config=SheetConfig(plane_z=16.0, chunk_rows=10**7), log=None)
    assert a.summary["attrs"]["iteration"] == 7
    assert a.summary["volume"] == b.summary["volume"]
    for k in a.components:
        np.testing.assert_allclose(a.components[k], b.components[k], rtol=1e-10, equal_nan=True)
    for k in ("hole1", "com"):
        np.testing.assert_allclose(a.summary["radial"][k]["sheet_volume"], b.summary["radial"][k]["sheet_volume"])
    np.testing.assert_allclose(a.summary["features"]["log_sigma"]["inside_volume"],
                               b.summary["features"]["log_sigma"]["inside_volume"])
    d = src.read(6, 0, 10, ["log_sigma", "log_beta_inv", "log_theta"])
    np.testing.assert_allclose(d["log_sigma"], LOG10_2 + d["log_beta_inv"] + d["log_theta"], rtol=1e-6)
    paths = write_outputs(a, tmp_path / "out", "toy")
    summ = json.loads(open(paths["json"]).read())
    assert summ["components"]["n_components"] == len(a.components["id"])
    with open(paths["csv"]) as fh:
        rows = list(csv.reader(fh))
    assert rows[0][0] == "id" and len(rows) == len(a.components["id"]) + 1
    planes = np.load(paths["planes"])
    assert "l5_sheet" in planes and "l6_sheet" in planes
    src.close()


def test_duplicate_points_rejected():
    xyz = _single(n=(4, 4, 4), dx=1.0)
    xyz = np.concatenate([xyz, xyz[:1]])
    labels = np.zeros(len(xyz), dtype=int)
    with pytest.raises(ValueError, match="duplicate"):
        _run(xyz, labels)


def test_no_sheet_cells():
    xyz = _single(n=(6, 6, 6), dx=1.0)
    res = _run(xyz, np.zeros(len(xyz), dtype=int), holes=np.array([[0.0, 0, 0], [5.0, 5, 5]]))
    assert res.summary["volume"]["sheet"] == 0.0
    assert res.summary["components"]["n_components"] == 0
    assert len(res.components["volume"]) == 0


def test_weighted_percentile():
    v = np.array([3.0, 1.0, 2.0, np.nan])
    w = np.array([1.0, 1.0, 2.0, 5.0])
    assert weighted_percentile(v, w, 50)[0] == 2.0
    assert weighted_percentile(v, w, [0, 100]).tolist() == [1.0, 3.0]
    assert np.isnan(weighted_percentile([], [], 50)[0])


def test_blocked_edt_matches_whole_box():
    from scipy import ndimage

    dx = 0.25
    n = (48, 40, 24)
    xyz = _single(n=n, dx=dx)
    rng = np.random.default_rng(3)
    field = ndimage.gaussian_filter(rng.normal(size=n), 2.0).ravel()
    labels = np.where(field > np.quantile(field, 0.85), SHEET, 0)
    whole = _run(xyz, labels, cfg=SheetConfig(plane_z=None, edt_block=None))
    blocked = _run(xyz, labels, cfg=SheetConfig(plane_z=None, edt_block=7, edt_overlap=12))
    ew, eb = whole.summary["thickness"]["edt"], blocked.summary["thickness"]["edt"]
    assert eb["n_ridge_beyond_overlap"] == 0
    assert eb["n_ridge_cells"] == ew["n_ridge_cells"] > 0
    assert eb["hist"] == ew["hist"]
    np.testing.assert_array_equal(blocked.components["thickness_edt"], whole.components["thickness_edt"])
    # a slab thicker than twice the overlap is flagged
    iz = np.rint(xyz[:, 2] / dx).astype(int)
    thick = np.where((iz >= 2) & (iz < 22), SHEET, 0)
    flagged = _run(xyz, thick, cfg=SheetConfig(plane_z=None, edt_block=8, edt_overlap=3))
    assert flagged.summary["thickness"]["edt"]["n_ridge_beyond_overlap"] > 0


def test_composite_mask_keeps_fine_background():
    from aweSOM.sheet_stats import LevelLattice, _composite_sheet

    coarse = LevelLattice(5, 1.0, (0, 0, 0), (1, 1, 1))
    fine = LevelLattice(7, 0.25, (0.25, 0, 0), (2, 1, 1))
    sheets = {5: np.ones(coarse.shape, bool), 7: np.zeros(fine.shape, bool)}
    owned = {5: np.ones(coarse.shape, bool), 7: np.ones(fine.shape, bool)}
    assert not _composite_sheet(fine, 0, {5: coarse, 7: fine}, sheets, owned).any()
    # and a fine sheet cell is seen from the coarse box
    sheets[7][:] = True
    comp = _composite_sheet(coarse, 1, {5: coarse, 7: fine}, sheets, owned)
    assert comp[1, 1, 1]  # the coarse cell itself


def test_grid_file_source_sharded_parts(tmp_path):
    from aweSOM.npy_parts import write_npy_manifest
    from aweSOM.stream import npy_crc32

    xyz, level, _, _ = _two_level()
    rng = np.random.default_rng(2)
    names = ["log_beta_inv", "log_theta", "log_chi_B"]
    feats = {n: rng.normal(size=len(xyz)).astype(np.float32) for n in names}
    labels = np.where((xyz[:, 2] >= 14) & (xyz[:, 2] < 16), SHEET, 1).astype(np.uint8)
    attrs = {"horizon_centres": np.array([[10.0, 16.0, 16.0], [22.0, 16.0, 16.0]]), "horizon_mass": np.array([0.5, 0.5])}
    _write_grid(tmp_path / "grid.h5", xyz, level, feats, names, attrs)
    np.save(tmp_path / "plain.npy", labels)

    pdir = tmp_path / "sharded.parts"
    pdir.mkdir()
    mid = len(labels) // 2
    entries = []
    for r, (a, b) in enumerate(((0, mid), (mid, len(labels)))):
        pf = pdir / f"part-{r:04d}.npy"
        np.save(pf, labels[a:b])
        entries.append(dict(rank=r, file=pf.name, start=a, stop=b, rows=b - a, bytes=b - a, crc32=int(npy_crc32(pf, 0, b - a))))
    write_npy_manifest(pdir, dtype=labels.dtype, shape=labels.shape, parts=entries)

    cfg = SheetConfig(plane_z=16.0, chunk_rows=5000)
    src_plain = GridFileSource(tmp_path / "grid.h5", tmp_path / "plain.npy")
    src_parts = GridFileSource(tmp_path / "grid.h5", tmp_path / "sharded.npy")  # resolves to sharded.parts
    try:
        res_plain = sheet_statistics(src_plain, [SHEET], config=cfg, log=None)
        res_parts = sheet_statistics(src_parts, [SHEET], config=cfg, log=None)
        assert res_parts.summary["volume"] == res_plain.summary["volume"]
        for k in res_plain.components:
            np.testing.assert_allclose(res_parts.components[k], res_plain.components[k], rtol=1e-12, equal_nan=True)
    finally:
        src_plain.close()
        src_parts.close()



def test_without_resource_module(monkeypatch):
    """No ``resource`` module (Windows): peak RSS is None and the log lines still format."""
    import aweSOM.sheet_stats as sheet_stats

    monkeypatch.setattr(sheet_stats, "resource", None)
    assert sheet_stats._rss_gb() is None
    xyz = _single(n=(12, 12, 8))
    iz = np.rint(xyz[:, 2] / 0.5).astype(int)
    labels = np.where((iz >= 3) & (iz < 5), SHEET, 0)
    lines = []
    src = ArraySource(xyz, np.full(len(xyz), 5, dtype=np.int8), labels)
    res = sheet_statistics(src, [SHEET], config=SheetConfig(plane_z=None), log=lines.append)
    assert res.summary["peak_rss_gb"] is None
    assert any("peak RSS unknown" in line for line in lines)
    json.dumps(res.summary)  # None is valid JSON (null)


def test_imports_without_resource_module(monkeypatch):
    """A fresh import with ``resource`` blocked takes the ImportError branch."""
    import importlib.util
    import sys

    import aweSOM.sheet_stats as sheet_stats

    monkeypatch.setitem(sys.modules, "resource", None)
    spec = importlib.util.spec_from_file_location("aweSOM._sheet_stats_without_resource", sheet_stats.__file__)
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, fresh)  # dataclasses look up their module
    spec.loader.exec_module(fresh)
    assert fresh.resource is None
    assert fresh._rss_gb() is None
