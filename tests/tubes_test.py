"""Tests of aweSOM.tubes: grouping, stitching across levels, descriptors, matching."""

import numpy as np
import pytest

from aweSOM import tubes


def _grid(n, dx=1.0, origin=0.0):
    g = origin + dx * np.arange(n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


# ----------------------------------------------------------------------------- group_cells


def test_group_cells_separates_blobs_and_orders_by_size():
    xyz = _grid(12)
    mask = np.zeros(len(xyz), bool)
    a = np.all((xyz >= 1) & (xyz <= 3), axis=1)  # 27 cells
    b = np.all((xyz >= 6) & (xyz <= 10), axis=1)  # 125 cells
    mask |= a | b
    ids = tubes.group_cells(mask, xyz, dx_per_level=1.0)
    assert set(np.unique(ids)) == {-1, 0, 1}
    assert np.all(ids[b] == 0) and np.all(ids[a] == 1)
    assert np.all(ids[~mask] == -1)


def test_group_cells_connectivity_diagonal():
    xyz = _grid(4)
    mask = np.zeros(len(xyz), bool)
    mask[np.all(xyz == [1, 1, 1], axis=1)] = True
    mask[np.all(xyz == [2, 2, 2], axis=1)] = True
    assert len(np.unique(tubes.group_cells(mask, xyz, connectivity=26))) == 2  # -1 and 0
    assert len(np.unique(tubes.group_cells(mask, xyz, connectivity=6))) == 3


def test_group_cells_labels_do_not_merge_across_values():
    xyz = _grid(6)
    lab = np.full(len(xyz), -1)
    lab[xyz[:, 0] <= 2] = 3
    lab[xyz[:, 0] >= 3] = 7
    ids = tubes.group_cells(lab, xyz)
    assert len(np.unique(ids[xyz[:, 0] <= 2])) == 1
    assert len(np.unique(ids[xyz[:, 0] >= 3])) == 1
    assert ids[0] != ids[-1]


def _two_level_rod(gap=False):
    """A rod along x through a coarse grid (dx 2) with a refined cube (dx 1) in the middle.
    The coarse level does not own points inside the fine region."""
    coarse = _grid(32, dx=2.0)  # 0..62
    fine_lo, fine_hi = 20.0, 43.0  # fine region [20, 43]
    fine = _grid(24, dx=1.0, origin=20.0)
    inside = np.all((coarse >= fine_lo) & (coarse <= fine_hi), axis=1)
    coarse = coarse[~inside]
    xyz = np.concatenate([coarse, fine])
    level = np.concatenate([np.full(len(coarse), 5), np.full(len(fine), 6)])
    rod = (np.abs(xyz[:, 1] - 32) <= 2) & (np.abs(xyz[:, 2] - 32) <= 2)
    if gap:
        rod &= ~((xyz[:, 0] > 16) & (xyz[:, 0] < 24))
    return xyz, level, rod


def test_group_cells_stitches_levels():
    xyz, level, rod = _two_level_rod()
    dx = {5: 2.0, 6: 1.0}
    unstitched = tubes.group_cells(rod, xyz, level, dx, stitch=False)
    assert len(np.unique(unstitched[rod])) == 3  # coarse left, fine, coarse right
    ids = tubes.group_cells(rod, xyz, level, dx)
    assert len(np.unique(ids[rod])) == 1
    assert np.all(ids[~rod] == -1)


def test_group_cells_does_not_stitch_across_a_gap():
    xyz, level, rod = _two_level_rod(gap=True)
    ids = tubes.group_cells(rod, xyz, level, {5: 2.0, 6: 1.0})
    assert len(np.unique(ids[rod])) == 2


def test_group_cells_rejects_off_grid_points():
    xyz = _grid(3)
    xyz[0] += 0.3
    with pytest.raises(ValueError):
        tubes.group_cells(np.ones(len(xyz), bool), xyz)


# ----------------------------------------------------------------------------- descriptors


def test_descriptors_straight_rod_and_single_cell():
    xyz = _grid(40)
    rod = (np.abs(xyz[:, 1] - 20) <= 1) & (np.abs(xyz[:, 2] - 20) <= 1) & (xyz[:, 0] >= 5) & (xyz[:, 0] < 35)
    single = np.all(xyz == [2, 2, 2], axis=1)
    ids = tubes.group_cells(rod | single, xyz)
    B = np.zeros_like(xyz)
    B[:, 0] = 1.0
    d = tubes.tube_descriptors(ids, xyz, B)
    k = ids[rod][0]
    assert d["n_cells"][k] == 30 * 9
    np.testing.assert_allclose(d["extents"][k], [30, 3, 3], rtol=1e-9)
    np.testing.assert_allclose(d["span"][k], [30, 3, 3], rtol=1e-9)
    np.testing.assert_allclose(np.abs(d["axes"][k, 0]), [1, 0, 0], atol=1e-12)
    assert d["elongation"][k] == pytest.approx(10.0)
    assert d["alignment"][k] == pytest.approx(1.0)
    assert d["alignment_local"][k] == pytest.approx(1.0)
    assert d["tube_score"][k] == pytest.approx(10.0)
    assert d["B_coherence"][k] == pytest.approx(1.0)
    s = ids[single][0]
    np.testing.assert_allclose(d["extents"][s], [1, 1, 1])
    assert d["elongation"][s] == pytest.approx(1.0)
    # field across the rod
    B2 = np.zeros_like(xyz)
    B2[:, 1] = -2.0
    d2 = tubes.tube_descriptors(ids, xyz, B2)
    assert d2["alignment"][k] == pytest.approx(0.0, abs=1e-12)
    assert d2["mean_B"][k] == pytest.approx(2.0)


def test_descriptors_sheet_has_field_in_plane():
    xyz = _grid(32)
    sheet = (np.abs(xyz[:, 2] - 16) <= 0) & (xyz[:, 0] >= 2) & (xyz[:, 0] < 30) & (xyz[:, 1] >= 2) & (xyz[:, 1] < 30)
    ids = tubes.group_cells(sheet, xyz)
    rng = np.random.default_rng(1)
    phi = rng.uniform(0, 2 * np.pi, len(xyz))
    B = np.stack([np.cos(phi), np.sin(phi), np.zeros(len(xyz))], 1)  # random in-plane
    d = tubes.tube_descriptors(ids, xyz, B)
    assert d["elongation"][0] < 1.2
    assert d["flatness"][0] > 20
    assert d["align_axes"][0, 2] == pytest.approx(0.0, abs=1e-12)
    assert 0.55 < d["alignment"][0] < 0.72  # ~ 2/pi


def test_descriptors_volume_weighting_by_level():
    xyz, level, rod = _two_level_rod()
    dx = {5: 2.0, 6: 1.0}
    ids = tubes.group_cells(rod, xyz, level, dx)
    d = tubes.tube_descriptors(ids, xyz, None, level=level, dx_per_level=dx)
    fine = rod & (level == 6)
    coarse = rod & (level == 5)
    assert d["volume"][0] == pytest.approx(fine.sum() * 1.0 + coarse.sum() * 8.0)
    assert d["cells_per_level"][6][0] == fine.sum()
    # the rod spans x in [0, 62] with cubes of side 2 at the ends: length ~ 64
    assert d["span"][0, 0] == pytest.approx(62 + 2, rel=0.05)
    assert np.isnan(d["alignment"][0])


def test_select_tubes_filters_and_sorts():
    d = {
        "n_cells": np.array([100, 10, 500, 400]),
        "elongation": np.array([5.0, 9.0, 2.0, 8.0]),
        "alignment": np.array([0.9, 0.95, 0.9, 0.8]),
    }
    d["tube_score"] = d["elongation"] * d["alignment"]
    out = tubes.select_tubes(d, min_cells=50, min_elongation=3, min_alignment=0.7)
    np.testing.assert_array_equal(out, [3, 0])


# ----------------------------------------------------------------------------- matching


def test_match_identical_and_permuted():
    rng = np.random.default_rng(0)
    ids = rng.integers(-1, 5, size=1000)
    perm = np.array([3, 0, 4, 1, 2])
    other = np.where(ids >= 0, perm[np.maximum(ids, 0)], -1)
    m = tubes.match_to_reference(ids, other)
    for a in range(5):
        assert m.best_b_for_a[a] == perm[a]
        assert m.best_iou_a[a] == pytest.approx(1.0)
    assert len(m.pairs) == 5
    np.testing.assert_allclose(m.recall_b, 1.0)


def test_match_partial_overlap_and_forms():
    n = 12
    xyz = _grid(n)
    a = np.where(xyz[:, 0] < 6, 0, -1)  # half the box
    # reference: x in [3, 9) as a boolean grid mask with origin and spacing
    mask = np.zeros((n, n, n), bool)
    mask[3:9] = True
    m = tubes.match_to_reference(a, {"mask": mask, "origin": [0, 0, 0], "spacing": 1.0}, xyz=xyz)
    assert m.iou[0, 0] == pytest.approx(3 / 9)
    # same reference as a list of row-index arrays and as a boolean row mask
    rows = np.flatnonzero((xyz[:, 0] >= 3) & (xyz[:, 0] < 9))
    m2 = tubes.match_to_reference(a, [rows], n_rows=len(xyz))
    m3 = tubes.match_to_reference(a, (xyz[:, 0] >= 3) & (xyz[:, 0] < 9))
    assert m2.iou[0, 0] == pytest.approx(3 / 9)
    assert m3.iou[0, 0] == pytest.approx(3 / 9)


def test_match_volume_weighted():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [4, 0, 0]], float)
    level = np.array([7, 7, 5])
    a = np.array([0, 0, 0])
    b = np.array([0, -1, -1])
    m = tubes.match_to_reference(a, b, level=level, dx_per_level={5: 2.0, 7: 1.0})
    assert m.iou[0, 0] == pytest.approx(1 / 10)
    m_count = tubes.match_to_reference(a, b)
    assert m_count.iou[0, 0] == pytest.approx(1 / 3)


def test_tubes_to_masks_capsule_and_sphere():
    xyz = _grid(20)
    tube = {"centres": [[2, 10, 10], [17, 10, 10]], "radius": 2.0}
    sphere = {"centre": [10, 10, 10], "radius": 1.0}
    rows_t, rows_s = tubes.tubes_to_masks([tube, sphere], xyz)
    d = np.hypot(xyz[rows_t, 1] - 10, xyz[rows_t, 2] - 10)
    assert d.max() <= 2.0
    assert rows_s.size == 7  # centre plus 6 face neighbours
    m = tubes.match_to_reference(np.where(np.isin(np.arange(len(xyz)), rows_t), 0, -1), [tube], xyz=xyz)
    assert m.iou[0, 0] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- synthetic tubes


def test_synthetic_tubes_recovered():
    syn = tubes.make_synthetic_tubes(n=64, n_tubes=3, radius=3.0, noise=0.08, seed=2)
    Bmag = np.linalg.norm(syn["B"], axis=1)
    ids = tubes.group_cells(Bmag > 0.5, syn["xyz"])
    d = tubes.tube_descriptors(ids, syn["xyz"], syn["B"])
    sel = tubes.select_tubes(d, min_cells=30, min_elongation=3.0, min_alignment=0.6)
    assert len(sel) == 3
    keep = np.where(np.isin(ids, sel), ids, -1)
    m = tubes.match_to_reference(keep, syn["truth"])
    assert len(m.pairs) == 3
    assert np.all(m.best_iou_b > 0.7)
    assert np.all(d["alignment_local"][sel] > 0.85)
