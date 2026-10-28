"""Cross-level adjacency rule (aweSOM.amr_adjacency) on the cases of the
2026-10-28 review of tubes.py and sheet_stats.py."""

import numpy as np

from aweSOM import tubes
from aweSOM.amr_adjacency import OwnedLevel, cross_level_pairs, touching
from aweSOM.sheet_stats import ArraySource, SheetConfig, sheet_statistics


def _level(points, dx):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    origin = pts.min(axis=0)
    idx = np.rint((pts - origin) / dx).astype(int)
    owned = np.zeros(tuple(idx.max(axis=0) + 1), dtype=bool)
    owned[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return OwnedLevel(dx, origin, owned)


def _vertex_centred_interface():
    """Fine box [0, 1]^3 at spacing 0.5 next to coarse points (2, y, z), spacing 1."""
    fine = np.indices((3, 3, 3)).reshape(3, -1).T * 0.5
    coarse = np.array([[2.0, y, z] for y in (0, 1, 2) for z in (0, 1, 2)])
    return fine, _level(fine, 0.5), coarse, _level(coarse, 1.0)


def test_fine_cell_between_keeps_levels_apart():
    # fine 0, 0.5, 1, 1.5 (dx 0.5) and coarse 2 (dx 1): 1 and 2 only touch through 1.5
    fine = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [1.5, 0, 0]], dtype=float)
    a, b = _level(fine, 0.5), _level([[2, 0, 0]], 1.0)
    assert not touching([[1, 0, 0]], [[2, 0, 0]], a, b, 26)[0]
    assert touching([[1.5, 0, 0]], [[2, 0, 0]], a, b, 6)[0]
    xyz = np.vstack([fine, [[2, 0, 0]]]).astype(np.float32)
    level = np.array([6, 6, 6, 6, 5])
    ids = tubes.group_cells(np.array([0, 0, 1, 0, 1], bool), xyz, level, {5: 1.0, 6: 0.5}, connectivity=6)
    assert len(set(ids[ids >= 0])) == 2
    ids = tubes.group_cells(np.array([0, 0, 1, 1, 1], bool), xyz, level, {5: 1.0, 6: 0.5}, connectivity=6)
    assert len(set(ids[ids >= 0])) == 1


def test_vertex_centred_gap_is_bridged_at_the_midpoint():
    fine, a, coarse, b = _vertex_centred_interface()
    # straight across the gap: face contact
    assert touching([[1, 0, 0]], [[2, 0, 0]], a, b, 6)[0]
    # from a fine point between two coarse rows: still a face (overlap in y)
    assert touching([[1, 0.5, 0.5]], [[2, 0, 0]], a, b, 6)[0]
    assert touching([[1, 0.5, 0.5]], [[2, 1, 1]], a, b, 6)[0]
    # a coarse diagonal from a coincident fine point: the fine point (1, 0.5, 0) lies between
    assert not touching([[1, 0, 0]], [[2, 1, 0]], a, b, 26)[0]
    # one row inside the fine box: nothing reaches the coarse level
    assert not touching([[0.5, 0, 0]], [[2, 0, 0]], a, b, 26)[0]
    pairs = cross_level_pairs(fine, a, coarse, b, 6)
    assert len(pairs) > 0
    assert np.all(fine[pairs[:, 0], 0] == 1.0)


def test_cell_centred_interface_touches_at_the_face():
    # fine centres at 0.25, 0.75 (dx 0.5) inside [0, 1]; coarse centres at 1.5, 2.5 (dx 1)
    fine = np.array([[x, y, z] for x in (0.25, 0.75) for y in (0.25, 0.75) for z in (0.25, 0.75)])
    coarse = np.array([[x, y, z] for x in (1.5, 2.5) for y in (-0.5, 0.5, 1.5) for z in (-0.5, 0.5, 1.5)])
    a, b = _level(fine, 0.5), _level(coarse, 1.0)
    assert touching([[0.75, 0.25, 0.25]], [[1.5, 0.5, 0.5]], a, b, 6)[0]
    assert touching([[0.75, 0.25, 0.25]], [[1.5, -0.5, 0.5]], a, b, 18)[0]  # edge: y faces at 0
    assert not touching([[0.75, 0.25, 0.25]], [[1.5, -0.5, 0.5]], a, b, 6)[0]
    assert not touching([[0.25, 0.25, 0.25]], [[1.5, 0.5, 0.5]], a, b, 26)[0]


def test_ratio_four_cells_separated_tangentially():
    a = _level([[1.5, 0.25, 0]], 0.25)
    b = _level([[2, 1, 0]], 1.0)
    assert not touching([[1.5, 0.25, 0]], [[2, 1, 0]], a, b, 26)[0]
    x = np.array([[1.5, 0.25, 0], [2, 1, 0]], np.float32)
    r = sheet_statistics(ArraySource(x, np.array([7, 5]), np.ones(2, int), spacing={5: 1, 7: 0.25}), [1],
                         config=SheetConfig(plane_z=None, min_cells=1), log=None)
    assert r.summary["components"]["n_components"] == 2


def test_sheet_interface_respects_connectivity():
    # fine 3x3x3 at 0.5 with its (1, 1, 1) corner coarse-owned; the centre and the corner are diagonal neighbours
    fine = np.indices((3, 3, 3)).reshape(3, -1).T * 0.5
    fine = fine[~np.all(fine == 1, axis=1)]
    x = np.vstack([fine, [[1, 1, 1]]])
    level = np.r_[np.full(len(fine), 6), 5]
    labels = (np.all(x == 0.5, axis=1) | np.all(x == 1, axis=1)).astype(int)
    for conn, expect in ((6, 2), (18, 2), (26, 1)):
        r = sheet_statistics(ArraySource(x, level, labels, spacing={5: 1, 6: 0.5}), [1],
                             config=SheetConfig(plane_z=None, min_cells=1, connectivity=conn), log=None)
        assert r.summary["components"]["n_components"] == expect, conn
    ids = tubes.group_cells(labels == 1, x.astype(np.float32), level, {5: 1.0, 6: 0.5}, connectivity=6)
    assert len(set(ids[ids >= 0])) == 2
    ids = tubes.group_cells(labels == 1, x.astype(np.float32), level, {5: 1.0, 6: 0.5}, connectivity=26)
    assert len(set(ids[ids >= 0])) == 1


def test_tube_span_projects_cube_corners():
    x = np.array([[0, 0, 0], [1, 1, 0]], float)
    d = tubes.tube_descriptors(np.zeros(2, int), x)
    e1 = d["axes"][0, 0]
    np.testing.assert_allclose(d["span"][0, 0], np.ptp(x @ e1) + np.abs(e1).sum())  # 2 sqrt 2, not 1 + sqrt 2
    single = tubes.tube_descriptors(np.zeros(1, int), np.zeros((1, 3)), cell_volume=8.0)
    np.testing.assert_allclose(single["span"][0], 2.0)  # one cube of side 2 along any axis
