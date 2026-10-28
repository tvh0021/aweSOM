import numpy as np
import pytest

from aweSOM.physics_score import VoxelGrid
from aweSOM.postprocess import clean_labels, majority_filter, remove_small_components


def slab_with_specks():
    m = np.zeros((16, 16, 16), dtype=np.int16)
    m[7:9] = 1  # 2-cell-thick sheet, 512 cells
    m[2, 2, 2] = 2  # isolated specks
    m[12, 12, 12] = 1
    m[3, 12, 4] = 1
    return m


def test_default_is_identity():
    m = slab_with_specks()
    assert np.array_equal(clean_labels(m), m)
    xyz = np.argwhere(m >= 0).astype(float)
    lab = m[tuple(xyz.astype(int).T)]
    assert np.array_equal(clean_labels(lab, xyz, 1.0), lab)


def test_min_size_removes_specks_keeps_thin_sheet():
    m = slab_with_specks()
    out = clean_labels(m, min_size=27)
    assert out[2, 2, 2] == 0 and out[12, 12, 12] == 0 and out[3, 12, 4] == 0
    assert (out[7:9] == 1).all()  # the sheet is large though thin: kept
    assert (out == 2).sum() == 0


def test_majority_erodes_thin_sheet():
    m = slab_with_specks()
    m[7] = 0  # one cell thick
    out = majority_filter(m, 3, 1)
    assert out[2, 2, 2] == 0
    assert (out[8] == 1).sum() == 0  # a 1-cell sheet loses the 3^3 vote


def test_majority_keeps_current_on_tie_and_background():
    m = np.array([[0, 1], [-1, -1]], dtype=np.int16)
    out = majority_filter(m, 3, 1)
    assert out[1, 0] == -1 and out[1, 1] == -1
    assert out[0, 0] == 0 and out[0, 1] == 1


def test_remove_small_nearest_fill():
    m = np.zeros((1, 9), dtype=np.int16)
    m[0, :4] = 3
    m[0, 4] = 5  # small, nearer to label 3 at distance 1 and to 0 at distance 1 -> any of them
    m[0, 5:] = 0
    out = remove_small_components(m, 2)
    assert out[0, 4] in (0, 3)
    assert (out[0, :4] == 3).all() and (out[0, 5:] == 0).all()
    # background is never a source
    m2 = np.array([[1, -1, -1, 2, 2, 2]], dtype=np.int16)
    out2 = remove_small_components(m2, 2)
    assert out2[0, 0] == 2 and (out2[0, 1:3] == -1).all()


def test_point_form_with_levels_matches_dense():
    m = slab_with_specks()
    ijk = np.argwhere(m >= 0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ijk))
    # level 0 at spacing 1 and a copy of it shifted far away at spacing 0.5
    xyz = np.concatenate([ijk * 1.0, 100 + ijk * 0.5])[np.concatenate([perm, perm + len(ijk)])]
    lev = np.repeat([0, 1], len(ijk))[np.concatenate([perm, perm + len(ijk)])]
    lab = np.concatenate([m[tuple(ijk.T)]] * 2)[np.concatenate([perm, perm + len(ijk)])]
    out = clean_labels(lab, xyz, {0: 1.0, 1: 0.5}, lev, min_size=27)
    dense = clean_labels(m, min_size=27)
    grid = VoxelGrid.from_points(xyz, {0: 1.0, 1: 0.5}, lev)
    maps = grid.to_maps(out, -1)
    assert np.array_equal(maps[0], dense) and np.array_equal(maps[1], dense)
    assert out.dtype == lab.dtype
    with pytest.raises(ValueError):
        clean_labels(lab, xyz, None, lev, min_size=27)
