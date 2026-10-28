import numpy as np
import pytest

from aweSOM.physics_score import (
    VoxelGrid,
    best_match_jaccard,
    component_stats,
    feature_summary,
    neighbour_agreement,
    physicality,
    reference_overlap,
    ridge_mask,
    score_clusters,
    volume_fractions,
)


def two_level_points():
    """Level 0: 4^3 cells of spacing 1; level 1: 4^3 cells of spacing 0.5, shuffled."""
    rng = np.random.default_rng(0)
    g0 = np.stack(np.meshgrid(*[np.arange(4.0)] * 3, indexing="ij"), -1).reshape(-1, 3)
    g1 = np.stack(np.meshgrid(*[np.arange(4) * 0.5 + 10] * 3, indexing="ij"), -1).reshape(-1, 3)
    xyz = np.concatenate([g0, g1])
    lev = np.repeat([0, 1], 64)
    perm = rng.permutation(128)
    return xyz[perm], lev[perm], perm


def test_voxelgrid_roundtrip_and_volume():
    xyz, lev, _ = two_level_points()
    grid = VoxelGrid.from_points(xyz, {0: 1.0, 1: 0.5}, lev)
    assert grid.shape == {0: (4, 4, 4), 1: (4, 4, 4)}
    vals = np.arange(128)
    maps = grid.to_maps(vals, -1)
    assert np.array_equal(grid.from_maps(maps), vals)
    vol = grid.cell_volume()
    assert np.allclose(vol[lev == 0], 1.0) and np.allclose(vol[lev == 1], 0.125)
    # a voxel's map value is the value of the point at that position
    i = int(np.flatnonzero((lev == 0) & np.all(xyz == [1, 2, 3], axis=1))[0])
    assert maps[0][1, 2, 3] == vals[i]


def test_voxelgrid_rejects_off_lattice_and_duplicates():
    with pytest.raises(ValueError):
        VoxelGrid.from_points(np.array([[0.0, 0, 0], [0.3, 0, 0]]), 1.0)
    with pytest.raises(ValueError):
        VoxelGrid.from_points(np.array([[0.0, 0, 0], [0.0, 0, 0]]), 1.0)


def test_volume_fractions_weighted():
    lab = np.array([0, 0, 1, -1])
    assert np.allclose(volume_fractions(lab), [2 / 3, 1 / 3])
    assert np.allclose(volume_fractions(lab, np.array([1.0, 1.0, 2.0, 5.0])), [0.5, 0.5])


def test_feature_summary_quantiles():
    lab = np.repeat([0, 1], 101)
    x = np.concatenate([np.linspace(0, 1, 101), np.linspace(10, 20, 101)])
    s = feature_summary(lab, {"x": x})["x"]
    assert np.allclose(s[:, 1], [0.5, 15.0])
    assert np.allclose(s[:, 2] - s[:, 0], [0.5, 5.0], rtol=0.02)


def test_neighbour_agreement_blocks_vs_noise():
    m = np.zeros((20, 20, 20), dtype=np.int16)
    m[10:] = 1
    r = neighbour_agreement({0: m}, 2)
    assert np.all(r["kappa"] > 0.9)
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 2, m.shape).astype(np.int16)
    r = neighbour_agreement({0: noise}, 2)
    assert np.all(np.abs(r["kappa"]) < 0.05)
    # background voxels are ignored
    m2 = m.copy()
    m2[:, :, :5] = -1
    assert np.all(neighbour_agreement([m2], 2)["kappa"] > 0.9)


def test_neighbour_agreement_exact_small_case():
    m = np.array([[0, 0, 1]], dtype=np.int16)
    r = neighbour_agreement([m], 2)
    # pairs: (0,0) same, (0,1) different -> label 0 has 3 ends, 2 agreeing; label 1 has 1 end, 0 agreeing
    assert np.allclose(r["agree"], [2 / 3, 0.0])
    assert np.allclose(r["p"], [0.75, 0.25])


def test_component_stats():
    m = np.full((10, 10, 10), 0, dtype=np.int16)
    m[0, 0, 0] = 1  # speck
    m[5:8, 5:8, 5:8] = 1  # 27-cell blob
    r = component_stats({0: m}, 2, small=27)
    assert r["n_components"][1] == 2 and r["n_small"][1] == 1
    assert np.isclose(r["small_frac"][1], 1 / 28)
    assert np.isclose(r["largest_frac"][1], 27 / 28)
    assert np.isclose(r["mean_cells"][1], (27**2 + 1) / 28)
    assert r["n_components"][0] == 1


def test_reference_overlap():
    lab = np.array([0, 0, 1, 1, 2, 2])
    ref = np.array([0, 0, 1, 1, 1, 0], bool)
    r = reference_overlap(lab, ref)
    assert np.allclose(r["precision"], [0, 1, 0.5])
    assert np.allclose(r["recall"], [0, 2 / 3, 1 / 3])
    assert r["best"] == 1 and np.isclose(r["best_jaccard"], 2 / 3)
    # union {1, 2}: 3 of 4 -> 0.75 > 2/3
    assert r["union"] == [1, 2] and np.isclose(r["union_jaccard"], 0.75)
    assert np.isclose(r["enrichment"][1], 1 / 0.5)


def test_ridge_mask_marks_sheet_midplane():
    x = np.arange(21)
    f = np.exp(-((x - 10) ** 2) / 4.0)[:, None, None] * np.ones((21, 9, 9))
    r = ridge_mask(f, threshold=0.5)
    assert r[10].all() and r.sum() == 81
    f2 = f.copy()
    f2[:, 0, :] = np.nan
    r2 = ridge_mask(f2, threshold=0.5)
    assert not r2[:, 0].any() and r2[10, 1:].all()


def test_best_match_jaccard_permutation_invariant():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 4, 1000)
    perm = np.array([2, 0, 3, 1])
    assert np.allclose(best_match_jaccard(a, perm[a]), 1.0)
    b = a.copy()
    b[a == 3] = 0  # merge 3 into 0
    j = best_match_jaccard(a, b)
    assert np.isclose(j[1], 1.0) and j[0] < 1 and j[3] < 1


def test_physicality_and_score_clusters():
    assert np.isclose(physicality([1, 1], 1.0, [1, 1]), 1.0)
    assert physicality([1, -0.5], 1.0, [1, 1]) == pytest.approx(np.cbrt(0.5))
    assert physicality([1, 1], 0.0, [1, 1]) == 0.0
    # several reference Jaccards are averaged
    assert physicality([1, 1], [0.2, 0.6], [1, 1]) == pytest.approx(np.cbrt(0.4))
    # a slab "sheet" cluster inside a bulk, with the reference equal to the slab
    xyz = np.stack(np.meshgrid(*[np.arange(12.0)] * 3, indexing="ij"), -1).reshape(-1, 3)
    lab = (np.abs(xyz[:, 0] - 6) < 1.5).astype(np.int64)
    grid = VoxelGrid.from_points(xyz, 1.0)
    out = score_clusters(
        lab, grid, features={"x": xyz[:, 0]}, reference=lab == 1, others=[lab], ridge=xyz[:, 0] == 6
    )
    assert out["n_clusters"] == 2
    assert np.isclose(out["reference"]["best_jaccard"], 1.0) and out["reference"]["best"] == 1
    assert np.isclose(out["features"]["x"]["median"][1], 6.0)
    assert np.allclose(out["stability"], 1.0)
    assert out["n_components"][1] == 1
    assert np.isclose(out["ridge_frac"][1], 1 / 3)
    assert 0.5 < out["physicality"] <= 1.0
