"""Synthetic checks of HHR plane geometry, criterion, and overlap conventions."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/grmhd-bbh'))
from tube_reference import criterion_mask, area_filter, plane_points
from tube_match_hhr import overlap_metrics, sample_owned


def test_criterion_sign_strict_cuts_and_invalid_values():
    bz = np.array([10., -10., 0., 10., 10., np.nan, 10.])
    p = np.array([1., 1., 1., 0., -1., 1., np.inf])
    signed, _, _ = criterion_mask(0., 0., bz, p)
    absolute, _, _ = criterion_mask(0., 0., bz, p, absolute=True)
    assert signed.tolist() == [True, False, False, False, False, False, False]
    assert absolute.tolist() == [True, True, False, False, False, False, False]
    assert not criterion_mask(0., 0., 1., 1/(8*np.pi))[0]
    assert not criterion_mask(np.sqrt(1-.7**2), 0., .7, .001)[0]


def test_area_filter_strict_and_diagonal_connectivity():
    mask = np.eye(3, dtype=bool)
    assert not area_filter(mask, .5, .5).any()
    assert np.count_nonzero(area_filter(mask, .5, .5, 8)) == 3
    with pytest.raises(ValueError):
        area_filter(mask, -1, 1)


def test_overlap_partial_missed_and_false_positive():
    p = np.array([4, 4, -1, 9, -1])
    r = np.array([1, 0, 1, 0, 2])
    m = overlap_metrics(p, r, .25)
    assert m['reference_coverage'] == pytest.approx(1/3)
    assert m['false_positive_area_M2'] == .5
    assert m['regions'][0]['best_jaccard'] == pytest.approx(1/3)
    assert m['regions'][0]['covered_fraction'] == .5
    assert m['regions'][1]['best_component_id'] is None
    assert overlap_metrics(np.full(3, -1), np.zeros(3), 1)['reference_coverage'] is None


def test_plane_layout_matches_etkplot_frame_points():
    raw = dict(orbital_a=np.array([1., 2.]), orbital_b=np.array([3., 4., 5.]), orbital_origin=np.zeros(3), orbital_e1=np.array([0, 1, 0]), orbital_e2=np.array([-1, 0, 0]))
    p = plane_points(raw)
    assert p.shape == (2, 3, 3)
    np.testing.assert_array_equal(p[1, 0], [-3, 2, 0])


def test_sampling_background_and_finest_owned_cell():
    xyz = np.array([[0., 0., 0.], [.25, 0, 0], [1, 0, 0]])
    ids, covered = sample_owned(np.array([[.25, 0, 0], [1, 0, 0], [5, 0, 0]]), xyz, np.array([5, 6, 5]), np.array([7, -1, 8]), {5:1., 6:.5})
    assert ids.tolist() == [-1, 8, -1]
    assert covered.tolist() == [True, True, False]
