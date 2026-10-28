import numpy as np
from scipy import ndimage
from aweSOM.sheet_stats import LevelLattice
from aweSOM.slab_thickness import ridge_thickness


def test_slab_ridges_match_full_edt():
    for width in (1, 3, 6):
        mask = np.zeros((31, 24, 27), bool)
        mask[3:28, 4:20, 12:12+width] = True
        lat = LevelLattice(8, 1/32, (0., 0., 0.), mask.shape)
        got = ridge_thickness(8, {8:lat}, {8:mask}, {8:np.ones_like(mask)}, block=4, overlap=8)
        dist = ndimage.distance_transform_edt(mask, sampling=lat.dx).astype(np.float32)
        ridge = mask & (dist >= ndimage.maximum_filter(dist, size=3))
        expected = 2*dist[ridge].astype(float)-lat.dx/2
        assert got['n_ridge'] == len(expected)
        assert got['median_M'] == np.median(expected)
        assert got['n_beyond_overlap'] == 0
        if width == 1:
            assert got['median_cells'] == 1.5


def test_composite_slabs_match_full_amr_distance():
    from aweSOM.sheet_stats import _composite_sheet
    coarse = LevelLattice(7, .0625, (0., 0., 0.), (18, 18, 18))
    fine = LevelLattice(8, .03125, (.25, .25, .25), (17, 17, 17))
    lats = {7:coarse, 8:fine}
    owned = {L:np.ones(lat.shape, bool) for L,lat in lats.items()}
    owned[7][4:13,4:13,4:13] = False
    sheets = {}
    for L,lat in lats.items():
        z = lat.origin[2] + np.arange(lat.shape[2])*lat.dx
        sheets[L] = np.broadcast_to((z>=.375)&(z<=.5),lat.shape).copy() & owned[L]
    for L,lat in lats.items():
        composite = _composite_sheet(lat, 8, lats, sheets, owned)
        dist = ndimage.distance_transform_edt(composite, sampling=lat.dx).astype(np.float32)
        mx = ndimage.maximum_filter(dist, size=3)
        core = tuple(slice(8,8+n) for n in lat.shape)
        ridge = sheets[L] & (dist[core]>=mx[core])
        expected = 2*dist[core][ridge].astype(float)-lat.dx/2
        got = ridge_thickness(L,lats,sheets,owned,block=3,overlap=8)
        assert got['n_ridge'] == len(expected)
        assert got['median_M'] == np.median(expected)
        assert got['n_beyond_overlap'] == 0
