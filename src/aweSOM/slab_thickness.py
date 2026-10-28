"""Bounded-memory EDT ridge measurements on disk-backed AMR masks."""
from collections import Counter

import numpy as np
from scipy import ndimage

from .sheet_stats import LevelLattice, _composite_sheet


def ridge_thickness(level, lattices, sheets, owned, block=64, overlap=16):
    """Measure ridges in x slabs, using the finest-owned composite AMR mask.

    Masks may be numpy memmaps. Only a block plus two halos is copied into RAM.
    A result is certified when ``n_beyond_overlap`` is zero: every core and
    neighbouring distance needed by the ridge filter is then inside the halo.
    Thickness follows sheet_stats: ``2*d - dx/2``, with float32 EDT distances.
    The histogram retains exact values and counts without retaining ridge cells.
    """
    if block < 1 or overlap < 2:
        raise ValueError('block must be positive and overlap at least two')
    lat = lattices[level]
    nx, ny, nz = lat.shape
    hist = Counter()
    deep = 0
    for start in range(0, nx, block):
        stop = min(start + block, nx)
        sublat = LevelLattice(level, lat.dx,
            (lat.origin[0] + (start-overlap)*lat.dx,
             lat.origin[1]-overlap*lat.dx, lat.origin[2]-overlap*lat.dx),
            (stop-start+2*overlap, ny+2*overlap, nz+2*overlap))
        sub = _composite_sheet(sublat, 0, lattices, sheets, owned)
        dist = ndimage.distance_transform_edt(sub, sampling=lat.dx).astype(np.float32)
        core = (slice(overlap, overlap+stop-start), slice(overlap, overlap+ny), slice(overlap, overlap+nz))
        # Certify all distances used by the maximum filter, not just ridges.
        near = dist[overlap-1:overlap+stop-start+1, overlap-1:overlap+ny+1, overlap-1:overlap+nz+1]
        deep += int(np.count_nonzero(near >= (overlap-1)*lat.dx))
        mx = ndimage.maximum_filter(dist, size=3, mode='constant', cval=0)
        d = dist[core]
        ridge = sheets[level][start:stop] & (d >= mx[core]) & (d > 0)
        values, counts = np.unique(2*d[ridge].astype(np.float64)-0.5*lat.dx, return_counts=True)
        hist.update(dict(zip(map(float, values), map(int, counts))))
    total = sum(hist.values())
    def quantile(q):
        target = q*(total-1)
        ranks = [int(np.floor(target)), int(np.ceil(target))]
        vals = []
        for rank in ranks:
            count = 0
            for value, n in sorted(hist.items()):
                count += n
                if count > rank:
                    vals.append(value)
                    break
        return vals[0]+(target-ranks[0])*(vals[1]-vals[0]) if total else float('nan')
    return {'median_M':quantile(.5), 'median_cells':quantile(.5)/lat.dx,
            'p90_cells':quantile(.9)/lat.dx, 'n_ridge':total,
            'n_beyond_overlap':deep, 'histogram_M':dict(sorted(hist.items()))}
