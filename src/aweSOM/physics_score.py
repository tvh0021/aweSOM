"""Physics-facing scores for cluster labels on regular or AMR grids.

There is no ground truth for which points belong to a current sheet, so no
single number here measures "correct" clusters. The scores test properties a
physical region should have, each on its own:

- volume fraction and the distribution (median, IQR) of chosen features;
- spatial coherence: a physical region is a connected structure, not salt and
  pepper. :func:`neighbour_agreement` gives, per cluster, the share of grid
  neighbours with the same label and its excess over chance (``kappa``);
  :func:`component_stats` the connected-component size distribution;
- sheet-likeness: overlap with a reference mask, e.g. a threshold-based
  current-sheet metric (:func:`reference_overlap`), and the share of cells on
  a ridge of a scalar such as chi_B (:func:`ridge_mask`);
- stability: best-match Jaccard against other labelings of the same points,
  e.g. neighbouring merge thresholds or other training seeds
  (:func:`best_match_jaccard`).

:func:`score_clusters` runs them all; :func:`physicality` folds three of them
into one scalar (see its docstring for the definition and its limits).

Points live on one or more uniform grids ("groups", e.g. AMR refinement
levels); :class:`VoxelGrid` maps them to one dense box per group so that
neighbours and connected components are those of the grid. Neighbours are
only ever taken within one group: a structure crossing a level boundary
counts as one component per level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

__all__ = [
    "VoxelGrid",
    "volume_fractions",
    "feature_summary",
    "neighbour_agreement",
    "component_stats",
    "reference_overlap",
    "ridge_mask",
    "best_match_jaccard",
    "physicality",
    "score_clusters",
]


# --------------------------------------------------------------------- grid


@dataclass
class VoxelGrid:
    """Points of one or more uniform grids, each embedded in a dense box.

    Attributes:
        spacing (dict): group -> grid spacing.
        origin (dict): group -> lower corner of the box (coordinates of voxel 0).
        shape (dict): group -> box shape.
        rows (dict): group -> indices (or a slice) of the group's points in the
            point arrays.
        flat (dict): group -> C-order flat voxel index of each of those points.
        n_points (int): total number of points.
        ndim (int): spatial dimension.
    """

    spacing: dict
    origin: dict
    shape: dict
    rows: dict
    flat: dict
    n_points: int
    ndim: int
    _volume: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def from_points(cls, xyz, spacing, groups=None, tol: float = 1e-3) -> "VoxelGrid":
        """Build the grid from point coordinates.

        Args:
            xyz (np.ndarray): [N, d] coordinates (d = 1, 2 or 3).
            spacing (float | dict): grid spacing, or group -> spacing.
            groups (np.ndarray, optional): [N] group id of each point (e.g. the
                refinement level). Defaults to one group, 0.
            tol (float, optional): largest allowed distance, in cells, of a point
                from its voxel centre. Defaults to 1e-3.

        Returns:
            VoxelGrid: the grid.

        Raises:
            ValueError: if a point is off its group's lattice or two points share a voxel.
        """
        xyz = np.asarray(xyz)
        if xyz.ndim != 2:
            raise ValueError("xyz must have shape (N, d)")
        n, d = xyz.shape
        if groups is None:
            groups = np.zeros(n, dtype=np.int8)
        groups = np.asarray(groups)
        if groups.shape != (n,):
            raise ValueError("groups must have one entry per point")
        ids = [g.item() for g in np.unique(groups)]
        if not isinstance(spacing, dict):
            spacing = {g: float(spacing) for g in ids}
        out = dict(spacing={}, origin={}, shape={}, rows={}, flat={})
        for g in ids:
            if g not in spacing:
                raise ValueError(f"no spacing for group {g}")
            idx = np.flatnonzero(groups == g)
            rows = slice(int(idx[0]), int(idx[-1]) + 1) if idx[-1] - idx[0] + 1 == idx.size else idx
            if isinstance(rows, slice):
                del idx
            dx = float(spacing[g])
            p = xyz[rows]
            lo = p.min(axis=0).astype(np.float64)
            hi = p.max(axis=0).astype(np.float64)
            shape = tuple(int(s) for s in np.rint((hi - lo) / dx) + 1)
            flat = np.empty(len(p), dtype=np.int64)
            chunk = 1 << 22  # bounded temporaries for large grids
            for a in range(0, len(p), chunk):
                f = (np.asarray(p[a : a + chunk], dtype=np.float64) - lo) / dx
                ijk = np.rint(f)
                if np.abs(f - ijk).max() > tol:
                    raise ValueError(f"group {g}: points are not on a lattice of spacing {dx}")
                ijk = ijk.astype(np.int64)
                flat[a : a + chunk] = np.ravel_multi_index(tuple(ijk.T), shape)
                del f, ijk
            del p
            seen = np.zeros(int(np.prod(shape)), dtype=bool)
            seen[flat] = True
            if np.count_nonzero(seen) != flat.size:
                raise ValueError(f"group {g}: two points share a voxel")
            del seen
            out["spacing"][g] = dx
            out["origin"][g] = lo
            out["shape"][g] = shape
            out["rows"][g] = rows
            out["flat"][g] = flat
        return cls(n_points=n, ndim=d, **out)

    @property
    def groups(self) -> list:
        """list: the group ids."""
        return list(self.shape)

    def cell_volume(self) -> np.ndarray:
        """np.ndarray: [N] volume (dx**d) of each point's cell."""
        if self._volume is None:
            v = np.empty(self.n_points)
            for g in self.groups:
                v[self.rows[g]] = self.spacing[g] ** self.ndim
            self._volume = v
        return self._volume

    def to_maps(self, values, fill) -> dict:
        """Scatter per-point ``values`` into one dense box per group.

        Args:
            values (np.ndarray): [N] values.
            fill: value of the voxels no point occupies.

        Returns:
            dict: group -> array of the group's box shape.
        """
        values = np.asarray(values)
        maps = {}
        for g in self.groups:
            m = np.full(self.shape[g], fill, dtype=values.dtype)
            m.ravel()[self.flat[g]] = values[self.rows[g]]
            maps[g] = m
        return maps

    def from_maps(self, maps: dict, dtype=None) -> np.ndarray:
        """Gather per-point values back from the boxes (inverse of :meth:`to_maps`)."""
        first = maps[self.groups[0]]
        out = np.empty(self.n_points, dtype=dtype or first.dtype)
        for g in self.groups:
            out[self.rows[g]] = maps[g].ravel()[self.flat[g]]
        return out


def _n_clusters(labels: np.ndarray, n_clusters: int | None) -> int:
    if n_clusters is not None:
        return int(n_clusters)
    return int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0


# ----------------------------------------------------------- per-point stats


def volume_fractions(labels, volume=None, n_clusters: int | None = None) -> np.ndarray:
    """Share of the total volume in each cluster (negative labels are ignored).

    Args:
        labels (np.ndarray): [N] int labels.
        volume (np.ndarray, optional): [N] cell volumes. Defaults to 1 per point.
        n_clusters (int, optional): number of clusters. Defaults to max label + 1.

    Returns:
        np.ndarray: [K] volume fractions.
    """
    labels = np.asarray(labels)
    k = _n_clusters(labels, n_clusters)
    ok = labels >= 0
    w = None if volume is None else np.asarray(volume, dtype=np.float64)[ok]
    v = np.bincount(labels[ok], weights=w, minlength=k)[:k].astype(np.float64)
    tot = v.sum()
    return v / tot if tot > 0 else v


def _weighted_quantiles(x: np.ndarray, w: np.ndarray, qs) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws)
    if cw[-1] <= 0:
        return np.full(len(qs), np.nan)
    # midpoint rule: the value at cumulative weight q * total
    pos = (cw - 0.5 * ws) / cw[-1]
    return np.interp(qs, pos, xs)


def feature_summary(labels, features: dict, volume=None, n_clusters=None, qs=(0.25, 0.5, 0.75)) -> dict:
    """Volume-weighted quantiles of each feature in each cluster.

    Args:
        labels (np.ndarray): [N] int labels.
        features (dict): name -> [N] values.
        volume (np.ndarray, optional): [N] weights. Defaults to 1 per point.
        n_clusters (int, optional): number of clusters.
        qs (tuple, optional): quantiles. Defaults to (0.25, 0.5, 0.75).

    Returns:
        dict: name -> [K, len(qs)] array (NaN for an empty cluster).
    """
    labels = np.asarray(labels)
    k = _n_clusters(labels, n_clusters)
    w_all = np.ones(labels.shape) if volume is None else np.asarray(volume, dtype=np.float64)
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(k + 1))
    out = {}
    for name, x in features.items():
        x = np.asarray(x)
        res = np.full((k, len(qs)), np.nan)
        for c in range(k):
            sel = order[bounds[c]:bounds[c + 1]]
            sel = sel[np.isfinite(x[sel])]
            if sel.size:
                res[c] = _weighted_quantiles(x[sel].astype(np.float64), w_all[sel], qs)
        out[name] = res
    return out


# ------------------------------------------------------- spatial coherence


def neighbour_agreement(maps, n_clusters: int, background: int = -1) -> dict:
    """Share of each cluster's face-neighbour pairs that have the same label.

    For every pair of face-adjacent occupied voxels (both labels >= 0 and not
    ``background``), each end counts one pair for its cluster, and a pair whose
    two labels agree counts as agreeing for both ends. ``agree[k]`` is the
    agreeing share of cluster k's pair ends. ``kappa[k] = (agree - p) / (1 - p)``
    is the excess over chance, with ``p[k]`` cluster k's share of all pair ends
    (what a spatially random labeling with the same proportions gives): 1 for
    clusters made of large compact regions, 0 for salt and pepper. A slab
    ``t`` cells thick has ``agree`` of about ``1 - 1/(3 t)``, so thin sheets
    score lower than blobs even when perfectly coherent.

    Args:
        maps (dict | list): label boxes (int arrays), e.g. ``VoxelGrid.to_maps``.
        n_clusters (int): number of clusters K.
        background (int, optional): label of empty voxels. Defaults to -1.

    Returns:
        dict: ``agree``, ``kappa``, ``p`` and ``pairs`` (pair ends), each [K].
    """
    k = int(n_clusters)
    ends = np.zeros(k)
    same = np.zeros(k)
    for m in maps.values() if isinstance(maps, dict) else maps:
        m = np.asarray(m)
        for ax in range(m.ndim):
            n = m.shape[ax]
            if n < 2:
                continue
            a = m.take(np.arange(n - 1), axis=ax).ravel()
            b = m.take(np.arange(1, n), axis=ax).ravel()
            ok = (a >= 0) & (b >= 0) & (a != background) & (b != background)
            a, b = a[ok], b[ok]
            ends += np.bincount(a, minlength=k)[:k] + np.bincount(b, minlength=k)[:k]
            same += 2 * np.bincount(a[a == b], minlength=k)[:k]
    with np.errstate(invalid="ignore", divide="ignore"):
        agree = same / ends
        p = ends / ends.sum() if ends.sum() > 0 else ends
        kappa = (agree - p) / (1 - p)
    return dict(agree=agree, kappa=kappa, p=p, pairs=ends)


def component_stats(maps, n_clusters: int, cell_volumes=None, small: int = 27, connectivity: int = 1) -> dict:
    """Connected-component size distribution of each cluster.

    Components are found per box (group) with ``scipy.ndimage.label``.

    Args:
        maps (dict): group -> label box (int; negative = empty).
        n_clusters (int): number of clusters K.
        cell_volumes (dict, optional): group -> cell volume, to pool groups by
            volume. Defaults to 1 per cell.
        small (int, optional): components with fewer cells than this count as
            speckle. Defaults to 27 (one 3^3 stencil).
        connectivity (int, optional): 1 = faces, ndim = faces, edges and corners.
            Defaults to 1.

    Returns:
        dict of [K] arrays: ``n_components``; ``n_small``; ``small_frac`` (volume
        share in components of fewer than ``small`` cells); ``largest_frac``
        (volume share of the largest component); ``mean_cells`` (volume-weighted
        mean component size in cells, sum s^2 / sum s, pooled over groups).
    """
    k = int(n_clusters)
    keys = list(maps) if isinstance(maps, dict) else list(range(len(maps)))
    maps = maps if isinstance(maps, dict) else dict(enumerate(maps))
    ncomp = np.zeros(k, dtype=np.int64)
    nsmall = np.zeros(k, dtype=np.int64)
    vol = np.zeros(k)
    vsmall = np.zeros(k)
    vlargest = np.zeros(k)
    s1 = np.zeros(k)
    s2 = np.zeros(k)
    for g in keys:
        m = maps[g]
        cv = 1.0 if cell_volumes is None else float(cell_volumes[g])
        struct = ndimage.generate_binary_structure(m.ndim, connectivity)
        present = np.unique(m)
        for c in present[(present >= 0) & (present < k)]:
            lab, n = ndimage.label(m == c, structure=struct)
            sizes = np.bincount(lab.ravel())[1:].astype(np.float64)
            del lab
            ncomp[c] += n
            nsmall[c] += int(np.count_nonzero(sizes < small))
            vol[c] += sizes.sum() * cv
            vsmall[c] += sizes[sizes < small].sum() * cv
            vlargest[c] = max(vlargest[c], sizes.max() * cv)
            s1[c] += sizes.sum()
            s2[c] += (sizes**2).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(
            n_components=ncomp,
            n_small=nsmall,
            small_frac=vsmall / vol,
            largest_frac=vlargest / vol,
            mean_cells=s2 / s1,
        )


# ------------------------------------------------------- sheet-likeness


def reference_overlap(labels, reference, volume=None, n_clusters=None) -> dict:
    """Overlap of each cluster with a reference mask (e.g. metric current sheets).

    Args:
        labels (np.ndarray): [N] int labels (negative = ignored).
        reference (np.ndarray): [N] bool mask.
        volume (np.ndarray, optional): [N] cell volumes. Defaults to 1.
        n_clusters (int, optional): number of clusters.

    Returns:
        dict: per cluster [K] ``precision`` (share of the cluster in the mask),
        ``recall`` (share of the mask in the cluster), ``jaccard`` and
        ``enrichment`` (precision over the mask's overall share); and scalars
        ``ref_frac``, ``best`` (cluster with the largest Jaccard), ``best_jaccard``,
        ``union`` (clusters added greedily by precision while the union's Jaccard
        grows) and ``union_jaccard``.
    """
    labels = np.asarray(labels)
    reference = np.asarray(reference, dtype=bool)
    k = _n_clusters(labels, n_clusters)
    ok = labels >= 0
    w = np.ones(labels.shape) if volume is None else np.asarray(volume, dtype=np.float64)
    vk = np.bincount(labels[ok], weights=w[ok], minlength=k)[:k]
    ik = np.bincount(labels[ok & reference], weights=w[ok & reference], minlength=k)[:k]
    vref = w[ok & reference].sum()
    vtot = w[ok].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = ik / vk
        rec = ik / vref
        jac = ik / (vk + vref - ik)
        enr = prec / (vref / vtot)
    best = int(np.nanargmax(jac)) if k and np.isfinite(jac).any() else -1
    union, uj, ui, uv = [], 0.0, 0.0, 0.0
    for c in np.argsort(-np.nan_to_num(prec, nan=-1)):
        ni, nv = ui + ik[c], uv + vk[c]
        nj = ni / (nv + vref - ni) if nv + vref - ni > 0 else 0.0
        if nj <= uj:
            break
        union.append(int(c))
        uj, ui, uv = nj, ni, nv
    return dict(
        precision=prec,
        recall=rec,
        jaccard=jac,
        enrichment=enr,
        ref_frac=vref / vtot if vtot > 0 else np.nan,
        best=best,
        best_jaccard=float(jac[best]) if best >= 0 else np.nan,
        union=union,
        union_jaccard=float(uj),
    )


def ridge_mask(field_map: np.ndarray, threshold: float = -np.inf) -> np.ndarray:
    """Cells on a ridge (sheet-like local maximum) of a scalar field.

    A cell is on a ridge if, along the grid axis where the field's second
    difference is most negative, it is at least as large as both neighbours,
    and it is above ``threshold``. On a sheet the field peaks across the sheet
    and not along it, so this marks the sheet's mid-surface. Cells with a
    non-finite value or a non-finite neighbour on that axis are False.

    Args:
        field_map (np.ndarray): dense field (NaN for empty voxels).
        threshold (float, optional): lower bound on the field. Defaults to -inf.

    Returns:
        np.ndarray: bool array of the same shape.
    """
    f = np.asarray(field_map, dtype=np.float32)
    best_d2 = np.full(f.shape, np.inf, dtype=np.float32)
    is_max = np.zeros(f.shape, dtype=bool)
    for ax in range(f.ndim):
        lo = np.full(f.shape, np.nan, dtype=np.float32)
        hi = np.full(f.shape, np.nan, dtype=np.float32)
        sl = [slice(None)] * f.ndim
        sr = [slice(None)] * f.ndim
        sl[ax], sr[ax] = slice(1, None), slice(None, -1)
        lo[tuple(sl)] = f[tuple(sr)]  # value at i - 1
        hi[tuple(sr)] = f[tuple(sl)]  # value at i + 1
        d2 = lo + hi - 2 * f
        d2 = np.where(np.isfinite(d2), d2, np.inf).astype(np.float32)
        here = (f >= lo) & (f >= hi)
        better = d2 < best_d2
        best_d2 = np.where(better, d2, best_d2)
        is_max = np.where(better, here, is_max)
        del lo, hi, d2, here, better
    return is_max & np.isfinite(best_d2) & (f > threshold)


# ------------------------------------------------------------- stability


def best_match_jaccard(a, b, volume=None, n_a=None, n_b=None) -> np.ndarray:
    """For each cluster of ``a``, its largest Jaccard index with a cluster of ``b``.

    Args:
        a (np.ndarray): [N] labels (negative = ignored).
        b (np.ndarray): [N] labels of the same points (negative = ignored).
        volume (np.ndarray, optional): [N] weights. Defaults to 1.
        n_a (int, optional): number of clusters of ``a``.
        n_b (int, optional): number of clusters of ``b``.

    Returns:
        np.ndarray: [K_a] best-match Jaccard (NaN for an empty cluster).
    """
    a, b = np.asarray(a), np.asarray(b)
    ka, kb = _n_clusters(a, n_a), _n_clusters(b, n_b)
    ok = (a >= 0) & (b >= 0)
    w = None if volume is None else np.asarray(volume, dtype=np.float64)[ok]
    joint = np.bincount(a[ok].astype(np.int64) * kb + b[ok], weights=w, minlength=ka * kb)
    joint = joint[: ka * kb].reshape(ka, kb).astype(np.float64)
    va, vb = joint.sum(1), joint.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = joint / (va[:, None] + vb[None, :] - joint)
    out = np.where(va > 0, np.nanmax(np.nan_to_num(jac, nan=0.0), axis=1) if kb else 0.0, np.nan)
    return out


# ----------------------------------------------------------- the scalar


def physicality(kappa, sheet_jaccard, stability) -> float:
    """One-number summary: geometric mean of coherence, sheet match and stability.

    ``P = (C * D * S) ** (1/3)`` with

    - ``C``: mean over clusters of ``max(kappa_k, 0)`` (unweighted, so many small
      fragmented clusters pull it down);
    - ``D``: ``sheet_jaccard``, the Jaccard index of the best single cluster
      with a reference sheet mask (1 = one cluster is exactly the sheets). A
      sequence is averaged: pass the Jaccards against masks at a loose and a
      strict threshold so that D does not reward matching one arbitrary
      threshold's volume;
    - ``S``: mean over clusters of the best-match Jaccard with other labelings
      (seeds or neighbouring merge thresholds).

    Each factor is in [0, 1] and the geometric mean is zero if any is zero, so
    no factor can be traded entirely for another. It is a ranking device for
    configurations of one data set, not an absolute measure; NaN entries of
    ``kappa`` and ``stability`` (empty clusters) are skipped.

    Args:
        kappa (np.ndarray): [K] per-cluster kappa from :func:`neighbour_agreement`.
        sheet_jaccard (float or sequence): best single-cluster Jaccard with the
            reference, or several (one per reference mask) to be averaged.
        stability (np.ndarray): [K] per-cluster best-match Jaccard.

    Returns:
        float: P in [0, 1].
    """
    c = float(np.nanmean(np.clip(np.asarray(kappa, dtype=np.float64), 0, 1)))
    s = float(np.nanmean(np.asarray(stability, dtype=np.float64)))
    d = float(np.mean(np.clip(np.asarray(sheet_jaccard, dtype=np.float64), 0, 1)))
    return float(np.cbrt(c * d * s))


# ------------------------------------------------------------ all at once


def score_clusters(
    labels,
    grid: VoxelGrid,
    features: dict | None = None,
    reference=None,
    others=(),
    ridge=None,
    n_clusters: int | None = None,
    components: bool = True,
    small: int = 27,
    stats_rows=None,
) -> dict:
    """Scoreboard of one labeling: every score of this module, per cluster.

    Args:
        labels (np.ndarray): [N] int labels on the points of ``grid``.
        grid (VoxelGrid): the points' grid (cell volumes, neighbours).
        features (dict, optional): name -> [N] values to summarise (median, IQR).
        reference (np.ndarray, optional): [N] bool reference sheet mask.
        others (sequence, optional): other [N] labelings of the same points, for
            stability.
        ridge (np.ndarray, optional): [N] bool ridge mask (e.g. of log chi_B);
            gives ``ridge_frac`` and ``ridge_enrichment``.
        n_clusters (int, optional): number of clusters.
        components (bool, optional): compute :func:`component_stats` (the slowest
            part on large grids). Defaults to True.
        small (int, optional): speckle size for :func:`component_stats`.
        stats_rows (np.ndarray, optional): indices of the points used for the
            feature quantiles and the stability (a subsample, to save time).
            Defaults to all points.

    Returns:
        dict: ``n_clusters``, ``volume_frac``, ``features`` (name -> {median, iqr}),
        ``agree``, ``kappa``, optionally the component stats, ``reference`` (the
        :func:`reference_overlap` dict), ``ridge_frac``, ``ridge_enrichment``,
        ``stability`` and ``physicality`` (NaN unless ``reference`` and ``others``
        are given).
    """
    labels = np.asarray(labels)
    k = _n_clusters(labels, n_clusters)
    vol = grid.cell_volume()
    out = dict(n_clusters=k, volume_frac=volume_fractions(labels, vol, k))
    rows = slice(None) if stats_rows is None else stats_rows
    if features:
        summ = feature_summary(labels[rows], {n: np.asarray(x)[rows] for n, x in features.items()}, vol[rows], k)
        out["features"] = {
            n: dict(median=q[:, 1], iqr=q[:, 2] - q[:, 0], q25=q[:, 0], q75=q[:, 2]) for n, q in summ.items()
        }
    maps = grid.to_maps(labels.astype(np.int16 if k < 32767 else np.int32), -1)
    out.update(neighbour_agreement(maps, k))
    if components:
        cv = {g: grid.spacing[g] ** grid.ndim for g in grid.groups}
        out.update(component_stats(maps, k, cv, small=small))
    del maps
    if reference is not None:
        out["reference"] = reference_overlap(labels, reference, vol, k)
    if ridge is not None:
        ridge = np.asarray(ridge, dtype=bool)
        r = reference_overlap(labels, ridge, vol, k)
        out["ridge_frac"] = r["precision"]
        out["ridge_enrichment"] = r["enrichment"]
    if len(others):
        stab = np.mean([best_match_jaccard(labels[rows], np.asarray(o)[rows], vol[rows], k) for o in others], axis=0)
        out["stability"] = stab
    if reference is not None and len(others):
        out["physicality"] = physicality(out["kappa"], out["reference"]["best_jaccard"], out["stability"])
    else:
        out["physicality"] = np.nan
    return out
