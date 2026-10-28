"""Group labelled cells into coherent 3D structures and describe them as flux-tube candidates.

aweSOM labels cells one at a time. A flux tube is a coherent, elongated,
strongly magnetized structure whose field runs along its length, so finding
tubes needs a second step that joins labelled cells into connected
components and measures their shape against the field:

1. :func:`group_cells` labels connected components of a cell mask (or of each
   value of a label array) on each refinement level's regular grid with
   ``scipy.ndimage.label``, then joins components of different levels that
   touch across a level boundary.
2. :func:`tube_descriptors` gives each component its size, volume, centroid,
   principal axes and extents (from the volume-weighted second moments), and
   the alignment of the magnetic field with its long axis.
3. :func:`select_tubes` keeps the components that are large, elongated and
   field-aligned enough.
4. :func:`match_to_reference` scores components against an independent set of
   tubes (intersection over union, by volume or by cell count).

Points are rows of a flat table: ``xyz`` (N, 3) coordinates, ``level`` (N,)
refinement level, and a spacing per level. Each level's points are assumed to
lie on a regular grid of that spacing; points may be missing (cells owned by a
finer level, horizon masks), so a level does not have to fill its bounding box.

Tube score
----------
``tube_score = elongation * alignment`` where

* ``elongation = extent_1 / extent_2``, the ratio of the longest to the middle
  principal extent. Extents are ``sqrt(12 * lambda_i)``, with ``lambda_i`` the
  eigenvalues of the volume-weighted position covariance of the component's
  cells, each cell counted as a uniform cube of side ``dx`` (so a single cell
  has extents ``(dx, dx, dx)`` and a straight row of ``n`` cells has
  ``extent_1 = n dx``). For a uniform rod of length L and width w,
  elongation = L / w.
* ``alignment = <|B . e_1| / |B|>``, the volume-weighted mean over the
  component's cells of the absolute cosine between the field and the longest
  principal axis ``e_1``: 1 for a straight tube with the field along it, 1/2
  for an isotropic random field, about 2/pi = 0.64 for a field at random
  angles within a plane that contains ``e_1``, 0 for a field across the axis.

A straight tube scores roughly its aspect ratio; a blob scores about 0.5-1; a
current sheet (two long axes, field in the sheet plane at changing angles)
has elongation near 1 and a low ``align_axes[:, 2]`` (field across the sheet
normal). Curved tubes lose global alignment: ``alignment_local`` repeats the
measurement with the axis of each of ``n_segments`` slices along ``e_1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

__all__ = [
    "level_indices",
    "group_cells",
    "tube_descriptors",
    "select_tubes",
    "grid_mask_to_rows",
    "tubes_to_masks",
    "as_membership",
    "MatchResult",
    "match_to_reference",
    "make_synthetic_tubes",
]

_TOL = 1e-3  # in units of a level's spacing


# --------------------------------------------------------------------------- grids


def _structure(connectivity: int) -> np.ndarray:
    rank = {6: 1, 18: 2, 26: 3}.get(int(connectivity))
    if rank is None:
        raise ValueError("connectivity must be 6, 18 or 26")
    return ndimage.generate_binary_structure(3, rank)


def _dx_lookup(dx_per_level, levels: np.ndarray) -> dict:
    """{level: dx} for every level in ``levels``."""
    if np.isscalar(dx_per_level):
        return {int(lv): float(dx_per_level) for lv in levels}
    if not isinstance(dx_per_level, Mapping):
        raise TypeError("dx_per_level must be a number or a mapping level -> dx")
    out = {}
    for lv in levels:
        if int(lv) not in dx_per_level:
            raise KeyError(f"no spacing given for level {int(lv)}")
        out[int(lv)] = float(dx_per_level[int(lv)])
    return out


def _levels(level, n: int) -> np.ndarray:
    if level is None:
        return np.zeros(n, dtype=np.int16)
    level = np.asarray(level)
    if level.shape != (n,):
        raise ValueError("level must have one entry per row")
    return level


def level_indices(xyz: np.ndarray, dx: float, origin=None) -> tuple[np.ndarray, np.ndarray]:
    """Integer grid indices of points on a regular grid of spacing ``dx``.

    Args:
        xyz (np.ndarray): (n, 3) coordinates.
        dx (float): Grid spacing.
        origin (array-like, optional): Coordinates of index (0, 0, 0). Defaults to
            the per-axis minimum of ``xyz``.

    Returns:
        tuple[np.ndarray, np.ndarray]: int64 (n, 3) indices and the float64 origin.

    Raises:
        ValueError: if a point is more than 1e-3 dx off the grid.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    origin = xyz.min(axis=0) if origin is None else np.asarray(origin, dtype=np.float64)
    f = (xyz - origin) / dx
    idx = np.rint(f)
    if f.size and np.max(np.abs(f - idx)) > _TOL:
        raise ValueError(f"points are not on a grid of spacing {dx} (off by "
                         f"{np.max(np.abs(f - idx)):.3g} dx)")
    return idx.astype(np.int64), origin


def _relabel_by_size(ids: np.ndarray, n: int) -> np.ndarray:
    """Renumber ids 0..n-1 so that 0 is the largest component (ties: lower old id)."""
    fg = ids >= 0
    counts = np.bincount(ids[fg], minlength=n)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty(n, dtype=ids.dtype)
    rank[order] = np.arange(n, dtype=ids.dtype)
    out = ids.copy()
    out[fg] = rank[ids[fg]]
    return out


def _touch_pairs(xa, xb, h, connectivity):
    """Index pairs (i, j) of points xa[i], xb[j] that are neighbours on a grid of
    spacing ``h`` under ``connectivity``."""
    if len(xa) == 0 or len(xb) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if connectivity == 26:
        p, r = np.inf, h * (1 + _TOL)
    elif connectivity == 18:
        p, r = 2.0, np.sqrt(2.0) * h * (1 + _TOL)
    else:
        p, r = 2.0, h * (1 + _TOL)
    ta, tb = cKDTree(xa), cKDTree(xb)
    pairs = ta.sparse_distance_matrix(tb, r, p=p, output_type="ndarray")
    return np.stack([pairs["i"], pairs["j"]], axis=1).astype(np.int64)


def group_cells(
    mask_or_labels: np.ndarray,
    xyz: np.ndarray,
    level: np.ndarray | None = None,
    dx_per_level: float | Mapping[int, float] = 1.0,
    connectivity: int = 26,
    stitch: bool = True,
) -> np.ndarray:
    """Connected components of selected cells, per level and then across levels.

    On each level the selected cells are placed on the level's regular grid
    (indices from ``xyz`` and the level's spacing) and labelled with
    ``scipy.ndimage.label``. With ``stitch``, a component on one level and a
    component on another are then merged when one of their cells are grid
    neighbours on the coarser of the two grids (Chebyshev distance at most the
    coarser spacing for 26-connectivity, Euclidean distance at most it for 6,
    at most sqrt(2) times it for 18). Only cells on the outer shell of each
    level's selection (found by binary erosion) are compared, so the cost is
    set by the size of the level boundaries, not of the components.

    Args:
        mask_or_labels (np.ndarray): (N,) bool mask of the cells to group, or an
            integer label per cell; with labels, each value >= 0 is grouped on
            its own (cells of different labels never join) and negative values
            are background.
        xyz (np.ndarray): (N, 3) cell coordinates.
        level (np.ndarray, optional): (N,) refinement level of each cell.
            Defaults to a single level.
        dx_per_level (float | Mapping[int, float], optional): Spacing of each
            level (a number if there is one level). Defaults to 1.0.
        connectivity (int, optional): 6, 18 or 26. Defaults to 26.
        stitch (bool, optional): Join components across levels. Defaults to True.

    Returns:
        np.ndarray: int64 (N,) component id per cell, -1 for unselected cells.
            Ids run from 0 in order of decreasing cell count.
    """
    sel = np.asarray(mask_or_labels)
    n = sel.shape[0]
    if sel.shape != (n,):
        raise ValueError("mask_or_labels must be one-dimensional")
    xyz = np.asarray(xyz)
    if xyz.shape != (n, 3):
        raise ValueError("xyz must have shape (N, 3)")
    level = _levels(level, n)
    structure = _structure(connectivity)
    if sel.dtype == bool:
        classes = [(None, sel)]
    else:
        values = np.unique(sel[sel >= 0])
        classes = [(v, sel == v) for v in values]

    all_levels = np.unique(level)
    dx = _dx_lookup(dx_per_level, all_levels)
    max_dx = max(dx.values()) if dx else 1.0
    ids = np.full(n, -1, dtype=np.int64)
    next_id = 0
    edges = []
    for _, mask in classes:
        shells = {}  # level -> rows on the outer shell of the selection
        for lv in all_levels:
            rows = np.flatnonzero(mask & (level == lv))
            if rows.size == 0:
                continue
            h = dx[int(lv)]
            idx, _ = level_indices(xyz[rows], h)
            shape = tuple(idx.max(axis=0) + 1)
            vol = np.zeros(shape, dtype=bool)
            vol[idx[:, 0], idx[:, 1], idx[:, 2]] = True
            lab, k = ndimage.label(vol, structure=structure)
            ids[rows] = lab[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int64) - 1 + next_id
            next_id += int(k)
            del lab
            if stitch and len(all_levels) > 1:
                # cells within the coarsest spacing of a non-selected position
                it = max(1, int(np.ceil(max_dx / h - _TOL)))
                inner = ndimage.binary_erosion(vol, structure=np.ones((3, 3, 3), bool),
                                               iterations=it, border_value=0)
                shells[int(lv)] = rows[~inner[idx[:, 0], idx[:, 1], idx[:, 2]]]
                del inner
            del vol
        if stitch:
            lvs = sorted(shells)
            for a in range(len(lvs)):
                for b in range(a + 1, len(lvs)):
                    ra, rb = shells[lvs[a]], shells[lvs[b]]
                    h = max(dx[lvs[a]], dx[lvs[b]])
                    pairs = _touch_pairs(xyz[ra].astype(np.float64), xyz[rb].astype(np.float64),
                                         h, connectivity)
                    if len(pairs):
                        edges.append(np.stack([ids[ra[pairs[:, 0]]], ids[rb[pairs[:, 1]]]], axis=1))
    if edges and next_id:
        e = np.unique(np.concatenate(edges), axis=0)
        graph = sparse.coo_matrix((np.ones(len(e), bool), (e[:, 0], e[:, 1])),
                                  shape=(next_id, next_id))
        k, root = connected_components(graph, directed=False)
        fg = ids >= 0
        ids[fg] = root[ids[fg]]
        next_id = int(k)
    return _relabel_by_size(ids, next_id)


# --------------------------------------------------------------------------- descriptors


def _cell_volume(level, dx_per_level, n, cell_volume):
    if dx_per_level is None:
        return np.full(n, float(cell_volume)), np.full(n, float(cell_volume) ** (1 / 3))
    lv = _levels(level, n)
    dx = _dx_lookup(dx_per_level, np.unique(lv))
    lut_levels = np.array(sorted(dx), dtype=np.int64)
    lut_dx = np.array([dx[k] for k in lut_levels])
    h = lut_dx[np.searchsorted(lut_levels, lv.astype(np.int64))]
    return h ** 3, h


def _group_minmax(key: np.ndarray, values: np.ndarray, n: int):
    lo = np.full(n, np.inf)
    hi = np.full(n, -np.inf)
    np.minimum.at(lo, key, values)
    np.maximum.at(hi, key, values)
    return lo, hi


def _moments(key, x, w, n, h2):
    """Weighted centroid and covariance (cells as uniform cubes) per key."""
    wsum = np.bincount(key, weights=w, minlength=n)
    safe = np.where(wsum > 0, wsum, 1.0)
    c = np.stack([np.bincount(key, weights=w * x[:, i], minlength=n) for i in range(3)], 1) / safe[:, None]
    d = x - c[key]
    cov = np.empty((n, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            s = np.bincount(key, weights=w * d[:, i] * d[:, j], minlength=n) / safe
            cov[:, i, j] = s
            cov[:, j, i] = s
    cube = np.bincount(key, weights=w * h2, minlength=n) / safe / 12.0
    cov[:, [0, 1, 2], [0, 1, 2]] += cube[:, None]
    return wsum, c, cov


def _eig_desc(cov):
    """Eigenvalues (descending) and axes as rows e_1, e_2, e_3, sign-fixed."""
    lam, vec = np.linalg.eigh(cov)
    lam = lam[:, ::-1]
    axes = np.transpose(vec[:, :, ::-1], (0, 2, 1)).copy()
    # deterministic sign: largest-|component| of each axis is positive
    big = np.take_along_axis(axes, np.argmax(np.abs(axes), axis=2)[..., None], axis=2)[..., 0]
    axes *= np.where(big < 0, -1.0, 1.0)[..., None]
    return np.clip(lam, 0.0, None), axes


def tube_descriptors(
    component_ids: np.ndarray,
    xyz: np.ndarray,
    B: np.ndarray | None = None,
    level: np.ndarray | None = None,
    dx_per_level: float | Mapping[int, float] | None = None,
    cell_volume: float = 1.0,
    n_segments: int = 4,
) -> dict:
    """Shape and field-alignment descriptors of each component.

    Every quantity is weighted by cell volume (``dx_level ** 3``), so cells of
    a coarse level count for what they cover. See the module docstring for the
    definitions of extent, elongation, alignment and tube score.

    Args:
        component_ids (np.ndarray): (N,) ids from :func:`group_cells`, -1 for
            background; ids must be 0..K-1.
        xyz (np.ndarray): (N, 3) cell coordinates.
        B (np.ndarray, optional): (N, 3) magnetic field. Without it the
            alignment entries are NaN.
        level (np.ndarray, optional): (N,) level of each cell.
        dx_per_level (float | Mapping[int, float], optional): Spacing per level;
            if None, every cell has volume ``cell_volume``.
        cell_volume (float, optional): Cell volume when ``dx_per_level`` is None.
        n_segments (int, optional): Slices along ``e_1`` for ``alignment_local``.
            Defaults to 4.

    Returns:
        dict: arrays indexed by component id ``k`` (length K):

        - ``id``: k; ``n_cells``; ``volume``; ``centroid`` (K, 3);
        - ``eigvals`` (K, 3) descending position variances (cube-corrected);
        - ``axes`` (K, 3, 3), ``axes[k, i]`` the unit principal axis ``e_{i+1}``;
        - ``extents`` (K, 3) = sqrt(12 eigvals); ``span`` (K, 3) max - min of
          the cell coordinates projected on each axis, plus one cell;
        - ``elongation`` = extents[:, 0] / extents[:, 1];
          ``flatness`` = extents[:, 1] / extents[:, 2];
        - ``alignment`` mean |cos(B, e_1)|; ``align_axes`` (K, 3) mean |cos| with
          each axis; ``alignment_local`` the same with each slice's own long axis;
        - ``B_coherence`` largest eigenvalue of the mean of b b^T (b = B/|B|):
          1 if the field keeps one direction (up to sign), 1/3 if isotropic;
        - ``mean_B`` volume-weighted mean |B|; ``max_B``;
        - ``tube_score`` = elongation * alignment;
        - ``cells_per_level`` dict level -> (K,) counts (if ``level`` given).
    """
    ids = np.asarray(component_ids)
    n = ids.shape[0]
    xyz = np.asarray(xyz)
    fg = np.flatnonzero(ids >= 0)
    key = ids[fg].astype(np.int64)
    K = int(key.max()) + 1 if key.size else 0
    x = xyz[fg].astype(np.float64)
    w, h = _cell_volume(None if level is None else _levels(level, n)[fg], dx_per_level,
                        len(fg), cell_volume)
    h2 = h * h

    n_cells = np.bincount(key, minlength=K)
    volume, centroid, cov = _moments(key, x, w, K, h2)
    lam, axes = _eig_desc(cov) if K else (np.zeros((0, 3)), np.zeros((0, 3, 3)))
    extents = np.sqrt(12.0 * lam)
    proj = np.einsum("ni,nji->nj", x - centroid[key], axes[key])  # (n, 3) along e_1..e_3
    span = np.empty((K, 3))
    for j in range(3):
        lo, hi = _group_minmax(key, proj[:, j], K)
        span[:, j] = hi - lo + np.bincount(key, weights=w * h, minlength=K) / np.where(volume > 0, volume, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        elongation = extents[:, 0] / extents[:, 1]
        flatness = extents[:, 1] / extents[:, 2]

    out = {
        "id": np.arange(K),
        "n_cells": n_cells,
        "volume": volume,
        "centroid": centroid,
        "eigvals": lam,
        "axes": axes,
        "extents": extents,
        "span": span,
        "elongation": elongation,
        "flatness": flatness,
    }

    nan = np.full(K, np.nan)
    if B is None:
        out.update(alignment=nan, align_axes=np.full((K, 3), np.nan), alignment_local=nan.copy(),
                   B_coherence=nan.copy(), mean_B=nan.copy(), max_B=nan.copy())
    else:
        Bf = np.asarray(B)[fg].astype(np.float64)
        bmag = np.linalg.norm(Bf, axis=1)
        ok = bmag > 0
        b = np.zeros_like(Bf)
        b[ok] = Bf[ok] / bmag[ok, None]
        wb = w * ok  # cells with zero field carry no direction
        safe = np.where(np.bincount(key, weights=wb, minlength=K) > 0,
                        np.bincount(key, weights=wb, minlength=K), np.nan)
        cosines = np.abs(np.einsum("ni,nji->nj", b, axes[key]))
        align_axes = np.stack([np.bincount(key, weights=wb * cosines[:, j], minlength=K)
                               for j in range(3)], 1) / safe[:, None]
        # orientation tensor of the field direction
        T = np.empty((K, 3, 3))
        for i in range(3):
            for j in range(i, 3):
                s = np.bincount(key, weights=wb * b[:, i] * b[:, j], minlength=K) / safe
                T[:, i, j] = s
                T[:, j, i] = s
        coherence = np.full(K, np.nan)
        good = np.isfinite(safe)
        if np.any(good):
            coherence[good] = np.linalg.eigvalsh(T[good])[:, -1]
        # local alignment: slices along e_1
        nseg = max(int(n_segments), 1)
        lo1, hi1 = _group_minmax(key, proj[:, 0], K)
        width = np.where(hi1 > lo1, hi1 - lo1, 1.0)
        seg = np.clip(((proj[:, 0] - lo1[key]) / width[key] * nseg).astype(np.int64), 0, nseg - 1)
        skey = key * nseg + seg
        sw, _, scov = _moments(skey, x, w, K * nseg, h2)
        _, saxes = _eig_desc(scov)
        scount = np.bincount(skey, minlength=K * nseg)
        loc_axis = np.where((scount >= 3)[:, None], saxes[:, 0], axes[np.arange(K * nseg) // nseg, 0])
        cos_loc = np.abs(np.einsum("ni,ni->n", b, loc_axis[skey]))
        alignment_local = np.bincount(key, weights=wb * cos_loc, minlength=K) / safe
        mean_B = np.bincount(key, weights=w * bmag, minlength=K) / np.where(volume > 0, volume, np.nan)
        max_B = np.full(K, -np.inf)
        np.maximum.at(max_B, key, bmag)
        out.update(alignment=align_axes[:, 0], align_axes=align_axes, alignment_local=alignment_local,
                   B_coherence=coherence, mean_B=mean_B, max_B=max_B)
    out["tube_score"] = out["elongation"] * out["alignment"]
    if level is not None:
        lv = _levels(level, n)[fg]
        out["cells_per_level"] = {int(v): np.bincount(key[lv == v], minlength=K) for v in np.unique(lv)}
    return out


def select_tubes(
    descriptors: dict,
    min_cells: int = 50,
    min_elongation: float = 3.0,
    min_alignment: float = 0.7,
    sort_by: str = "tube_score",
) -> np.ndarray:
    """Ids of the components that pass a tube selection.

    Args:
        descriptors (dict): Output of :func:`tube_descriptors`.
        min_cells (int, optional): Minimum cell count. Defaults to 50.
        min_elongation (float, optional): Minimum extent_1 / extent_2. Defaults to 3.
        min_alignment (float, optional): Minimum mean |cos(B, e_1)|. Defaults to 0.7.
        sort_by (str, optional): Descriptor to sort by, descending. Defaults to
            "tube_score".

    Returns:
        np.ndarray: int64 ids of the passing components, best first.
    """
    keep = (
        (descriptors["n_cells"] >= min_cells)
        & (np.nan_to_num(descriptors["elongation"], nan=0.0) >= min_elongation)
        & (np.nan_to_num(descriptors["alignment"], nan=0.0) >= min_alignment)
    )
    ids = np.flatnonzero(keep)
    score = np.nan_to_num(np.asarray(descriptors[sort_by])[ids], nan=-np.inf)
    return ids[np.argsort(-score, kind="stable")].astype(np.int64)


# --------------------------------------------------------------------------- reference tubes


def grid_mask_to_rows(mask: np.ndarray, origin, spacing, xyz: np.ndarray) -> np.ndarray:
    """Sample a 3D boolean (or integer label) grid at each row's position.

    Each row takes the value of the nearest grid point; rows outside the grid
    get False (or 0). Use it for a tube mask on one of the simulation's level
    grids (rows of that level then hit grid points exactly) or on any other
    uniform grid.

    Args:
        mask (np.ndarray): (nx, ny, nz) array, indexed [ix, iy, iz].
        origin (array-like): Coordinates of mask[0, 0, 0].
        spacing (float | array-like): Grid spacing, one value or one per axis.
        xyz (np.ndarray): (N, 3) row coordinates.

    Returns:
        np.ndarray: (N,) values of ``mask`` at the rows (dtype of ``mask``).
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError("mask must be 3D")
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (3,))
    out = np.zeros(len(xyz), dtype=mask.dtype)
    step = 1 << 22
    for a in range(0, len(xyz), step):
        p = np.asarray(xyz[a:a + step], dtype=np.float64)
        idx = np.rint((p - origin) / spacing).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < np.array(mask.shape)), axis=1)
        i = idx[inside]
        out[a:a + step][inside] = mask[i[:, 0], i[:, 1], i[:, 2]]
    return out


def _segment_distance(p, a, b):
    """Distance from points p (n, 3) to segment [a, b], and the segment parameter."""
    ab = b - a
    L2 = float(ab @ ab)
    t = np.zeros(len(p)) if L2 == 0 else np.clip((p - a) @ ab / L2, 0.0, 1.0)
    d = p - (a + t[:, None] * ab)
    return np.sqrt(np.einsum("ni,ni->n", d, d)), t


def tubes_to_masks(tubes: Sequence[Mapping], xyz: np.ndarray) -> list[np.ndarray]:
    """Rows inside tubes given as centre lines with a radius.

    Each tube is a mapping with ``"centres"`` (M, 3) points along its centre
    line (M = 1 gives a sphere; also accepted: ``"centre"`` (3,)) and
    ``"radius"``, a number or one value per centre (linearly interpolated
    along each segment). A row belongs to the tube if it lies within the radius
    of the piecewise-linear centre line.

    Args:
        tubes (Sequence[Mapping]): The tubes.
        xyz (np.ndarray): (N, 3) row coordinates.

    Returns:
        list[np.ndarray]: int64 row indices of each tube, sorted.
    """
    out = []
    for tube in tubes:
        c = np.asarray(tube["centres"] if "centres" in tube else tube["centre"], dtype=np.float64)
        c = c.reshape(-1, 3)
        r = np.broadcast_to(np.asarray(tube["radius"], dtype=np.float64), (len(c),))
        lo, hi = c.min(axis=0) - r.max(), c.max(axis=0) + r.max()
        cand = []
        step = 1 << 22
        for a in range(0, len(xyz), step):
            p = np.asarray(xyz[a:a + step])
            box = np.all((p >= lo) & (p <= hi), axis=1)
            cand.append(np.flatnonzero(box) + a)
        rows = np.concatenate(cand) if cand else np.zeros(0, np.int64)
        p = np.asarray(xyz[rows], dtype=np.float64)
        inside = np.zeros(len(rows), bool)
        if len(c) == 1:
            inside = np.linalg.norm(p - c[0], axis=1) <= r[0]
        for k in range(len(c) - 1):
            d, t = _segment_distance(p, c[k], c[k + 1])
            inside |= d <= r[k] + t * (r[k + 1] - r[k])
        out.append(np.sort(rows[inside]).astype(np.int64))
    return out


def as_membership(components, n_rows: int, xyz: np.ndarray | None = None):
    """Turn any supported description of a set of components into a sparse
    (n_rows, K) boolean membership matrix.

    Accepted forms:

    - (N,) integer array aligned with the grid rows: component id per row,
      negative for background (the output of :func:`group_cells`);
    - (N,) boolean array: one component;
    - a sequence of (N,) boolean arrays, or of integer row-index arrays: one
      component each (components may overlap);
    - a mapping ``{"mask": 3D array, "origin": (3,), "spacing": dx}``: a
      boolean mask (one component) or integer label grid (each value > 0 a
      component) on a uniform grid, sampled at the rows with
      :func:`grid_mask_to_rows` (needs ``xyz``);
    - a sequence of mappings with ``"centres"``/``"centre"`` and ``"radius"``:
      tubes converted with :func:`tubes_to_masks` (needs ``xyz``).

    Args:
        components: One of the forms above.
        n_rows (int): Number of grid rows N.
        xyz (np.ndarray, optional): (N, 3) row coordinates, for the grid and tube forms.

    Returns:
        tuple[scipy.sparse.csc_matrix, np.ndarray]: membership (N, K) and the
            label of each column (component id, mask value, or list position).
    """

    def from_ids(ids):
        ids = np.asarray(ids)
        rows = np.flatnonzero(ids >= 0)
        vals = ids[rows]
        labels, cols = np.unique(vals, return_inverse=True)
        m = sparse.csc_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(n_rows, len(labels)))
        return m, labels

    def from_row_lists(lists):
        rows = np.concatenate([np.asarray(r, dtype=np.int64) for r in lists]) if lists else np.zeros(0, np.int64)
        cols = np.concatenate([np.full(len(r), k) for k, r in enumerate(lists)]) if lists else np.zeros(0, np.int64)
        m = sparse.csc_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(n_rows, len(lists)))
        m.sum_duplicates()
        return m, np.arange(len(lists))

    if isinstance(components, Mapping):
        if xyz is None:
            raise ValueError("a grid mask needs xyz")
        vals = grid_mask_to_rows(components["mask"], components["origin"], components["spacing"], xyz)
        if vals.dtype == bool:
            return from_row_lists([np.flatnonzero(vals)])
        vals = vals.astype(np.int64)
        vals[vals <= 0] = -1
        return from_ids(vals)
    if isinstance(components, np.ndarray) and components.ndim == 1 and len(components) == n_rows:
        if components.dtype == bool:
            return from_row_lists([np.flatnonzero(components)])
        return from_ids(components)
    items = list(components)
    if items and isinstance(items[0], Mapping):
        if xyz is None:
            raise ValueError("tubes given as centre lines need xyz")
        return from_row_lists(tubes_to_masks(items, xyz))
    lists = []
    for it in items:
        it = np.asarray(it)
        if it.dtype == bool:
            if it.shape != (n_rows,):
                raise ValueError("boolean masks must have one entry per row")
            lists.append(np.flatnonzero(it))
        else:
            lists.append(it.astype(np.int64))
    return from_row_lists(lists)


@dataclass
class MatchResult:
    """Overlap of two sets of components.

    Attributes:
        iou (np.ndarray | scipy.sparse.csr_matrix): (Ka, Kb) intersection over
            union; dense when Ka * Kb <= 2e7, else sparse (zeros not stored).
        intersection (scipy.sparse.csr_matrix): (Ka, Kb) overlap (volume or cells).
        size_a (np.ndarray): (Ka,) size of each component of a.
        size_b (np.ndarray): (Kb,) size of each component of b.
        labels_a (np.ndarray): label of each row of the matrices.
        labels_b (np.ndarray): label of each column.
        best_a_for_b (np.ndarray): (Kb,) row index of a with the highest IoU, -1 if none overlaps.
        best_iou_b (np.ndarray): (Kb,) that IoU.
        best_b_for_a (np.ndarray): (Ka,) likewise for each component of a.
        best_iou_a (np.ndarray): (Ka,).
        pairs (np.ndarray): (P, 2) mutual best matches (row of a, column of b)
            with IoU >= ``min_iou``.
        recall_b (np.ndarray): (Kb,) fraction of each b covered by the union of a.
        precision_a (np.ndarray): (Ka,) fraction of each a inside the union of b.
    """

    iou: object
    intersection: object
    size_a: np.ndarray
    size_b: np.ndarray
    labels_a: np.ndarray
    labels_b: np.ndarray
    best_a_for_b: np.ndarray
    best_iou_b: np.ndarray
    best_b_for_a: np.ndarray
    best_iou_a: np.ndarray
    pairs: np.ndarray
    recall_b: np.ndarray
    precision_a: np.ndarray


def match_to_reference(
    components_a,
    components_b,
    xyz: np.ndarray | None = None,
    level: np.ndarray | None = None,
    dx_per_level: float | Mapping[int, float] | None = None,
    n_rows: int | None = None,
    min_iou: float = 0.0,
) -> MatchResult:
    """Intersection over union between two sets of components on the same rows.

    Designed for comparing aweSOM tubes (``components_a``, usually the ids from
    :func:`group_cells`, possibly with non-tubes set to -1) with an independent
    identification (``components_b``) in any form :func:`as_membership`
    accepts: an id per grid row, a list of boolean row masks or row-index
    arrays, a mask or label cube on a uniform grid with its origin and
    spacing, or a list of centre lines with radii.

    Sizes are cell volumes (``dx_level ** 3``) when ``dx_per_level`` is given,
    otherwise cell counts.

    Args:
        components_a: The first set (see :func:`as_membership`).
        components_b: The second set.
        xyz (np.ndarray, optional): (N, 3) row coordinates (needed for grid
            masks and centre lines).
        level (np.ndarray, optional): (N,) level of each row, for volumes.
        dx_per_level (float | Mapping[int, float], optional): Spacing per level.
        n_rows (int, optional): N, if neither ``xyz`` nor an id array gives it.
        min_iou (float, optional): Threshold for ``pairs``. Defaults to 0.

    Returns:
        MatchResult: matrices and best matches.
    """
    if n_rows is None:
        if xyz is not None:
            n_rows = len(xyz)
        else:
            for c in (components_a, components_b):
                if isinstance(c, np.ndarray) and c.ndim == 1:
                    n_rows = len(c)
                    break
    if n_rows is None:
        raise ValueError("give n_rows or xyz")
    A, la = as_membership(components_a, n_rows, xyz)
    Bm, lb = as_membership(components_b, n_rows, xyz)
    if dx_per_level is None:
        w = np.ones(n_rows)
    else:
        w, _ = _cell_volume(level, dx_per_level, n_rows, 1.0)
    A = A.astype(np.float64)
    Bm = Bm.astype(np.float64)
    W = sparse.diags(w)
    inter = (A.T @ W @ Bm).tocsr()
    size_a = np.asarray(A.T @ w).ravel()
    size_b = np.asarray(Bm.T @ w).ravel()
    ka, kb = len(size_a), len(size_b)
    coo = inter.tocoo()
    iou_vals = coo.data / (size_a[coo.row] + size_b[coo.col] - coo.data)
    iou_sparse = sparse.csr_matrix((iou_vals, (coo.row, coo.col)), shape=(ka, kb))
    best_a_for_b = np.full(kb, -1, dtype=np.int64)
    best_iou_b = np.zeros(kb)
    best_b_for_a = np.full(ka, -1, dtype=np.int64)
    best_iou_a = np.zeros(ka)
    order = np.lexsort((-iou_vals, coo.col))  # per column, best first
    first = np.ones(len(order), bool)
    first[1:] = coo.col[order][1:] != coo.col[order][:-1]
    sel = order[first]
    best_a_for_b[coo.col[sel]] = coo.row[sel]
    best_iou_b[coo.col[sel]] = iou_vals[sel]
    order = np.lexsort((-iou_vals, coo.row))
    first = np.ones(len(order), bool)
    first[1:] = coo.row[order][1:] != coo.row[order][:-1]
    sel = order[first]
    best_b_for_a[coo.row[sel]] = coo.col[sel]
    best_iou_a[coo.row[sel]] = iou_vals[sel]
    pairs = [(best_a_for_b[j], j) for j in range(kb)
             if best_a_for_b[j] >= 0 and best_b_for_a[best_a_for_b[j]] == j and best_iou_b[j] >= min_iou]
    pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    union_a = np.asarray(A.sum(axis=1)).ravel() > 0
    union_b = np.asarray(Bm.sum(axis=1)).ravel() > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        recall_b = np.asarray(Bm.T @ (w * union_a)).ravel() / size_b
        precision_a = np.asarray(A.T @ (w * union_b)).ravel() / size_a
    iou = iou_sparse.toarray() if ka * kb <= 2e7 else iou_sparse
    return MatchResult(iou=iou, intersection=inter, size_a=size_a, size_b=size_b, labels_a=la,
                       labels_b=lb, best_a_for_b=best_a_for_b, best_iou_b=best_iou_b,
                       best_b_for_a=best_b_for_a, best_iou_a=best_iou_a, pairs=pairs,
                       recall_b=recall_b, precision_a=precision_a)


# --------------------------------------------------------------------------- synthetic data


def _centre_lines(n: int, n_tubes: int):
    """Curved centre lines in a box [0, n)^3: arcs and helices, kept apart."""
    t = np.linspace(0.0, 1.0, 1500)
    s = n / 128.0
    lines = []
    # 1: arc of a circle (radius 40) in a plane tilted about x
    th = np.deg2rad(-60 + 120 * t)
    tilt = np.deg2rad(25)
    arc = np.stack([40 * np.sin(th), 40 * (np.cos(th) - 1) * np.cos(tilt),
                    40 * (np.cos(th) - 1) * np.sin(tilt)], 1)
    lines.append(arc * s + np.array([64, 104, 40]) * s)
    # 2: helix along z, radius 5, pitch 90
    ph = 2 * np.pi * 1.1 * t
    lines.append(np.stack([5 * np.cos(ph) + 26, 5 * np.sin(ph) + 30, 14 + 100 * t], 1) * s)
    # 3: gently bent tube along x-y diagonal (quadratic bow)
    u = t - 0.5
    lines.append(np.stack([36 + 70 * t, 40 + 50 * t, 92 + 50 * u ** 2 - 12], 1) * s)
    # 4: arc in the x-z plane (radius 30)
    th = np.deg2rad(-50 + 100 * t)
    lines.append(np.stack([96 + 30 * np.sin(th), 60 + 0 * t, 20 + 30 * np.cos(th)], 1) * s)
    # 5: S-curve along y
    lines.append(np.stack([108 + 8 * np.sin(2 * np.pi * t), 12 + 100 * t, 104 + 0 * t], 1) * s)
    return lines[:n_tubes]


def make_synthetic_tubes(n: int = 128, n_tubes: int = 4, radius: float = 3.0, noise: float = 0.1,
                         seed: int = 0, threshold: float = 0.5):
    """A box of curved Gaussian flux tubes in a noisy background field.

    Each tube has a curved centre line (arc, helix, bow or S-curve), field
    strength ``exp(-d^2 / (2 radius^2))`` at distance d from it (scaled to the
    box size ``n / 128``), and field direction along the local tangent. The
    background is a smoothed Gaussian random vector field of rms ``noise``.
    The truth mask of a tube is the cells where its own noise-free strength is
    above ``threshold``.

    Args:
        n (int, optional): Cells per side. Defaults to 128.
        n_tubes (int, optional): 1 to 5. Defaults to 4.
        radius (float, optional): Gaussian width in cells at n = 128. Defaults to 3.
        noise (float, optional): rms of the background field. Defaults to 0.1.
        seed (int, optional): Random seed. Defaults to 0.
        threshold (float, optional): |B| threshold of the truth masks. Defaults to 0.5.

    Returns:
        dict: ``xyz`` (n^3, 3) float32 cell centres (C order, dx = 1), ``B``
            (n^3, 3) float32, ``truth`` (n^3,) int64 tube index or -1,
            ``centre_lines`` list of (M, 3), ``radius`` (scaled), ``shape``.
    """
    if not 1 <= n_tubes <= 5:
        raise ValueError("n_tubes must be between 1 and 5")
    rng = np.random.default_rng(seed)
    lines = _centre_lines(n, n_tubes)
    r0 = radius * n / 128.0
    g = np.arange(n, dtype=np.float64)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    del X, Y, Z
    B = np.zeros_like(pts)
    best = np.zeros(len(pts))
    truth = np.full(len(pts), -1, dtype=np.int64)
    for k, c in enumerate(lines):
        tang = np.gradient(c, axis=0)
        tang /= np.linalg.norm(tang, axis=1)[:, None]
        d, j = cKDTree(c).query(pts, distance_upper_bound=5 * r0)
        near = np.isfinite(d)
        amp = np.zeros(len(pts))
        amp[near] = np.exp(-d[near] ** 2 / (2 * r0 ** 2))
        B[near] += amp[near, None] * tang[j[near]]
        own = amp > np.maximum(best, threshold)
        truth[own] = k
        best = np.maximum(best, amp)
    if noise > 0:
        for i in range(3):
            f = ndimage.gaussian_filter(rng.standard_normal((n, n, n)), 1.5, mode="wrap")
            B[:, i] += noise * f.ravel() / f.std()
    return {"xyz": pts.astype(np.float32), "B": B.astype(np.float32), "truth": truth,
            "centre_lines": lines, "radius": r0, "shape": (n, n, n)}
