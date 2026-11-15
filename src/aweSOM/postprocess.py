"""Optional spatial clean-up of cluster labels. Nothing here runs by default.

aweSOM clusters points in feature space only; it never sees where a point is.
Physical regions are spatially connected, so isolated specks of a label inside
another region are usually noise at a cluster boundary in feature space.
:func:`clean_labels` removes them on the grid the points come from:

- ``min_size``: connected components of a label with fewer cells than this
  are dissolved, and each of their cells takes the label of the nearest cell
  (Euclidean, in cells) that is not in such a component. This is
  shape-agnostic: a thin but extended sheet survives as long as it is
  connected and large;
- ``majority_iters``: that many passes of a majority (mode) filter over a
  ``majority_size``-wide box. It smooths boundaries but also erodes structures
  thinner than about half the box, i.e. thin current sheets; the 2026-10-28
  BBH study recommends ``min_size`` alone.

Points on several uniform grids (AMR levels) are cleaned one grid at a time
(``groups``), with neighbours and component sizes counted in cells of each
point's own grid.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .physics_score import VoxelGrid

__all__ = ["clean_labels", "clean_label_map", "majority_filter", "remove_small_components"]


def majority_filter(m: np.ndarray, size: int = 3, iters: int = 1, background: int = -1) -> np.ndarray:
    """Mode filter on a label box; ``background`` voxels neither vote nor change.

    Each occupied voxel takes the label most frequent among the occupied voxels
    of the ``size``-wide box around it (itself included); a tie keeps the
    current label if it is among the most frequent, else the smallest label.

    Args:
        m (np.ndarray): int label array (any dimension).
        size (int, optional): box width (odd). Defaults to 3.
        iters (int, optional): number of passes. Defaults to 1.
        background (int, optional): label of empty voxels. Defaults to -1.

    Returns:
        np.ndarray: the filtered labels (a new array).
    """
    m = np.array(m, copy=True)
    occupied = (m != background) & (m >= 0)
    labels = np.unique(m[occupied])
    if labels.size < 2:
        return m
    for _ in range(int(iters)):
        best = np.full(m.shape, -1.0, dtype=np.float32)
        out = m.copy()
        for c in labels:
            cnt = ndimage.uniform_filter((m == c).astype(np.float32), size=size, mode="constant")
            cnt += 0.5 / size**m.ndim * (m == c)  # tie-break towards the current label
            better = occupied & (cnt > best)
            best[better] = cnt[better]
            out[better] = c
            del cnt, better
        changed = np.count_nonzero(out != m)
        m = out
        if changed == 0:
            break
    return m


def remove_small_components(
    m: np.ndarray, min_size: int, connectivity: int = 1, background: int = -1
) -> np.ndarray:
    """Dissolve every connected component smaller than ``min_size`` cells.

    The cells of such components take the label of the nearest occupied cell
    outside them (Euclidean distance transform). If every component is small,
    the labels are returned unchanged.

    Args:
        m (np.ndarray): int label array.
        min_size (int): smallest component kept, in cells.
        connectivity (int, optional): 1 = faces; ``m.ndim`` = faces, edges and
            corners. Defaults to 1.
        background (int, optional): label of empty voxels. Defaults to -1.

    Returns:
        np.ndarray: the cleaned labels (a new array).
    """
    m = np.array(m, copy=True)
    if min_size <= 1:
        return m
    occupied = (m != background) & (m >= 0)
    struct = ndimage.generate_binary_structure(m.ndim, connectivity)
    small = np.zeros(m.shape, dtype=bool)
    for c in np.unique(m[occupied]):
        lab, n = ndimage.label(m == c, structure=struct)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        drop = sizes < min_size
        drop[0] = False
        if drop.any():
            small |= drop[lab]
        del lab
    if not small.any():
        return m
    source = occupied & ~small
    if not source.any():
        return m
    nearest = ndimage.distance_transform_edt(~source, return_distances=False, return_indices=True)
    sel = np.nonzero(small)
    m[sel] = m[tuple(ix[sel] for ix in nearest)]
    return m


def clean_label_map(
    m: np.ndarray,
    min_size: int = 0,
    majority_iters: int = 0,
    majority_size: int = 3,
    connectivity: int = 1,
    background: int = -1,
) -> np.ndarray:
    """:func:`majority_filter` (if ``majority_iters``) then
    :func:`remove_small_components` (if ``min_size``) on one label box."""
    out = np.asarray(m)
    if majority_iters:
        out = majority_filter(out, majority_size, majority_iters, background)
    if min_size and min_size > 1:
        out = remove_small_components(out, min_size, connectivity, background)
    return np.array(out, copy=True) if out is m else out


def clean_labels(
    labels,
    xyz=None,
    spacing=None,
    groups=None,
    *,
    min_size: int = 0,
    majority_iters: int = 0,
    majority_size: int = 3,
    connectivity: int = 1,
    background: int = -1,
    grid: VoxelGrid | None = None,
) -> np.ndarray:
    """Spatial clean-up of cluster labels (off by default: with the default
    ``min_size=0`` and ``majority_iters=0`` the labels come back unchanged).

    Two input forms:

    - a dense label array (2D or 3D image of the grid) with ``xyz=None``;
      ``background`` marks voxels that are not data;
    - per-point labels [N] with coordinates ``xyz`` [N, d] on uniform grids of
      spacing ``spacing`` (a float, or group -> spacing), optionally split into
      ``groups`` (e.g. AMR refinement levels; each is cleaned on its own grid),
      or a prebuilt ``grid`` (:class:`aweSOM.physics_score.VoxelGrid`).

    Recommended by the 2026-10-28 BBH study:
    ``min_size=27`` (one 3^3 stencil), ``majority_iters=0``.

    Args:
        labels (np.ndarray): label image, or [N] labels.
        xyz (np.ndarray, optional): [N, d] coordinates of the points.
        spacing (float | dict, optional): grid spacing (per group).
        groups (np.ndarray, optional): [N] group (e.g. level) of each point.
        min_size (int, optional): dissolve components smaller than this many
            cells. Defaults to 0 (off).
        majority_iters (int, optional): majority-filter passes. Defaults to 0 (off).
        majority_size (int, optional): majority box width. Defaults to 3.
        connectivity (int, optional): neighbourhood of the components
            (1 = faces). Defaults to 1.
        background (int, optional): label of empty voxels (dense form) and of
            points to leave alone (point form, e.g. unassigned -1). Defaults to -1.
        grid (VoxelGrid, optional): prebuilt grid for the point form.

    Returns:
        np.ndarray: cleaned labels, same shape and dtype as ``labels``.
    """
    labels = np.asarray(labels)
    if xyz is None and grid is None:
        return clean_label_map(labels, min_size, majority_iters, majority_size, connectivity, background)
    if grid is None:
        if spacing is None:
            raise ValueError("spacing is required with xyz")
        grid = VoxelGrid.from_points(xyz, spacing, groups)
    if labels.shape != (grid.n_points,):
        raise ValueError("labels must have one entry per point")
    if min_size <= 1 and not majority_iters:
        return labels.copy()
    dtype = np.int16 if labels.size == 0 or (labels.max() < 32767 and labels.min() >= -32768) else np.int64
    maps = grid.to_maps(labels.astype(dtype), background)
    for g in grid.groups:
        maps[g] = clean_label_map(maps[g], min_size, majority_iters, majority_size, connectivity, background)
    return grid.from_maps(maps).astype(labels.dtype)
