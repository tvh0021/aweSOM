"""Adjacency of cells across refinement levels.

Cells of one level are grid neighbours in the usual sense (6, 18 or 26
neighbours). Across levels there is no common grid, so two cells are taken to
be neighbours when the regions of space they stand for touch. Along each axis,
toward the other cell, a cell's region reaches

* half its own spacing when the next position on its own lattice along that
  axis is owned by its level (an interior cell);
* otherwise half-way to the nearest owned point of the other level on that ray,
  if there is one within the larger of the two spacings. On a vertex-centred
  interface the last fine point sits on a coarse position and the first coarse
  point is a full coarse spacing away, so the two regions meet at the mid-point
  of that gap; on a cell-centred interface they meet at the shared face;
* otherwise half the larger spacing when the cell sits on the other level's
  lattice along that axis (a vertex-centred interface seen from a fine point
  that lies tangentially between two coarse rows, whose ray holds no coarse
  point);
* otherwise half its own spacing (a detached cell).

Two cells touch when the regions touch or overlap along all three axes. The
contact is a face when they overlap along two axes and only touch along one,
an edge when they touch along two, a corner when they touch along all three.
Connectivity 6 accepts faces, 18 faces and edges, 26 all three.

This rule neither bridges a cell that lies between the two (its region
squeezes theirs) nor joins regions that are separated, which the rule
"within the coarser spacing" did; see :func:`touching`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

_MAX_TOUCH = {6: 1, 18: 2, 26: 3}


@dataclass(frozen=True)
class OwnedLevel:
    """The cells one refinement level owns, on its lattice.

    Attributes:
        dx (float): Lattice spacing.
        origin (np.ndarray): Position of lattice index (0, 0, 0).
        owned (np.ndarray): Dense bool array; ``owned[i, j, k]`` is whether the
            level owns the point ``origin + (i, j, k) * dx``.
    """

    dx: float
    origin: np.ndarray
    owned: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=np.float64).reshape(3))
        object.__setattr__(self, "dx", float(self.dx))
        if self.owned.ndim != 3:
            raise ValueError("owned must be a 3-d array")

    @property
    def shape(self) -> tuple:
        return self.owned.shape

    def fractional(self, xyz: np.ndarray) -> np.ndarray:
        """Positions in lattice units, float64 [n, 3]."""
        return (np.asarray(xyz, dtype=np.float64) - self.origin) / self.dx

    def is_owned(self, idx: np.ndarray) -> np.ndarray:
        """Whether integer lattice indices [n, 3] are owned (False outside the box)."""
        idx = np.asarray(idx)
        out = np.zeros(len(idx), dtype=bool)
        ok = np.all((idx >= 0) & (idx < np.asarray(self.shape)), axis=1)
        if ok.any():
            i = idx[ok]
            out[ok] = self.owned[i[:, 0], i[:, 1], i[:, 2]]
        return out


def _half_widths(x: np.ndarray, a: OwnedLevel, b: OwnedLevel, sign: np.ndarray, tol: float) -> np.ndarray:
    """Half-widths [n, 3] of the regions of cells ``x`` (owned by level ``a``)
    toward level ``b``, in direction ``sign`` (entries -1, 0 or 1) per axis."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    w = np.full((n, 3), a.dx / 2)
    cap = max(a.dx, b.dx)
    ia = np.rint(a.fractional(x)).astype(np.int64)
    fb = b.fractional(x)
    ib = np.rint(fb)
    on_b = np.abs(fb - ib) < tol
    ib = ib.astype(np.int64)
    steps = max(1, int(round(cap / b.dx)))  # lattice positions of b within the cap
    for k in range(3):
        s = sign[:, k]
        cand = s != 0
        if not cand.any():
            continue
        nb = ia.copy()
        nb[:, k] += s
        cand &= ~a.is_owned(nb)  # an interior cell keeps its half spacing
        if not cand.any():
            continue
        tangential = np.ones(n, dtype=bool)
        for j in range(3):
            if j != k:
                tangential &= on_b[:, j]
        rows = np.flatnonzero(cand & tangential)
        found = np.zeros(n, dtype=bool)
        if rows.size:
            f = fb[rows, k]
            sr = s[rows]
            first = np.where(sr > 0, np.floor(f + tol) + 1, np.ceil(f - tol) - 1).astype(np.int64)
            idx = ib[rows].copy()
            pending = np.ones(rows.size, dtype=bool)
            for m in range(steps):
                ik = first + m * sr
                d = np.abs(ik * b.dx + b.origin[k] - x[rows, k])
                idx[:, k] = ik
                hit = pending & (d <= cap * (1 + tol)) & b.is_owned(idx)
                if hit.any():
                    w[rows[hit], k] = d[hit] / 2
                    found[rows[hit]] = True
                    pending &= ~hit
                if not pending.any():
                    break
        gap = cand & ~found & on_b[:, k]
        w[gap, k] = cap / 2
    return w


def touching(p: np.ndarray, q: np.ndarray, a: OwnedLevel, b: OwnedLevel, connectivity: int = 26,
             tol: float = 1e-3) -> np.ndarray:
    """Whether cells ``p[i]`` of level ``a`` and ``q[i]`` of level ``b`` are
    neighbours under ``connectivity`` (see the module docstring).

    Args:
        p (np.ndarray): [n, 3] positions of cells owned by ``a``.
        q (np.ndarray): [n, 3] positions of cells owned by ``b``.
        a (OwnedLevel): Level of ``p``.
        b (OwnedLevel): Level of ``q``.
        connectivity (int, optional): 6, 18 or 26. Defaults to 26.
        tol (float, optional): Tolerance in units of the spacings. Defaults to 1e-3.

    Returns:
        np.ndarray: bool [n].
    """
    if connectivity not in _MAX_TOUCH:
        raise ValueError("connectivity must be 6, 18 or 26")
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    q = np.asarray(q, dtype=np.float64).reshape(-1, 3)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if len(p) == 0:
        return np.zeros(0, dtype=bool)
    d = q - p
    sign = np.sign(d).astype(np.int64)
    ad = np.abs(d)
    reach = _half_widths(p, a, b, sign, tol) + _half_widths(q, b, a, -sign, tol)
    touch = np.all(ad <= reach * (1 + tol), axis=1)
    n_touch = np.sum(ad >= reach * (1 - tol), axis=1)  # axes along which the regions only touch
    return touch & (n_touch <= _MAX_TOUCH[connectivity])


def cross_level_pairs(xa: np.ndarray, a: OwnedLevel, xb: np.ndarray, b: OwnedLevel,
                      connectivity: int = 26, tol: float = 1e-3) -> np.ndarray:
    """Index pairs (i, j) of cells ``xa[i]`` (level ``a``) and ``xb[j]`` (level
    ``b``) that are neighbours under ``connectivity``.

    Candidates are found with a KD-tree within the larger spacing (Chebyshev),
    which bounds the reach of two regions, then kept by :func:`touching`.

    Returns:
        np.ndarray: int64 [m, 2].
    """
    xa = np.asarray(xa, dtype=np.float64).reshape(-1, 3)
    xb = np.asarray(xb, dtype=np.float64).reshape(-1, 3)
    if len(xa) == 0 or len(xb) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    r = max(a.dx, b.dx) * (1 + tol)
    pairs = cKDTree(xa).sparse_distance_matrix(cKDTree(xb), r, p=np.inf, output_type="ndarray")
    i, j = pairs["i"].astype(np.int64), pairs["j"].astype(np.int64)
    if len(i) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    keep = touching(xa[i], xb[j], a, b, connectivity, tol)
    return np.stack([i[keep], j[keep]], axis=1)
