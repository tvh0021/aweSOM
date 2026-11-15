"""Current-sheet statistics from SOM labels on a nested Cartesian grid.

The input is one point per owned grid cell (the finest refinement level that
holds it), as written by ``etkplot som-features --grid-out``: positions,
refinement level, the SOM cluster label and the features. Each level is a
regular lattice of spacing ``dx``; its points may have holes (cells owned by a
finer level, or masked near a horizon). Every cell carries the volume
``dx**3`` of its level.

A "sheet" cell is one whose label is in ``sheet_clusters``. The statistics:

* volume and volume fraction of sheet cells, per level and in total;
* connected components (face connectivity by default) of the sheet cells.
  Components are found on each level's lattice and then joined across level
  interfaces: a sheet cell at the edge of what its level owns is connected to
  the sheet cells of a coarser level whose regions touch its own under the
  same connectivity (:mod:`aweSOM.amr_adjacency`: a cell between two others
  keeps them apart, regions that do not touch are not joined);
* per component: volume, centroid, principal extents from the inertia tensor
  (``L_i = sqrt(12 lambda_i)``, exact for a uniform box, cells counted as cubes),
  aspect ratios, and two thickness estimates:

  - ``thickness_inertia`` = smallest principal extent; exact for a flat slab,
    an overestimate for a curved or branched sheet;
  - ``thickness_edt`` = median over the component's ridge cells of
    ``2 d - dx / 2``, where ``d`` is the Euclidean distance from a sheet cell
    centre to the nearest non-sheet cell centre and ridge cells are local
    maxima of ``d`` (3x3x3). For a slab of ``n`` cells this is ``n + 1/2``
    cells for odd ``n`` and ``n - 1/2`` for even ``n``: a half-cell bias
    either way, which is 50 % for a one-cell sheet. Where the two estimates
    differ, quote ``thickness_inertia`` for thin or unresolved sheets and
    ``thickness_edt`` only as a local (ridge-wise) measure. The distance map is
    computed on each level's box (padded by ``edt_margin`` cells) with the
    other levels' cells resampled onto it, the finest owner of each point
    deciding whether it is sheet or background, so level interfaces do not
    cut sheets;

* volume-weighted histograms of features inside and outside sheets
  (``log_sigma = log10 2 + log_beta_inv + log_theta``, i.e. ``b^2 / rho``, is
  derived when the source does not have it);
* radial profiles of sheet and total volume about each hole and the binary
  centre of mass;
* the share of sheet volume in boxes around the segment joining the holes
  (``0 <= s <= 1`` along the segment, ``|a|, |b| <= w`` across it, with ``a``
  in the orbital plane and ``b`` along the orbital axis) for several ``w``,
  plus the profile along the segment.

Memory: the dense per-level boolean masks and int32 component labels are kept
(about 6 bytes per lattice cell of every level's bounding box); point data are
read in chunks of ``SheetConfig.chunk_rows`` rows, one level at a time; the
sheet cells (position, level, three features) are collected for the
per-component moments.

Use :class:`GridFileSource` for an etkplot grid HDF5 file and a label array,
:class:`ArraySource` for in-memory arrays, then :func:`sheet_statistics` and
:func:`write_outputs`.
"""

from __future__ import annotations

import csv
import json
import math
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .amr_adjacency import OwnedLevel, touching

__all__ = [
    "SheetConfig",
    "SheetResult",
    "LevelLattice",
    "level_lattice",
    "ArraySource",
    "GridFileSource",
    "sheet_statistics",
    "write_outputs",
    "weighted_percentile",
]

LOG10_2 = math.log10(2.0)
SIGMA_INPUTS = ("log_beta_inv", "log_theta")

# fixed histogram ranges (lo, hi, bin width) so that runs share bins
DEFAULT_FEATURE_BINS = {
    "log_chi_B": (-3.1, 2.0, 0.02),
    "log_beta_inv": (-4.0, 5.0, 0.02),
    "log_sigma": (-9.0, 5.0, 0.02),
    "log_theta": (-7.0, 1.5, 0.02),
    "log_grad_ln_b2": (-2.0, 2.5, 0.02),
    "asinh_div_v": (-6.0, 6.0, 0.02),
}
# features averaged per component (when available)
COMPONENT_FEATURES = ("log_chi_B", "log_beta_inv", "log_sigma")


def _log_sigma(log_beta_inv, log_theta) -> np.ndarray:
    """log10(b^2 / rho) = log10(2) + log10(b^2 / 2P) + log10(P / rho), float32."""
    return (LOG10_2 + np.asarray(log_beta_inv, dtype=np.float64) + np.asarray(log_theta)).astype(np.float32)


@dataclass
class SheetConfig:
    """Parameters of :func:`sheet_statistics`.

    Attributes:
        connectivity (int): 6 (faces), 18 or 26 neighbours within a level.
        min_cells (int): Components with fewer cells are left out of the shape
            summaries (inertia thickness, aspect ratios); they are still counted.
        edt_margin (int): Padding, in cells, of each level's box for the
            distance transform.
        edt_block (int | None): The distance transform runs on slabs of this
            many cells along x (None: the whole box at once), which bounds its
            memory (about 60 bytes per cell of a slab plus overlap).
        edt_overlap (int): Cells added on each side of a slab. Distances up to
            ``edt_overlap * dx`` are exact; ridge cells deeper than that are
            counted in ``thickness.edt.n_ridge_beyond_overlap``.
        feature_bins (dict): name -> (lo, hi, width) of the inside/outside
            histograms; values outside go to the end bins.
        radial_max (float): Outer edge of the radial profiles, M.
        radial_step (float): Radial bin width, M.
        between_widths (tuple): Half-widths ``w`` (M) of the boxes around the
            segment joining the holes.
        between_default (float): The ``w`` used for the per-component
            ``between_fraction`` and the along-segment profile.
        segment_bins (int): Bins of the along-segment profile on ``s`` in
            [-0.5, 1.5].
        volume_bins (tuple): (log10 lo, log10 hi, dex) of the component-volume
            histogram, M^3.
        thickness_bins (tuple): (log10 lo, log10 hi, dex) of the thickness
            histograms, M.
        count_thresholds (tuple): Component volumes (M^3) above which
            components are counted.
        plane_z (float | None): z of the slice saved for plotting (None: no slice).
        plane_background (str): Feature drawn under the sheet cells in the slice.
        chunk_rows (int): Rows read per chunk.
    """

    connectivity: int = 6
    min_cells: int = 64
    edt_margin: int = 8
    edt_block: int | None = 128
    edt_overlap: int = 32
    feature_bins: dict = field(default_factory=lambda: dict(DEFAULT_FEATURE_BINS))
    radial_max: float = 66.0
    radial_step: float = 0.5
    between_widths: tuple = (2.0, 4.0, 8.0)
    between_default: float = 4.0
    segment_bins: int = 100
    volume_bins: tuple = (-4.0, 5.0, 0.25)
    thickness_bins: tuple = (-2.0, 1.5, 0.05)
    count_thresholds: tuple = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    plane_z: float | None = 0.0
    plane_background: str = "log_beta_inv"
    chunk_rows: int = 4_000_000


@dataclass
class SheetResult:
    """Output of :func:`sheet_statistics`.

    Attributes:
        summary (dict): JSON-serialisable statistics.
        components (dict[str, np.ndarray]): One row per connected component,
            sorted by decreasing volume (column name -> array).
        planes (dict[int, dict]): Per level, the slice at ``plane_z``: ``sheet``
            (int8, -1 not owned, 0 no sheet, 1 sheet), ``background`` (float32,
            NaN where not owned) and ``extent`` [x0, x1, y0, y1] (cell edges).
    """

    summary: dict
    components: dict
    planes: dict


# --------------------------------------------------------------------------- lattices


@dataclass(frozen=True)
class LevelLattice:
    """Regular lattice of one refinement level.

    Attributes:
        level (int): Refinement level.
        dx (float): Spacing (cells are cubes of side ``dx``).
        origin (tuple): Position of lattice index (0, 0, 0).
        shape (tuple): Lattice size along x, y, z.
    """

    level: int
    dx: float
    origin: tuple
    shape: tuple

    @property
    def size(self) -> int:
        """int: Number of lattice cells (owned or not)."""
        return int(np.prod(self.shape))

    def index(self, xyz: np.ndarray) -> np.ndarray:
        """Integer lattice index of points.

        Args:
            xyz (np.ndarray): [n, 3] positions on the lattice.

        Returns:
            np.ndarray: int64 [n, 3].

        Raises:
            ValueError: if a point is off the lattice or outside the box.
        """
        f = (np.asarray(xyz, dtype=np.float64) - np.asarray(self.origin)) / self.dx
        i = np.rint(f)
        if f.size and np.max(np.abs(f - i)) > 1e-3:
            raise ValueError(f"level {self.level}: points are not on a lattice of spacing {self.dx}")
        i = i.astype(np.int64)
        if f.size and (np.any(i < 0) or np.any(i >= np.asarray(self.shape))):
            raise ValueError(f"level {self.level}: points outside the lattice box")
        return i

    def flat(self, xyz: np.ndarray) -> np.ndarray:
        """C-order flat index of points (see :meth:`index`)."""
        i = self.index(xyz)
        return np.ravel_multi_index((i[:, 0], i[:, 1], i[:, 2]), self.shape)

    def centres(self, idx: np.ndarray) -> np.ndarray:
        """Positions of integer lattice indices [n, 3]."""
        return np.asarray(self.origin) + np.asarray(idx, dtype=np.float64) * self.dx


def level_lattice(source, level: int, chunk: int = 4_000_000) -> LevelLattice:
    """Lattice (origin, shape, spacing) of one level of a source, from its points.

    The spacing is ``source.spacing(level)`` or, if None, the smallest gap
    between distinct coordinates within a chunk of rows.
    """
    n = source.n_points(level)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    dx = source.spacing(level)
    step = np.inf
    for a in range(0, n, chunk):
        p = np.asarray(source.read(level, a, min(a + chunk, n), ("xyz",))["xyz"], dtype=np.float64)
        lo = np.minimum(lo, p.min(0))
        hi = np.maximum(hi, p.max(0))
        if dx is None:
            for k in range(3):
                u = np.unique(p[:, k])
                if u.size > 1:
                    step = min(step, float(np.min(np.diff(u))))
    if dx is None:
        if not np.isfinite(step):
            raise ValueError(f"level {level}: cannot infer the spacing from a single point")
        dx = step
    shape = tuple(int(s) for s in np.rint((hi - lo) / dx).astype(np.int64) + 1)
    return LevelLattice(int(level), float(dx), tuple(float(v) for v in lo), shape)


# --------------------------------------------------------------------------- sources


class ArraySource:
    """Point data held in memory.

    Args:
        xyz (np.ndarray): [N, 3] positions.
        level (np.ndarray): [N] refinement level of each point.
        labels (np.ndarray): [N] cluster labels.
        features (dict[str, np.ndarray], optional): name -> [N] values.
        spacing (dict[int, float], optional): level -> dx; inferred if missing.
        attrs (dict, optional): metadata (e.g. ``horizon_centres``, ``horizon_mass``,
            ``iteration``, ``time``).
    """

    def __init__(self, xyz, level, labels, features=None, spacing=None, attrs=None):
        self.xyz = np.asarray(xyz)
        level = np.asarray(level)
        self.labels = np.asarray(labels)
        self.features = {k: np.asarray(v) for k, v in (features or {}).items()}
        self._spacing = dict(spacing or {})
        self.attrs = dict(attrs or {})
        self.levels = [int(v) for v in np.unique(level)]
        self._rows = {L: np.flatnonzero(level == L) for L in self.levels}

    def n_points(self, level: int) -> int:
        return int(self._rows[level].size)

    def spacing(self, level: int):
        return self._spacing.get(level)

    def has(self, name: str) -> bool:
        if name == "log_sigma" and name not in self.features:
            return all(n in self.features for n in SIGMA_INPUTS)
        return name in self.features

    def read(self, level: int, start: int, stop: int, names) -> dict:
        rows = self._rows[level][start:stop]
        out = {}
        for name in names:
            if name == "xyz":
                out[name] = self.xyz[rows]
            elif name == "labels":
                out[name] = self.labels[rows]
            elif name in self.features:
                out[name] = self.features[name][rows]
            elif name == "log_sigma":
                out[name] = _log_sigma(self.features["log_beta_inv"][rows], self.features["log_theta"][rows])
            else:
                raise KeyError(name)
        return out


class GridFileSource:
    """Grid HDF5 file from ``etkplot som-features --grid-out`` plus labels.

    The file holds ``xyz`` [N, 3], ``level`` [N], ``features`` [N, F] and ``names``;
    root attributes give the horizons (``horizon_centres``, ``horizon_mass``),
    ``iteration`` and ``time``. Rows of one level should be contiguous (they are
    in files from etkplot); otherwise each level's rows are gathered from the
    span that holds them.

    Args:
        path (str | Path): Grid file.
        labels (np.ndarray | str | Path | None): [N] labels, or a ``.npy`` path
            (memory-mapped). May be None when only geometry or features are read.
        spacing (dict[int, float], optional): level -> dx; inferred if missing.
    """

    def __init__(self, path, labels=None, spacing=None):
        import h5py

        self.path = Path(path)
        self.h5 = h5py.File(self.path, "r")
        level = self.h5["level"][:]
        self.levels = [int(v) for v in np.unique(level)]
        self._rows = {}
        for L in self.levels:
            rows = np.flatnonzero(level == L)
            if rows[-1] - rows[0] + 1 == rows.size:
                self._rows[L] = (int(rows[0]), int(rows[-1]) + 1, None)
            else:
                self._rows[L] = (int(rows[0]), int(rows[-1]) + 1, rows - rows[0])
        del level
        self.names = [n.decode() if isinstance(n, bytes) else str(n) for n in self.h5["names"][()]]
        self.attrs = {k: _plain(v) for k, v in self.h5.attrs.items()}
        self._spacing = dict(spacing or {})
        self._owned_labels = False
        if isinstance(labels, (str, Path)):
            from .npy_parts import open_npy

            labels = open_npy(labels, mmap_mode="r")
            self._owned_labels = True
        if labels is not None and len(labels) != self.h5["level"].shape[0]:
            if self._owned_labels and hasattr(labels, "close"):
                labels.close()
            raise ValueError(f"{len(labels)} labels for {self.h5['level'].shape[0]} grid points")
        self.labels = labels

    @property
    def n_total(self) -> int:
        return int(self.h5["level"].shape[0])

    def n_points(self, level: int) -> int:
        a, b, rows = self._rows[level]
        return b - a if rows is None else int(rows.size)

    def spacing(self, level: int):
        return self._spacing.get(level)

    def has(self, name: str) -> bool:
        if name == "log_sigma" and name not in self.names:
            return all(n in self.names for n in SIGMA_INPUTS)
        return name in self.names

    def row_range(self, level: int, start: int, stop: int):
        """Absolute row slice (or index array) of a level's rows [start, stop)."""
        a, b, rows = self._rows[level]
        if rows is None:
            return slice(a + start, a + stop)
        return a + rows[start:stop]

    def _get(self, dset, sel):
        if isinstance(sel, slice):
            return dset[sel]
        lo, hi = int(sel[0]), int(sel[-1]) + 1
        return dset[lo:hi][sel - lo]

    def read(self, level: int, start: int, stop: int, names) -> dict:
        sel = self.row_range(level, start, stop)
        out = {}
        feats = None
        for name in names:
            if name == "xyz":
                out[name] = self._get(self.h5["xyz"], sel)
            elif name == "labels":
                if self.labels is None:
                    raise ValueError("no labels given")
                out[name] = np.asarray(self._get(self.labels, sel))
            elif name in self.names or name == "log_sigma":
                if feats is None:
                    feats = self._get(self.h5["features"], sel)
                if name in self.names:
                    out[name] = feats[:, self.names.index(name)]
                else:
                    i, j = (self.names.index(n) for n in SIGMA_INPUTS)
                    out[name] = _log_sigma(feats[:, i], feats[:, j])
            else:
                raise KeyError(name)
        return out

    def close(self):
        if getattr(self, "_owned_labels", False) and hasattr(self.labels, "close"):
            self.labels.close()
        self.h5.close()


def _plain(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "S":
            return [x.decode() for x in v.ravel()]
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


# --------------------------------------------------------------------------- helpers


def weighted_percentile(values, weights, q) -> np.ndarray:
    """Weighted percentiles (inverted CDF of the weights, no interpolation).

    Args:
        values (array): Sample values.
        weights (array): Non-negative weights.
        q (float | array): Percentiles in [0, 100].

    Returns:
        np.ndarray: Smallest value whose cumulative weight reaches q% (NaN if empty).
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), values.shape).ravel()
    q = np.atleast_1d(np.asarray(q, dtype=np.float64))
    ok = np.isfinite(values) & (weights > 0)
    if not ok.any():
        return np.full(q.shape, np.nan)
    v, w = values[ok], weights[ok]
    o = np.argsort(v, kind="stable")
    v, cw = v[o], np.cumsum(w[o])
    i = np.searchsorted(cw, q / 100.0 * cw[-1], side="left")
    return v[np.clip(i, 0, v.size - 1)]


def _hist_percentiles(counts, edges, q):
    """Percentiles from binned weights, linear within the bin."""
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum()
    if tot <= 0:
        return [float("nan")] * len(q)
    cdf = np.concatenate([[0.0], np.cumsum(counts)]) / tot
    return [float(np.interp(p / 100.0, cdf, edges)) for p in q]


def _structure(connectivity: int) -> np.ndarray:
    rank = {6: 1, 18: 2, 26: 3}.get(connectivity)
    if rank is None:
        raise ValueError("connectivity must be 6, 18 or 26")
    return ndimage.generate_binary_structure(3, rank)


def _bins(v, lo, width, nb):
    i = np.floor((np.asarray(v, dtype=np.float64) - lo) / width)
    i = np.nan_to_num(i, nan=-1, posinf=nb - 1, neginf=0)
    return np.clip(i, 0, nb - 1).astype(np.int64)


def _interface_pairs(fine: LevelLattice, owned_f, comp_f, coarse: LevelLattice, owned_c, comp_c,
                     connectivity: int = 6) -> np.ndarray:
    """(fine component, coarse component) pairs whose cells touch across the
    edge of the fine level's owned region, under ``connectivity`` and the
    cross-level rule of :mod:`aweSOM.amr_adjacency`."""
    shape = owned_f.shape
    inner = ndimage.binary_erosion(owned_f, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    cells = np.flatnonzero((comp_f > 0) & ~inner)  # sheet cells with a non-owned position next to them
    del inner
    if cells.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    a = OwnedLevel(fine.dx, fine.origin, owned_f)
    b = OwnedLevel(coarse.dx, coarse.origin, owned_c)
    i3 = np.stack(np.unravel_index(cells, shape), axis=1)
    p = fine.centres(i3)
    f = b.fractional(p)
    lo = np.ceil(f - 1 - 1e-3).astype(np.int64)  # coarse cells within the coarse spacing
    hi = np.floor(f + 1 + 1e-3).astype(np.int64)
    cf = comp_f.ravel()[cells]
    cshape = np.asarray(coarse.shape)
    out = []
    for da in range(3):
        for db in range(3):
            for dc in range(3):
                idx = lo + np.array([da, db, dc])
                ok = np.all((idx <= hi) & (idx >= 0) & (idx < cshape), axis=1)
                if not ok.any():
                    continue
                cc = comp_c[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
                keep = cc > 0
                if not keep.any():
                    continue
                rows = np.flatnonzero(ok)[keep]
                t = touching(p[rows], coarse.centres(idx[rows]), a, b, connectivity)
                if t.any():
                    out.append(np.stack([cf[rows][t], cc[keep][t]], axis=1))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.concatenate(out).astype(np.int64), axis=0)


def _axis_map(xs, origin, dx, n, finer):
    f = (xs - origin) / dx
    if finer:
        i = np.rint(f)
        ok = np.abs(f - i) < 1e-6
    else:
        i = np.floor(f + 0.5 + 1e-9)
        ok = np.ones(xs.shape, dtype=bool)
    i = i.astype(np.int64)
    ok &= (i >= 0) & (i < n)
    return i, ok


def _composite_sheet(L: LevelLattice, margin: int, lattices, sheets, owned) -> np.ndarray:
    """Sheet mask on level L's box padded by ``margin`` cells, each point taking
    the value (sheet or background) of the finest level that owns it: coarser
    levels nearest-neighbour, finer levels at coincident points. Points no
    level owns are background."""
    shape = tuple(s + 2 * margin for s in L.shape)
    origin = np.asarray(L.origin) - margin * L.dx
    out = np.zeros(shape, dtype=bool)
    for K, latK in sorted(lattices.items(), key=lambda kv: -kv[1].dx):  # coarsest first, finest wins
        maps = []
        for a in range(3):
            xs = origin[a] + np.arange(shape[a]) * L.dx
            i, ok = _axis_map(xs, latK.origin[a], latK.dx, latK.shape[a], finer=latK.dx < L.dx * (1 - 1e-9))
            maps.append((np.flatnonzero(ok), i[ok]))
        if any(m[0].size == 0 for m in maps):
            continue
        (wx, ix), (wy, iy), (wz, iz) = maps
        src = np.ix_(ix, iy, iz)
        dst = np.ix_(wx, wy, wz)
        own = owned[K][src]
        block = out[dst]
        block[own] = sheets[K][src][own]
        out[dst] = block
        del block, own
    return out


def _rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


# --------------------------------------------------------------------------- main entry


def sheet_statistics(
    source,
    sheet_clusters,
    holes=None,
    masses=None,
    config: SheetConfig | None = None,
    log: Callable[[str], None] | None = print,
) -> SheetResult:
    """Current-sheet statistics of one snapshot.

    Args:
        source: :class:`GridFileSource`, :class:`ArraySource` or any object with
            ``levels``, ``n_points(level)``, ``spacing(level)``, ``has(name)``,
            ``read(level, start, stop, names)`` and ``attrs``.
        sheet_clusters (iterable[int]): Labels that count as sheet.
        holes (array, optional): [2, 3] hole positions; default
            ``source.attrs["horizon_centres"]``. Without holes the radial and
            between-holes statistics are skipped.
        masses (array, optional): [2] masses for the centre of mass; default
            ``source.attrs["horizon_mass"]`` or equal.
        config (SheetConfig, optional): Parameters.
        log (callable, optional): Progress printer (None: silent).

    Returns:
        SheetResult: summary, per-component table and slices.
    """
    cfg = config or SheetConfig()
    say = log or (lambda s: None)
    clusters = np.asarray(sorted({int(c) for c in sheet_clusters}), dtype=np.int64)
    t_start = time.perf_counter()
    timings = {}
    chunk = int(cfg.chunk_rows)
    levels = sorted(int(L) for L in source.levels)
    attrs = getattr(source, "attrs", {}) or {}

    if holes is None and "horizon_centres" in attrs:
        holes = attrs["horizon_centres"]
    holes = None if holes is None else np.asarray(holes, dtype=np.float64).reshape(-1, 3)
    if holes is not None and len(holes) != 2:
        raise ValueError("need exactly two hole positions")
    if holes is not None:
        if masses is None:
            masses = attrs.get("horizon_mass", [1.0, 1.0])
        masses = np.asarray(masses, dtype=np.float64)
        com = (masses[:, None] * holes).sum(0) / masses.sum()
        centres = {"hole1": holes[0], "hole2": holes[1], "com": com}
        seg = holes[1] - holes[0]
        seg_len = float(np.linalg.norm(seg))
        u = seg / seg_len
        ez = np.array([0.0, 0.0, 1.0])
        v = np.cross(ez, u)
        if np.linalg.norm(v) < 1e-12:  # segment along z
            v = np.array([1.0, 0.0, 0.0])
        v /= np.linalg.norm(v)
        wv = np.cross(u, v)
    hist_feats = [n for n in cfg.feature_bins if source.has(n)]
    comp_feats = [n for n in COMPONENT_FEATURES if source.has(n)]
    bg_name = cfg.plane_background if source.has(cfg.plane_background) else None

    # ---- pass 1: lattices, owned and sheet masks
    t0 = time.perf_counter()
    lat, owned, sheet = {}, {}, {}
    for L in levels:
        lat[L] = level_lattice(source, L, chunk)
        own = np.zeros(lat[L].shape, dtype=bool)
        sh = np.zeros(lat[L].shape, dtype=bool)
        n = source.n_points(L)
        for a in range(0, n, chunk):
            d = source.read(L, a, min(a + chunk, n), ("xyz", "labels"))
            flat = lat[L].flat(d["xyz"])
            own.ravel()[flat] = True
            sh.ravel()[flat[np.isin(d["labels"], clusters)]] = True
        if int(own.sum()) != n:
            raise ValueError(f"level {L}: {n - int(own.sum())} duplicate points")
        owned[L], sheet[L] = own, sh
        say(f"[sheet_stats] level {L}: dx {lat[L].dx:g}, box {lat[L].shape}, {n} points, "
            f"{int(sh.sum())} sheet cells")
    timings["masks"] = time.perf_counter() - t0

    # ---- connected components per level, then across level interfaces
    t0 = time.perf_counter()
    struct = _structure(cfg.connectivity)
    comp, ncomp, offs = {}, {}, {}
    total = 0
    for L in levels:
        comp[L], ncomp[L] = ndimage.label(sheet[L], structure=struct, output=np.int32)
        offs[L] = total
        total += ncomp[L]
    rows, cols = [], []
    for i, L in enumerate(levels):
        for K in levels[:i]:
            pairs = _interface_pairs(lat[L], owned[L], comp[L], lat[K], owned[K], comp[K], cfg.connectivity)
            if pairs.size:
                rows.append(offs[L] + pairs[:, 0] - 1)
                cols.append(offs[K] + pairs[:, 1] - 1)
    n_links = int(sum(r.size for r in rows))
    if total == 0:
        root, n_comp = np.zeros(0, dtype=np.int64), 0
    else:
        r = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        c = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
        graph = coo_matrix((np.ones(r.size, dtype=np.int8), (r, c)), shape=(total, total))
        n_comp, root = connected_components(graph, directed=False)
        root = root.astype(np.int64)
    say(f"[sheet_stats] components: {total} per-level, {n_links} interface links, {n_comp} joined")
    timings["components"] = time.perf_counter() - t0

    def merged(L, local):
        return root[offs[L] + local.astype(np.int64) - 1]

    # ---- distance-transform thickness at ridge cells
    t0 = time.perf_counter()
    ridge_id, ridge_t, ridge_dx = [], [], []
    m = int(cfg.edt_margin)
    h = int(cfg.edt_overlap)
    n_deep = 0
    for L in levels:
        if ncomp[L] == 0:
            continue
        comp_mask = _composite_sheet(lat[L], m, lat, sheet, owned)
        nx, ny, nz = lat[L].shape
        bx = int(cfg.edt_block) if cfg.edt_block else nx
        for x0 in range(0, nx, bx):
            x1 = min(x0 + bx, nx)
            a = 0 if bx >= nx else max(x0 + m - h, 0)
            b = nx + 2 * m if bx >= nx else min(x1 + m + h, nx + 2 * m)
            sub = comp_mask[a:b]
            if sub.all():  # no background anywhere in the block: distances undefined
                dist = np.full(sub.shape, np.inf, dtype=np.float32)
            else:
                dist = ndimage.distance_transform_edt(sub, sampling=lat[L].dx).astype(np.float32)
            mx = ndimage.maximum_filter(dist, size=3, mode="constant", cval=0.0)
            core = (slice(x0 + m - a, x1 + m - a), slice(m, m + ny), slice(m, m + nz))
            d_in = dist[core]
            ridge = sheet[L][x0:x1] & (d_in >= mx[core]) & (d_in > 0)
            del mx
            idx = np.flatnonzero(ridge)
            d_r = d_in.ravel()[idx].astype(np.float64)
            if bx < nx:
                n_deep += int(np.sum(d_r > h * lat[L].dx))
            ridge_id.append(merged(L, comp[L][x0:x1].ravel()[idx]))
            ridge_t.append(2.0 * d_r - 0.5 * lat[L].dx)
            ridge_dx.append(np.full(idx.size, lat[L].dx))
            del dist, d_in, ridge, sub
        del comp_mask
    ridge_id = np.concatenate(ridge_id) if ridge_id else np.zeros(0, dtype=np.int64)
    ridge_t = np.concatenate(ridge_t) if ridge_t else np.zeros(0)
    ridge_dx = np.concatenate(ridge_dx) if ridge_dx else np.zeros(0)
    timings["distance_transform"] = time.perf_counter() - t0

    # ---- pass 2: per-point statistics
    t0 = time.perf_counter()
    fb = {n: cfg.feature_bins[n] for n in hist_feats}
    nbins = {n: int(round((hi - lo) / w)) for n, (lo, hi, w) in fb.items()}
    h_in = {n: np.zeros(nbins[n]) for n in hist_feats}
    h_out = {n: np.zeros(nbins[n]) for n in hist_feats}
    s_in = {n: [0.0, 0.0] for n in hist_feats}  # sum v dV, sum dV (finite values)
    s_out = {n: [0.0, 0.0] for n in hist_feats}
    n_rad = int(math.ceil(cfg.radial_max / cfg.radial_step))
    rad_tot = {k: np.zeros(n_rad) for k in (centres if holes is not None else {})}
    rad_sh = {k: np.zeros(n_rad) for k in rad_tot}
    widths = [float(w) for w in cfg.between_widths]
    if cfg.between_default not in widths:
        widths.append(float(cfg.between_default))
    btw_tot = np.zeros(len(widths))
    btw_sh = np.zeros(len(widths))
    seg_edges = np.linspace(-0.5, 1.5, cfg.segment_bins + 1)
    seg_tot = np.zeros(cfg.segment_bins)
    seg_sh = np.zeros(cfg.segment_bins)
    per_level = {}
    cells = {"id": [], "pos": [], "level": [], "between": [], "edge": []}
    for name in comp_feats:
        cells[name] = []
    planes = {}
    coarsest = levels[0]
    names_needed = ["xyz", "labels"] + sorted(set(hist_feats) | set(comp_feats) | ({bg_name} - {None}))
    for L in levels:
        la = lat[L]
        vol = la.dx ** 3
        n = source.n_points(L)
        n_sheet = 0
        k0 = None
        if cfg.plane_z is not None:
            f = (cfg.plane_z - la.origin[2]) / la.dx
            if abs(f - round(f)) < 1e-6 and 0 <= round(f) < la.shape[2]:
                k0 = int(round(f))
                planes[L] = {
                    "sheet": np.full(la.shape[:2], -1, dtype=np.int8),
                    "background": np.full(la.shape[:2], np.nan, dtype=np.float32),
                    "extent": [la.origin[0] - la.dx / 2, la.origin[0] + (la.shape[0] - 0.5) * la.dx,
                               la.origin[1] - la.dx / 2, la.origin[1] + (la.shape[1] - 0.5) * la.dx],
                }
        for a in range(0, n, chunk):
            d = source.read(L, a, min(a + chunk, n), names_needed)
            pos = np.asarray(d["xyz"], dtype=np.float64)
            i3 = la.index(pos)
            pos = la.centres(i3)  # exact lattice positions
            ins = np.isin(d["labels"], clusters)
            n_sheet += int(ins.sum())
            for name in hist_feats:
                lo, hi, w = fb[name]
                val = np.asarray(d[name], dtype=np.float64)
                fin = np.isfinite(val)
                b = _bins(val, lo, w, nbins[name])
                h_in[name] += np.bincount(b[ins & fin], minlength=nbins[name]) * vol
                h_out[name] += np.bincount(b[~ins & fin], minlength=nbins[name]) * vol
                s_in[name][0] += float(val[ins & fin].sum()) * vol
                s_in[name][1] += float((ins & fin).sum()) * vol
                s_out[name][0] += float(val[~ins & fin].sum()) * vol
                s_out[name][1] += float((~ins & fin).sum()) * vol
            if holes is not None:
                for key, c in centres.items():
                    r = np.sqrt(((pos - c) ** 2).sum(1))
                    b = np.minimum((r / cfg.radial_step).astype(np.int64), n_rad - 1)
                    rad_tot[key] += np.bincount(b, minlength=n_rad) * vol
                    rad_sh[key] += np.bincount(b[ins], minlength=n_rad) * vol
                rel = pos - holes[0]
                s = rel @ u / seg_len
                pa = np.abs(rel @ v)
                pb = np.abs(rel @ wv)
                inside_w = []
                for j, w in enumerate(widths):
                    reg = (s >= 0) & (s <= 1) & (pa <= w) & (pb <= w)
                    btw_tot[j] += reg.sum() * vol
                    btw_sh[j] += (reg & ins).sum() * vol
                    inside_w.append(reg)
                reg_def = inside_w[widths.index(float(cfg.between_default))]
                band = (pa <= cfg.between_default) & (pb <= cfg.between_default)
                sb = np.searchsorted(seg_edges, s, side="right") - 1
                okb = band & (sb >= 0) & (sb < cfg.segment_bins)
                seg_tot += np.bincount(sb[okb], minlength=cfg.segment_bins) * vol
                seg_sh += np.bincount(sb[okb & ins], minlength=cfg.segment_bins) * vol
            else:
                reg_def = np.zeros(pos.shape[0], dtype=bool)
            if k0 is not None:
                on = i3[:, 2] == k0
                planes[L]["sheet"][i3[on, 0], i3[on, 1]] = ins[on].astype(np.int8)
                if bg_name is not None:
                    planes[L]["background"][i3[on, 0], i3[on, 1]] = d[bg_name][on]
            if ins.any():
                flat = np.ravel_multi_index((i3[ins, 0], i3[ins, 1], i3[ins, 2]), la.shape)
                cells["id"].append(merged(L, comp[L].ravel()[flat]))
                cells["pos"].append(pos[ins])
                cells["level"].append(np.full(int(ins.sum()), L, dtype=np.int8))
                cells["between"].append(reg_def[ins])
                if L == coarsest:
                    e = np.any((i3[ins] == 0) | (i3[ins] == np.asarray(la.shape) - 1), axis=1)
                else:
                    e = np.zeros(int(ins.sum()), dtype=bool)
                cells["edge"].append(e)
                for name in comp_feats:
                    cells[name].append(np.asarray(d[name][ins], dtype=np.float32))
        per_level[L] = {
            "dx": la.dx,
            "n_points": n,
            "n_sheet_points": n_sheet,
            "owned_volume": n * vol,
            "sheet_volume": n_sheet * vol,
            "sheet_fraction": n_sheet / n if n else float("nan"),
            "box_shape": list(la.shape),
            "box_origin": list(la.origin),
            "n_components_on_level": int(ncomp[L]),
        }
        say(f"[sheet_stats] level {L}: sheet fraction {per_level[L]['sheet_fraction']:.4%} "
            f"({time.perf_counter() - t0:.0f}s, peak RSS {_rss_gb():.2f} GB)")
    timings["point_pass"] = time.perf_counter() - t0
    del comp, owned, sheet

    # ---- per-component table
    t0 = time.perf_counter()
    for k in list(cells):
        if cells[k]:
            cells[k] = np.concatenate(cells[k])
        else:
            cells[k] = np.zeros((0, 3)) if k == "pos" else np.zeros(0)
    dxc = np.zeros(cells["level"].shape)
    for L in levels:
        dxc[cells["level"] == L] = lat[L].dx
    w = dxc ** 3
    ids = cells["id"].astype(np.int64)
    nc = int(n_comp)
    V = np.bincount(ids, weights=w, minlength=nc).astype(np.float64)
    cen = np.stack([np.bincount(ids, weights=w * cells["pos"][:, k], minlength=nc) for k in range(3)], 1)
    cen = cen.astype(np.float64).reshape(nc, 3)
    cen /= np.maximum(V, 1e-300)[:, None]
    dpos = cells["pos"] - cen[ids]
    C = np.empty((nc, 3, 3))
    cube = np.bincount(ids, weights=w * dxc ** 2, minlength=nc) / 12.0
    for i in range(3):
        for j in range(i, 3):
            C[:, i, j] = np.bincount(ids, weights=w * dpos[:, i] * dpos[:, j], minlength=nc)
            if i == j:
                C[:, i, j] += cube
            C[:, j, i] = C[:, i, j]
    C /= np.maximum(V, 1e-300)[:, None, None]
    lam = np.linalg.eigvalsh(C) if nc else np.zeros((0, 3))
    ext = np.sqrt(12.0 * np.clip(lam, 0.0, None))[:, ::-1]  # descending
    del dpos, C
    order = np.argsort(-V, kind="stable")
    rank = np.empty(nc, dtype=np.int64)
    rank[order] = np.arange(nc)
    table = {"id": np.arange(nc)}
    table["n_cells"] = np.bincount(ids, minlength=nc)[order]
    for L in levels:
        table[f"n_cells_l{L}"] = np.bincount(ids[cells["level"] == L], minlength=nc)[order]
    table["volume"] = V[order]
    for k, ax in enumerate("xyz"):
        table[ax] = cen[order, k]
    if holes is not None:
        for key, c in centres.items():
            table[f"r_{key}"] = np.sqrt(((cen[order] - c) ** 2).sum(1))
    for k in range(3):
        table[f"extent_{k + 1}"] = ext[order, k]
    table["thickness_inertia"] = ext[order, 2]
    # EDT ridge thickness: median per component
    t_edt = np.full(nc, np.nan)
    if ridge_id.size:
        o = np.lexsort((ridge_t, ridge_id))
        rid, rt = ridge_id[o], ridge_t[o]
        uid, start, cnt = np.unique(rid, return_index=True, return_counts=True)
        t_edt[uid] = 0.5 * (rt[start + (cnt - 1) // 2] + rt[start + cnt // 2])
    table["thickness_edt"] = t_edt[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        table["aspect_12"] = ext[order, 0] / ext[order, 1]
        table["aspect_13"] = ext[order, 0] / ext[order, 2]
        table["aspect_23"] = ext[order, 1] / ext[order, 2]
    for name in comp_feats:
        table[f"mean_{name}"] = (np.bincount(ids, weights=w * cells[name], minlength=nc) / np.maximum(V, 1e-300))[order]
    table["between_fraction"] = (np.bincount(ids, weights=w * cells["between"], minlength=nc) / np.maximum(V, 1e-300))[order]
    table["touches_outer_boundary"] = (np.bincount(ids, weights=cells["edge"].astype(np.float64), minlength=nc) > 0)[order].astype(np.int8)
    timings["components_table"] = time.perf_counter() - t0

    # ---- summary
    total_vol = sum(p["owned_volume"] for p in per_level.values())
    sheet_vol = sum(p["sheet_volume"] for p in per_level.values())
    Vs = table["volume"]
    resolved = table["n_cells"] >= cfg.min_cells
    vlo, vhi, vd = cfg.volume_bins
    v_edges = 10.0 ** np.arange(vlo, vhi + 1e-9, vd)
    tlo, thi, td = cfg.thickness_bins
    t_edges = 10.0 ** np.arange(tlo, thi + 1e-9, td)
    qs = [5, 25, 50, 75, 95, 99]

    def vw_median(vals):
        return float(weighted_percentile(vals, vals, 50)[0]) if vals.size else float("nan")

    comps = {
        "n_components": nc,
        "n_resolved": int(resolved.sum()),
        "min_cells_resolved": cfg.min_cells,
        "connectivity": cfg.connectivity,
        "n_interface_links": n_links,
        "n_above_volume": {f"{t:g}": int((Vs > t).sum()) for t in cfg.count_thresholds},
        "volume_percentiles": dict(zip(map(str, qs), np.percentile(Vs, qs).tolist())) if nc else {},
        "volume_weighted_median": vw_median(Vs),
        "largest_volume": float(Vs[0]) if nc else 0.0,
        "largest_fraction_of_sheet_volume": float(Vs[0] / sheet_vol) if nc and sheet_vol > 0 else float("nan"),
        "top10_fraction_of_sheet_volume": float(Vs[:10].sum() / sheet_vol) if nc and sheet_vol > 0 else float("nan"),
        "volume_hist": {
            "edges": v_edges.tolist(),
            "count": np.histogram(Vs, bins=v_edges)[0].tolist(),
            "volume": np.histogram(Vs, bins=v_edges, weights=Vs)[0].tolist(),
        },
    }
    area_w = ridge_dx ** 2
    thick = {
        "edt": {
            "definition": "2 d - dx/2 at ridge cells (local maxima of the distance d to the nearest non-sheet cell)",
            "n_ridge_cells": int(ridge_t.size),
            "n_ridge_beyond_overlap": n_deep,
            "median_area_weighted": float(weighted_percentile(ridge_t, area_w, 50)[0]),
            "percentiles_area_weighted": dict(zip(map(str, qs), weighted_percentile(ridge_t, area_w, qs).tolist())),
            "per_level": {
                str(L): {
                    "median": float(np.median(ridge_t[ridge_dx == lat[L].dx])) if np.any(ridge_dx == lat[L].dx) else float("nan"),
                    "median_in_cells": float(np.median(ridge_t[ridge_dx == lat[L].dx]) / lat[L].dx) if np.any(ridge_dx == lat[L].dx) else float("nan"),
                    "n_ridge_cells": int(np.sum(ridge_dx == lat[L].dx)),
                }
                for L in levels
            },
            "hist": {"edges": t_edges.tolist(),
                     "area": np.histogram(ridge_t, bins=t_edges, weights=area_w)[0].tolist()},
            "component_median": float(np.nanmedian(table["thickness_edt"][resolved])) if resolved.any() else float("nan"),
            "component_median_volume_weighted": float(weighted_percentile(table["thickness_edt"][resolved], Vs[resolved], 50)[0]) if resolved.any() else float("nan"),
        },
        "inertia": {
            "definition": "smallest principal extent sqrt(12 lambda_min) of components with >= min_cells cells",
            "median": float(np.median(table["thickness_inertia"][resolved])) if resolved.any() else float("nan"),
            "median_volume_weighted": float(weighted_percentile(table["thickness_inertia"][resolved], Vs[resolved], 50)[0]) if resolved.any() else float("nan"),
            "hist": {"edges": t_edges.tolist(),
                     "count": np.histogram(table["thickness_inertia"][resolved], bins=t_edges)[0].tolist()},
        },
        "aspect": {
            "median_13": float(np.median(table["aspect_13"][resolved])) if resolved.any() else float("nan"),
            "median_23": float(np.median(table["aspect_23"][resolved])) if resolved.any() else float("nan"),
            "median_12": float(np.median(table["aspect_12"][resolved])) if resolved.any() else float("nan"),
        },
    }
    feats = {}
    for name in hist_feats:
        lo, hi, wd = fb[name]
        edges = np.linspace(lo, lo + nbins[name] * wd, nbins[name] + 1)
        feats[name] = {
            "edges": edges.tolist(),
            "inside_volume": h_in[name].tolist(),
            "outside_volume": h_out[name].tolist(),
            "inside_mean": s_in[name][0] / s_in[name][1] if s_in[name][1] else float("nan"),
            "outside_mean": s_out[name][0] / s_out[name][1] if s_out[name][1] else float("nan"),
            "inside_percentiles": dict(zip(map(str, qs), _hist_percentiles(h_in[name], edges, qs))),
            "outside_percentiles": dict(zip(map(str, qs), _hist_percentiles(h_out[name], edges, qs))),
        }
    summary = {
        "sheet_clusters": clusters.tolist(),
        "attrs": {k: attrs[k] for k in ("iteration", "time", "segment", "horizon_centres", "horizon_mass",
                                        "horizon_radius", "b2_min", "s_v") if k in attrs},
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "volume": {
            "total_owned": total_vol,
            "sheet": sheet_vol,
            "sheet_fraction": sheet_vol / total_vol if total_vol else float("nan"),
            "per_level": {str(L): p for L, p in per_level.items()},
        },
        "components": comps,
        "thickness": thick,
        "features": feats,
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
    }
    if holes is not None:
        r_edges = np.arange(n_rad + 1) * cfg.radial_step
        summary["holes"] = {"positions": holes.tolist(), "masses": masses.tolist(), "com": com.tolist(),
                            "separation": seg_len}
        summary["radial"] = {
            "edges": r_edges.tolist(),
            "note": "last bin holds everything beyond its inner edge",
            **{k: {"total_volume": rad_tot[k].tolist(), "sheet_volume": rad_sh[k].tolist()} for k in rad_tot},
        }
        summary["between"] = {
            "definition": "0 <= s <= 1 along hole1->hole2, |a| <= w in the orbital plane and |b| <= w along the orbital axis",
            "widths": {
                f"{w:g}": {
                    "region_volume": float(btw_tot[j]),
                    "region_fraction_of_total": float(btw_tot[j] / total_vol) if total_vol else float("nan"),
                    "sheet_volume": float(btw_sh[j]),
                    "fraction_of_sheet_volume": float(btw_sh[j] / sheet_vol) if sheet_vol else float("nan"),
                    "sheet_filling": float(btw_sh[j] / btw_tot[j]) if btw_tot[j] else float("nan"),
                    "enhancement": float((btw_sh[j] / btw_tot[j]) / (sheet_vol / total_vol)) if btw_tot[j] and sheet_vol else float("nan"),
                }
                for j, w in enumerate(widths)
            },
            "default_width": float(cfg.between_default),
            "segment_profile": {"s_edges": seg_edges.tolist(), "total_volume": seg_tot.tolist(),
                                "sheet_volume": seg_sh.tolist()},
        }
    timings["total"] = time.perf_counter() - t_start
    summary["timings_s"] = {k: round(v, 2) for k, v in timings.items()}
    summary["peak_rss_gb"] = round(_rss_gb(), 3)
    say(f"[sheet_stats] done in {timings['total']:.0f}s: sheet fraction {summary['volume']['sheet_fraction']:.4%}, "
        f"{nc} components, peak RSS {summary['peak_rss_gb']:.2f} GB")
    return SheetResult(summary=summary, components=table, planes=planes)


def write_outputs(result: SheetResult, out_dir, name: str) -> dict:
    """Write ``<name>_sheet_stats.json``, ``<name>_components.csv`` and
    ``<name>_planes.npz`` into ``out_dir``.

    Args:
        result (SheetResult): Output of :func:`sheet_statistics`.
        out_dir (str | Path): Directory (created if needed).
        name (str): Run name used as file prefix.

    Returns:
        dict: Paths written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out / f"{name}_sheet_stats.json",
        "csv": out / f"{name}_components.csv",
        "planes": out / f"{name}_planes.npz",
    }
    paths["json"].write_text(json.dumps(result.summary, indent=1, default=_plain) + "\n")
    cols = list(result.components)
    with open(paths["csv"], "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        arrays = [result.components[c] for c in cols]
        fmt = [(lambda x: str(int(x))) if np.issubdtype(np.asarray(a).dtype, np.integer) else (lambda x: f"{x:.6g}")
               for a in arrays]
        for i in range(len(arrays[0]) if arrays else 0):
            wr.writerow([f(a[i]) for f, a in zip(fmt, arrays)])
    np.savez_compressed(
        paths["planes"],
        **{f"l{L}_{k}": np.asarray(v) for L, p in result.planes.items() for k, v in p.items()},
    )
    return {k: str(v) for k, v in paths.items()}
