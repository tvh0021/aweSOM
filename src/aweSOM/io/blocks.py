"""The block interface shared by the readers.

A *level block* is a box of one refinement level, small enough to hold in
memory, with every input variable on it plus a halo of ``halo`` points on
each side so that centred differences can be taken at every interior point.
Readers yield blocks one at a time, so peak memory is one block (plus a
bounded read cache) whatever the size of the snapshot.

``iter_level_blocks(source, variables, iteration)`` yields
:class:`LevelBlock` objects; each unpacks as the 4-tuple
``(level, block_origin, spacing, {var: ndarray})``, and carries more: which
interior points the level *owns* (the finest level covering them, outside any
excluded region), their coordinates, and a key that names the block so a
driver can skip blocks it has already done.

Arrays are float64 in (x, y, z) order; points with no data are NaN.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["LevelBlock", "iter_level_blocks", "open_reader", "slabs", "ArrayCache", "Sphere"]

Sphere = Tuple[Sequence[float], float]  # (centre xyz, radius)


@dataclass
class LevelBlock:
    """One block of one refinement level.

    Attributes:
        level: refinement level.
        origin: (3,) coordinate of ``data[v][0, 0, 0]`` (halo included).
        spacing: (3,) grid spacing.
        data: variable -> float64 (nx, ny, nz) array, halo included; NaN = no data.
        halo: points of halo on each side, per axis (an int means all three).
            The halo holds real data where the level has it, NaN elsewhere.
        key: name of the block, unique within (source, iteration).
        iteration: the snapshot's iteration.
        time: the snapshot's time (NaN if unknown).
        owned: bool array of the interior's shape: points this level owns.
            None means every interior point.
        component: int array of the interior's shape: the reader's component
            (Carpet component, AMReX box) that holds each owned point, -1 elsewhere.
        xyz_base, xyz_offset: interior point (i, j, k) sits at
            ``xyz_base + (xyz_offset + (i, j, k)) * spacing``; readers set these
            to reproduce their reference tool's coordinates bitwise.
        info: reader-specific extras (counts, provenance).
    """

    level: int
    origin: np.ndarray
    spacing: np.ndarray
    data: Dict[str, np.ndarray]
    halo: object
    key: str
    iteration: int = 0
    time: float = float("nan")
    owned: Optional[np.ndarray] = None
    component: Optional[np.ndarray] = None
    xyz_base: Optional[np.ndarray] = None
    xyz_offset: Optional[np.ndarray] = None
    info: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=np.float64)
        self.spacing = np.asarray(self.spacing, dtype=np.float64)
        self.halo = np.broadcast_to(np.asarray(self.halo, dtype=np.int64), (3,)).copy()
        if self.xyz_base is None:
            self.xyz_base = self.origin + self.halo * self.spacing
            self.xyz_offset = np.zeros(3, np.int64)
        self.xyz_base = np.asarray(self.xyz_base, dtype=np.float64)
        self.xyz_offset = np.asarray(self.xyz_offset, dtype=np.int64)

    def __iter__(self):
        """Unpack as ``(level, origin, spacing, data)``."""
        return iter((self.level, self.origin, self.spacing, self.data))

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the data arrays (halo included)."""
        return next(iter(self.data.values())).shape

    @property
    def interior_shape(self) -> Tuple[int, ...]:
        return tuple(int(n - 2 * h) for n, h in zip(self.shape, self.halo))

    @property
    def interior(self) -> Tuple[slice, ...]:
        """Index of the interior in the data arrays."""
        return tuple(slice(int(h), int(n - h)) for n, h in zip(self.shape, self.halo))

    @property
    def n_owned(self) -> int:
        return int(np.prod(self.interior_shape)) if self.owned is None else int(np.count_nonzero(self.owned))

    @property
    def owned_mask(self) -> np.ndarray:
        """Bool mask of the interior's shape (all True if ``owned`` is None)."""
        if self.owned is None:
            return np.ones(self.interior_shape, bool)
        return self.owned

    def full_mask(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """``mask`` (default: owned) of the interior, embedded in the data's shape."""
        out = np.zeros(self.shape, bool)
        out[self.interior] = self.owned_mask if mask is None else mask
        return out

    def xyz(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """(N, 3) float64 coordinates of the interior points in ``mask`` (default: owned), C order."""
        mask = self.owned_mask if mask is None else mask
        idx = np.stack(np.nonzero(mask), axis=1)
        return self.xyz_base + (self.xyz_offset + idx) * self.spacing


def slabs(n: int, width: int) -> List[Tuple[int, int]]:
    """[a, b) ranges of at most ``width`` points covering 0..n."""
    width = max(int(width), 1)
    return [(a, min(a + width, n)) for a in range(0, n, width)]


def paste(dst: np.ndarray, dst_lo, src: np.ndarray, src_lo, where_finite: bool = False) -> bool:
    """Copy the overlap of ``src`` (first point at index ``src_lo``) into ``dst``
    (first point at ``dst_lo``), both in one integer index space. Returns
    whether they overlapped."""
    dst_lo = np.asarray(dst_lo, dtype=np.int64)
    src_lo = np.asarray(src_lo, dtype=np.int64)
    lo = np.maximum(dst_lo, src_lo)
    hi = np.minimum(dst_lo + dst.shape, src_lo + src.shape)
    if np.any(hi <= lo):
        return False
    d = tuple(slice(a, b) for a, b in zip(lo - dst_lo, hi - dst_lo))
    s = tuple(slice(a, b) for a, b in zip(lo - src_lo, hi - src_lo))
    if where_finite:
        np.copyto(dst[d], src[s], where=~np.isnan(src[s]))
    else:
        dst[d] = src[s]
    return True


class ArrayCache:
    """A least-recently-used cache of arrays with a byte budget.

    Compressed HDF5 datasets stored as one chunk are decompressed whole on
    every read; neighbouring blocks read the same datasets, so readers keep
    the most recent ones here."""

    def __init__(self, max_bytes: float = 2e9):
        self.max_bytes = float(max_bytes)
        self._d: "OrderedDict[object, np.ndarray]" = OrderedDict()
        self.nbytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key, load: Callable[[], np.ndarray]) -> np.ndarray:
        if key in self._d:
            self._d.move_to_end(key)
            self.hits += 1
            return self._d[key]
        self.misses += 1
        a = load()
        if a.nbytes <= self.max_bytes:
            self._d[key] = a
            self.nbytes += a.nbytes
            while self.nbytes > self.max_bytes and len(self._d) > 1:
                _, old = self._d.popitem(last=False)
                self.nbytes -= old.nbytes
        return a

    def clear(self) -> None:
        self._d.clear()
        self.nbytes = 0


def sphere_mask(xyz_base, xyz_offset, spacing, shape, spheres: Sequence[Sphere]) -> np.ndarray:
    """Bool mask (shape) of the points strictly inside any of ``spheres``."""
    out = np.zeros(shape, bool)
    if not spheres:
        return out
    ax = [xyz_base[d] + (xyz_offset[d] + np.arange(shape[d])) * spacing[d] for d in range(3)]
    for c, r in spheres:
        c = np.asarray(c, dtype=float)
        d2 = ((ax[0] - c[0]) ** 2)[:, None, None] + ((ax[1] - c[1]) ** 2)[None, :, None] + ((ax[2] - c[2]) ** 2)[
            None, None, :
        ]
        out |= d2 < float(r) * float(r)
    return out


def open_reader(source, **kwargs):
    """A reader for ``source``: a reader object is returned as is; a path is
    inspected (Carpet ``*.xyz.file_N.h5`` directory or etkplot index ``.json``
    -> :class:`~aweSOM.io.carpet.CarpetHDF5Reader`; CarpetX Silo or openPMD
    output -> :class:`~aweSOM.io.carpetx.CarpetXReader`)."""
    if hasattr(source, "iter_level_blocks"):
        return source
    p = Path(source)
    from .carpet import CarpetHDF5Reader, looks_like_carpet
    from .carpetx import CarpetXReader, looks_like_carpetx

    if looks_like_carpet(p):
        return CarpetHDF5Reader(p, **kwargs)
    if looks_like_carpetx(p):
        return CarpetXReader(p, **kwargs)
    raise ValueError(f"{source}: neither Carpet HDF5 nor CarpetX (Silo / openPMD-HDF5) output")


def iter_level_blocks(source, variables: Sequence[str], iteration: int, **kwargs) -> Iterator[LevelBlock]:
    """Yield the blocks of ``iteration`` with ``variables`` on them.

    Args:
        source: a reader, or a path for :func:`open_reader`.
        variables: input variables, in the canonical (etkplot) names, e.g.
            :data:`aweSOM.features_grmhd.INPUTS`.
        iteration: the snapshot.
        **kwargs: ``levels``, ``halo``, ``slab`` and reader options.

    Yields:
        LevelBlock: unpacks as ``(level, block_origin, spacing, {var: ndarray})``.
    """
    block_kw = {k: kwargs.pop(k) for k in ("levels", "halo", "slab", "skip") if k in kwargs}
    reader = open_reader(source, **kwargs)
    yield from reader.iter_level_blocks(variables, iteration, **block_kw)
