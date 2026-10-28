"""Level blocks from Carpet (CarpetIOHDF5) 3D output.

Carpet writes each refinement level as many components spread over the
``<var>.xyz.file_<N>.h5`` files of a run; each component is a dataset named
``THORN::var it=N tl=0 rl=L c=C`` stored (z, y, x) with attributes
``origin``, ``delta``, ``time``, ``cctk_nghostzones``, ``cctk_bbox``,
``active`` and ``iorigin``.

:class:`CarpetHDF5Reader` yields the same blocks, with the same values, the
same owned points in the same order and the same coordinates, as etkplot's
``som-features`` uses to write its grid files, so labels predicted from a
reader's blocks match labels predicted from an etkplot grid file exactly:

* a level's components (ghost zones stripped, buffer zones kept) are grouped
  into connected boxes ("patches"), sorted by the x coordinate of their
  centre;
* a level owns the points in its components' ``active`` regions that lie in
  no finer level's ``active`` region and outside the excluded spheres (the
  horizons); coordinates are matched with etkplot's tolerance;
* each patch is cut into slabs of ``slab`` x points; a block is one slab
  plus ``halo`` x points on each side (the y and z extent is the whole
  patch, whose buffer zones serve as halo there). Points no component
  covers are NaN.

Two backends do this. ``backend="etkplot"`` calls etkplot itself
(``ownership.level_ownership``, ``somfeatures._read_slab``); ``"native"`` is a
self-contained h5py reimplementation with a bounded cache of decompressed
components (the BBH run stores each dataset as one gzip chunk, so a
hyperslab read decompresses the whole component). ``"auto"`` uses etkplot
when it can be imported. Both are tested to give identical blocks.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from .blocks import ArrayCache, LevelBlock, Sphere, slabs

__all__ = ["CarpetHDF5Reader", "looks_like_carpet", "DEFAULT_SLAB"]

DEFAULT_SLAB = 64  # etkplot.features.SLAB
_TOL = 1e-2  # etkplot.ownership._TOL, in units of the level's spacing
INDEX_FORMAT = "etkplot-index-1"

DATASET_RE = re.compile(
    r"^(?P<thorn>[^:]+)::(?P<var>\S+) it=(?P<it>\d+) tl=(?P<tl>\d+)"
    r"(?: m=(?P<m>\d+))?(?: rl=(?P<rl>\d+))?(?: c=(?P<c>\d+))?$"
)
FILE_RE = re.compile(r"^(?P<var>[A-Za-z0-9_\[\]]+?)\.xyz(?:\.file_(?P<filenum>\d+))?\.h5$")


def looks_like_carpet(path) -> bool:
    """An etkplot index (``.json``) or a directory of ``<var>.xyz[.file_N].h5`` files."""
    p = Path(path)
    if p.suffix == ".json" and p.is_file():
        try:
            with open(p) as f:
                return json.load(f).get("format") == INDEX_FORMAT
        except (OSError, ValueError):
            return False
    return p.is_dir() and any(FILE_RE.match(q.name) for q in p.iterdir())


# --------------------------------------------------------------------- metadata
_VEC = r"\[(-?\d+(?:,-?\d+)*)\]"
_BOX = rf"\({_VEC}:{_VEC}:{_VEC}(?:/[^()]*)?\)"
_BOXES_RE = re.compile(rf"(?:{_BOX}(?:,{_BOX})*)?")


def parse_active(text, iorigin) -> Optional[tuple]:
    """Carpet ``bboxset`` string -> local index boxes ((lo), (hi)), hi exclusive
    (etkplot.carpet.parse_active). None if it can't be read."""
    if isinstance(text, bytes):
        text = text.decode()
    body = re.search(r"\{([^{}]*)\}", re.sub(r"\s", "", text))
    if body is None or not _BOXES_RE.fullmatch(body.group(1)):
        return None
    iorigin = np.asarray(iorigin, dtype=int)
    boxes = []
    for lo, hi, st in re.findall(_BOX, body.group(1)):
        lo, hi, st = (np.array([int(v) for v in s.split(",")]) for s in (lo, hi, st))
        if not len(lo) == len(hi) == len(st) == len(iorigin) == 3 or np.any(st <= 0) or np.any(hi < lo):
            return None
        boxes.append((tuple(int(v) for v in lo // st - iorigin), tuple(int(v) for v in hi // st - iorigin + 1)))
    return tuple(boxes)


@dataclass(frozen=True)
class Component:
    """One Carpet dataset's metadata, (x, y, z) order."""

    path: str
    dataset: str
    it: int
    rl: int
    c: int
    time: float
    origin: Tuple[float, ...]
    delta: Tuple[float, ...]
    shape: Tuple[int, ...]
    nghost: Tuple[int, ...]
    bbox: Tuple[int, ...]
    active: Optional[tuple] = None

    def interior(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Ghost-stripped local index box [lo, hi) (buffer zones kept)."""
        lo = np.array([0 if self.bbox[2 * d] else self.nghost[d] for d in range(3)])
        hi = np.array([self.shape[d] - (0 if self.bbox[2 * d + 1] else self.nghost[d]) for d in range(3)])
        return None if np.any(hi <= lo) else (lo, hi)

    def active_boxes(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Local index boxes of the ghost-stripped active region (etkplot.ownership._active_boxes)."""
        rng = self.interior()
        if rng is None:
            return
        ilo, ihi = rng
        for lo, hi in [(ilo, ihi)] if self.active is None else self.active:
            lo, hi = np.maximum(lo, ilo), np.minimum(hi, ihi)
            if np.all(hi > lo):
                yield lo, hi


def _attr(a, key, default, cast):
    return tuple(np.atleast_1d(a[key]).astype(cast).tolist()) if key in a else default


def _component(path, name, dset) -> Optional[Component]:
    m = DATASET_RE.match(name)
    if m is None or int(m["tl"]) != 0 or dset.ndim != 3:
        return None
    a = dset.attrs
    active = None
    if "active" in a and "iorigin" in a:
        active = parse_active(a["active"], _attr(a, "iorigin", (0, 0, 0), int))
    return Component(
        path=str(path),
        dataset=name,
        it=int(m["it"]),
        rl=int(m["rl"] or 0),
        c=int(m["c"] or 0),
        time=float(np.atleast_1d(a["time"])[0]) if "time" in a else float("nan"),
        origin=_attr(a, "origin", (0.0,) * 3, float),
        delta=_attr(a, "delta", (1.0,) * 3, float),
        shape=tuple(int(s) for s in dset.shape[::-1]),
        nghost=_attr(a, "cctk_nghostzones", (0,) * 3, int),
        bbox=_attr(a, "cctk_bbox", (0,) * 6, int),
        active=active,
    )


class _Catalogue:
    """Components per (var, it, rl), from an etkplot index or a directory scan,
    in etkplot's order (index order; else file name order, then dataset order)."""

    def __init__(self, path):
        self.path = Path(path)
        self.index = None
        self._dir_files: Dict[str, List[Path]] = {}
        self._groups: Dict[tuple, List[Component]] = {}
        if self.path.suffix == ".json":
            with open(self.path) as f:
                self.index = json.load(f)
            if self.index.get("format") != INDEX_FORMAT:
                raise ValueError(f"{path} is not an {INDEX_FORMAT} file")
            self.root = Path(self.index["directory"])
        else:
            self.root = self.path
            for p in sorted(self.root.iterdir()):
                m = FILE_RE.match(p.name)
                if m and p.is_file():
                    self._dir_files.setdefault(m["var"], []).append(p)

    def variables(self) -> List[str]:
        return sorted(self.index["variables"]) if self.index is not None else sorted(self._dir_files)

    def _check(self, var):
        if var not in self.variables():
            raise KeyError(f"variable {var!r} not in {self.path}; available: {', '.join(self.variables())}")

    def iterations(self, var) -> List[int]:
        self._check(var)
        if self.index is not None:
            return sorted(set(self.index["components"]["it"]))
        its = set()
        with h5py.File(self._dir_files[var][0], "r") as f:
            for name in f.keys():
                m = DATASET_RE.match(name)
                if m and m["var"] == var and int(m["tl"]) == 0:
                    its.add(int(m["it"]))
        return sorted(its)

    def levels(self, var, it) -> List[int]:
        self._check(var)
        if self.index is not None:
            g = self.index["components"]
            return sorted({rl for i, rl in zip(g["it"], g["rl"]) if i == it})
        return sorted({c.rl for c in self._scan(var, it)})

    def _scan(self, var, it) -> List[Component]:
        key = (var, it, None)
        if key not in self._groups:
            out = []
            tag = f"::{var} it={it} "
            for p in self._dir_files[var]:
                with h5py.File(p, "r") as f:
                    for name in f.keys():
                        if tag in name:
                            c = _component(p, name, f[name])
                            if c is not None and c.it == it:
                                out.append(c)
            self._groups[key] = out
        return self._groups[key]

    def components(self, var, it, rl) -> List[Component]:
        self._check(var)
        key = (var, it, rl)
        if key not in self._groups:
            if self.index is None:
                comps = [c for c in self._scan(var, it) if c.rl == rl]
            else:
                thorn = self.index["variables"][var]["thorn"]
                g = self.index["components"]
                active = g.get("active", [None] * len(g["it"]))
                comps = []
                for k in range(len(g["it"])):
                    if g["it"][k] != it or g["rl"][k] != rl:
                        continue
                    comps.append(
                        Component(
                            path=str(self.root / (var + g["file_suffix"][k])),
                            dataset=f"{thorn}::{var} {g['name_tail'][k]}",
                            it=it,
                            rl=rl,
                            c=g["c"][k],
                            time=g["time"][k],
                            origin=tuple(g["origin"][k]),
                            delta=tuple(g["delta"][k]),
                            shape=tuple(g["shape"][k]),
                            nghost=tuple(g["nghost"][k]),
                            bbox=tuple(g["bbox"][k]),
                            active=None
                            if active[k] is None
                            else tuple((tuple(lo), tuple(hi)) for lo, hi in active[k]),
                        )
                    )
            self._groups[key] = comps
        return self._groups[key]


# --------------------------------------------------------------- level layout
@dataclass
class _Patch:
    origin: np.ndarray  # coordinate of point (0, 0, 0)
    delta: np.ndarray
    component: np.ndarray  # int16 (x, y, z): owning component, -1 = not owned

    def index_range(self, x0, x1) -> Optional[tuple]:
        """etkplot.ownership._index_range."""
        a = np.ceil((x0 - self.origin) / self.delta - _TOL).astype(int)
        b = np.floor((x1 - self.origin) / self.delta + _TOL).astype(int) + 1
        a = np.maximum(a, 0)
        b = np.minimum(b, self.component.shape)
        if np.any(b <= a):
            return None
        return tuple(slice(i, j) for i, j in zip(a, b))


def _connected_groups(lo: np.ndarray, hi: np.ndarray) -> List[List[int]]:
    """etkplot.assemble._connected_groups: boxes that overlap or touch."""
    n = len(lo)
    if n == 0:
        return []
    touch = np.all((lo[:, None] <= hi[None]) & (lo[None] <= hi[:, None]), axis=-1)
    label = np.arange(n)
    while True:
        new = np.where(touch, label[None, :], n).min(axis=1)
        if np.array_equal(new, label):
            break
        label = new
    groups: Dict[int, List[int]] = {}
    for i, g in enumerate(label.tolist()):
        groups.setdefault(g, []).append(i)
    return list(groups.values())


def _level_patches(comps: Sequence[Component]) -> List[_Patch]:
    """The level's connected boxes (buffer zones kept), sorted by x centre,
    with nothing owned yet (etkplot.assemble.level_boxes)."""
    hits = [(c, c.interior()) for c in comps]
    hits = [(c, r) for c, r in hits if r is not None]
    if not hits:
        return []
    dx = np.array(hits[0][0].delta)
    ref = np.array(hits[0][0].origin)
    lo, hi = [], []
    for c, (ilo, ihi) in hits:
        if not np.allclose(c.delta, dx):
            raise ValueError(f"{c.dataset}: spacing {c.delta} differs from the level's {tuple(dx)}")
        start = (np.array(c.origin) + ilo * dx - ref) / dx
        b = np.rint(start).astype(int)
        if np.any(np.abs(start - b) > 1e-6):
            raise ValueError(f"{c.dataset}: origin {c.origin} is not on the level's grid")
        lo.append(b)
        hi.append(b + (ihi - ilo))
    lo, hi = np.array(lo), np.array(hi)
    boxes = []
    for g in _connected_groups(lo, hi):
        blo = ref + dx * lo[g].min(axis=0)
        bhi = ref + dx * (hi[g].max(axis=0) - 1)
        boxes.append((blo, bhi))
    boxes.sort(key=lambda b: 0.5 * (b[0][0] + b[1][0]))
    out = []
    for blo, bhi in boxes:
        shape = tuple(np.rint((bhi - blo) / dx).astype(int) + 1)
        out.append(_Patch(origin=blo, delta=dx, component=np.full(shape, -1, np.int16)))
    return out


def _mask_spheres(patches: List[_Patch], spheres: Sequence[Sphere]) -> int:
    """etkplot.ownership._mask_horizons."""
    masked = 0
    for c, r in spheres:
        c = np.asarray(c, dtype=float)
        r = float(r)
        for p in patches:
            sl = p.index_range(c - r, c + r)
            if sl is None:
                continue
            d = [p.origin[k] + p.delta[k] * np.arange(s.start, s.stop) - c[k] for k, s in enumerate(sl)]
            d2 = d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2 + d[2][None, None, :] ** 2
            sub = p.component[sl]
            inside = (d2 < r * r) & (sub >= 0)
            masked += int(np.count_nonzero(inside))
            sub[inside] = -1
    return masked


class DoubleCover(ValueError):
    """Two components of one level hold the same point in their active regions."""


def _native_ownership(cat: _Catalogue, var: str, it: int, rl: int, spheres) -> Tuple[List[_Patch], dict]:
    """etkplot.ownership.level_ownership."""
    comps = cat.components(var, it, rl)
    if not comps:
        raise ValueError(f"no components of {var} at it={it}, rl={rl}; levels present: {cat.levels(var, it)}")
    dx = np.array(comps[0].delta, dtype=float)
    patches = _level_patches(comps)
    cover = [np.zeros(p.component.shape, np.uint8) for p in patches]
    for c in comps:
        for lo, hi in c.active_boxes():
            x0 = np.array(c.origin) + lo * dx
            x1 = np.array(c.origin) + (hi - 1) * dx
            for p, n in zip(patches, cover):
                sl = p.index_range(x0, x1)
                if sl is not None:
                    p.component[sl] = c.c
                    n[sl] += 1
    double = sum(int(np.count_nonzero(n > 1)) for n in cover)
    del cover
    if double:
        raise DoubleCover(f"rl={rl} of {var} at it={it}: {double} points lie in two components' active regions")
    for rf in cat.levels(var, it):
        if rf <= rl:
            continue
        for c in cat.components(var, it, rf):
            fdx = np.array(c.delta, dtype=float)
            for lo, hi in c.active_boxes():
                x0 = np.array(c.origin) + lo * fdx
                x1 = np.array(c.origin) + (hi - 1) * fdx
                for p in patches:
                    sl = p.index_range(x0, x1)
                    if sl is not None:
                        p.component[sl] = -1
    masked = _mask_spheres(patches, spheres) if spheres else 0
    owned = sum(int(np.count_nonzero(p.component >= 0)) for p in patches)
    return patches, {"owned": owned, "masked": masked, "time": comps[0].time}


# ------------------------------------------------------------------- reader
class CarpetHDF5Reader:
    """Level blocks of Carpet 3D output (see the module docstring).

    Args:
        path: an etkplot index (``.json``) or a directory of ``<var>.xyz.file_N.h5`` files.
        backend: "etkplot", "native" or "auto" (etkplot if importable).
        spheres: regions no level owns, as (centre, radius) pairs: the horizons.
            For the BBH run these come from a grid or training file's
            ``horizon_centres``/``horizon_radius`` (one iteration) or from the
            trajectory via ``parfile`` (etkplot backend, any iteration).
        parfile: Cactus parameter file whose trajectory table gives the
            horizons at each iteration (etkplot backend only).
        etkplot_path: directory to add to ``sys.path`` to import etkplot.
        cache_bytes: budget of the native backend's component cache.
        reference_var: variable whose layout stands for all (default: the first requested).
    """

    def __init__(
        self,
        path,
        backend: str = "auto",
        spheres: Optional[Sequence[Sphere]] = None,
        parfile=None,
        etkplot_path=None,
        cache_bytes: float = 1e9,
        reference_var: Optional[str] = None,
    ):
        self.path = Path(path)
        self.spheres = [(np.asarray(c, float), float(r)) for c, r in (spheres or [])]
        self.parfile = parfile
        self.reference_var = reference_var
        self._etk = None
        if backend not in ("auto", "etkplot", "native"):
            raise ValueError(f"backend must be auto, etkplot or native, not {backend!r}")
        if backend in ("auto", "etkplot"):
            self._etk = _import_etkplot(etkplot_path)
            if self._etk is None and backend == "etkplot":
                raise ImportError("etkplot is not importable; pass etkplot_path= or use backend='native'")
        if parfile is not None and self._etk is None:
            raise ValueError("horizons from a parfile need the etkplot backend; pass spheres= instead")
        self.backend = "etkplot" if self._etk is not None else "native"
        if self.backend == "etkplot":
            self._src = self._etk["source"].open_source(str(self.path))
        else:
            self._cat = _Catalogue(self.path)
        self.cache = ArrayCache(cache_bytes)

    # ------------------------------------------------------------ queries
    def variables(self) -> List[str]:
        return list(self._src.variables()) if self.backend == "etkplot" else self._cat.variables()

    def iterations(self, var: Optional[str] = None) -> List[int]:
        var = var or self.reference_var or self.variables()[0]
        return self._src.iterations(var) if self.backend == "etkplot" else self._cat.iterations(var)

    def levels(self, iteration: int, var: Optional[str] = None) -> List[int]:
        var = var or self.reference_var or self.variables()[0]
        return self._src.levels(var, iteration) if self.backend == "etkplot" else self._cat.levels(var, iteration)

    # ------------------------------------------------------------- blocks
    def ownership(self, ref: str, iteration: int, rl: int):
        """(patches, stats): each patch has ``origin``, ``delta`` and
        ``component`` (int16, -1 where the level owns nothing)."""
        if self.backend == "native":
            return _native_ownership(self._cat, ref, iteration, rl, self.spheres)
        own_mod = self._etk["ownership"]
        hz = None
        if self.parfile is not None:
            opts = SimpleNamespace(parfile=str(self.parfile), traj=None, t0=None, horizon_mask=True)
            hz = own_mod.horizons(self._src, ref, iteration, opts)
        elif self.spheres:
            nan = np.full(len(self.spheres), np.nan)
            hz = own_mod.Horizons(
                centres=np.array([c for c, _ in self.spheres]),
                radius=np.array([r for _, r in self.spheres]),
                mass=nan,
                spin=nan,
            )
        own = own_mod.level_ownership(self._src, ref, iteration, rl, horizons=hz)
        stats = dict(own.stats, time=own.time)
        if hz is not None:
            stats["spheres"] = [(c.tolist(), float(r)) for c, r in zip(hz.centres, hz.radius)]
        return own.patches, stats

    def _read_slab(self, var, it, rl, patch, a, b, halo, comps) -> np.ndarray:
        if self.backend == "etkplot":
            return self._etk["somfeatures"]._read_slab(self._src, var, it, rl, patch, a, b, halo)
        nx = patch.component.shape[0]
        shape = (b - a + 2 * halo,) + patch.component.shape[1:]
        out = np.full(shape, np.nan)
        # the slab in the patch's index frame; out[0] is x index a - halo
        slo = np.array([max(a - halo, 0), 0, 0])
        shi = np.array([min(b + halo, nx), shape[1], shape[2]])
        dx = patch.delta
        for c in comps:
            rng = c.interior()
            if rng is None:
                continue
            start = (np.array(c.origin) - patch.origin) / dx
            o = np.rint(start).astype(int)
            if np.any(np.abs(start - o) > 1e-6):
                raise ValueError(f"{c.dataset}: origin {c.origin} is not on the level's grid")
            lo = np.maximum(o + rng[0], slo)
            hi = np.minimum(o + rng[1], shi)
            if np.any(hi <= lo):
                continue
            data = self.cache.get((c.path, c.dataset), lambda c=c, rng=rng: _read_interior(c, rng))
            src = tuple(slice(l - o[d] - rng[0][d], h - o[d] - rng[0][d]) for d, (l, h) in enumerate(zip(lo, hi)))
            dst_lo = lo - np.array([a - halo, 0, 0])
            dst = tuple(slice(l, l + (h - lo_)) for l, h, lo_ in zip(dst_lo, hi, lo))
            piece = data[src]
            np.copyto(out[dst], piece, where=~np.isnan(piece))
        return out

    def iter_level_blocks(
        self,
        variables: Sequence[str],
        iteration: int,
        levels: Optional[Sequence[int]] = None,
        halo: int = 2,
        slab: int = DEFAULT_SLAB,
        skip=None,
    ) -> Iterator[LevelBlock]:
        """Yield the blocks of ``iteration`` (see the module docstring).

        Args:
            variables: the variables to read (their layout comes from the first,
                or ``reference_var``).
            iteration: the snapshot.
            levels: levels to visit (default: all present).
            halo: x points of halo on each side (the stencil half-width).
            slab: x points per block.
            skip: container of block keys not to read (they are not yielded).
        """
        variables = list(variables)
        ref = self.reference_var or variables[0]
        present = self.levels(iteration, ref)
        levels = present if levels is None else list(levels)
        missing = [rl for rl in levels if rl not in present]
        if missing:
            raise ValueError(f"no level {missing} of {ref} at it={iteration}; levels present: {present}")
        skip = set() if skip is None else skip
        for rl in levels:
            patches, stats = self.ownership(ref, iteration, rl)
            comps = {}
            if self.backend == "native":
                comps = {v: self._cat.components(v, iteration, rl) for v in variables}
            for pi, p in enumerate(patches):
                for a, b in slabs(p.component.shape[0], slab):
                    key = f"rl{rl}/p{pi}/x{a:05d}"
                    owned_c = p.component[a:b]
                    owned = owned_c >= 0
                    if key in skip or not owned.any():
                        continue
                    data = {v: self._read_slab(v, iteration, rl, p, a, b, halo, comps.get(v)) for v in variables}
                    origin = np.array(p.origin, dtype=float)
                    origin[0] = p.origin[0] + (a - halo) * p.delta[0]
                    yield LevelBlock(
                        level=rl,
                        origin=origin,
                        spacing=np.array(p.delta, dtype=float),
                        data=data,
                        halo=(halo, 0, 0),
                        key=key,
                        iteration=iteration,
                        time=float(stats["time"]),
                        owned=owned,
                        component=np.where(owned, owned_c, -1).astype(np.int16),
                        xyz_base=np.array(p.origin, dtype=float),
                        xyz_offset=np.array([a, 0, 0]),
                        info={"patch": pi, "slab": (a, b), "level_owned": stats["owned"], "level_masked": stats["masked"]},
                    )
            del patches
            self.cache.clear()


def _read_interior(c: Component, rng) -> np.ndarray:
    """A component's ghost-stripped interior, float64 (x, y, z)."""
    lo, hi = rng
    with h5py.File(c.path, "r") as f:
        d = f[c.dataset][lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]]
    return np.ascontiguousarray(d.T, dtype=np.float64)


def _import_etkplot(path=None) -> Optional[dict]:
    """etkplot's modules, or None if it can't be imported."""
    if path is not None and str(path) not in sys.path:
        sys.path.append(str(path))
    try:
        from etkplot import ownership, somfeatures, source  # type: ignore
    except Exception:
        return None
    return {"ownership": ownership, "somfeatures": somfeatures, "source": source}
