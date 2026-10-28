"""Level blocks from CarpetX (AsterX) output: Silo and openPMD-HDF5, plus TSV lines.

CarpetX keeps each refinement level as an AMReX ``BoxArray``: disjoint
boxes of cell-centred data, refinement ratio 2, all levels sharing one
domain origin (``ProbLo``). A cell of level L with integer index i (counted
from the domain's first cell) has centre ``x0 + (i + 1/2) dx_L``.

Silo (``CarpetX::out_silo_vars``), as found in the FM torus run
(``eg-liwei-fm/FM_4lvl_Noble``; HDF5 driver, so h5py reads it):

* per iteration a metafile ``<sim>[.<plane>].it%08d.silo`` and data files
  ``<sim>[.<plane>].it%08d.silo.dir/<sim>[.<plane>].it%08d.p%06d.silo``
  (one per writing process);
* every Silo object is an HDF5 *named datatype* carrying a compound
  attribute ``silo`` and an int ``silo_type`` (130 quadmesh, 501 quadvar,
  520 multimesh, 521 multivar); array members are paths of raw datasets
  ``/.silo/#NNNNNN``. Groups hold ``..`` links to their parent;
* the metafile's multivar ``<thorn>_<var>`` (lower case, e.g.
  ``hydrobasex_rho``, ``asterx_b2small``) lists its blocks as
  ``<file>:<thorn>_<var>_m0000_rlLL_cCCCCCCCC`` joined by ``;``; the
  multimesh ``gh_ghosts03_03_03`` lists the meshes
  ``box_ghosts03_03_03_m0000_rlLL_cCCCCCCCC``; ``CarpetX/FabArrayBase_m0000_rlLL``
  holds the level's valid boxes [ncomp, 6] (lo xyz, hi xyz, inclusive);
  ``CarpetX/nlevels``, ``cycle``, ``dtime`` the rest;
* a quadmesh has 1D node coordinates ``coord0..2`` including the ghosts;
  ``min_index``/``max_index`` bracket the valid nodes (so ``min_index`` is
  the ghost width);
* a quadvar's ``value0`` is the whole box *including ghosts*; ``dims`` are
  cells for cell-centred variables (``centering`` 111, ``align`` 0.5) and
  nodes for vertex-centred ones (110). The raw dataset is in Fortran order:
  read it as ``ds[()].ravel().reshape(dims[::-1]).T`` to get (x, y, z).
  Checked against the TSV lines of the same run: identical values.
* 2D output (``.xy``/``.xz``/``.yz``) is the same with one cell (or node)
  along the normal.

openPMD-HDF5 (``CarpetX::out_openpmd_vars`` with ``openpmd_format = "HDF5"``;
from CarpetX's ``io_openpmd.cxx``; there is no local example):

* file based, ``<sim>.it%08d.h5``; iteration group ``/data/<it>`` with
  attributes ``time``, ``dt``, ``numLevels_patch00`` and
  ``chunkInfo_patch00_levLL`` (int64 [nboxes * 6]: lo z, y, x then hi z, y, x
  exclusive, level cell indices);
* mesh ``/data/<it>/meshes/<thorn>_<group>_patch00_levLL`` (lower case,
  e.g. ``hydrobasex_vel_patch00_lev01``), components ``<thorn>_<var>``
  (``hydrobasex_velx``); attributes ``gridSpacing`` and ``gridGlobalOffset``
  in (z, y, x) order, component attribute ``position`` (0.5 for cells);
* each level is one dense dataset (z, y, x) of the level's whole domain
  (cells + 1 per axis); boxes are stored at their index without ghosts, so
  points outside the boxes read back as 0, not NaN: :attr:`chunkInfo` is
  the only record of where the data is.

Blocks: one x slab (``slab`` cells) of one box, with ``halo`` cells on every
side. The halo comes from the level's boxes (valid cells), then, Silo only,
from the box's own ghost zones, and what is still missing, at the level's
outer edge, from the next coarser level by injection (``fill="coarse"``,
the default) or is left NaN (``fill="nan"``). A level owns the valid cells
of its boxes that no finer level's boxes cover and that lie outside
``spheres``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

from .blocks import ArrayCache, LevelBlock, Sphere, slabs, sphere_mask

__all__ = [
    "CarpetXReader",
    "ASTERX_NAMES",
    "read_tsv",
    "looks_like_carpetx",
    "silo_object",
    "SILO_QUADMESH",
    "SILO_QUADVAR",
    "SILO_MULTIMESH",
    "SILO_MULTIVAR",
]

SILO_QUADMESH, SILO_QUADVAR, SILO_MULTIMESH, SILO_MULTIVAR = 130, 501, 520, 521
SILO_ZONECENT, SILO_NODECENT = 111, 110

# canonical (etkplot / IllinoisGRMHD) name -> (openPMD mesh group, component = Silo variable)
ASTERX_NAMES: Dict[str, Tuple[str, str]] = {
    "rho_b": ("hydrobasex_rho", "hydrobasex_rho"),
    "P": ("hydrobasex_press", "hydrobasex_press"),
    "smallb2": ("asterx_diagnostics", "asterx_b2small"),
    "Bx": ("hydrobasex_bvec", "hydrobasex_bvecx"),
    "By": ("hydrobasex_bvec", "hydrobasex_bvecy"),
    "Bz": ("hydrobasex_bvec", "hydrobasex_bvecz"),
    "vel[0]": ("hydrobasex_vel", "hydrobasex_velx"),
    "vel[1]": ("hydrobasex_vel", "hydrobasex_vely"),
    "vel[2]": ("hydrobasex_vel", "hydrobasex_velz"),
}

SILO_FILE_RE = re.compile(r"^(?P<sim>.+?)(?:\.(?P<plane>xy|xz|yz|x|y|z))?\.it(?P<it>\d+)\.silo$")
OPMD_FILE_RE = re.compile(r"^(?P<sim>.+?)\.it(?P<it>\d+)\.h5$")
BLOCK_RE = re.compile(r"_rl(?P<rl>\d+)_c(?P<c>\d+)$")
MESH_RE = re.compile(r"^(?P<group>.+?)(?:_patch(?P<patch>\d+))?_lev(?P<lev>\d+)$")


# ----------------------------------------------------------------------- TSV
def read_tsv(path) -> Dict[str, np.ndarray]:
    """A CarpetX TSV file (``<thorn>-<group>.it%06d.{x,y,z}.tsv``) as columns.

    The header is ``# 1:iteration 2:time 3:patch 4:level 5:i 6:j 7:k 8:x 9:y
    10:z 11:<var> ...``; each file holds 1D lines (one per level and box),
    ghost points included, cell-centre coordinates.

    Returns:
        dict: column name -> array (iteration, patch, level, i, j, k as int64).
    """
    names = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                toks = re.findall(r"(\d+):(\S+)", line)
                if toks:
                    names = [n for _, n in toks]
                    break
    if names is None:
        raise ValueError(f"{path}: no '# 1:iteration 2:time ...' header")
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != len(names):
        raise ValueError(f"{path}: {data.shape[1]} columns but the header names {len(names)}")
    out = {}
    for j, n in enumerate(names):
        col = data[:, j]
        out[n] = col.astype(np.int64) if n in ("iteration", "patch", "level", "i", "j", "k") else col
    return out


# ---------------------------------------------------------------------- Silo
def silo_object(f: h5py.File, name: str) -> Tuple[int, Dict[str, object]]:
    """(silo_type, fields) of a Silo object; fields that point at ``/.silo``
    datasets are returned as the datasets' values."""
    obj = f[name]
    st = int(obj.attrs["silo_type"])
    rec = obj.attrs["silo"]
    out: Dict[str, object] = {}
    for k in rec.dtype.names:
        v = rec[k]
        if isinstance(v, bytes):
            s = v.decode()
            out[k] = f[s][()] if s.startswith("/.silo/") and s in f else s
        else:
            out[k] = np.asarray(v)
    return st, out


def _silo_names(raw) -> List[str]:
    """A multimesh/multivar name list (bytes dataset) -> entries."""
    s = bytes(np.asarray(raw, dtype=np.uint8)).decode().strip("\x00").strip()
    return [t for t in s.split(";") if t]


def silo_array(f: h5py.File, fields: Mapping[str, object]) -> np.ndarray:
    """A quadvar's data as float64 (x, y, z) (Silo writes Fortran order)."""
    dims = [int(d) for d in np.asarray(fields["dims"]).ravel()]
    val = fields["value0"]
    a = np.asarray(val if not isinstance(val, str) else f[val][()])
    return np.ascontiguousarray(a.ravel().reshape(dims[::-1]).T, dtype=np.float64)


@dataclass
class _Box:
    level: int
    c: int
    lo: np.ndarray  # first valid cell, level index
    hi: np.ndarray  # exclusive
    ghosts: np.ndarray = field(default_factory=lambda: np.zeros(3, int))
    where: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # Silo: var -> (file, object)
    coords: Optional[List[np.ndarray]] = None  # Silo: node coordinates, ghosts included


@dataclass
class _Level:
    level: int
    dx: np.ndarray
    boxes: List[_Box]


@dataclass
class _Snapshot:
    iteration: int
    time: float
    x0: np.ndarray  # domain lower edge (ProbLo)
    levels: Dict[int, _Level]
    path: Path
    kind: str  # "silo" or "openpmd"
    meshes: Dict[Tuple[str, int], str] = field(default_factory=dict)  # openPMD: (component, level) -> dataset path


def _scan_silo(meta: Path, iteration: int) -> _Snapshot:
    """The layout of one Silo iteration from its metafile and the block meshes."""
    root = meta.parent
    blocks: Dict[Tuple[int, int], _Box] = {}
    time = float("nan")
    with h5py.File(meta, "r") as f:
        if "dtime" in f:
            time = float(np.ravel(f["dtime"][()])[0])
        varnames = {}
        meshnames = None
        fab = {}
        for name in f:
            if name in ("..", ".silo") or not isinstance(f[name], h5py.Datatype):
                if name == "CarpetX":
                    for k in f[name]:
                        m = re.match(r"FabArrayBase_m0000_rl(\d+)$", k)
                        if m:
                            fab[int(m.group(1))] = np.asarray(f[name][k][()]).reshape(-1, 6)
                continue
            st = int(f[name].attrs.get("silo_type", -1))
            if st == SILO_MULTIVAR:
                varnames[name] = _silo_names(silo_object(f, name)[1]["varnames"])
            elif st == SILO_MULTIMESH and meshnames is None:
                meshnames = _silo_names(silo_object(f, name)[1]["meshnames"])
    if meshnames is None:
        raise ValueError(f"{meta}: no multimesh")
    files: Dict[str, h5py.File] = {}
    try:
        for entry in meshnames:
            fname, obj = entry.rsplit(":", 1)
            m = BLOCK_RE.search(obj)
            if m is None:
                continue
            if fname not in files:
                files[fname] = h5py.File(root / fname, "r")
            st, mesh = silo_object(files[fname], obj)
            coords = [np.asarray(mesh[f"coord{d}"], dtype=float).ravel() for d in range(3)]
            lo_n = np.asarray(mesh["min_index"]).astype(int).ravel()
            hi_n = np.asarray(mesh["max_index"]).astype(int).ravel()
            blocks[int(m["rl"]), int(m["c"])] = _Box(
                level=int(m["rl"]), c=int(m["c"]), lo=lo_n, hi=hi_n, ghosts=lo_n.copy(), coords=coords
            )
    finally:
        for h in files.values():
            h.close()
    for var, entries in varnames.items():
        for entry in entries:
            fname, obj = entry.rsplit(":", 1)
            m = BLOCK_RE.search(obj)
            if m and (int(m["rl"]), int(m["c"])) in blocks:
                blocks[int(m["rl"]), int(m["c"])].where[var] = (fname, obj)
    # spacing per level and the domain origin, from the valid node coordinates
    levels: Dict[int, _Level] = {}
    for (rl, _), b in sorted(blocks.items()):
        dx = np.array(
            [
                (b.coords[d][b.hi[d]] - b.coords[d][b.lo[d]]) / (b.hi[d] - b.lo[d]) if b.hi[d] > b.lo[d] else np.nan
                for d in range(3)
            ]
        )
        levels.setdefault(rl, _Level(rl, dx, []))
    lev0 = min(levels)
    b0 = [b for (rl, _), b in blocks.items() if rl == lev0]
    x0 = np.array([min(b.coords[d][b.lo[d]] for b in b0) for d in range(3)])
    for (rl, c), b in sorted(blocks.items()):
        L = levels[rl]
        first = np.array([b.coords[d][b.lo[d]] for d in range(3)])
        with np.errstate(invalid="ignore"):
            ilo = np.rint((first - x0) / L.dx)
        ilo = np.where(np.isfinite(ilo), ilo, 0).astype(int)
        ncell = np.maximum(b.hi - b.lo, 1)  # a 2D slice has one cell along its normal
        b.lo, b.hi = ilo, ilo + ncell
        L.boxes.append(b)
    for rl, rows in fab.items():  # cross-check with CarpetX's own record
        if rl in levels:
            got = sorted((tuple(b.lo), tuple(b.hi - 1)) for b in levels[rl].boxes)
            want = sorted((tuple(r[:3]), tuple(r[3:])) for r in rows)
            planar = any(np.any(b.hi - b.lo == 1) for b in levels[rl].boxes)
            if not planar and got != want:
                raise ValueError(f"{meta}: level {rl} boxes {got} differ from CarpetX/FabArrayBase {want}")
    return _Snapshot(iteration, time, x0, levels, meta, "silo")


def _scan_openpmd(path: Path, iteration: int) -> _Snapshot:
    """The layout of one openPMD-HDF5 iteration.

    TODO(AsterX run): checked against CarpetX's writer source and a synthetic
    file only. On the first real file confirm (1) the ``chunkInfo`` order
    (z, y, x lo then hi) and that it lists every box, (2) that
    ``gridGlobalOffset`` is ProbLo and ``position`` 0.5 for cell-centred
    data, (3) the mesh and component names of the asterx diagnostics group
    (``asterx_diagnostics_patch00_levLL/asterx_b2small``)."""
    with h5py.File(path, "r") as f:
        base = f.attrs.get("basePath", b"/data/%T/")
        base = base.decode() if isinstance(base, bytes) else str(base)
        meshes_path = f.attrs.get("meshesPath", b"meshes/")
        meshes_path = meshes_path.decode() if isinstance(meshes_path, bytes) else str(meshes_path)
        it_path = base.replace("%T", str(iteration)).rstrip("/")
        if it_path not in f:
            raise KeyError(f"{path}: no iteration {iteration} ({it_path})")
        g = f[it_path]
        time = float(np.ravel(g.attrs.get("time", np.nan))[0])
        levels: Dict[int, _Level] = {}
        meshes: Dict[Tuple[str, int], str] = {}
        x0 = None
        mg = g[meshes_path.rstrip("/")]
        for mname in mg:
            m = MESH_RE.match(mname)
            if m is None or int(m["patch"] or 0) != 0:
                continue
            rl = int(m["lev"])
            mesh = mg[mname]
            dx = np.asarray(mesh.attrs["gridSpacing"], dtype=float)[::-1].copy()
            off = np.asarray(mesh.attrs["gridGlobalOffset"], dtype=float)[::-1].copy()
            if x0 is None:
                x0 = off
            if rl not in levels:
                levels[rl] = _Level(rl, dx, [])
            comps = {mname: mesh} if isinstance(mesh, h5py.Dataset) else dict(mesh.items())
            for cname, ds in comps.items():
                if not isinstance(ds, h5py.Dataset):
                    continue
                pos = np.asarray(ds.attrs.get("position", [0.5, 0.5, 0.5]), dtype=float)
                if not np.allclose(pos, 0.5):
                    continue  # vertex- or face-centred: not a feature input
                key = m["group"] if isinstance(mesh, h5py.Dataset) else cname
                meshes[key, rl] = ds.name
        for rl, L in levels.items():
            key = f"chunkInfo_patch00_lev{rl:02d}"
            if key not in g.attrs:
                raise KeyError(f"{path}: iteration {iteration} has no {key}")
            ci = np.asarray(g.attrs[key], dtype=np.int64).reshape(-1, 6)
            for c, row in enumerate(ci):
                L.boxes.append(_Box(level=rl, c=c, lo=row[2::-1].copy(), hi=row[5:2:-1].copy()))
    if x0 is None:
        raise ValueError(f"{path}: iteration {iteration} has no meshes")
    return _Snapshot(iteration, time, x0, levels, path, "openpmd", meshes)


def looks_like_carpetx(path) -> bool:
    p = Path(path)
    if p.is_file():
        return p.suffix == ".silo" or _is_openpmd(p)
    if not p.is_dir():
        return False
    return any(SILO_FILE_RE.match(q.name) or (OPMD_FILE_RE.match(q.name) and _is_openpmd(q)) for q in p.iterdir())


def _is_openpmd(p: Path) -> bool:
    try:
        with h5py.File(p, "r") as f:
            return "openPMD" in f.attrs
    except OSError:
        return False


# -------------------------------------------------------------------- reader
class CarpetXReader:
    """Level blocks of CarpetX 3D output (see the module docstring).

    Args:
        path: a directory of Silo or openPMD-HDF5 output, or one file of it.
        format: "silo", "openpmd" or "auto".
        plane: Silo only: read the 2D files of this plane ("xy", ...) instead
            of the 3D ones (for checks; features need 3D data).
        names: canonical variable -> (openPMD group, component / Silo name).
        spheres: (centre, radius) regions no level owns (excision, horizons).
        fill: how halo cells outside the level are filled: "coarse"
            (injection from the next coarser level) or "nan".
        cache_bytes: budget for decompressed Silo boxes.
    """

    def __init__(
        self,
        path,
        format: str = "auto",
        plane: Optional[str] = None,
        names: Mapping[str, Tuple[str, str]] = ASTERX_NAMES,
        spheres: Optional[Sequence[Sphere]] = None,
        fill: str = "coarse",
        cache_bytes: float = 1e9,
    ):
        self.path = Path(path)
        self.names = dict(names)
        self.plane = plane
        self.spheres = [(np.asarray(c, float), float(r)) for c, r in (spheres or [])]
        if fill not in ("coarse", "nan"):
            raise ValueError(f"fill must be 'coarse' or 'nan', not {fill!r}")
        self.fill = fill
        self.cache = ArrayCache(cache_bytes)
        self._files: Dict[int, Path] = {}
        root = self.path if self.path.is_dir() else self.path.parent
        cands = [self.path] if self.path.is_file() else sorted(root.iterdir())
        silo, opmd = {}, {}
        for q in cands:
            m = SILO_FILE_RE.match(q.name)
            if m and q.is_file() and m["plane"] == plane:
                silo[int(m["it"])] = q
            m = OPMD_FILE_RE.match(q.name)
            if m and q.is_file() and plane is None and _is_openpmd(q):
                opmd[int(m["it"])] = q
        if format == "auto":
            format = "silo" if silo else "openpmd" if opmd else None
            if format is None:
                raise ValueError(f"{path}: no CarpetX Silo or openPMD-HDF5 output" + (f" on plane {plane}" if plane else ""))
        if format not in ("silo", "openpmd"):
            raise ValueError(f"format must be silo, openpmd or auto, not {format!r}")
        self.format = format
        self._files = silo if format == "silo" else opmd
        self._snap: Optional[_Snapshot] = None

    # ------------------------------------------------------------ queries
    def iterations(self) -> List[int]:
        if self.format == "openpmd" and len(self._files) == 1:  # one file, several iterations
            p = next(iter(self._files.values()))
            with h5py.File(p, "r") as f:
                if "data" in f:
                    return sorted(int(k) for k in f["data"])
        return sorted(self._files)

    def snapshot(self, iteration: int) -> _Snapshot:
        if self._snap is None or self._snap.iteration != iteration:
            self.cache.clear()
            if self.format == "silo":
                self._snap = _scan_silo(self._file(iteration), iteration)
            else:
                self._snap = _scan_openpmd(self._file(iteration), iteration)
        return self._snap

    def _file(self, iteration: int) -> Path:
        if iteration in self._files:
            return self._files[iteration]
        if self.format == "openpmd" and len(self._files) == 1:
            return next(iter(self._files.values()))
        raise KeyError(f"no iteration {iteration} in {self.path}; have {self.iterations()[:5]}...")

    def levels(self, iteration: int) -> List[int]:
        return sorted(self.snapshot(iteration).levels)

    def _component(self, var: str) -> Tuple[str, str]:
        if var in self.names:
            return self.names[var]
        return (var, var)  # a native name

    # ---------------------------------------------------------- box data
    def read_box(self, var: str, iteration: int, level: int, c: int) -> Tuple[np.ndarray, np.ndarray]:
        """Silo: one box of ``var`` with its ghosts, float64 (x, y, z), and the
        level index of its first point."""
        if self.format != "silo":
            raise ValueError("read_box is for Silo; openPMD stores levels whole")
        snap = self.snapshot(iteration)
        box = next(b for b in snap.levels[level].boxes if b.c == c)
        name = self._component(var)[1]
        if name not in box.where:
            raise KeyError(f"{name} is not in the Silo output at it={iteration} (rl={level}, c={c})")
        fname, obj = box.where[name]

        def load():
            with h5py.File(snap.path.parent / fname, "r") as f:
                st, fields = silo_object(f, obj)
                cent = int(np.ravel(fields.get("centering", SILO_ZONECENT))[0])
                if cent != SILO_ZONECENT:
                    raise ValueError(f"{name} is not cell centred (centering {cent})")
                return silo_array(f, fields)

        a = self.cache.get((str(snap.path), fname, obj), load)
        return a, box.lo - box.ghosts

    def _region(self, var: str, snap: _Snapshot, level: int, lo, hi, own: Optional[_Box] = None, depth: int = 0):
        """``var`` on level cells [lo, hi): the level's valid cells, then
        (Silo) ``own``'s ghosts, then the coarser level or NaN."""
        lo = np.asarray(lo, int)
        hi = np.asarray(hi, int)
        out = np.full(tuple(hi - lo), np.nan)
        have = np.zeros(out.shape, bool)
        L = snap.levels[level]
        if snap.kind == "silo":
            if own is not None:
                a, alo = self.read_box(var, snap.iteration, level, own.c)
                _paste(out, have, lo, a, alo)
            for b in L.boxes:
                if np.all(b.lo < hi) and np.all(b.hi > lo):
                    a, alo = self.read_box(var, snap.iteration, level, b.c)
                    g = b.ghosts
                    _paste(out, have, lo, a[g[0] : a.shape[0] - g[0], g[1] : a.shape[1] - g[1], g[2] : a.shape[2] - g[2]], b.lo)
        else:
            comp = self._component(var)[1]
            key = (comp, level) if (comp, level) in snap.meshes else (self._component(var)[0], level)
            if key not in snap.meshes:
                raise KeyError(f"{comp} is not in the openPMD output at it={snap.iteration}, level {level}")
            with h5py.File(snap.path, "r") as f:
                ds = f[snap.meshes[key]]
                n = np.array(ds.shape[::-1])
                clo, chi = np.maximum(lo, 0), np.minimum(hi, n)
                if np.all(chi > clo):
                    data = ds[clo[2] : chi[2], clo[1] : chi[1], clo[0] : chi[0]].T.astype(np.float64)
                    valid = np.zeros(data.shape, bool)
                    for b in L.boxes:
                        blo, bhi = np.maximum(b.lo, clo), np.minimum(b.hi, chi)
                        if np.all(bhi > blo):
                            valid[tuple(slice(x - y, z - y) for x, y, z in zip(blo, clo, bhi))] = True
                    sl = tuple(slice(x - y, z - y) for x, y, z in zip(clo, lo, chi))
                    np.copyto(out[sl], data, where=valid)
                    have[sl] |= valid
        if self.fill == "coarse" and level - 1 in snap.levels and not have.all():
            ratio = np.rint(snap.levels[level - 1].dx / L.dx).astype(int)
            ratio = np.where(ratio > 0, ratio, 1)
            clo = np.floor_divide(lo, ratio)
            chi = np.floor_divide(hi - 1, ratio) + 1
            coarse = self._region(var, snap, level - 1, clo, chi, None, depth + 1)
            idx = np.ix_(*[np.arange(lo[d], hi[d]) // ratio[d] - clo[d] for d in range(3)])
            np.copyto(out, coarse[idx], where=~have)
        return out

    def owned(self, snap: _Snapshot, level: int, lo, hi) -> np.ndarray:
        """Cells [lo, hi) of ``level`` that no finer level's boxes cover."""
        lo = np.asarray(lo, int)
        hi = np.asarray(hi, int)
        out = np.ones(tuple(hi - lo), bool)
        L = snap.levels[level]
        for rf in sorted(snap.levels):
            if rf <= level:
                continue
            F = snap.levels[rf]
            ratio = np.rint(L.dx / F.dx).astype(int)
            ratio = np.where(ratio > 0, ratio, 1)
            for b in F.boxes:
                clo = np.maximum(np.floor_divide(b.lo, ratio), lo)
                chi = np.minimum(np.floor_divide(b.hi - 1, ratio) + 1, hi)
                if np.all(chi > clo):
                    out[tuple(slice(x - y, z - y) for x, y, z in zip(clo, lo, chi))] = False
        return out

    # -------------------------------------------------------------- blocks
    def iter_level_blocks(
        self,
        variables: Sequence[str],
        iteration: int,
        levels: Optional[Sequence[int]] = None,
        halo: int = 2,
        slab: int = 64,
        skip=None,
    ) -> Iterator[LevelBlock]:
        """Yield one block per x slab of each box of each level (see the module docstring)."""
        snap = self.snapshot(iteration)
        levels = sorted(snap.levels) if levels is None else list(levels)
        skip = set() if skip is None else skip
        for rl in levels:
            L = snap.levels[rl]
            for b in sorted(L.boxes, key=lambda b: b.c):
                nx = int(b.hi[0] - b.lo[0])
                for a, e in slabs(nx, slab):
                    key = f"rl{rl}/b{b.c}/x{a:05d}"
                    if key in skip:
                        continue
                    ilo = b.lo + np.array([a, 0, 0])
                    ihi = np.array([b.lo[0] + e, b.hi[1], b.hi[2]])
                    owned = self.owned(snap, rl, ilo, ihi)
                    base = snap.x0 + 0.5 * L.dx
                    if self.spheres:
                        owned &= ~sphere_mask(base, ilo, L.dx, owned.shape, self.spheres)
                    if not owned.any():
                        continue
                    data = {v: self._region(v, snap, rl, ilo - halo, ihi + halo, own=b) for v in variables}
                    yield LevelBlock(
                        level=rl,
                        origin=base + (ilo - halo) * L.dx,
                        spacing=L.dx.copy(),
                        data=data,
                        halo=halo,
                        key=key,
                        iteration=iteration,
                        time=snap.time,
                        owned=owned,
                        component=np.where(owned, b.c, -1).astype(np.int32),
                        xyz_base=base,
                        xyz_offset=ilo,
                        info={"box": b.c, "slab": (a, e), "box_lo": b.lo.tolist(), "box_hi": b.hi.tolist()},
                    )


def _paste(out: np.ndarray, have: np.ndarray, lo, src: np.ndarray, src_lo) -> None:
    """Copy the overlap of ``src`` (first cell at level index ``src_lo``) into
    ``out`` (first cell at ``lo``) and mark it present."""
    lo = np.asarray(lo, int)
    src_lo = np.asarray(src_lo, int)
    a = np.maximum(lo, src_lo)
    b = np.minimum(lo + out.shape, src_lo + src.shape)
    if np.any(b <= a):
        return
    d = tuple(slice(x - y, z - y) for x, y, z in zip(a, lo, b))
    s = tuple(slice(x - y, z - y) for x, y, z in zip(a, src_lo, b))
    out[d] = src[s]
    have[d] = True
