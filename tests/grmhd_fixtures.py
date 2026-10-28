"""Synthetic Carpet, CarpetX Silo, CarpetX openPMD-HDF5 and TSV output for the reader tests.

Every variable is an analytic function of position (:func:`field`), ghosts
included, so a reader's values can be checked anywhere it claims data.
"""

from pathlib import Path

import h5py
import numpy as np

CARPET_VARS = {  # canonical name -> Carpet thorn
    "smallb2": "SMALLBPOYNET",
    "P": "ILLINOISGRMHD",
    "rho_b": "ILLINOISGRMHD",
    "Bx": "ILLINOISGRMHD",
    "By": "ILLINOISGRMHD",
    "Bz": "ILLINOISGRMHD",
    "vel[0]": "HYDROBASE",
    "vel[1]": "HYDROBASE",
    "vel[2]": "HYDROBASE",
}
_K = {v: k for k, v in enumerate(CARPET_VARS)}


def field(var: str, x, y, z):
    """A smooth positive (for the scalars) test field, different per variable."""
    k = _K.get(var, len(_K))
    base = np.sin(0.3 * x + 0.1 * k) * np.cos(0.2 * y - 0.05 * k) + 0.1 * z * (k + 1)
    if var in ("smallb2", "P", "rho_b"):
        return np.exp(0.5 * base) * (1.0 + 0.1 * k)
    return base


# ------------------------------------------------------------------- Carpet
def _active(boxes):
    vec = lambda v: "[" + ",".join(str(int(a)) for a in v) + "]"  # noqa: E731
    body = ",".join(f"({vec(lo)}:{vec(np.asarray(hi) - 1)}:[1,1,1])" for lo, hi in boxes)
    return f"bboxset<CCTK_INT4,3>(set<bbox>:{{{body}}},stride:[1,1,1],offset:[0,0,0])"


def carpet_components(it=0):
    """Two levels: rl0 dx 0.5 on [-4, 4]^3 cut into two components along x
    (ghosts 2, outer boundary faces without ghosts); rl1 dx 0.25, one component
    around (0.5, 0, 0) with 2 ghost and 2 buffer points on every face."""
    comps = []
    g = 2
    # rl0: global points 0..16 (17), x split at 8
    for c, (xlo, xhi) in enumerate([(0, 8), (8, 17)]):
        lo = np.array([xlo, 0, 0])
        hi = np.array([xhi, 17, 17])
        bbox = [1 if xlo == 0 else 0, 1 if xhi == 17 else 0, 1, 1, 1, 1]
        glo = lo - np.array([0 if bbox[0] else g, 0, 0])
        ghi = hi + np.array([0 if bbox[1] else g, 0, 0])
        shape = ghi - glo
        act_lo = lo - glo + np.array([1 if bbox[0] else 0, 1, 1])  # outer boundary points are not active
        act_hi = hi - glo - np.array([1 if bbox[1] else 0, 1, 1])
        comps.append(dict(rl=0, c=c, origin=-4.0 + 0.5 * glo, delta=0.5, shape=shape, nghost=g, bbox=bbox,
                          active=[(act_lo, act_hi)]))
    # rl1: points 0..20 at dx 0.25 starting at x=-2, y=z=-2.5 (interior incl. buffers), ghosts outside
    lo = np.array([-2.0, -2.5, -2.5])
    n = np.array([21, 21, 21])
    shape = n + 2 * g
    comps.append(dict(rl=1, c=0, origin=lo - g * 0.25, delta=0.25, shape=shape, nghost=g, bbox=[0] * 6,
                      active=[(np.array([g + 2] * 3), shape - g - 2)]))
    return comps


def write_carpet(directory: Path, it=0, time=1.5, variables=CARPET_VARS):
    """Carpet-style ``<var>.xyz.file_0.h5`` files; returns the component list."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    comps = carpet_components(it)
    for var, thorn in variables.items():
        with h5py.File(directory / f"{var}.xyz.file_0.h5", "w") as f:
            for cp in comps:
                ax = [cp["origin"][d] + cp["delta"] * np.arange(cp["shape"][d]) for d in range(3)]
                X, Y, Z = np.meshgrid(*ax, indexing="ij")
                data = field(var, X, Y, Z)
                name = f"{thorn}::{var} it={it} tl=0 rl={cp['rl']} c={cp['c']}"
                ds = f.create_dataset(name, data=data.T)  # Carpet stores (z, y, x)
                ds.attrs["origin"] = np.asarray(cp["origin"], float)
                ds.attrs["delta"] = np.full(3, cp["delta"])
                ds.attrs["time"] = time
                ds.attrs["cctk_nghostzones"] = np.full(3, cp["nghost"], np.int32)
                ds.attrs["cctk_bbox"] = np.asarray(cp["bbox"], np.int32)
                ds.attrs["iorigin"] = np.zeros(3, np.int32)
                ds.attrs["active"] = _active(cp["active"])
    return comps


# --------------------------------------------------------------------- CarpetX
CX_GHOSTS = 3
CX_X0 = np.array([-8.0, -8.0, -8.0])


def carpetx_levels():
    """rl0: 16^3 cells of dx 1 in two boxes split at x cell 8; rl1: dx 0.5,
    one box of cells [10, 22)^3 (covering rl0 cells 5..10)."""
    return {
        0: (np.array([1.0, 1.0, 1.0]), [(np.array([0, 0, 0]), np.array([8, 16, 16])),
                                         (np.array([8, 0, 0]), np.array([16, 16, 16]))]),
        1: (np.array([0.5, 0.5, 0.5]), [(np.array([10, 10, 10]), np.array([22, 22, 22]))]),
    }


def cell_centres(level_dx, lo, n):
    return [CX_X0[d] + (lo[d] + np.arange(n[d]) + 0.5) * level_dx[d] for d in range(3)]


def _silo_obj(f, name, silo_type, fields):
    """A Silo object: a committed datatype with a compound ``silo`` attribute."""
    f[name] = np.dtype("int32")
    dt, vals = [], []
    for k, v in fields.items():
        if isinstance(v, (str, bytes)):
            b = v.encode() if isinstance(v, str) else v
            dt.append((k, f"S{len(b) + 1}"))
            vals.append(b)
        else:
            a = np.asarray(v)
            dt.append((k, a.dtype, a.shape) if a.shape else (k, a.dtype))
            vals.append(a)
    rec = np.array(tuple(vals), dtype=dt)
    f[name].attrs["silo"] = rec
    f[name].attrs["silo_type"] = np.int32(silo_type)


def write_silo(directory: Path, sim="run", it=0, time=2.0, names=None):
    """CarpetX-style 3D Silo output: metafile plus one data file."""
    from aweSOM.io import ASTERX_NAMES

    names = names or [c for _, c in ASTERX_NAMES.values()]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = directory / f"{sim}.it{it:08d}.silo"
    sub = f"{sim}.it{it:08d}.silo.dir"
    (directory / sub).mkdir(exist_ok=True)
    dname = f"{sim}.it{it:08d}.p000000.silo"
    g = CX_GHOSTS
    counter = [0]

    def raw(f, data):
        counter[0] += 1
        p = f"/.silo/#{counter[0]:06d}"
        f.create_dataset(p, data=data)
        return p

    canon = {c: k for k, (_, c) in ASTERX_NAMES.items()}
    meshes, varlists = [], {n: [] for n in names}
    with h5py.File(directory / sub / dname, "w") as f:
        f.create_group(".silo")
        for rl, (dx, boxes) in carpetx_levels().items():
            for c, (lo, hi) in enumerate(boxes):
                n = hi - lo
                tag = f"m0000_rl{rl:02d}_c{c:08d}"
                mesh = f"box_ghosts03_03_03_{tag}"
                coords = [CX_X0[d] + (lo[d] - g + np.arange(n[d] + 2 * g + 1)) * dx[d] for d in range(3)]
                _silo_obj(f, mesh, 130, {
                    "coord0": raw(f, coords[0]), "coord1": raw(f, coords[1]), "coord2": raw(f, coords[2]),
                    "ndims": np.int32(3), "dims": (n + 2 * g + 1).astype(np.int32),
                    "min_index": np.full(3, g, np.int32), "max_index": (n + g).astype(np.int32),
                })
                meshes.append(f"{sub}/{dname}:{mesh}")
                X, Y, Z = np.meshgrid(*cell_centres(dx, lo - g, n + 2 * g), indexing="ij")
                for name in names:
                    data = field(canon.get(name, name), X, Y, Z)
                    dims = (n + 2 * g).astype(np.int32)
                    obj = f"{name}_{tag}"
                    _silo_obj(f, obj, 501, {
                        "meshid": mesh, "value0": raw(f, np.ascontiguousarray(data.T).reshape(tuple(dims))),
                        "ndims": np.int32(3), "centering": np.int32(111), "dims": dims,
                        "align": np.full(3, 0.5),
                    })
                    varlists[name].append(f"{sub}/{dname}:{obj}")
    with h5py.File(meta, "w") as f:
        f.create_group(".silo")
        counter[0] = 0
        lst = ";".join(meshes).encode()
        _silo_obj(f, "gh_ghosts03_03_03", 520, {"nblocks": np.int32(len(meshes)),
                                                  "meshnames": raw(f, np.frombuffer(lst, np.uint8))})
        for name, entries in varlists.items():
            lst = ";".join(entries).encode()
            _silo_obj(f, name, 521, {"nvars": np.int32(len(entries)), "varnames": raw(f, np.frombuffer(lst, np.uint8))})
        cx = f.create_group("CarpetX")
        for rl, (dx, boxes) in carpetx_levels().items():
            cx.create_dataset(f"FabArrayBase_m0000_rl{rl:02d}",
                              data=np.array([np.r_[lo, hi - 1] for lo, hi in boxes], np.int32))
        cx.create_dataset("nlevels", data=np.array([2], np.int32))
        f.create_dataset("dtime", data=np.array([time]))
        f.create_dataset("cycle", data=np.array([it], np.int32))
    return meta


def write_openpmd(directory: Path, sim="run", it=0, time=2.0):
    """CarpetX-style openPMD-HDF5 output (file based), from CarpetX's io_openpmd.cxx."""
    from aweSOM.io import ASTERX_NAMES

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sim}.it{it:08d}.h5"
    canon = {c: k for k, (_, c) in ASTERX_NAMES.items()}
    groups = {}
    for _, (grp, comp) in ASTERX_NAMES.items():
        groups.setdefault(grp, []).append(comp)
    with h5py.File(path, "w") as f:
        f.attrs.update(openPMD="1.1.0", basePath="/data/%T/", meshesPath="meshes/", iterationEncoding="fileBased",
                       iterationFormat=f"{sim}.it%08T.h5")
        itg = f.create_group(f"data/{it}")
        itg.attrs.update(time=time, dt=0.1, numLevels_patch00=np.int64(2))
        levels = carpetx_levels()
        for rl, (dx, boxes) in levels.items():
            ncell = (16 * 2 ** rl) * np.ones(3, int)
            itg.attrs[f"chunkInfo_patch00_lev{rl:02d}"] = np.concatenate(
                [np.r_[lo[::-1], hi[::-1]] for lo, hi in boxes]).astype(np.int64)
            for grp, comps in groups.items():
                mg = itg.create_group(f"meshes/{grp}_patch00_lev{rl:02d}")
                mg.attrs.update(axisLabels=np.array([b"z", b"y", b"x"]), gridSpacing=dx[::-1],
                                gridGlobalOffset=CX_X0[::-1], geometry="cartesian")
                for comp in comps:
                    dense = np.zeros(tuple(ncell[::-1] + 1))
                    for c, (lo, hi) in enumerate(boxes):
                        X, Y, Z = np.meshgrid(*cell_centres(dx, lo, hi - lo), indexing="ij")
                        dense[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = field(canon[comp], X, Y, Z).T
                    ds = mg.create_dataset(comp, data=dense)
                    ds.attrs["position"] = np.full(3, 0.5)
    return path


def write_tsv(path: Path, var="hydrobasex_rho"):
    rows = []
    for lev in (0, 1):
        for i in range(-3, 5):
            x = -1.0 + (i + 0.5) * 0.5 ** lev
            rows.append((1024, 2.5, 0, lev, i, 7, 7, x, -0.25, -0.25, field("rho_b", x, -0.25, -0.25)))
    with open(path, "w") as f:
        f.write("# hydrobasex-rho.it001024.x.tsv\n")
        f.write("# 1:iteration\t2:time\t3:patch\t4:level\t5:i\t6:j\t7:k\t8:x\t9:y\t10:z\t11:rho\n")
        for r in rows:
            f.write("\t".join(str(v) if isinstance(v, int) else repr(float(v)) for v in r) + "\n")
    return np.array(rows)
