"""Per-rank sharded ``.npy`` outputs, lazy reader, and verified stitcher.

Multi-rank SOM labelling writes one directory ``<stem>.parts/`` per output array:
each rank ``r`` writes only its own contiguous row slice into ``part-RRRR.npy``
with a single writer per inode (preserving network filesystem write caching),
and rank 0 atomically publishes ``manifest.json`` last after every rank has
``fsync``ed and closed its part file.

:func:`open_npy` opens either a plain ``.npy`` file or a ``<stem>.parts/``
directory transparently. :class:`ShardedNpyArray` exposes one read-only array
over the parts using lazy per-part memory maps and ``O(log R)`` binary search on
row boundaries. :func:`stitch_npy` (and ``python -m aweSOM.npy_parts stitch``)
assembles a ``<stem>.parts/`` directory into a single ``.npy`` file bit-for-bit
identical to :func:`numpy.save`, checking every part's CRC-32 against
``manifest.json``.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MANIFEST_NAME",
    "PARTS_SUFFIX",
    "ShardedNpyArray",
    "copy_npy_output",
    "npy_file_for",
    "npy_output_exists",
    "open_npy",
    "part_filename",
    "parts_dir_for",
    "remove_npy_output",
    "stitch_npy",
    "verify_npy_parts",
    "write_npy_manifest",
]

PARTS_SUFFIX = ".parts"
MANIFEST_NAME = "manifest.json"
FORMAT_NAME = "awesom-npy-parts"
FORMAT_VERSION = 1


def part_filename(rank: int, ranks: int) -> str:
    """Zero-padded part file name for ``rank`` of ``ranks`` (at least 4 digits)."""
    rank, ranks = int(rank), int(ranks)
    if ranks < 1 or rank < 0 or rank >= ranks:
        raise ValueError(f"invalid rank {rank} for {ranks} rank(s)")
    width = max(4, len(str(ranks - 1)))
    return f"part-{rank:0{width}d}.npy"


def _check_bare_filename(name: Any) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or Path(name).name != name
        or Path(name).is_absolute()
    ):
        raise ValueError(f"invalid part filename in manifest: {name!r}")
    return name


def _parse_npy_target(path: str | os.PathLike[str]) -> tuple[str, str]:
    """Return ``(kind, stem)`` where ``kind`` is ``'npy'`` or ``'parts'``.

    Uses exact suffix stripping (never :meth:`Path.with_suffix`) so dotted
    legacy stems such as ``labels.bbh_0003-91-64-0.5-1u.npy`` stay intact.
    Bare stems without ``.npy`` or ``.parts`` are rejected.
    """
    raw = os.fsdecode(path)
    if len(raw) > 1:
        raw = raw.rstrip("/")
        if os.sep != "/":
            raw = raw.rstrip(os.sep)
    for sep in ("/", os.sep):
        suffix = sep + MANIFEST_NAME
        if raw.endswith(suffix):
            parent = raw[: -len(suffix)]
            if not parent.endswith(PARTS_SUFFIX):
                raise ValueError(f"{path}: {MANIFEST_NAME} must live inside a '*{PARTS_SUFFIX}' directory")
            stem = parent[: -len(PARTS_SUFFIX)]
            if not Path(stem).name:
                raise ValueError(f"{path}: empty output stem")
            return "parts", stem
    if raw.endswith(PARTS_SUFFIX):
        stem = raw[: -len(PARTS_SUFFIX)]
        if not Path(stem).name:
            raise ValueError(f"{path}: empty output stem")
        return "parts", stem
    if raw.endswith(".npy"):
        stem = raw[:-4]
        if not Path(stem).name:
            raise ValueError(f"{path}: empty output stem")
        return "npy", stem
    raise ValueError(f"{path}: expected a '*.npy' or '*{PARTS_SUFFIX}' path")


def parts_dir_for(path: str | os.PathLike[str]) -> Path:
    """Map ``<stem>.npy``, ``<stem>.parts``, or ``<stem>.parts/manifest.json`` to ``Path('<stem>.parts')``."""
    _, stem = _parse_npy_target(path)
    return Path(stem + PARTS_SUFFIX)


def npy_file_for(path: str | os.PathLike[str]) -> Path:
    """Map ``<stem>.npy``, ``<stem>.parts``, or ``<stem>.parts/manifest.json`` to ``Path('<stem>.npy')``."""
    _, stem = _parse_npy_target(path)
    return Path(stem + ".npy")


def _fsync_dir(directory: Path) -> bool:
    """fsync ``directory``; tolerate filesystems that return EINVAL/ENOTSUP on directories."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
        return True
    except OSError as exc:
        allowed = {errno.EINVAL, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)}
        if exc.errno in allowed:
            return False
        raise
    finally:
        os.close(fd)


def _read_npy_header(fh) -> tuple[tuple[int, ...], bool, np.dtype, int]:
    version = np.lib.format.read_magic(fh)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
    elif version in ((2, 0), (3, 0)):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
    else:
        raise ValueError(f"unsupported .npy format version {version}")
    return tuple(int(d) for d in shape), bool(fortran), np.dtype(dtype), int(fh.tell())


def _check_part_header(part_path: Path, expected_dtype: np.dtype, expected_shape: tuple[int, ...]) -> int:
    """Validate a part ``.npy`` file's header and exact byte size; return payload byte offset."""
    with open(part_path, "rb") as fh:
        shape, fortran, dtype, offset = _read_npy_header(fh)
        actual_size = os.fstat(fh.fileno()).st_size
    if fortran:
        raise ValueError(f"{part_path}: Fortran-order array is not supported")
    if dtype != expected_dtype or shape != expected_shape:
        raise ValueError(
            f"{part_path}: header has shape {shape} {dtype}, expected {expected_shape} {expected_dtype}"
        )
    expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * expected_dtype.itemsize
    if actual_size != offset + expected_bytes:
        raise ValueError(
            f"{part_path}: file is {actual_size} bytes, expected {offset + expected_bytes}"
        )
    return offset


def _plain_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain_json(value.tolist())
    if isinstance(value, np.generic):
        return _plain_json(value.item())
    return value


def _validate_manifest_dict(data: Any, manifest_path: Path) -> tuple[np.dtype, tuple[int, ...], list[dict[str, Any]]]:
    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path}: manifest must be a JSON object")
    fmt = data.get("format")
    if fmt != FORMAT_NAME:
        raise ValueError(f"{manifest_path}: unsupported manifest format {fmt!r}, expected {FORMAT_NAME!r}")
    ver = data.get("version")
    if type(ver) is not int or ver != FORMAT_VERSION:
        raise ValueError(f"{manifest_path}: unsupported manifest version {ver!r}, expected {FORMAT_VERSION}")
    raw_dtype = data.get("dtype")
    if not isinstance(raw_dtype, str):
        raise ValueError(f"{manifest_path}: invalid dtype {raw_dtype!r}; expected scalar dtype string")
    try:
        dtype = np.dtype(raw_dtype)
    except Exception as exc:
        raise ValueError(f"{manifest_path}: invalid dtype {raw_dtype!r}") from exc
    if dtype.names is not None or dtype.subdtype is not None or dtype.hasobject:
        raise ValueError(f"{manifest_path}: non-scalar dtypes are not supported: {dtype!r}")
    raw_shape = data.get("shape")
    if not isinstance(raw_shape, list) or len(raw_shape) < 1 or any(type(d) is not int or d < 0 for d in raw_shape):
        raise ValueError(f"{manifest_path}: invalid shape {raw_shape!r}")
    shape = tuple(raw_shape)
    total_rows = data.get("total_rows")
    if type(total_rows) is not int or total_rows != shape[0]:
        raise ValueError(f"{manifest_path}: total_rows {total_rows!r} does not match shape {shape}")
    ranks = data.get("ranks")
    raw_parts = data.get("parts")
    if type(ranks) is not int or ranks < 1 or not isinstance(raw_parts, list) or len(raw_parts) != ranks:
        raise ValueError(f"{manifest_path}: expected {ranks!r} part entries")
    row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
    seen_files: set[str] = set()
    validated: list[dict[str, Any]] = []
    pos = 0
    for expected_rank, entry in enumerate(raw_parts):
        if not isinstance(entry, dict):
            raise ValueError(f"{manifest_path}: part {expected_rank} must be an object")
        r = entry.get("rank")
        if type(r) is not int or r != expected_rank:
            raise ValueError(f"{manifest_path}: part {expected_rank} has rank {r!r}")
        fname = _check_bare_filename(entry.get("file"))
        if fname in seen_files:
            raise ValueError(f"{manifest_path}: duplicate part file {fname!r}")
        seen_files.add(fname)
        start, stop, rows = entry.get("start"), entry.get("stop"), entry.get("rows")
        if any(type(x) is not int for x in (start, stop, rows)):
            raise ValueError(f"{manifest_path}: part {r} start/stop/rows must be integers")
        if start != pos or stop < start or rows != stop - start:
            raise ValueError(
                f"{manifest_path}: part {r} range [{start}, {stop}) (rows={rows}) is not contiguous at row {pos}"
            )
        nbytes = entry.get("bytes")
        if type(nbytes) is not int or nbytes != rows * row_bytes:
            raise ValueError(f"{manifest_path}: part {r} bytes {nbytes!r} != {rows * row_bytes}")
        crc = entry.get("crc32")
        if type(crc) is not int or not (0 <= crc < 2**32):
            raise ValueError(f"{manifest_path}: part {r} has invalid crc32 {crc!r}")
        if rows == 0 and crc != 0:
            raise ValueError(f"{manifest_path}: empty part {r} must have crc32 0, got {crc}")
        validated.append(
            dict(rank=r, file=fname, start=start, stop=stop, rows=rows, bytes=nbytes, crc32=crc)
        )
        pos = stop
    if pos != total_rows:
        raise ValueError(f"{manifest_path}: parts end at row {pos}, expected total_rows={total_rows}")
    return dtype, shape, validated


def _load_and_validate_manifest(parts_dir: Path) -> tuple[dict[str, Any], np.dtype, tuple[int, ...], list[dict[str, Any]]]:
    if not parts_dir.is_dir():
        raise FileNotFoundError(parts_dir)
    manifest_path = parts_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {MANIFEST_NAME} in {parts_dir}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: invalid JSON: {exc}") from exc
    dtype, shape, parts = _validate_manifest_dict(data, manifest_path)
    tail = shape[1:]
    for p in parts:
        _check_part_header(parts_dir / p["file"], dtype, (p["rows"], *tail))
    return data, dtype, shape, parts


def write_npy_manifest(
    parts_dir: str | os.PathLike[str],
    *,
    dtype: Any,
    shape: Sequence[int],
    parts: Sequence[dict[str, Any]],
    provenance: Optional[dict[str, Any]] = None,
    sync: bool = True,
) -> tuple[Path, bool]:
    """Validate part headers/lengths and contiguous coverage, then atomically publish ``manifest.json``.

    Returns ``(manifest_path, directory_fsync)`` where ``directory_fsync`` is
    True when ``os.fsync`` succeeded on ``parts_dir`` (and False when skipped
    or unsupported by the filesystem).
    """
    directory = Path(parts_dir)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    manifest_path = directory / MANIFEST_NAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    dt = np.dtype(dtype)
    if dt.names is not None or dt.subdtype is not None or dt.hasobject:
        raise ValueError(f"{manifest_path}: non-scalar dtypes are not supported: {dt!r}")
    descr = np.lib.format.dtype_to_descr(dt)
    if not isinstance(descr, str):
        raise ValueError(f"{manifest_path}: non-scalar dtype descriptor is not supported: {descr!r}")
    shp = [int(d) for d in shape]
    doc: dict[str, Any] = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "dtype": descr,
        "shape": shp,
        "total_rows": shp[0] if shp else 0,
        "ranks": len(parts),
        "parts": _plain_json(list(parts)),
        "provenance": _plain_json(dict(provenance or {})),
    }
    payload_str = json.dumps(doc, indent=2, allow_nan=False) + "\n"
    _, clean_shape, clean_parts = _validate_manifest_dict(json.loads(payload_str), manifest_path)
    tail = clean_shape[1:]
    for p in clean_parts:
        _check_part_header(directory / p["file"], dt, (p["rows"], *tail))

    payload = payload_str.encode("utf-8")
    partial = directory / (MANIFEST_NAME + ".partial")
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
    try:
        view = memoryview(payload)
        while len(view):
            wrote = os.write(fd, view)
            view = view[wrote:]
        if sync:
            os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise
    os.close(fd)
    try:
        os.link(partial, manifest_path)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
    dir_synced = _fsync_dir(directory) if sync else False
    return manifest_path, dir_synced


class ShardedNpyArray(np.lib.mixins.NDArrayOperatorsMixin):
    """Lazy read-only array view over ``<stem>.parts/manifest.json`` and its part ``.npy`` files."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        parts_dir = parts_dir_for(path)
        manifest, dtype, shape, parts = _load_and_validate_manifest(parts_dir)
        self.path = parts_dir
        self.manifest_path = parts_dir / MANIFEST_NAME
        self.filename = parts_dir
        self.manifest = manifest
        self.dtype = dtype
        self.shape = shape
        self._tail = shape[1:]
        self.ndim = len(shape)
        self.size = int(np.prod(shape, dtype=np.int64))
        self.itemsize = int(dtype.itemsize)
        self.nbytes = self.size * self.itemsize
        self._starts = np.array([p["start"] for p in parts], dtype=np.int64)
        self._stops = np.array([p["stop"] for p in parts], dtype=np.int64)
        self._files = [parts_dir / p["file"] for p in parts]
        self._mmaps: list[Optional[np.ndarray]] = [None] * len(parts)

    def _part(self, r: int) -> np.ndarray:
        mm = self._mmaps[r]
        if mm is None:
            mm = np.load(self._files[r], mmap_mode="r", allow_pickle=False)
            self._mmaps[r] = mm
        return mm

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"ShardedNpyArray(path={str(self.path)!r}, shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, key: Any) -> Any:
        n = self.shape[0]
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            idx = int(key)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"index {int(key)} is out of bounds for axis 0 with size {n}")
            r = int(np.searchsorted(self._stops, idx, side="right"))
            return self._part(r)[idx - int(self._starts[r])]

        if isinstance(key, slice) and key.step in (None, 1):
            start, stop, _ = key.indices(n)
            if stop <= start:
                return np.empty((0, *self._tail), dtype=self.dtype)
            r0 = int(np.searchsorted(self._stops, start, side="right"))
            s0 = int(self._starts[r0])
            if stop <= int(self._stops[r0]):
                return self._part(r0)[start - s0 : stop - s0]
            r1 = int(np.searchsorted(self._starts, stop, side="left"))
            out = np.empty((stop - start, *self._tail), dtype=self.dtype)
            pos = 0
            for r in range(r0, r1):
                sr, er = int(self._starts[r]), int(self._stops[r])
                a = max(start, sr) - sr
                b = min(stop, er) - sr
                if b > a:
                    out[pos : pos + (b - a)] = self._part(r)[a:b]
                    pos += b - a
            return out

        if isinstance(key, np.ndarray) and key.ndim == 1 and np.issubdtype(key.dtype, np.integer):
            if len(key) == 0:
                return np.empty((0, *self._tail), dtype=self.dtype)
            idx = key.astype(np.int64, copy=False)
            if np.any(idx < 0):
                idx = np.where(idx < 0, idx + n, idx)
            if np.any((idx < 0) | (idx >= n)):
                bad = int(key[np.flatnonzero((idx < 0) | (idx >= n))[0]])
                raise IndexError(f"index {bad} is out of bounds for axis 0 with size {n}")
            part_ix = np.searchsorted(self._stops, idx, side="right")
            out = np.empty((len(idx), *self._tail), dtype=self.dtype)
            for r in np.unique(part_ix):
                r_int = int(r)
                mask = part_ix == r_int
                out[mask] = self._part(r_int)[idx[mask] - int(self._starts[r_int])]
            return out

        return np.asarray(self)[key]

    def __array__(self, dtype: Any = None, copy: Optional[bool] = None) -> np.ndarray:
        non_empty = [r for r in range(len(self._files)) if self._stops[r] > self._starts[r]]
        target_dt = self.dtype if dtype is None else np.dtype(dtype)
        if copy is False:
            if len(non_empty) == 1 and target_dt == self.dtype:
                return np.asarray(self._part(non_empty[0]))
            if self.size == 0 and len(self._files) >= 1 and target_dt == self.dtype:
                return np.asarray(self._part(0))
            raise ValueError("unable to avoid copy while materializing multi-part ShardedNpyArray")
        out = np.empty(self.shape, dtype=self.dtype)
        for r in non_empty:
            out[int(self._starts[r]) : int(self._stops[r])] = self._part(r)
        if target_dt != self.dtype:
            return out.astype(target_dt)
        return out

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        outs = kwargs.get("out")
        if outs is not None:
            out_tuple = outs if isinstance(outs, tuple) else (outs,)
            if any(isinstance(o, ShardedNpyArray) for o in out_tuple):
                raise ValueError("ShardedNpyArray is read-only")
        converted = tuple(np.asarray(x) if isinstance(x, ShardedNpyArray) else x for x in inputs)
        return getattr(ufunc, method)(*converted, **kwargs)

    def reshape(self, *shape: Any) -> ShardedNpyArray | np.ndarray:
        new_shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else tuple(shape)
        if self.ndim == 1 and new_shape in ((-1,), (self.shape[0],)):
            return self
        return np.asarray(self).reshape(*shape)

    def _reduce_minmax(self, op_name: str, axis: Any = None, **kwargs: Any) -> Any:
        if axis is not None or kwargs:
            return getattr(np.asarray(self), op_name)(axis=axis, **kwargs)
        if self.size == 0:
            return getattr(np.empty((0,), dtype=self.dtype), op_name)()
        tail_elems = max(1, int(np.prod(self._tail, dtype=np.int64)))
        step = max(1, (1 << 24) // tail_elems)
        combine = np.minimum if op_name == "min" else np.maximum
        acc = None
        for r in range(len(self._files)):
            m = int(self._stops[r] - self._starts[r])
            if not m:
                continue
            part = self._part(r)
            for a in range(0, m, step):
                val = getattr(part[a : min(a + step, m)], op_name)()
                acc = val if acc is None else combine(acc, val)
        return acc

    def min(self, axis: Any = None, **kwargs: Any) -> Any:
        return self._reduce_minmax("min", axis=axis, **kwargs)

    def max(self, axis: Any = None, **kwargs: Any) -> Any:
        return self._reduce_minmax("max", axis=axis, **kwargs)

    def flush(self) -> None:
        return None

    def release_mapping(self) -> None:
        import mmap

        if not hasattr(mmap, "MADV_DONTNEED"):
            return
        for part in self._mmaps:
            if part is None:
                continue
            mapping = getattr(part, "_mmap", None)
            if mapping is not None and hasattr(mapping, "madvise"):
                mapping.madvise(mmap.MADV_DONTNEED)

    def close(self) -> None:
        for i, part in enumerate(self._mmaps):
            if part is None:
                continue
            mapping = getattr(part, "_mmap", None)
            self._mmaps[i] = None
            del part
            if mapping is not None:
                mapping.close()


def npy_output_exists(path: str | os.PathLike[str]) -> bool:
    """True when ``<stem>.npy`` is a file or ``<stem>.parts/manifest.json`` is a file.

    Raises :class:`FileNotFoundError` if ``<stem>.parts`` exists as a directory
    without ``manifest.json`` (an incomplete or failed sharded write).
    """
    kind, stem = _parse_npy_target(path)
    parts_dir = Path(stem + PARTS_SUFFIX)
    if kind == "npy":
        npy_path = Path(stem + ".npy")
        if npy_path.is_file():
            return True
    if parts_dir.is_dir():
        manifest_path = parts_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing {MANIFEST_NAME} in {parts_dir}")
        return True
    return False


def open_npy(path: str | os.PathLike[str], mmap_mode: Optional[str] = "r") -> np.ndarray | ShardedNpyArray:
    """Open a plain ``.npy`` file or a sharded ``<stem>.parts/`` dataset.

    Mode contract:
    - ``mmap_mode="r"`` (default): returns a read-only ``numpy.memmap`` for a
      plain ``.npy`` file or a lazy :class:`ShardedNpyArray` for ``.parts``.
    - ``mmap_mode=None``: returns an eager in-memory ``numpy.ndarray`` in both
      layouts.
    - Any other ``mmap_mode`` is forwarded to ``numpy.load`` on plain ``.npy``
      files and raises :class:`ValueError` on ``.parts`` datasets.
    """
    kind, stem = _parse_npy_target(path)
    if kind == "npy":
        npy_path = Path(stem + ".npy")
        if npy_path.is_file():
            return np.load(npy_path, mmap_mode=mmap_mode, allow_pickle=False)
        parts_dir = Path(stem + PARTS_SUFFIX)
        if not parts_dir.is_dir():
            raise FileNotFoundError(npy_path)
    else:
        parts_dir = Path(stem + PARTS_SUFFIX)

    if mmap_mode == "r":
        return ShardedNpyArray(parts_dir)
    if mmap_mode is None:
        sharded = ShardedNpyArray(parts_dir)
        try:
            return np.asarray(sharded)
        finally:
            sharded.close()
    raise ValueError(
        f"unsupported mmap_mode={mmap_mode!r} for sharded dataset {parts_dir}; expected 'r' or None"
    )


def verify_npy_parts(path: str | os.PathLike[str], *, block: int = 2**24) -> dict[str, Any]:
    """Verify every part's header, length, and payload CRC-32 against ``manifest.json``."""
    if int(block) < 1:
        raise ValueError("block must be positive")
    t0 = time.perf_counter()
    parts_dir = parts_dir_for(path)
    manifest, dtype, shape, parts = _load_and_validate_manifest(parts_dir)
    tail = shape[1:]
    row_bytes = int(np.prod(tail, dtype=np.int64)) * dtype.itemsize
    total_bytes = 0
    for p in parts:
        part_path = parts_dir / p["file"]
        n = p["rows"]
        crc = 0
        with open(part_path, "rb", buffering=0) as fh:
            _read_npy_header(fh)
            if n:
                buf = np.empty((min(n, int(block)), *tail), dtype=dtype)
                for a in range(0, n, int(block)):
                    m = min(int(block), n - a)
                    view = memoryview(buf[:m]).cast("B")
                    got = 0
                    while got < len(view):
                        k = fh.readinto(view[got:])
                        if not k:
                            raise IOError(f"{part_path}: short read")
                        got += k
                    crc = zlib.crc32(buf[:m], crc)
        if crc != p["crc32"]:
            raise ValueError(
                f"{part_path}: rank {p['rank']} CRC-32 {crc} differs from manifest {p['crc32']}"
            )
        total_bytes += n * row_bytes
    return dict(
        path=str(parts_dir),
        ranks=len(parts),
        total_rows=shape[0],
        bytes=total_bytes,
        seconds=time.perf_counter() - t0,
        manifest=manifest,
    )


def _remove_parts_dir(parts_dir: Path) -> None:
    """Remove ``parts_dir`` safely without ``shutil.rmtree`` (never following a symlink root)."""
    if parts_dir.is_symlink():
        parts_dir.unlink()
        return
    manifest_path = parts_dir / MANIFEST_NAME
    part_names: list[str] = []
    if manifest_path.is_symlink():
        manifest_path.unlink()
    elif manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        _, _, parts = _validate_manifest_dict(data, manifest_path)
        part_names = [p["file"] for p in parts]
        manifest_path.unlink()
    partial = parts_dir / (MANIFEST_NAME + ".partial")
    if partial.is_symlink() or partial.is_file():
        partial.unlink()
    if part_names:
        for fname in part_names:
            target = parts_dir / fname
            if target.is_symlink() or target.is_file():
                target.unlink()
    else:
        for entry in parts_dir.iterdir():
            if entry.name.startswith("part-") and entry.name.endswith(".npy") and (entry.is_symlink() or entry.is_file()):
                entry.unlink()
    parts_dir.rmdir()


def remove_npy_output(path: str | os.PathLike[str]) -> None:
    """Remove ``<stem>.npy`` and/or ``<stem>.parts`` (unlinking ``manifest.json`` first).

    Never calls ``shutil.rmtree`` and never follows a ``.parts`` directory symlink.
    When both ``<stem>.npy`` and ``<stem>.parts`` exist, removes both.
    """
    _, stem = _parse_npy_target(path)
    npy_path = Path(stem + ".npy")
    parts_dir = Path(stem + PARTS_SUFFIX)
    removed = False
    if npy_path.is_symlink() or npy_path.is_file():
        npy_path.unlink()
        removed = True
    if parts_dir.is_symlink():
        parts_dir.unlink()
        removed = True
    elif parts_dir.is_dir():
        _remove_parts_dir(parts_dir)
        removed = True
    if not removed:
        raise FileNotFoundError(path)


def copy_npy_output(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> Path:
    """Copy ``src`` to ``dst``'s stem, copying ``.npy`` when present (as :func:`open_npy` opens)
    or ``.parts`` otherwise (publishing ``manifest.json`` last)."""
    src_kind, src_stem = _parse_npy_target(src)
    _, dst_stem = _parse_npy_target(dst)
    src_npy = Path(src_stem + ".npy")
    src_parts = Path(src_stem + PARTS_SUFFIX)
    dst_npy = Path(dst_stem + ".npy")
    dst_parts = Path(dst_stem + PARTS_SUFFIX)
    if dst_npy.exists() or dst_npy.is_symlink() or dst_parts.exists() or dst_parts.is_symlink():
        raise FileExistsError(f"destination already exists for stem {dst_stem}")

    if src_kind == "npy" and src_npy.is_file():
        shutil.copyfile(src_npy, dst_npy)
        return dst_npy

    _, _, _, parts = _load_and_validate_manifest(src_parts)
    staging = Path(dst_stem + PARTS_SUFFIX + ".partial")
    if staging.is_symlink():
        staging.unlink()
    elif staging.is_dir():
        _remove_parts_dir(staging)
    elif staging.exists():
        staging.unlink()
    staging.mkdir()
    try:
        for p in parts:
            shutil.copyfile(src_parts / p["file"], staging / p["file"])
        shutil.copyfile(src_parts / MANIFEST_NAME, staging / (MANIFEST_NAME + ".partial"))
        os.link(staging / (MANIFEST_NAME + ".partial"), staging / MANIFEST_NAME)
        (staging / (MANIFEST_NAME + ".partial")).unlink()
        os.rename(staging, dst_parts)
    except BaseException:
        if staging.is_symlink():
            staging.unlink()
        elif staging.is_dir():
            _remove_parts_dir(staging)
        raise
    return dst_parts


def stitch_npy(
    source: str | os.PathLike[str],
    dest: Optional[str | os.PathLike[str]] = None,
    *,
    delete_parts: bool = False,
    block: int = 2**24,
) -> dict[str, Any]:
    """Assemble ``<stem>.parts`` into a single ``.npy`` file, checking every part's CRC-32.

    Checks that ``dest`` does not exist before reading any part bytes, truncates
    and reuses any stale ``<dest>.partial`` file, ``fsync``s the stitched file,
    atomically links it to ``dest``, ``fsync``s ``dest.parent`` when supported,
    and deletes ``<stem>.parts`` afterwards only when ``delete_parts=True``.
    """
    if int(block) < 1:
        raise ValueError("block must be positive")
    from . import stream

    t0 = time.perf_counter()
    parts_dir = parts_dir_for(source)
    _, dtype, shape, parts = _load_and_validate_manifest(parts_dir)
    dest_path = npy_file_for(parts_dir) if dest is None else Path(dest)
    if dest_path.exists() or dest_path.is_symlink():
        raise FileExistsError(f"destination already exists: {dest_path}")

    partial = Path(os.fsdecode(dest_path) + ".partial")
    if partial.exists() or partial.is_symlink():
        partial.unlink()

    writer = None
    copied: dict[str, Any] = {"read_seconds": 0.0, "write_seconds": 0.0, "bytes": 0}
    try:
        made = np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=shape)
        del made
        writer = stream.NpyWriter(partial)
        for p in parts:
            part_path = parts_dir / p["file"]
            crc = stream.copy_npy_rows(
                part_path, writer, p["start"], block=int(block), stats=copied, rows=p["rows"]
            )
            if crc != p["crc32"]:
                raise ValueError(
                    f"{part_path}: rank {p['rank']} CRC-32 {crc} differs from manifest {p['crc32']}"
                )
        writer.close(sync=True)
        fsync_seconds = writer.fsync_seconds
        writer = None
        os.link(partial, dest_path)
    except BaseException:
        if writer is not None:
            writer.close(sync=False)
        if partial.exists() or partial.is_symlink():
            partial.unlink()
        raise
    finally:
        if partial.exists() or partial.is_symlink():
            partial.unlink()

    dir_synced = _fsync_dir(dest_path.parent)
    if delete_parts:
        _remove_parts_dir(parts_dir)
    return dict(
        source=str(parts_dir),
        dest=str(dest_path),
        ranks=len(parts),
        rows=shape[0],
        bytes=copied["bytes"],
        read_seconds=copied["read_seconds"],
        write_seconds=copied["write_seconds"],
        fsync_seconds=fsync_seconds,
        directory_fsync=dir_synced,
        deleted_parts=bool(delete_parts),
        seconds=time.perf_counter() - t0,
    )


def _cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m aweSOM.npy_parts",
        description="Verify or stitch per-rank aweSOM .parts outputs.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    st = sub.add_parser("stitch", help="Assemble <stem>.parts into a single .npy file with CRC-32 checks.")
    st.add_argument("source", type=Path, help="Path to <stem>.parts, <stem>.parts/manifest.json, or <stem>.npy")
    st.add_argument("-o", "--output", type=Path, default=None, help="Output .npy path (default: <stem>.npy)")
    st.add_argument("--delete-parts", action="store_true", help="Remove <stem>.parts after publishing the .npy")
    st.add_argument("--block", type=int, default=2**24, help="Rows copied per block (default: 2^24)")

    vf = sub.add_parser("verify", help="Verify part headers, byte sizes, and CRC-32 checksums against manifest.json.")
    vf.add_argument("source", type=Path, help="Path to <stem>.parts, <stem>.parts/manifest.json, or <stem>.npy")
    vf.add_argument("--block", type=int, default=2**24, help="Rows read per block (default: 2^24)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _cli_parser().parse_args(argv)
    if args.command == "stitch":
        res = stitch_npy(args.source, args.output, delete_parts=args.delete_parts, block=args.block)
        print(json.dumps(res), flush=True)
        return 0
    res = verify_npy_parts(args.source, block=args.block)
    summary = {k: v for k, v in res.items() if k != "manifest"}
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
