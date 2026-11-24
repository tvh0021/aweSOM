"""Unit tests for aweSOM.npy_parts: sharded .parts layout, manifest validation,
ShardedNpyArray lazy indexing/operators, open_npy modes, and stitch_npy."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import aweSOM
from aweSOM import ShardedNpyArray, open_npy, stitch_npy
from aweSOM.npy_parts import (
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST_NAME,
    PARTS_SUFFIX,
    copy_npy_output,
    npy_file_for,
    npy_output_exists,
    part_filename,
    parts_dir_for,
    remove_npy_output,
    verify_npy_parts,
    write_npy_manifest,
)
from aweSOM.stream import NpyWriter


def make_sharded(base_npy: Path, arr: np.ndarray, edges: list[int], provenance: dict | None = None) -> Path:
    """Write ``arr`` across ``edges`` into ``<stem>.parts/`` and publish ``manifest.json``."""
    parts_dir = parts_dir_for(base_npy)
    parts_dir.mkdir(parents=True, exist_ok=False)
    ranks = len(edges) - 1
    tail = arr.shape[1:]
    row_bytes = int(np.prod(tail, dtype=np.int64)) * arr.dtype.itemsize
    entries = []
    for r, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        fname = part_filename(r, ranks)
        part_path = parts_dir / fname
        made = np.lib.format.open_memmap(part_path, mode="w+", dtype=arr.dtype, shape=(b - a, *tail))
        del made
        w = NpyWriter(part_path)
        w[0 : b - a] = arr[a:b]
        w.close()
        entries.append(
            dict(
                rank=r,
                file=fname,
                start=a,
                stop=b,
                rows=b - a,
                bytes=(b - a) * row_bytes,
                crc32=w.crc32,
            )
        )
    write_npy_manifest(parts_dir, dtype=arr.dtype, shape=arr.shape, parts=entries, provenance=provenance)
    return parts_dir


def test_part_filename_padding():
    assert part_filename(0, 1) == "part-0000.npy"
    assert part_filename(7, 8) == "part-0007.npy"
    assert part_filename(9999, 10000) == "part-9999.npy"
    assert part_filename(12, 10001) == "part-00012.npy"
    for bad in [(-1, 2), (2, 2), (0, 0)]:
        with pytest.raises(ValueError):
            part_filename(*bad)


def test_path_mapping_preserves_dotted_legacy_names_and_rejects_bare_stems(tmp_path):
    stem = "labels.bbh_0003-91-64-0.5-10000000-1u-mc0.05"
    npy = tmp_path / f"{stem}.npy"
    parts = tmp_path / f"{stem}.parts"
    manifest = parts / "manifest.json"
    for target in (npy, parts, str(parts) + "/", manifest):
        assert parts_dir_for(target) == parts
        assert npy_file_for(target) == npy

    for bare in (tmp_path / stem, tmp_path / "plain_stem", Path(".npy"), Path(".parts")):
        with pytest.raises(ValueError):
            parts_dir_for(bare)
        with pytest.raises(ValueError):
            open_npy(bare)

    ref = np.arange(37, dtype=np.uint8)
    make_sharded(npy, ref, [0, 15, 37])
    lazy = open_npy(npy, mmap_mode="r")
    eager = open_npy(npy, mmap_mode=None)
    assert isinstance(lazy, ShardedNpyArray)
    assert isinstance(eager, np.ndarray) and not isinstance(eager, np.memmap)
    np.testing.assert_array_equal(lazy, ref)
    np.testing.assert_array_equal(eager, ref)
    lazy.close()


@pytest.mark.parametrize(
    "dtype,shape,edges",
    [
        ("|u1", (1003,), [0, 502, 1003]),
        ("<u2", (1003,), [0, 0, 40, 41, 1003]),
        ("<i4", (2,), [0, 0, 1, 2]),
        ("<f8", (1003,), [0, 334, 668, 1003]),
        ("<f4", (101, 7), [0, 30, 30, 101]),
        ("|u1", (0,), [0, 0, 0]),
    ],
)
def test_round_trip_and_stitch_bitwise_equal_to_np_save(tmp_path, dtype, shape, edges):
    rng = np.random.default_rng(19)
    if np.dtype(dtype).kind in "iu":
        ref = rng.integers(0, 200, size=shape, dtype=dtype)
    else:
        ref = rng.normal(size=shape).astype(dtype)
    ref_path = tmp_path / "ref.npy"
    np.save(ref_path, ref)

    logical = tmp_path / "out-labels.npy"
    parts_dir = make_sharded(logical, ref, edges, provenance={"tag": "t", "seed": 42})
    assert npy_output_exists(logical) and npy_output_exists(parts_dir)

    arr = open_npy(logical)
    assert isinstance(arr, ShardedNpyArray)
    assert arr.shape == shape and arr.dtype == np.dtype(dtype)
    assert arr.ndim == len(shape) and len(arr) == shape[0]
    assert arr.size == ref.size and arr.itemsize == ref.itemsize and arr.nbytes == ref.nbytes
    assert arr.filename == parts_dir
    assert all(mm is None for mm in arr._mmaps)  # lazy: no mmap opened until indexed

    got = np.asarray(arr)
    assert got.dtype == ref.dtype and got.shape == ref.shape
    np.testing.assert_array_equal(got.view(np.uint8), ref.view(np.uint8))
    arr.release_mapping()
    arr.flush()
    arr.close()

    verified = verify_npy_parts(logical, block=13)
    assert verified["ranks"] == len(edges) - 1 and verified["total_rows"] == shape[0]

    stitched = stitch_npy(logical, block=17)
    assert Path(stitched["dest"]) == logical
    assert logical.read_bytes() == ref_path.read_bytes()


def test_lazy_slicing_indexing_and_reductions_match_ndarray(tmp_path):
    rng = np.random.default_rng(7)
    ref = rng.integers(0, 15, size=(127,), dtype=np.uint16)
    edges = [0, 0, 31, 32, 90, 127]
    parts_dir = make_sharded(tmp_path / "x.npy", ref, edges)
    arr = ShardedNpyArray(parts_dir)

    # Single-part slice only opens that single part's memmap and returns a view.
    sub = arr[35:50]
    np.testing.assert_array_equal(sub, ref[35:50])
    assert [mm is not None for mm in arr._mmaps] == [False, False, False, True, False]

    keys = [
        0,
        30,
        31,
        32,
        126,
        -1,
        -127,
        slice(None),
        slice(0, 0),
        slice(50, 20),
        slice(0, 31),
        slice(30, 33),
        slice(10, 120),
        slice(-20, -2),
        slice(5, 110, 3),
        slice(110, 5, -4),
        np.array([], dtype=np.int64),
        np.array([126, 0, 31, 31, 32, 89, 90, -3], dtype=np.int64),
        ref % 3 == 0,
        (slice(10, 50), None),
    ]
    for key in keys:
        np.testing.assert_array_equal(arr[key], ref[key])

    for bad_idx in (127, -128, np.array([0, 127]), np.array([-129])):
        with pytest.raises(IndexError):
            _ = arr[bad_idx]

    assert arr.min() == ref.min() and type(arr.min()) is type(ref.min())
    assert arr.max() == ref.max() and type(arr.max()) is type(ref.max())
    assert arr.reshape(-1) is arr and arr.reshape((127,)) is arr
    np.testing.assert_array_equal(arr.reshape(127, 1), ref.reshape(127, 1))
    arr.close()

    # Float reductions with NaN and 2-D axis reductions.
    fref = rng.normal(size=(45, 4)).astype(np.float64)
    f2 = ShardedNpyArray(make_sharded(tmp_path / "f2.npy", fref, [0, 20, 45]))
    assert f2.min() == fref.min() and f2.max() == fref.max()
    np.testing.assert_array_equal(f2.min(axis=0), fref.min(axis=0))
    np.testing.assert_array_equal(f2.max(axis=1), fref.max(axis=1))
    f2.close()

    fref_nan = fref.copy()
    fref_nan[33, 2] = np.nan
    fnan = ShardedNpyArray(make_sharded(tmp_path / "fnan.npy", fref_nan, [0, 20, 45]))
    assert np.isnan(fnan.min()) and np.isnan(fnan.max())
    fnan.close()


def test_operators_do_not_return_python_bool(tmp_path):
    ref = np.array([0, 1, 2, 1, 3, 1, 0], dtype=np.uint8)
    arr = open_npy(make_sharded(tmp_path / "ops.npy", ref, [0, 3, 7]))
    eq = arr == 1
    ne = arr != 1
    lt = arr < 2
    ge = 2 <= arr
    plus = arr + 10
    masked = np.where(arr == 1, 5.0, 0.0)
    assert isinstance(eq, np.ndarray) and eq.dtype == bool
    assert isinstance(ne, np.ndarray) and ne.dtype == bool
    assert isinstance(lt, np.ndarray) and lt.dtype == bool
    np.testing.assert_array_equal(eq, ref == 1)
    np.testing.assert_array_equal(ne, ref != 1)
    np.testing.assert_array_equal(lt, ref < 2)
    np.testing.assert_array_equal(ge, 2 <= ref)
    np.testing.assert_array_equal(plus, ref + 10)
    np.testing.assert_array_equal(masked, np.where(ref == 1, 5.0, 0.0))

    with pytest.raises(ValueError, match="read-only"):
        np.add(arr, 1, out=arr)
    arr.close()


def test_open_npy_modes_and_coexistence(tmp_path):
    ref = np.arange(50, dtype=np.int32)
    npy = tmp_path / "data.npy"
    parts = make_sharded(npy, ref, [0, 20, 50])

    # .parts modes: 'r' -> ShardedNpyArray, None -> ndarray, writable -> ValueError.
    a_r = open_npy(parts, mmap_mode="r")
    a_none = open_npy(parts, mmap_mode=None)
    assert isinstance(a_r, ShardedNpyArray)
    assert isinstance(a_none, np.ndarray) and not isinstance(a_none, np.memmap)
    np.testing.assert_array_equal(a_r, ref)
    np.testing.assert_array_equal(a_none, ref)
    a_r.close()
    for bad_mode in ("r+", "w+", "c"):
        with pytest.raises(ValueError, match="unsupported mmap_mode"):
            open_npy(parts, mmap_mode=bad_mode)

    # Coexistence after in-place stitch without deleting parts:
    stitch_npy(parts)
    assert npy.is_file() and parts.is_dir()
    opened_npy = open_npy(npy, mmap_mode="r")
    opened_parts = open_npy(parts, mmap_mode="r")
    assert isinstance(opened_npy, np.memmap)
    assert isinstance(opened_parts, ShardedNpyArray)
    np.testing.assert_array_equal(opened_npy, ref)
    np.testing.assert_array_equal(opened_parts, ref)
    opened_parts.close()


def test_manifest_missing_part_missing_and_validation_errors(tmp_path):
    ref = np.arange(20, dtype=np.uint16)
    incomplete = tmp_path / "inc.parts"
    incomplete.mkdir()
    np.save(incomplete / "part-0000.npy", ref)
    with pytest.raises(FileNotFoundError, match=MANIFEST_NAME):
        open_npy(tmp_path / "inc.npy")
    with pytest.raises(FileNotFoundError, match=MANIFEST_NAME):
        npy_output_exists(tmp_path / "inc.npy")

    # Unknown format / version and path-separator filename in manifest.
    parts = make_sharded(tmp_path / "v.npy", ref, [0, 10, 20])
    manifest_path = parts / MANIFEST_NAME
    orig = json.loads(manifest_path.read_text())

    for patch, match in [
        ({"format": "other"}, "unsupported manifest format"),
        ({"version": 99}, "unsupported manifest version"),
        ({"total_rows": 19}, "total_rows"),
    ]:
        bad = dict(orig, **patch)
        manifest_path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=match):
            open_npy(parts)

    for bad_file in ("../part-0000.npy", "sub/part-0000.npy", "..", ""):
        bad = json.loads(json.dumps(orig))
        bad["parts"][0]["file"] = bad_file
        manifest_path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="invalid part filename"):
            open_npy(parts)

    # Restore valid manifest, then test custom part filename from manifest (not rebuilt from rank count).
    renamed = parts / "custom_rank_zero.npy"
    (parts / "part-0000.npy").rename(renamed)
    custom = json.loads(json.dumps(orig))
    custom["parts"][0]["file"] = "custom_rank_zero.npy"
    manifest_path.write_text(json.dumps(custom))
    np.testing.assert_array_equal(open_npy(parts, mmap_mode=None), ref)

    # Truncated part file rejected on open.
    raw_bytes = renamed.read_bytes()
    renamed.write_bytes(raw_bytes[:-2])
    with pytest.raises(ValueError, match="bytes, expected"):
        open_npy(parts)

    # Missing part file rejected on open and stitch.
    renamed.unlink()
    with pytest.raises(FileNotFoundError):
        open_npy(parts)
    with pytest.raises(FileNotFoundError):
        stitch_npy(parts, tmp_path / "stitched.npy")


def test_stitch_existing_dest_stale_partial_and_corrupt_crc(tmp_path):
    ref = np.arange(40, dtype=np.int32)
    parts = make_sharded(tmp_path / "s.npy", ref, [0, 17, 40])

    # Existing destination is refused before reading/copying.
    existing = tmp_path / "already.npy"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="already exists"):
        stitch_npy(parts, existing)
    assert existing.read_bytes() == b"keep"

    # Stale <dest>.partial from a killed stitch is truncated and reused, and delete_parts removes .parts only.
    dest = tmp_path / "s.npy"
    stale_partial = tmp_path / "s.npy.partial"
    stale_partial.write_bytes(b"stale partial garbage" * 100)
    res = stitch_npy(parts, dest, delete_parts=True, block=9)
    assert res["deleted_parts"] is True
    assert dest.is_file() and not stale_partial.exists() and not parts.exists()
    np.testing.assert_array_equal(np.load(dest), ref)

    # Corrupt payload byte caught by verify_npy_parts and stitch_npy; partial cleaned up, parts kept.
    parts2 = make_sharded(tmp_path / "corrupt.npy", ref, [0, 17, 40])
    p1 = parts2 / "part-0001.npy"
    data = bytearray(p1.read_bytes())
    data[-1] ^= 0x01
    p1.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="CRC-32"):
        verify_npy_parts(parts2)
    out_corrupt = tmp_path / "corrupt.npy"
    with pytest.raises(ValueError, match="CRC-32"):
        stitch_npy(parts2, out_corrupt)
    assert not out_corrupt.exists() and not (tmp_path / "corrupt.npy.partial").exists()
    assert p1.is_file()


def test_copy_remove_and_directory_fsync_tolerance(tmp_path, monkeypatch):
    ref = np.arange(25, dtype=np.uint8)
    parts = make_sharded(tmp_path / "a.npy", ref, [0, 10, 25])

    copied = copy_npy_output(tmp_path / "a.npy", tmp_path / "b.npy")
    assert copied == tmp_path / "b.parts"
    np.testing.assert_array_equal(open_npy(tmp_path / "b.npy", mmap_mode=None), ref)

    # When both .npy and .parts exist at src, copy_npy_output copies the .npy; remove_npy_output removes both.
    stitch_npy(parts)
    copied_both = copy_npy_output(tmp_path / "a.npy", tmp_path / "c.npy")
    assert copied_both == tmp_path / "c.npy" and not (tmp_path / "c.parts").exists()
    remove_npy_output(tmp_path / "a.npy")
    assert not (tmp_path / "a.npy").exists() and not parts.exists()

    # Directory fsync returning EINVAL/ENOTSUP is tolerated and recorded as False, while file fsync succeeds.
    import stat

    real_fsync = os.fsync

    def rejecting_dir_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "Invalid argument")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", rejecting_dir_fsync)

    pdir = tmp_path / "nofsync.parts"
    pdir.mkdir()
    np.save(pdir / "part-0000.npy", ref)
    _, dir_synced = write_npy_manifest(
        pdir,
        dtype=ref.dtype,
        shape=ref.shape,
        parts=[dict(rank=0, file="part-0000.npy", start=0, stop=25, rows=25, bytes=25, crc32=int(aweSOM.stream.npy_crc32(pdir / "part-0000.npy", 0, 25)))],
    )
    assert dir_synced is False
    st = stitch_npy(pdir)
    assert st["directory_fsync"] is False


def test_cli_verify_and_stitch(tmp_path):
    ref = np.arange(33, dtype=np.int16)
    parts = make_sharded(tmp_path / "cli.npy", ref, [0, 12, 33])
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    v = subprocess.run(
        [sys.executable, "-m", "aweSOM.npy_parts", "verify", str(parts), "--block", "8"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert v.returncode == 0, v.stderr
    assert json.loads(v.stdout)["total_rows"] == 33

    out = tmp_path / "custom-out.npy"
    s = subprocess.run(
        [sys.executable, "-m", "aweSOM.npy_parts", "stitch", str(parts), "-o", str(out), "--delete-parts"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert s.returncode == 0, s.stderr
    assert out.is_file() and not parts.exists()
    np.testing.assert_array_equal(np.load(out), ref)


def test_symlinked_parts_cleanup_preserves_target_dataset(tmp_path):
    ref = np.arange(12, dtype=np.uint8)
    source = make_sharded(tmp_path / "source.npy", ref, [0, 5, 12])
    other = make_sharded(tmp_path / "other.npy", ref + 10, [0, 6, 12])

    # Direct .parts directory symlink removal unlinks only the alias.
    alias = tmp_path / "alias.parts"
    alias.symlink_to(source, target_is_directory=True)
    remove_npy_output(alias)
    assert not alias.exists() and not alias.is_symlink()
    assert sorted(p.name for p in source.iterdir()) == ["manifest.json", "part-0000.npy", "part-0001.npy"]
    np.testing.assert_array_equal(open_npy(source, mmap_mode=None), ref)

    # Stem .npy target with a sibling .parts directory symlink also unlinks only the alias.
    alias2 = tmp_path / "alias2.parts"
    alias2.symlink_to(source, target_is_directory=True)
    remove_npy_output(tmp_path / "alias2.npy")
    assert not alias2.exists() and not alias2.is_symlink()
    np.testing.assert_array_equal(open_npy(source, mmap_mode=None), ref)

    # Dangling .parts symlink is removed cleanly instead of raising FileNotFoundError.
    broken = tmp_path / "broken.parts"
    broken.symlink_to(tmp_path / "missing.parts", target_is_directory=True)
    remove_npy_output(broken)
    assert not broken.is_symlink()

    # Stale <dst>.parts.partial symlink to source.parts does not delete source.parts during copy.
    stale_staging = tmp_path / "copied.parts.partial"
    stale_staging.symlink_to(source, target_is_directory=True)
    copied = copy_npy_output(other, tmp_path / "copied.npy")
    assert copied == tmp_path / "copied.parts"
    np.testing.assert_array_equal(open_npy(copied, mmap_mode=None), ref + 10)
    np.testing.assert_array_equal(open_npy(source, mmap_mode=None), ref)

    # stitch_npy(..., delete_parts=True) on a .parts symlink unlinks the symlink and keeps source.parts intact.
    alias3 = tmp_path / "alias3.parts"
    alias3.symlink_to(source, target_is_directory=True)
    stitched = tmp_path / "stitched_from_alias.npy"
    stitch_npy(alias3, stitched, delete_parts=True)
    assert not alias3.exists() and not alias3.is_symlink()
    np.testing.assert_array_equal(np.load(stitched), ref)
    np.testing.assert_array_equal(open_npy(source, mmap_mode=None), ref)


def test_structured_and_nonscalar_dtypes_rejected_before_publication(tmp_path):
    rec = np.zeros(4, dtype=[("id", "u1"), ("value", "f8")])
    with pytest.raises(ValueError, match="non-scalar dtypes are not supported"):
        make_sharded(tmp_path / "records.npy", rec, [0, 2, 4])
    rec_dir = tmp_path / "records.parts"
    assert not (rec_dir / "manifest.json").exists()
    assert not (rec_dir / "manifest.json.partial").exists()

    sub_dir = tmp_path / "subarray.parts"
    sub_dir.mkdir()
    with pytest.raises(ValueError, match="non-scalar dtypes are not supported"):
        write_npy_manifest(
            sub_dir,
            dtype=np.dtype("(2,)f4"),
            shape=(4,),
            parts=[dict(rank=0, file="part-0000.npy", start=0, stop=4, rows=4, bytes=32, crc32=0)],
        )
    assert not (sub_dir / "manifest.json").exists()

    # A hand-edited manifest with a JSON list-of-lists dtype descriptor is rejected cleanly.
    bad_dir = make_sharded(tmp_path / "bad_descr.npy", np.arange(4, dtype=np.uint8), [0, 2, 4])
    doc = json.loads((bad_dir / "manifest.json").read_text(encoding="utf-8"))
    doc["dtype"] = [["id", "|u1"], ["value", "<f8"]]
    (bad_dir / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    for fn in (open_npy, verify_npy_parts, stitch_npy):
        with pytest.raises(ValueError, match="expected scalar dtype string"):
            fn(bad_dir)
