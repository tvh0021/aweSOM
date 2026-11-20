"""Single-process tests of NpyWriter and copy_npy_rows."""
import os
import zlib

import numpy as np
import pytest

from aweSOM.stream import NpyWriter, copy_npy_rows, npy_crc32


@pytest.mark.parametrize("dtype,shape", [("i4", (1003,)), ("f8", (1003,)), ("f4", (101, 7))])
def test_npy_writer_matches_numpy(tmp_path, dtype, shape):
    rng = np.random.default_rng(5)
    ref = rng.normal(size=shape).astype(dtype) if dtype != "i4" else rng.integers(-9, 9, shape).astype(dtype)
    np.save(tmp_path / "ref.npy", ref)
    path = tmp_path / "got.npy"
    created = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    del created
    a = NpyWriter(path)
    assert len(a) == shape[0]
    a[0:40] = ref[0:40]
    a[40:41] = ref[40:41]
    a[41:] = ref[41:]
    a.close()
    a.close()  # twice is fine
    assert path.read_bytes() == (tmp_path / "ref.npy").read_bytes()
    assert a.crc32 == zlib.crc32(np.ascontiguousarray(ref)) == npy_crc32(path, 0, shape[0], block=17)
    assert (a.first, a.end) == (0, shape[0])
    b = NpyWriter(path)  # a middle range, written out of order: no CRC
    b[50:60] = ref[50:60]
    assert b.crc32 == zlib.crc32(np.ascontiguousarray(ref[50:60])) == npy_crc32(path, 50, 60)
    b[10:20] = ref[10:20]
    assert b.crc32 is None
    b.close()
    assert path.read_bytes() == (tmp_path / "ref.npy").read_bytes()


def test_npy_writer_times_writes_and_fsync(tmp_path):
    path = tmp_path / "t.npy"
    np.save(path, np.zeros(1000, "f8"))
    w = NpyWriter(path)
    assert w.pwrite_seconds == w.fsync_seconds == 0.0
    w[0:1000] = np.ones(1000)
    w.close()
    assert w.pwrite_seconds > 0 and w.fsync_seconds > 0
    w = NpyWriter(path)
    w.close(sync=False)
    assert w.fsync_seconds == 0.0


@pytest.mark.parametrize("dtype,shape", [("i4", (1003,)), ("f8", (1003,)), ("f4", (101, 7))])
def test_copy_npy_rows_into_place(tmp_path, dtype, shape):
    """Parts copied in rank order give np.save's bytes; each copy's CRC is its writer's CRC."""
    rng = np.random.default_rng(9)
    ref = rng.normal(size=shape).astype(dtype)
    np.save(tmp_path / "ref.npy", ref)
    final = tmp_path / "final.npy"
    created = np.lib.format.open_memmap(final, mode="w+", dtype=dtype, shape=shape)
    del created
    n = shape[0]
    edges = [0, 0, 40, 41, n]  # an empty part, a one-row part, uneven parts
    stats = {}
    dest = NpyWriter(final)
    dest[0:0] = ref[0:0]
    for r, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        part = tmp_path / f"part{r}.npy"
        made = np.lib.format.open_memmap(part, mode="w+", dtype=dtype, shape=(b - a, *shape[1:]))
        del made
        w = NpyWriter(part)
        w[0:b - a] = ref[a:b]
        w.close()
        assert copy_npy_rows(part, dest, a, block=7, stats=stats) == w.crc32 == zlib.crc32(np.ascontiguousarray(ref[a:b]))
    assert dest.crc32 is None  # the copies are checked by their own CRC, not the writer's
    dest.close()
    assert final.read_bytes() == (tmp_path / "ref.npy").read_bytes()
    assert stats["bytes"] == ref.nbytes and stats["read_seconds"] >= 0 and stats["write_seconds"] > 0


def test_copy_npy_rows_refusals(tmp_path):
    final = tmp_path / "final.npy"
    np.save(final, np.zeros(10, "i4"))
    dest = NpyWriter(final)
    np.save(tmp_path / "f8.npy", np.zeros(3, "f8"))
    np.save(tmp_path / "2d.npy", np.zeros((3, 2), "i4"))
    np.save(tmp_path / "ok.npy", np.arange(3, dtype="i4"))
    np.save(tmp_path / "0d.npy", np.int32(5))
    for name, start in (("f8.npy", 0), ("2d.npy", 0), ("0d.npy", 0), ("ok.npy", 8), ("ok.npy", -1)):
        with pytest.raises(ValueError):
            copy_npy_rows(tmp_path / name, dest, start)
    with pytest.raises(ValueError, match="holds 3 rows, expected 4"):
        copy_npy_rows(tmp_path / "ok.npy", dest, 0, rows=4)
    assert copy_npy_rows(tmp_path / "ok.npy", dest, 0, rows=3) == zlib.crc32(np.arange(3, dtype="i4"))
    raw = (tmp_path / "ok.npy").read_bytes()
    (tmp_path / "short.npy").write_bytes(raw[:-4])  # header says 3 rows, 2 present
    with pytest.raises(IOError, match="short read"):
        copy_npy_rows(tmp_path / "short.npy", dest, 0)
    dest.close()


def test_npy_writer_refusals(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.zeros(10, "i4"))
    w = NpyWriter(path)
    with pytest.raises(TypeError):
        w[3] = 1
    with pytest.raises(TypeError):
        w[0:10:2] = np.zeros(5, "i4")
    with pytest.raises(ValueError):
        w[0:4] = np.zeros(3, "i4")
    w.close()
    np.save(tmp_path / "f.npy", np.asfortranarray(np.zeros((3, 2))))
    with pytest.raises(ValueError):
        NpyWriter(tmp_path / "f.npy")
    with pytest.raises(FileNotFoundError):
        NpyWriter(tmp_path / "missing.npy")


def test_npy_writer_close_failure_does_not_close_twice(tmp_path, monkeypatch):
    final = tmp_path / "final.npy"
    np.save(final, np.zeros(4, "i4"))
    w = NpyWriter(final)
    fd, real_close, calls = w.fd, os.close, []

    def failing_close(f):
        calls.append(f)
        real_close(f)
        raise OSError("close failed")

    monkeypatch.setattr(os, "close", failing_close)
    with pytest.raises(OSError, match="close failed"):
        w.close(sync=False)
    w.close()  # the descriptor number may already belong to another file: must not be closed again
    assert calls == [fd] and w.fd is None
