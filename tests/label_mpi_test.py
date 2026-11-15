"""Multi-rank labelling: row split, GPU device choice, NpyWriter, BMUSearcher,
predict_stream(comm=...), and the production_pool driver labelling on every rank."""
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import traceback
import types
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aweSOM import _gpu_screen, distributed
from aweSOM.inference import best_match_nodes, best_match_nodes_patch
from aweSOM.stream import BMUSearcher, NpyWriter, copy_npy_rows, npy_crc32, predict_stream

import mpi_label_rows_worker as rows_worker
import mpi_label_worker as worker
import output_guard
from production_pool_mpi_test import FakeComm, prepared  # noqa: F401  (module fixture)
from production_pool_test import pool

HERE = Path(__file__).parent
ROW_BYTES = sum(d.itemsize for d in worker.DTYPES.values())  # bytes per row of the three label files


def bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


# ------------------------------------------------------------------ row split


@pytest.mark.parametrize("n,size", [(10, 3), (1003, 2), (1003, 3), (7, 4), (2, 3), (0, 3), (5, 1), (1, 8)])
def test_row_range_covers_rows_in_rank_order(n, size):
    parts = [distributed.row_range(n, r, size) for r in range(size)]
    assert parts[0][0] == 0 and parts[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))  # contiguous, in rank order
    sizes = [b - a for a, b in parts]
    assert min(sizes) >= 0 and max(sizes) - min(sizes) <= 1  # balanced to one row


def test_row_range_uneven_and_fewer_rows_than_ranks():
    assert [distributed.row_range(10, r, 3) for r in range(3)] == [(0, 3), (3, 6), (6, 10)]
    assert [distributed.row_range(2, r, 3) for r in range(3)] == [(0, 0), (0, 1), (1, 2)]
    for bad in [(-1, 0, 1), (5, 1, 1), (5, -1, 2), (5, 0, 0)]:
        with pytest.raises(ValueError):
            distributed.row_range(*bad)


# ------------------------------------------------------------- device choice


class NodeComm:
    """Split_type of a fake world: ``nodes[r]`` is the node of rank r."""

    def __init__(self, rank, nodes):
        self.rank, self.nodes, self.freed = rank, nodes, 0

    def Get_rank(self):
        return self.rank

    def Split_type(self, kind, key=0):
        assert kind == "shared" and key == self.rank
        local = [r for r, n in enumerate(self.nodes) if n == self.nodes[self.rank]].index(self.rank)
        parent = self

        class Local:
            def Get_rank(self):
                return local

            def Free(self):
                parent.freed += 1

        return Local()


@pytest.fixture
def shared_split(monkeypatch):
    monkeypatch.setattr(distributed, "_shared_split_type", lambda: "shared")


def test_node_local_rank(shared_split):
    nodes = ["a", "a", "b", "a", "b"]
    comms = [NodeComm(r, nodes) for r in range(5)]
    assert [distributed.node_local_rank(c) for c in comms] == [0, 1, 0, 2, 1]
    assert all(c.freed == 1 for c in comms)
    assert distributed.node_local_rank(None) == 0


def test_gpu_device_round_robin_per_node(shared_split):
    nodes = ["a"] * 5 + ["b"] * 3
    got = [distributed.gpu_device(NodeComm(r, nodes), device_count=4) for r in range(8)]
    assert got == [0, 1, 2, 3, 0, 0, 1, 2]


def test_gpu_device_one_visible_device_per_rank(shared_split, monkeypatch):
    """Slurm binding (CUDA_VISIBLE_DEVICES with one GPU per task): every rank uses index 0."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    runtime = SimpleNamespace(getDeviceCount=lambda: 1)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))
    nodes = ["a"] * 4
    assert [distributed.gpu_device(NodeComm(r, nodes)) for r in range(4)] == [0, 0, 0, 0]
    assert distributed.gpu_device(None) == 0


def test_gpu_device_without_devices(shared_split):
    with pytest.raises(RuntimeError, match="no CUDA device"):
        distributed.gpu_device(NodeComm(0, ["a"]), device_count=0)


def test_cuda_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)  # not installed
    assert _gpu_screen.cuda_available() is False
    for count in (0, 2):
        runtime = SimpleNamespace(getDeviceCount=lambda count=count: count)
        monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))
        assert _gpu_screen.cuda_available() is (count > 0)

    def broken():
        raise RuntimeError("cudaErrorNoDevice")

    runtime = SimpleNamespace(getDeviceCount=broken)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))
    assert _gpu_screen.cuda_available() is False


# ------------------------------------------------------------------ NpyWriter


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
    crcs, stats = [], {}
    dest = NpyWriter(final)
    dest[0:0] = ref[0:0]
    for r, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        part = tmp_path / f"part{r}.npy"
        made = np.lib.format.open_memmap(part, mode="w+", dtype=dtype, shape=(b - a, *shape[1:]))
        del made
        w = NpyWriter(part)
        w[0:b - a] = ref[a:b]
        w.close()
        crcs.append(w.crc32)
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


# ---------------------------------------------------------------- BMUSearcher


class StandIn:
    """CPU stand-in for GpuScreenSearcher: the float32-screen reference and its float64 recompute."""

    devices = []

    def __init__(self, weights, xdim, ydim, device=None, **kwargs):
        self.w = weights
        type(self).devices.append(device)

    def __call__(self, x, idx, d2):
        idx[:], d2[:], fb = _gpu_screen._cuda_screen_reference(self.w, x, return_fallbacks=True)
        return fb

    def cluster_map(self, clusters, dtype=np.int32):
        return np.asarray(clusters).astype(dtype)

    def submit(self, x, out, d2=None, clusters=None):  # synchronous: wait has nothing to do
        idx, dist = _gpu_screen._cuda_screen_reference(self.w, x)
        out[:] = idx if clusters is None else clusters[idx]
        if d2 is not None:
            d2[:] = dist
        return 0

    def wait(self, token=None):
        pass

    def cancel(self):
        pass


def test_bmu_searcher_matches_reference(monkeypatch):
    monkeypatch.setattr(_gpu_screen, "GpuScreenSearcher", StandIn)
    StandIn.devices = []
    w, x = worker.lattice(), worker.data(1003)
    ref = best_match_nodes(w, x)
    assert (ref[0][::5] == 3).all()  # the tie between nodes 3 and 7 goes to the lowest index
    for backend in ("patch", "screen", "numba", "cuda_screen"):
        search = BMUSearcher(w, backend, lattice_shape=worker.SHAPE, device=1 if backend == "cuda_screen" else None)
        for rows in (x, x.astype(np.float32)):
            got = search(rows)
            bitwise(got[0], best_match_nodes(w, rows)[0])
            bitwise(got[1], best_match_nodes(w, rows)[1])
        idx, d2 = np.empty(3, np.int32), np.empty(3)
        assert search(x[:3], idx, d2)[0] is idx
        assert search(x[:0])[0].shape == (0,)
    assert StandIn.devices == [1]
    with pytest.raises(ValueError):
        BMUSearcher(w, "patch")
    with pytest.raises(ValueError):
        BMUSearcher(w, "kdtree", lattice_shape=worker.SHAPE)
    with pytest.raises(ValueError):
        BMUSearcher(w, "screen")(x[:, :3])


# ------------------------------------------- predict_stream with a communicator


class RankComm:
    """Rank r of ``size`` for predict_stream calls that need no collectives (CPU backends),
    plus Split_type (one node) for the GPU device choice."""

    def __init__(self, rank, size):
        self.rank, self.size = rank, size

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Split_type(self, kind, key=0):
        rank = self.rank
        return SimpleNamespace(Get_rank=lambda: rank, Free=lambda: None)


@pytest.mark.parametrize("size", [2, 3, 5])
@pytest.mark.parametrize("name", list(worker.cases()))
def test_shared_mode_ranks_in_turn(name, size, tmp_path):
    make, n, backend = worker.cases()[name]
    ref_lab, ref_st = worker.label(make(), backend, n_samples=n)
    ref_node, ref_d2 = best_match_nodes(worker.lattice(), worker.data(n))  # every case labels data(n)
    files = {}
    for key, dtype in worker.DTYPES.items():
        created = np.lib.format.open_memmap(tmp_path / f"{key}.npy", mode="w+", dtype=dtype, shape=(n,))
        del created
    for r in range(size):
        files = {k: NpyWriter(tmp_path / f"{k}.npy") for k in ("labels", "nodes", "distances")}
        _, st = worker.label(make(), backend, RankComm(r, size), files["labels"], files["nodes"],
                             files["distances"], n_samples=n)
        assert (st["start"], st["stop"]) == distributed.row_range(n, r, size) and st["rows"] == st["stop"] - st["start"]
        for f in files.values():
            f.close()
        local, lst = worker.label(make(), backend, RankComm(r, size), n_samples=n)
        bitwise(local, ref_lab[lst["start"]:lst["stop"]])
    bitwise(np.load(tmp_path / "labels.npy"), ref_lab)
    bitwise(np.load(tmp_path / "nodes.npy"), ref_node.astype(worker.DTYPES["nodes"]))
    bitwise(np.load(tmp_path / "distances.npy"), np.sqrt(ref_d2))
    assert ref_st["start"] == 0 and ref_st["stop"] == n and ref_st["ranks"] == 1


@pytest.mark.parametrize("size", [1, 2, 3, 5])
@pytest.mark.parametrize("name", list(worker.cases()))
def test_rank_local_outputs(name, size):
    """Each output may hold this rank's rows only (as the driver's part files do), alone or
    mixed with full-length outputs."""
    make, n, backend = worker.cases()[name]
    ref_lab, _ = worker.label(make(), backend, n_samples=n)
    ref_node, ref_d2 = best_match_nodes(worker.lattice(), worker.data(n))
    for mixed in (False, True):
        labels, nodes, dist = [], np.zeros(n, np.int32), []
        for r in range(size):
            lo, hi = distributed.row_range(n, r, size)
            mine = {"out": np.full(hi - lo, -1, np.int32), "out_distance": np.full(hi - lo, -1.0),
                    "out_node": nodes if mixed else np.full(hi - lo, -1, np.int32)}
            _, st = predict_stream(worker.lattice(), np.arange(len(worker.lattice()), dtype=np.int32) % 5,
                                   make(), chunk=worker.CHUNK, backend=backend, lattice_shape=worker.SHAPE,
                                   comm=RankComm(r, size), n_samples=n, **mine)
            assert (st["start"], st["stop"]) == (lo, hi)
            labels.append(mine["out"])
            dist.append(mine["out_distance"])
            if not mixed:
                nodes[lo:hi] = mine["out_node"]
        bitwise(np.concatenate(labels), ref_lab.astype(np.int32))  # an explicit int32 out
        bitwise(nodes, ref_node)
        bitwise(np.concatenate(dist), np.sqrt(ref_d2))


def test_shared_mode_refuses_other_lengths():
    x = worker.data(10)
    for out, kw in ((np.empty(4, np.int32), {}), (None, {"out_node": np.empty(9, np.int32)}),
                    (None, {"out_distance": np.empty(6)})):
        with pytest.raises(ValueError, match="must hold all 10 rows or this rank's 5"):
            worker.label(x, "patch", RankComm(1, 2), out=out, **kw)


@pytest.mark.parametrize("size", [1, 3])
def test_cuda_screen_path_matches_patch(monkeypatch, size):
    """The cuda_screen route (searcher, device choice per rank) through the CPU stand-in:
    nodes and float64 distances bitwise equal to patch."""
    monkeypatch.setattr(_gpu_screen, "GpuScreenSearcher", StandIn)
    monkeypatch.setattr(distributed, "_shared_split_type", lambda: "shared")
    runtime = SimpleNamespace(getDeviceCount=lambda: 2)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))
    StandIn.devices = []
    w, x = worker.lattice(), worker.data(1003)
    ref_node, ref_d2 = best_match_nodes_patch(w, x, *worker.SHAPE)
    clusters = np.arange(len(w)) % 5
    nodes, dist = np.zeros(len(x), np.int32), np.zeros(len(x))
    stats = []
    for r in range(size):
        comm = None if size == 1 else RankComm(r, size)
        _, st = predict_stream(w, clusters, x, out_node=nodes, out_distance=dist, chunk=97,
                               backend="cuda_screen", comm=comm, prefetch=False,
                               out=None if comm is None else np.zeros(len(x), np.int32))
        stats.append(st)
    bitwise(nodes, ref_node)
    bitwise(dist, np.sqrt(ref_d2))
    assert StandIn.devices == ([None] if size == 1 else [0, 1, 0])
    assert [s["device"] for s in stats] == StandIn.devices


# ----------------------------------------------------------------- real mpirun


def _mpirun():
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return None
    return shutil.which("mpirun")


@pytest.mark.skipif(_mpirun() is None, reason="mpi4py or mpirun not available")
@pytest.mark.parametrize("ranks", [2, 3])
def test_mpirun_files_equal_single_process(tmp_path, ranks):
    env = dict(__import__("os").environ, NUMBA_NUM_THREADS="2", OMP_NUM_THREADS="1",
               PYTHONPATH=str(HERE.parent / "src"))
    cmd = [_mpirun(), "-n", str(ranks), "--oversubscribe", sys.executable, "-m", "mpi4py",
           str(HERE / "mpi_label_worker.py"), str(tmp_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    report = json.loads((tmp_path / "report.json").read_text())
    for name, (make, n, backend) in worker.cases().items():
        lab, _ = worker.label(make(), backend, n_samples=n)
        single = {k: tmp_path / f"single-{name}-{k}.npy" for k in ("labels", "nodes", "distances")}
        nodes, d2 = best_match_nodes(worker.lattice(), worker.data(n))
        np.save(single["labels"], lab)
        np.save(single["nodes"], nodes.astype(worker.DTYPES["nodes"]))
        np.save(single["distances"], np.sqrt(d2))
        for k, path in single.items():
            assert (tmp_path / f"{name}-{k}.npy").read_bytes() == path.read_bytes(), (name, k)
        parts = [rep[name] for rep in report]
        assert [(p["start"], p["stop"]) for p in parts] == [distributed.row_range(n, r, ranks) for r in range(ranks)]
        assert sum(p["rows"] for p in parts) == n
        for p in parts:
            got = [npy_crc32(tmp_path / f"{name}-{k}.npy", p["start"], p["stop"]) for k in ("labels", "nodes", "distances")]
            assert got == p["crc"]
        local = np.concatenate([np.load(tmp_path / f"{name}-local-rank{r}.npz")["labels"] for r in range(ranks)])
        bitwise(local, lab)
    print(f"{ranks} mpirun ranks: labels, nodes and distances files byte-identical to one process")


def single_files(tmp_path, tag, n):
    """np.save of the single-process labels, nodes and distances: today's one-rank bytes."""
    lab, _ = worker.label(worker.data(n), "patch", n_samples=n)
    nodes, d2 = best_match_nodes(worker.lattice(), worker.data(n))
    ref = {"labels": lab, "nodes": nodes.astype(worker.DTYPES["nodes"]), "distances": np.sqrt(d2)}
    paths = {}
    for k, a in ref.items():
        paths[k] = tmp_path / f"np-save-{tag}-{k}.npy"
        np.save(paths[k], a)
    return paths


def test_label_rows_one_rank_bytes(tmp_path):
    """One rank (no comm): the final files are np.save's bytes and no part file is made."""
    pool = rows_worker.load_driver()
    for tag, n in rows_worker.CASES.items():
        rec = rows_worker.label_case(pool, tmp_path, tag, n, None)
        assert rec["crc_checked_ranks"] == [] and rec["assemble_bytes"] == 0
        assert [r["part_files"] for r in rec["ranks"]] == [False]
        for k, path in single_files(tmp_path, tag, n).items():
            assert (tmp_path / f"{tag}-{k}.npy").read_bytes() == path.read_bytes(), (tag, k)
    assert not list(tmp_path.glob("*.part*"))


@pytest.mark.skipif(_mpirun() is None, reason="mpi4py or mpirun not available")
@pytest.mark.parametrize("ranks", [2, 3])
def test_mpirun_label_rows_one_writer_per_file(tmp_path, ranks):
    """The driver's label_rows on real MPI: final files equal one rank's bit for bit (uneven
    and empty ranges), no two ranks ever hold one output file open, only rank 0 opens a
    final file, and the parts are gone afterwards."""
    env = dict(__import__("os").environ, NUMBA_NUM_THREADS="2", OMP_NUM_THREADS="1",
               PYTHONPATH=str(HERE.parent / "src"))
    cmd = [_mpirun(), "-n", str(ranks), "--oversubscribe", sys.executable, "-m", "mpi4py",
           str(HERE / "mpi_label_rows_worker.py"), str(tmp_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    report = json.loads((tmp_path / "report.json").read_text())
    for tag, n in rows_worker.CASES.items():
        for k, path in single_files(tmp_path, tag, n).items():
            assert (tmp_path / f"{tag}-{k}.npy").read_bytes() == path.read_bytes(), (tag, k)
        rec = report["records"][tag]
        assert [(r["start"], r["stop"]) for r in rec["ranks"]] == [distributed.row_range(n, r, ranks) for r in range(ranks)]
        assert rec["crc_checked_ranks"] == list(range(1, ranks))
        assert [r["part_files"] for r in rec["ranks"]] == [False] + [True] * (ranks - 1)
        assert rec["assemble_bytes"] == ROW_BYTES * sum(r["rows"] for r in rec["ranks"][1:])
    assert not list(tmp_path.glob("*.part*"))
    for r in report["ranks"]:
        finals = [p for p in r["opened"] if output_guard.is_final(p)]
        parts = [p for p in r["opened"] if not output_guard.is_final(p)]
        if r["rank"]:
            assert not finals and r["caught"] is True
            assert {Path(p).name for p in parts} == {f"{tag}-{k}.part{r['rank']}.npy" for tag in rows_worker.CASES
                                                     for k in ("labels", "nodes", "distances")}
        else:
            assert finals and all(".part0." not in p for p in parts)
    print(f"{ranks} mpirun ranks: driver label_rows files byte-identical to one rank, one writer per file")


# ------------------------------------------------------------- the driver


def test_resolve_label_backend(monkeypatch):
    import aweSOM.stream as stream
    monkeypatch.setattr(stream, "cuda_available", lambda: False)
    assert pool.resolve_label_backend("auto") == "patch"  # CPU-only: auto falls back to patch
    assert pool.resolve_label_backend("patch") == "patch"
    with pytest.raises(RuntimeError, match=r"no usable CUDA device .* rank\(s\) \[0\]"):
        pool.resolve_label_backend("cuda_screen")
    monkeypatch.setattr(stream, "cuda_available", lambda: True)
    assert pool.resolve_label_backend("auto") == "cuda_screen"
    assert pool.resolve_label_backend("patch") == "patch"
    one_without = SimpleNamespace(allgather=lambda ok: [True, ok, False])
    assert pool.resolve_label_backend("auto", one_without) == "patch"  # every rank agrees on patch
    with pytest.raises(RuntimeError, match=r"rank\(s\) \[2\]"):
        pool.resolve_label_backend("cuda_screen", one_without)
    with pytest.raises(ValueError):
        pool.resolve_label_backend("cuda")


def test_auto_falls_back_to_patch_without_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert pool.resolve_label_backend("auto") == "patch"


class SharedNodeComm(FakeComm):
    """FakeComm with Split_type: every fake rank is on one node."""

    def Split_type(self, kind, key=0):
        rank = self.rank
        return SimpleNamespace(Get_rank=lambda: rank, Free=lambda: None)


def driver_rank(conn, rank, size, barrier, slots, argv, mode):
    """One fake rank of production_pool.main, with every open of an output file during
    label_rows guarded (output_guard). mode: 'cpu', 'gpu' (fake CuPy with two devices and the CPU stand-in
    searcher), 'no-shared-out' (rank 1 cannot open OUT files), 'bad-header' (rank 1 hits a
    malformed .npy header) or 'bad-crc' (rank 1 corrupts a byte of its labels part after
    writing it). Sends (status, payload, guarded opens)."""
    try:
        output_guard.install()
        from aweSOM import distributed as dist, _gpu_screen as screen, stream
        dist.get_comm = lambda: SharedNodeComm(rank, size, barrier, slots)
        dist._shared_split_type = lambda: "shared"
        if mode == "gpu":
            runtime = SimpleNamespace(getDeviceCount=lambda: 2)
            sys.modules["cupy"] = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
            screen.GpuScreenSearcher = StandIn
            StandIn.devices = []
        else:  # a CPU-only machine, also on a GPU node
            sys.modules["cupy"] = None
        if mode == "no-shared-out" and rank == 1:
            def missing(path):
                raise FileNotFoundError(2, "No such file or directory", str(path))
            stream.NpyWriter = missing
        if mode == "bad-header" and rank == 1:
            def malformed(path):
                raise ValueError(f"{path}: malformed .npy header")
            stream.NpyWriter = malformed
        if mode == "bad-crc" and rank == 1:  # only rank 0 copies the parts: every rank's failure comes from rank 0
            close = stream.NpyWriter.close

            def corrupt(self, sync=True):
                close(self, sync)
                if self.path.endswith("-labels.part1.npy"):
                    with open(self.path, "r+b") as fh:
                        fh.seek(self.offset)
                        byte = fh.read(1)
                        fh.seek(self.offset)
                        fh.write(bytes([byte[0] ^ 1]))
            stream.NpyWriter.close = corrupt
        pool.label_rows = output_guard.locking_during(pool.label_rows)
        pool.main(argv)
        conn.send(("ok", StandIn.devices, output_guard.opened))
    except BaseException:
        conn.send(("error", traceback.format_exc(), output_guard.opened))
    finally:
        conn.close()


def run_driver(size, argv, mode):
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        barrier, slots = ctx.Barrier(size), manager.dict()
        pipes, procs = [], []
        for r in range(size):
            receive, send = ctx.Pipe(duplex=False)
            p = ctx.Process(target=driver_rank, args=(send, r, size, barrier, slots, argv, mode))
            p.start()
            send.close()
            pipes.append(receive)
            procs.append(p)
        results = [receive.recv() for receive in pipes]
        for p in procs:
            p.join(60)
    return results


def relabel(results, tag="4x3-s42"):
    """Single-process patch labels of the run's own saved weights over the whole pool."""
    import h5py
    from aweSOM import SOMModel
    model = SOMModel.load(results / f"{tag}.h5")
    with h5py.File(results / "pool.h5") as h:
        raw = h["raw"][:]
    nodes, d2 = best_match_nodes_patch(model.weights, model.scaler.transform(raw.astype("f8")), model.xdim, model.ydim)
    return np.asarray(model.node_cluster).ravel()[nodes], nodes, np.sqrt(d2)


@pytest.mark.parametrize("size,mode", [(2, "cpu"), (3, "cpu"), (2, "gpu"), (3, "gpu")])
def test_driver_ranks_label_their_rows(prepared, monkeypatch, size, mode):  # noqa: F811
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(prepared.tmp / "cache"))
    out = prepared.tmp / f"label-{mode}-{size}"
    argv = [*prepared.base, "--out", str(out), "--reuse-pool", str(prepared.pool), "--fit", "4x3", "42",
            "--mpi", "--statistics", "later"]
    results = run_driver(size, argv, mode)
    assert all(status == "ok" for status, _, _ in results), results
    res = out / "results"
    assert not list(res.glob("*.part*"))  # rank 0 deleted the parts after copying them
    for r, (_, _, opened) in enumerate(results):  # only rank 0 opens a final file
        assert any(output_guard.is_final(p) for p in opened) == (r == 0)
        if r:  # every other rank opens only its own part files
            assert {Path(p).name for p in opened} == {f"4x3-s42-{k}.part{r}.npy" for k in ("labels", "nodes", "distances")}
    lab, nodes, dist = relabel(res)
    bitwise(np.load(res / "4x3-s42-labels.npy"), lab.astype(np.uint8))  # narrow: 12 nodes, few clusters
    bitwise(np.load(res / "4x3-s42-nodes.npy"), nodes.astype(np.uint8))
    bitwise(np.load(res / "4x3-s42-distances.npy"), dist)
    rec = json.loads((res / "4x3-s42.json").read_text())["labelling"]
    n = len(lab)
    assert rec["requested"] == "auto" and rec["backend"] == ("cuda_screen" if mode == "gpu" else "patch")
    assert [(r["start"], r["stop"]) for r in rec["ranks"]] == [distributed.row_range(n, r, size) for r in range(size)]
    assert rec["crc_checked_ranks"] == list(range(1, size))
    assert rec["assemble_bytes"] == 10 * (n - rec["ranks"][0]["rows"])  # uint8 + uint8 + float64
    for key in ("assemble_seconds", "assemble_read_seconds", "assemble_write_seconds", "assemble_fsync_seconds"):
        assert rec[key] >= 0
    for r in rec["ranks"]:
        assert r["part_files"] == (r["rank"] > 0) and r["bytes_written"] == 10 * r["rows"]
        assert r["pwrite_seconds"] >= 0 and r["fsync_seconds"] >= 0
    devices = [payload for _, payload, _ in results]
    if mode == "gpu":
        assert [r["device"] for r in rec["ranks"]] == [r % 2 for r in range(size)]
        assert devices == [[r % 2] for r in range(size)]  # one searcher per rank, on its device
    else:
        assert all(r["device"] is None for r in rec["ranks"])


def test_driver_refuses_out_other_ranks_cannot_open(prepared, monkeypatch):  # noqa: F811
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(prepared.tmp / "cache"))
    out = prepared.tmp / "label-node-local"
    argv = [*prepared.base, "--out", str(out), "--reuse-pool", str(prepared.pool), "--fit", "4x3", "42", "--mpi"]
    results = run_driver(2, argv, "no-shared-out")
    assert [status for status, _, _ in results] == ["error", "error"]
    for _, message, _ in results:
        assert "multi-rank labelling needs OUT on a file system every rank can open" in message
        assert "rank 1: cannot open" in message
    assert not (out / "results").exists()


def test_driver_header_error_stops_every_rank(prepared, monkeypatch):  # noqa: F811
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(prepared.tmp / "cache"))
    out = prepared.tmp / "label-bad-header"
    argv = [*prepared.base, "--out", str(out), "--reuse-pool", str(prepared.pool), "--fit", "4x3", "42", "--mpi"]
    results = run_driver(2, argv, "bad-header")
    assert [status for status, _, _ in results] == ["error", "error"]
    for _, message, _ in results:
        assert "rank 1: cannot open the label files: ValueError" in message


def test_driver_readback_failure_stops_every_rank(prepared, monkeypatch):  # noqa: F811
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(prepared.tmp / "cache"))
    out = prepared.tmp / "label-bad-crc"
    argv = [*prepared.base, "--out", str(out), "--reuse-pool", str(prepared.pool), "--fit", "4x3", "42",
            "--mpi", "--statistics", "later"]
    results = run_driver(2, argv, "bad-crc")
    assert [status for status, _, _ in results] == ["error", "error"]
    for _, message, _ in results:
        assert "labelling readback check on rank 0 failed" in message
        assert "rank 1 labels rows" in message and "differ on disk from what it wrote" in message
    assert (out / "results.partial" / "4x3-s42-labels.part1.npy").exists()  # kept for a look


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
