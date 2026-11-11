"""Multi-rank labelling: row split, GPU device choice, NpyWriter, BMUSearcher,
predict_stream(comm=...), and the production_pool driver labelling on every rank."""
import json
import multiprocessing as mp
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

from aweSOM import _gpu, _gpu_screen, distributed
from aweSOM.inference import best_match_nodes, best_match_nodes_patch
from aweSOM.stream import BMUSearcher, NpyWriter, npy_crc32, predict_stream

import mpi_label_worker as worker
from production_pool_mpi_test import FakeComm, prepared  # noqa: F401  (module fixture)
from production_pool_test import pool

HERE = Path(__file__).parent


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
    assert _gpu.cuda_available() is False
    for count in (0, 2):
        runtime = SimpleNamespace(getDeviceCount=lambda count=count: count)
        monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))
        assert _gpu.cuda_available() is (count > 0)

    def broken():
        raise RuntimeError("cudaErrorNoDevice")

    runtime = SimpleNamespace(getDeviceCount=broken)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))
    assert _gpu.cuda_available() is False


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
    for key, dtype in (("labels", "i4"), ("nodes", "i4"), ("distances", "f8")):
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
    bitwise(np.load(tmp_path / "nodes.npy"), ref_node)
    bitwise(np.load(tmp_path / "distances.npy"), np.sqrt(ref_d2))
    assert ref_st["start"] == 0 and ref_st["stop"] == n and ref_st["ranks"] == 1


def test_shared_mode_refuses_short_outputs():
    x = worker.data(10)
    with pytest.raises(ValueError, match="must hold all 10 rows"):
        worker.label(x, "patch", RankComm(0, 2), out=np.empty(5, np.int32))


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
        np.save(single["nodes"], nodes)
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
    """One fake rank of production_pool.main. mode: 'cpu', 'gpu' (fake CuPy with two
    devices and the CPU stand-in searcher) or 'no-shared-out' (rank 1 cannot open OUT files)."""
    try:
        from aweSOM import distributed as dist, _gpu_screen as screen, stream
        dist.get_comm = lambda: SharedNodeComm(rank, size, barrier, slots)
        dist._shared_split_type = lambda: "shared"
        if mode == "gpu":
            runtime = SimpleNamespace(getDeviceCount=lambda: 2)
            sys.modules["cupy"] = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
            screen.GpuScreenSearcher = StandIn
            StandIn.devices = []
        if mode == "no-shared-out" and rank == 1:
            def missing(path):
                raise FileNotFoundError(2, "No such file or directory", str(path))
            stream.NpyWriter = missing
        pool.main(argv)
        conn.send(("ok", StandIn.devices))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
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
    assert all(status == "ok" for status, _ in results), results
    res = out / "results"
    lab, nodes, dist = relabel(res)
    bitwise(np.load(res / "4x3-s42-labels.npy"), lab.astype(np.int32))
    bitwise(np.load(res / "4x3-s42-nodes.npy"), nodes)
    bitwise(np.load(res / "4x3-s42-distances.npy"), dist)
    rec = json.loads((res / "4x3-s42.json").read_text())["labelling"]
    n = len(lab)
    assert rec["requested"] == "auto" and rec["backend"] == ("cuda_screen" if mode == "gpu" else "patch")
    assert [(r["start"], r["stop"]) for r in rec["ranks"]] == [distributed.row_range(n, r, size) for r in range(size)]
    assert rec["crc_checked_ranks"] == list(range(1, size))
    devices = [payload for _, payload in results]
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
    assert [status for status, _ in results] == ["error", "error"]
    for _, message in results:
        assert "multi-rank labelling needs OUT on a file system every rank can open" in message
        assert "rank 1: cannot open" in message
    assert not (out / "results").exists()
