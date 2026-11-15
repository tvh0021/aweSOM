"""GPU device choice, cuda_available, BMUSearcher, and the cuda_screen path of
predict_stream, run on the CPU through a CuPy stand-in."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from aweSOM import _gpu_screen, distributed
from aweSOM.inference import best_match_nodes, best_match_nodes_patch
from aweSOM.stream import BMUSearcher, predict_stream

import mpi_label_worker as worker
from fake_comm import NodeComm, RankComm, shared_split  # noqa: F401  (fixture)
from fake_cupy import StandIn


def bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


# ------------------------------------------------------------- device choice


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


# ---------------------------------------------------------------- BMUSearcher


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
        BMUSearcher(w, "bogus", lattice_shape=worker.SHAPE)
    with pytest.raises(ValueError):
        BMUSearcher(w, "screen")(x[:, :3])


# ------------------------------------------------------ cuda_screen with comm


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
