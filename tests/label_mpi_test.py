"""Multi-rank labelling on the CPU: row split, node-local rank, and
predict_stream(comm=...) with shared and rank-local outputs, under fake
communicators and under mpirun."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aweSOM import distributed
from aweSOM.inference import best_match_nodes
from aweSOM.stream import NpyWriter, npy_crc32, predict_stream

import mpi_label_worker as worker
from fake_comm import NodeComm, RankComm, shared_split  # noqa: F401  (fixture)

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


# ------------------------------------------------------------- node-local rank


def test_node_local_rank(shared_split):
    nodes = ["a", "a", "b", "a", "b"]
    comms = [NodeComm(r, nodes) for r in range(5)]
    assert [distributed.node_local_rank(c) for c in comms] == [0, 1, 0, 2, 1]
    assert all(c.freed == 1 for c in comms)
    assert distributed.node_local_rank(None) == 0


# ------------------------------------------- predict_stream with a communicator


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
