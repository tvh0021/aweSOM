"""Multi-process runs of aweSOM.stream.train_batch_stream under mpirun, compared
with the single-process reference."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aweSOM.inference import best_match_nodes

sys.path.insert(0, str(Path(__file__).parent))
import mpi_stream_worker as wk  # noqa: E402

WORKER = Path(__file__).parent / "mpi_stream_worker.py"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_import_does_not_load_mpi4py():
    code = "import sys, aweSOM.stream, aweSOM.distributed; print('mpi4py' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# ------------------------------------------------------------------ MPI

def _mpirun():
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return None
    return shutil.which("mpirun")


MPIRUN = _mpirun()
needs_mpi = pytest.mark.skipif(MPIRUN is None, reason="mpi4py or mpirun not available")


def run_worker(n, out, scenario):
    env = dict(os.environ, NUMBA_NUM_THREADS="2", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    cmd = [MPIRUN, "-n", str(n), "--oversubscribe", sys.executable, "-m", "mpi4py", str(WORKER),
           scenario, str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return [dict(np.load(p, allow_pickle=False)) for p in sorted(Path(out).glob("rank*.npz"))]


def rel_diff(a, b):
    """Max relative difference |a - b| / max(|a|, |b|), with a 1e-8 floor on the denominator."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)))


def _record(name, value):
    print(f"{name} = {value}")
    path = os.environ.get("AWESOM_MPI_TEST_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps({name: value}) + "\n")


@needs_mpi
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("ranks", [2, 4])
def test_mpi_array_source_matches_single_process(tmp_path, ranks, dtype):
    x = wk.data(dtype)
    outs = run_worker(ranks, tmp_path, "array" if dtype == "float32" else "array64")
    w_single, i_single, history = wk.run_array(x, None)
    for o in outs:  # every rank holds the same weights, bitwise
        assert sha(o["weights"]) == sha(outs[0]["weights"])
        assert int(o["ranks"]) == ranks
    o = outs[0]
    d = rel_diff(o["weights"], w_single)
    _record(f"array_{dtype}_P{ranks}_max_rel", d)
    assert np.all(np.abs(o["weights"] - w_single) <= 1e-4 * np.maximum(np.abs(o["weights"]), np.abs(w_single)) + 1e-12)
    np.testing.assert_array_equal(o["hits"], i_single["hits"])
    np.testing.assert_array_equal(o["rows_per_epoch"], i_single["rows_per_epoch"])
    assert o["rank_rows"].shape == (5, ranks) and (o["rank_rows"].sum(axis=1) == len(x)).all()
    # per-epoch BMUs of every row: the lattice entering each epoch under MPI vs single process
    flips = 0
    for e in range(5):
        a, _ = best_match_nodes(o["history"][e], x)
        b, _ = best_match_nodes(history[e], x)
        flips += int(np.count_nonzero(a != b))
    _record(f"array_{dtype}_P{ranks}_bmu_flips", flips)
    assert flips == 0


@needs_mpi
def test_mpi_fewer_chunks_than_ranks_and_sampling(tmp_path):
    x = wk.data()
    outs = run_worker(4, tmp_path, "uneven")
    w_single, i_single = wk.run_uneven(x, None)
    o = outs[0]
    assert (o["rank_rows"][:, 3] == 0).all()  # 3 chunks, rank 3 owns none
    d = rel_diff(o["weights"], w_single)
    _record("uneven_P4_max_rel", d)
    assert np.all(np.abs(o["weights"] - w_single) <= 1e-4 * np.maximum(np.abs(o["weights"]), np.abs(w_single)) + 1e-12)
    np.testing.assert_array_equal(o["rows_per_epoch"], i_single["rows_per_epoch"])
    s = outs[0]
    ws, is_ = wk.run_sampled(x, None)
    d = rel_diff(s["sampled_weights"], ws)
    _record("sampled_P4_max_rel", d)
    assert d < 1e-4
    np.testing.assert_array_equal(s["sampled_rows"], is_["rows_per_epoch"])


@needs_mpi
@pytest.mark.parametrize("ranks", [2, 4])
def test_mpi_callable_resume(tmp_path, ranks):
    outs = run_worker(ranks, tmp_path, "resume")
    o = outs[0]
    assert sha(o["resumed"]) == sha(o["uninterrupted"])  # bitwise for a fixed rank count
    assert int(o["resumed_from"]) == 2 and int(o["calls"]) == 7  # skipped counting pass, rows(), 5 epochs
    assert all(int(r["calls"]) == 7 for r in outs)
    w_single = wk.run_callable(wk.data(), None)
    d = rel_diff(o["uninterrupted"], w_single)
    _record(f"callable_P{ranks}_max_rel", d)
    assert d < 1e-4


@needs_mpi
def test_mpi_fixed_rank_count_is_bitwise_reproducible(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
    a = run_worker(3, tmp_path / "a", "array64")
    b = run_worker(3, tmp_path / "b", "array64")
    assert sha(a[0]["weights"]) == sha(b[0]["weights"])


@needs_mpi
def test_mpi_rank0_stop_or_failure_ends_every_rank(tmp_path):
    """A callback stop or checkpoint failure on rank 0 must not leave rank 1 in a collective."""
    outs = run_worker(2, tmp_path, "stop")
    assert len(outs) == 2
    r0, r1 = (list(o["outcome"]) for o in outs)
    assert r0 == ["Stop", "OSError"]
    assert r1[0].startswith("rank 0 stopped training after epoch 1: Stop(")
    assert r1[1].startswith("rank 0 stopped training after epoch 1: FileNotFoundError(")
