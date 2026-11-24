"""Real-MPI test of :func:`aweSOM.distributed.node_local_rank` under ``mpirun -n 2``."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WORKER = HERE / "mpi_node_local_worker.py"


def _mpirun():
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return None
    return shutil.which("mpirun")


MPIRUN = _mpirun()
needs_mpi = pytest.mark.skipif(MPIRUN is None, reason="mpi4py or mpirun not available")


@needs_mpi
def test_node_local_rank_under_real_mpirun(tmp_path):
    from mpi4py import MPI

    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(HERE.parent / "src"),
    )
    cmd = [
        MPIRUN,
        "-n",
        "2",
        "--oversubscribe",
        sys.executable,
        "-m",
        "mpi4py",
        str(WORKER),
        str(tmp_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]

    records = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("rank*.json"))]
    assert len(records) == 2
    assert [r["rank"] for r in records] == [0, 1]
    assert [r["size"] for r in records] == [2, 2]
    assert {r["local_rank"] for r in records} == {0, 1}
    assert [r["local_rank"] for r in records] == [0, 1]
    assert all(r["split_type"] == int(MPI.COMM_TYPE_SHARED) for r in records)
