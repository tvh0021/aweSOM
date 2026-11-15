"""Run by tests/node_local_rank_mpi_test.py under ``mpirun -n 2 python -m mpi4py``.

Calls :func:`aweSOM.distributed.node_local_rank` (and therefore
:func:`aweSOM.distributed._shared_split_type`) against real ``MPI.COMM_WORLD``
without fake communicators, writing ``OUT/rank<r>.json`` from every rank.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from aweSOM import distributed


def main(out_dir: str) -> None:
    comm = distributed.get_comm()
    if comm is None:
        raise RuntimeError("expected a multi-rank MPI communicator")
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    local_rank = distributed.node_local_rank(comm)
    payload = {
        "rank": rank,
        "size": size,
        "local_rank": local_rank,
        "split_type": int(distributed._shared_split_type()),
    }
    (Path(out_dir) / f"rank{rank}.json").write_text(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
