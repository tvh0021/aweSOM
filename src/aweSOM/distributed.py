"""Multi-process reduction for :func:`aweSOM.stream.train_batch_stream`.

One rank per node: every rank searches the BMUs of the chunks it owns
(round-robin by chunk index) with its own numba threads, then the per-node
sums are exchanged and every rank applies the same neighbourhood update.

The exchange is an ``Allgather`` of one float64 vector per rank (``S``, ``n``,
QE sum, rows, fallbacks and timings: 373 KB at 91x64 and F = 7), summed in rank
order on every rank. Every rank therefore holds bitwise-identical totals, and a
run with a fixed number of ranks is bitwise reproducible (``MPI_Allreduce``
does not promise a fixed order). Different rank counts add the float64 sums in
a different order, so their weights differ in the last bits.

mpi4py is imported only by :func:`get_comm`; ``import aweSOM.stream`` never
imports it. MPI calls are made on the main thread only.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

__all__ = ["get_comm", "Reducer"]

_PER_RANK = (  # per-rank fields after S, n and the QE sum, in this order
    "rows", "fallbacks", "io_wait", "bmu", "acc", "pass_time")


def get_comm() -> Optional[Any]:
    """``MPI.COMM_WORLD`` if mpi4py imports and more than one rank runs, else None.

    Importing mpi4py initialises MPI; call this once, from the main thread.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    comm = MPI.COMM_WORLD
    return comm if comm.Get_size() > 1 else None


class Reducer:
    """Rank-ordered reduction and broadcasts on an mpi4py communicator."""

    def __init__(self, comm: Any):
        self.comm = comm
        self.rank = int(comm.Get_rank())
        self.size = int(comm.Get_size())

    def bcast(self, array: Optional[np.ndarray], shape: tuple[int, ...], root: int = 0) -> np.ndarray:
        """Float64 ``array`` from ``root`` on every rank (other ranks pass None)."""
        buf = np.ascontiguousarray(array, np.float64) if self.rank == root else np.empty(shape, np.float64)
        if buf.shape != tuple(shape):
            raise ValueError(f"broadcast shape {buf.shape} != {tuple(shape)}")
        self.comm.Bcast(buf, root=root)
        return buf

    def bcast_object(self, obj: Any, root: int = 0) -> Any:
        return self.comm.bcast(obj, root=root)

    def barrier(self) -> None:
        self.comm.Barrier()

    def reduce_pass(self, st: dict) -> tuple[dict, dict]:
        """Totals of one :func:`aweSOM.stream._pass` result over all ranks.

        Returns ``(total, per_rank)``: ``total`` has ``s``, ``n``, ``qe_sum``,
        ``rows`` and ``fallbacks`` summed in rank order (bitwise equal on every
        rank); ``per_rank`` maps each of ``rows``, ``fallbacks``, ``io_wait``,
        ``bmu``, ``acc`` and ``pass_time`` to a ``(size,)`` array.
        """
        s = np.ascontiguousarray(st["s"], np.float64)
        n = np.ascontiguousarray(st["n"], np.float64)
        n_s, n_n = s.size, n.size
        tail = [st["qe_sum"]] + [st.get(k, 0.0) for k in _PER_RANK]
        send = np.concatenate([s.ravel(), n, np.asarray(tail, np.float64)])
        recv = np.empty((self.size, send.size), np.float64)
        self.comm.Allgather(send, recv)
        total = recv[0].copy()
        for r in range(1, self.size):
            total += recv[r]
        k = n_s + n_n
        out = {
            "s": total[:n_s].reshape(s.shape),
            "n": total[n_s:k],
            "qe_sum": float(total[k]),
            "rows": int(round(total[k + 1])),
            "fallbacks": int(round(total[k + 2])),
        }
        per_rank = {name: recv[:, k + 1 + j].copy() for j, name in enumerate(_PER_RANK)}
        per_rank["rows"] = per_rank["rows"].round().astype(np.int64)
        per_rank["fallbacks"] = per_rank["fallbacks"].round().astype(np.int64)
        return out, per_rank
