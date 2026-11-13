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

Labelling (:func:`aweSOM.stream.predict_stream` with ``comm``) needs no
reduction: every rank labels the contiguous row range :func:`row_range` gives
it. On a node with several GPUs, :func:`gpu_device` gives each rank its own.

mpi4py is imported only by :func:`get_comm` and :func:`node_local_rank` (which
needs a communicator, so MPI is initialised already); ``import aweSOM.stream``
never imports it. MPI calls are made on the main thread only.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

__all__ = ["get_comm", "Reducer", "row_range", "node_local_rank", "gpu_device"]

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


def row_range(n: int, rank: int, size: int) -> tuple[int, int]:
    """Rows ``[start, stop)`` of ``rank`` when ``n`` rows are split over ``size`` ranks.

    Contiguous, in rank order, and deterministic: rank r gets
    ``[r * n // size, (r + 1) * n // size)``. Sizes differ by at most one row.
    With fewer rows than ranks some ranks get an empty range.
    """
    n, rank, size = int(n), int(rank), int(size)
    if n < 0 or size < 1 or not 0 <= rank < size:
        raise ValueError(f"need n >= 0 and 0 <= rank < size, got n={n}, rank={rank}, size={size}")
    return rank * n // size, (rank + 1) * n // size


def _shared_split_type() -> Any:
    from mpi4py import MPI  # a communicator exists, so MPI is initialised already

    return MPI.COMM_TYPE_SHARED


def node_local_rank(comm: Any = None) -> int:
    """Rank of this process among the ranks of ``comm`` on the same node (0 without ``comm``).

    Uses ``comm.Split_type(COMM_TYPE_SHARED)``, a collective call: every rank of
    ``comm`` must call it. Reads no system information.
    """
    if comm is None:
        return 0
    local = comm.Split_type(_shared_split_type(), key=comm.Get_rank())
    try:
        return int(local.Get_rank())
    finally:
        local.Free()


def gpu_device(comm: Any = None, device_count: Optional[int] = None) -> int:
    """CUDA device index for this rank: node-local rank modulo the visible device count.

    CUDA numbers only the devices in ``CUDA_VISIBLE_DEVICES``. When Slurm gives
    each task its own GPU (``--gpus-per-task=1``, or one task per node on a
    one-GPU GH200 node), every rank sees one device and gets index 0, which is
    its own GPU. When ranks on a node share a list of several devices, rank k
    on the node takes device ``k % count``. ``device_count`` defaults to the
    CuPy device count. Collective when ``comm`` is given (see
    :func:`node_local_rank`).
    """
    local = node_local_rank(comm)
    if device_count is None:
        import cupy

        device_count = cupy.cuda.runtime.getDeviceCount()
    if device_count < 1:
        raise RuntimeError("no CUDA device is visible to this rank")
    return local % int(device_count)


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
