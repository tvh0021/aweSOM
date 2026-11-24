"""Fake mpi4py communicators for tests that run without MPI.

``FakeComm`` runs real collectives over processes that share a barrier and a
manager dict, and puts every fake rank on one node. ``NodeComm`` and ``RankComm``
answer the rank and node queries of single-process calls. Processes that reach
:func:`aweSOM.distributed.node_local_rank` with a fake communicator (the
``cuda_screen`` device choice) must replace ``distributed._shared_split_type``
first (``shared_split`` fixture, or ``lambda: "shared"`` in a spawned rank), so
no fake run imports ``mpi4py.MPI``. Test modules that use the ``shared_split`` fixture import
it by name.
"""
from types import SimpleNamespace

import numpy as np
import pytest


class FakeComm:
    """The mpi4py calls of the driver and train_batch_stream, over processes sharing a
    barrier and a manager dict. Collective k of rank r posts its payload under (k, r)."""

    def __init__(self, rank, size, barrier, slots):
        self.rank, self.size, self.barrier, self.slots, self.k = rank, size, barrier, slots, 0

    def Get_rank(self): return self.rank
    def Get_size(self): return self.size
    def Barrier(self): self.barrier.wait(120)

    def _exchange(self, obj):
        key = self.k; self.k += 1
        self.slots[(key, self.rank)] = obj
        self.barrier.wait(120)
        every = [self.slots[(key, r)] for r in range(self.size)]
        self.barrier.wait(120)
        del self.slots[(key, self.rank)]
        return every

    def Allgather(self, send, recv): recv[...] = np.stack(self._exchange(np.array(send)))
    def Bcast(self, buf, root=0): buf[...] = self._exchange(np.array(buf) if self.rank == root else None)[root]
    def bcast(self, obj, root=0): return self._exchange(obj if self.rank == root else None)[root]
    def allgather(self, obj): return self._exchange(obj)

    def gather(self, obj, root=0):
        every = self._exchange(obj)
        return every if self.rank == root else None

    def Split_type(self, kind, key=0):
        """One node: fake rank r is node-local rank r (what gpu_device needs on a GPU host)."""
        assert kind == "shared", f"replace distributed._shared_split_type first, got {kind!r}"
        rank = self.rank
        return SimpleNamespace(Get_rank=lambda: rank, Free=lambda: None)


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
    from aweSOM import distributed

    monkeypatch.setattr(distributed, "_shared_split_type", lambda: "shared")


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
