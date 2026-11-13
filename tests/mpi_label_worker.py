"""Run by tests/label_mpi_test.py under ``mpirun -n P python -m mpi4py``.

``python mpi_label_worker.py OUT`` labels the cases of :func:`cases` with
``predict_stream(..., comm=COMM_WORLD)``. Shared mode: rank 0 creates
``OUT/<case>-{labels,nodes,distances}.npy``, every rank fills its rows through
:class:`aweSOM.stream.NpyWriter`. Local mode (``out=None``): every rank writes
its own rows to ``OUT/<case>-local-rank<r>.npz``. The test compares both with a
single-process run.
"""
import json
import sys
from pathlib import Path

import numpy as np

from aweSOM.stream import NpyWriter, predict_stream

SHAPE = (8, 6)
CHUNK = 97


def lattice():
    rng = np.random.default_rng(3)
    u, v = np.meshgrid(np.linspace(-1, 1, SHAPE[0]), np.linspace(-1, 1, SHAPE[1]), indexing="ij")
    w = np.column_stack([u.ravel(order="F"), v.ravel(order="F")]
                        + [0.3 * rng.normal(size=u.size) for _ in range(5)])
    w[7] = w[3]  # equal nodes: the lowest index must win
    return w


def data(n):
    rng = np.random.default_rng(n)
    w = lattice()
    x = w[rng.integers(len(w), size=n)] + rng.normal(scale=0.05, size=(n, 7))
    x[::5] = w[3]  # exact ties between nodes 3 and 7
    return x


class Chunks:
    """Callable source with uneven chunks and lazy reads (like a reader of AMR blocks)."""

    def __init__(self, x):
        self.x, self.n_feat = x, x.shape[1]
        edges = np.unique(np.r_[0, np.random.default_rng(1).integers(0, len(x), 9), len(x)])
        self.parts = list(zip(edges[:-1], edges[1:]))

    def __call__(self):
        return (self.x[a:b] for a, b in self.parts)

    def lazy_chunks(self):
        for a, b in self.parts:
            yield b - a, (lambda a=a, b=b: self.x[a:b])


def cases():
    """name -> (source factory, N, backend)."""
    big, tiny = data(1003), data(2)
    return {
        "array-patch": (lambda: big, len(big), "patch"),
        "array-screen": (lambda: big, len(big), "screen"),
        "callable-patch": (lambda: Chunks(big), len(big), "patch"),
        "tiny-patch": (lambda: tiny, len(tiny), "patch"),
    }


def label(source, backend, comm=None, out=None, out_node=None, out_distance=None, n_samples=None):
    w = lattice()
    clusters = np.arange(len(w), dtype=np.int32) % 5
    return predict_stream(w, clusters, source, out=out, out_node=out_node, out_distance=out_distance,
                          chunk=CHUNK, backend=backend, lattice_shape=SHAPE, comm=comm,
                          n_samples=n_samples)


def main(out):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    out = Path(out)
    report = {}
    for name, (make, n, backend) in cases().items():
        paths = {k: out / f"{name}-{k}.npy" for k in ("labels", "nodes", "distances")}
        if rank == 0:
            for key, dtype in (("labels", "i4"), ("nodes", "i4"), ("distances", "f8")):
                created = np.lib.format.open_memmap(paths[key], mode="w+", dtype=dtype, shape=(n,))
                del created
        comm.Barrier()
        writers = {k: NpyWriter(p) for k, p in paths.items()}
        _, st = label(make(), backend, comm, writers["labels"], writers["nodes"], writers["distances"],
                      n_samples=n)
        for wr in writers.values():
            wr.close()
        comm.Barrier()
        local, lst = label(make(), backend, comm, n_samples=n)  # out=None: this rank's rows only
        np.savez(out / f"{name}-local-rank{rank}.npz", labels=local, start=lst["start"], stop=lst["stop"])
        report[name] = dict(start=st["start"], stop=st["stop"], rows=st["rows"],
                            crc=[writers[k].crc32 for k in ("labels", "nodes", "distances")])
    every = comm.gather(report, root=0)
    if rank == 0:
        (out / "report.json").write_text(json.dumps(every))


if __name__ == "__main__":
    main(sys.argv[1])
