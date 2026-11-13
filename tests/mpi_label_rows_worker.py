"""Run by tests/label_mpi_test.py under ``mpirun -n P python -m mpi4py``.

``python mpi_label_rows_worker.py OUT`` runs the driver's ``label_rows``
(examples/grmhd-bbh/production_pool.py) with ``COMM_WORLD`` on the cases of
:data:`CASES`, with every open of an output file guarded by
:mod:`output_guard` (two ranks holding one output file open fail). Rank 0
creates the final files first, as ``fit`` does. A control at the end checks
that the guard catches two ranks holding one file open. Rank 0 writes
``OUT/report.json``: rank 0's records, every rank's guarded opens and the
control outcome.
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import mpi_label_worker as worker  # noqa: E402
import output_guard  # noqa: E402

CASES = {"array": 1003, "tiny": 2, "one": 1}  # uneven ranges; with 3 ranks "tiny" leaves rank 0 empty


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "production_pool", HERE.parent / "examples/grmhd-bbh/production_pool.py")
    pool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pool)
    return pool


def label_case(pool, out, tag, n, comm):
    w = worker.lattice()
    nc = np.arange(len(w), dtype=np.int32) % 5
    rank = 0 if comm is None else comm.Get_rank()
    if not rank:
        for key, dtype in pool.LABEL_FILES:
            made = np.lib.format.open_memmap(out / f"{tag}-{key}.npy", mode="w+", dtype=dtype, shape=(n,))
            del made
    if comm is not None:
        comm.Barrier()
    return pool.label_rows(SimpleNamespace(chunk=worker.CHUNK), w, nc, worker.SHAPE, worker.data(n),
                           lambda block: np.asarray(block, "f8"), out, tag, "patch", comm)


def main(out):
    from mpi4py import MPI

    from aweSOM.stream import NpyWriter

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    out = Path(out)
    pool = load_driver()
    output_guard.install()
    records = {tag: label_case(pool, out, tag, n, comm) for tag, n in CASES.items()}
    opened = list(output_guard.opened)
    # Control: rank 0 holds a writer open; every other rank's open must fail.
    control = out / "control-labels.npy"
    if not rank:
        np.save(control, np.zeros(4, "i4"))
    comm.Barrier()
    held = NpyWriter(control) if not rank else None
    comm.Barrier()
    caught = None
    if rank:
        try:
            NpyWriter(control).close()
            caught = False
        except output_guard.Overlap:
            caught = True
    comm.Barrier()
    if held is not None:
        held.close()
    every = comm.gather(dict(rank=rank, opened=opened, caught=caught), root=0)
    if not rank:
        (out / "report.json").write_text(json.dumps(dict(records=records, ranks=every)))


if __name__ == "__main__":
    main(sys.argv[1])
