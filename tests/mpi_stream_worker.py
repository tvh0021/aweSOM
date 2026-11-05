"""Run by tests/stream_scaleout_test.py under ``mpirun -n P python -m mpi4py``.

``python mpi_stream_worker.py SCENARIO OUT`` writes ``OUT/rank<r>.npz`` from every
rank. The ``run_*`` helpers are also called by the tests with ``comm=None`` to get
the single-process reference with identical settings.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import gmm_bbh7  # noqa: E402

from aweSOM.stream import train_batch_stream  # noqa: E402

LATTICE = (20, 14)
SEED = 42


class Stop(Exception):
    pass


class Windows:
    """Stateful callable like production_pool.EpochSource: call k reads random windows."""

    def __init__(self, x, rows, chunk, seed=7, fail_at_call=None):
        self.x, self.rows_per_call, self.chunk, self.seed = x, rows, chunk, seed
        self.calls = 0
        self.n_feat = x.shape[1]
        self.fail_at_call = fail_at_call

    def _windows(self, call):
        rng = np.random.default_rng([self.seed, call])
        out, left = [], self.rows_per_call
        while left:
            n = min(left, self.chunk)
            out.append((int(rng.integers(0, len(self.x) - n + 1)), n))
            left -= n
        return out

    def lazy_chunks(self):
        if self.calls == self.fail_at_call:
            raise Stop
        windows = self._windows(self.calls)
        self.calls += 1
        for a, n in windows:
            yield n, (lambda a=a, n=n: self.x[a : a + n])

    def __call__(self):
        for _, read in self.lazy_chunks():
            yield read()

    def rows(self, ix):
        stream_rows = np.concatenate([np.arange(a, a + n) for a, n in self._windows(self.calls)])
        return self.x[stream_rows[ix]]


def data(dtype="float32"):
    """30000 gmm_bbh7 rows. Float32 rows sum exactly in float64 at this size (the
    rank order then does not change a bit); float64 rows get a full mantissa."""
    x = gmm_bbh7.rows(30_000)
    if dtype == "float64":
        x = x + np.random.default_rng(5).normal(0.0, 1e-3, x.shape)
    return x


def run_array(x, comm):
    history = [x[np.random.default_rng(SEED).integers(0, len(x), LATTICE[0] * LATTICE[1])].astype(np.float64)]
    w, info = train_batch_stream(x, *LATTICE, epochs=5, chunk=2048, r_min=3, seed=SEED, final_pass="full",
                                 comm=comm, callback=lambda e, w: history.append(w))
    return w, info, np.array(history[:5])


def run_uneven(x, comm):
    return train_batch_stream(x[:5000], *LATTICE, epochs=4, chunk=2048, r_min=3, seed=SEED,
                              final_pass="full", comm=comm)


def run_sampled(x, comm):
    return train_batch_stream(x, *LATTICE, epochs=4, chunk=2048, r_min=3, seed=SEED, sample_fraction=0.5,
                              sample_block=256, final_pass="sample", comm=comm)


def run_callable(x, comm, checkpoint=None, resume=False, fail_at_call=None):
    src = Windows(x, 12000, 3000, fail_at_call=fail_at_call)
    w, info = train_batch_stream(src, *LATTICE, epochs=5, r_min=2, seed=SEED, n_samples=12000,
                                 final_pass="none", comm=comm, checkpoint=checkpoint, resume=resume)
    return (w, info, src) if comm is not None else w


def main(scenario, out):
    from aweSOM.distributed import get_comm

    comm = get_comm()
    rank = comm.Get_rank()
    x = data("float64" if scenario.endswith("64") else "float32")
    res = {"ranks": comm.Get_size()}
    if scenario in ("array", "array64"):
        w, info, hist = run_array(x, comm)
        res.update(weights=w, hits=info["hits"], rows_per_epoch=info["rows_per_epoch"],
                   rank_rows=np.array(info["rank_rows"]), history=hist if rank == 0 else np.zeros(0))
    elif scenario == "uneven":
        w, info = run_uneven(x, comm)
        ws, iss = run_sampled(x, comm)
        res.update(weights=w, rows_per_epoch=info["rows_per_epoch"], rank_rows=np.array(info["rank_rows"]),
                   sampled_weights=ws, sampled_rows=iss["rows_per_epoch"])
    elif scenario == "resume":
        ck = str(Path(out) / "ck.npz")
        try:
            run_callable(x, comm, checkpoint=ck, resume=True, fail_at_call=2)
            raise RuntimeError("the interrupted run did not stop")
        except Stop:
            pass
        w, info, src = run_callable(x, comm, checkpoint=ck, resume=True)
        w_full, _, _ = run_callable(x, comm)
        res.update(resumed=w, uninterrupted=w_full, resumed_from=info["resumed_from"], calls=src.calls)
    else:
        raise SystemExit(f"unknown scenario {scenario}")
    np.savez(Path(out) / f"rank{rank}.npz", **res)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
