"""One timed training run of the streaming SOM on synthetic gmm_bbh7 rows, 1 or more ranks.

Every rank generates only the chunks it owns (chunk ``i`` belongs to rank
``i % ranks``, as in ``train_batch_stream(comm=...)``), so a P-rank run holds N/P
rows per node and reads nothing from disk. Block ``b`` of ``GEN_BLOCK`` rows comes
from ``default_rng([data_seed, b])`` with elementwise float32 arithmetic (the
generator of studies/2026-11-03/Z-scale-1e9-gp, job 8238), so the rows, and the
1-rank weights, equal those of an in-memory array of all N rows. The sampled
initial lattice is regenerated row by row on rank 0 (``GMM.rows``), so every rank
count starts from the same lattice.

Writes, under ``--out``: ``run.json`` (rank 0: settings, per-epoch timings and
rows, per-rank rows and pass times, reduction time, node names, peak RSS per
rank, git commit), ``rank<r>.json`` (every rank), ``weights.npy``,
``checkpoint.npz`` and ``progress.log``.

Usage (one rank):  python scale_out.py --out DIR --n 1e8 --lattice 91x64 --threads 32
       (P ranks):  srun ... python -m mpi4py scale_out.py --mpi --out DIR ...
"""
from __future__ import annotations

import os
import sys


def _launcher_rank() -> int:
    for k in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID"):
        if os.environ.get(k, "").isdigit():
            return int(os.environ[k])
    return 0


# One numba cache per rank (set before numba is imported; never shared across ranks).
if os.environ.get("SCALE_CACHE_ROOT"):
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.environ["SCALE_CACHE_ROOT"], f"rank{_launcher_rank()}")

import argparse  # noqa: E402
import concurrent.futures  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

KIT = Path(__file__).resolve().parent
GEN_BLOCK = 2**20


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


# ------------------------------------------------------------------ generator

class GMM:
    """gmm_bbh7 mixture; ``block`` is the Z-scale kit generator, ``rows`` gives single rows of it."""

    def __init__(self, path: Path):
        z = np.load(path)
        w = z["weights"].astype(np.float64)
        self.mu = np.ascontiguousarray(z["means"], np.float32)
        self.L = np.ascontiguousarray(z["cholesky"], np.float32)
        self.lo = z["clip_lo"].astype(np.float32)
        self.hi = z["clip_hi"].astype(np.float32)
        self.k, self.f = self.mu.shape
        self.cumw = np.cumsum(w / w.sum())
        self.cumw[-1] = 1.0
        self.Lij = {(i, j): np.ascontiguousarray(self.L[:, i, j]) for i in range(self.f) for j in range(i + 1)}
        self.mui = [np.ascontiguousarray(self.mu[:, i]) for i in range(self.f)]

    def _draw(self, seed: int, b: int):
        rng = np.random.default_rng([seed, b])
        comp = np.searchsorted(self.cumw, rng.random(GEN_BLOCK), side="right")
        np.minimum(comp, self.k - 1, out=comp)
        z = rng.standard_normal((self.f, GEN_BLOCK), dtype=np.float32)
        return comp, z

    def _mix(self, comp, z):
        """Rows for components ``comp`` and normals ``z`` (F, m), transposed; fixed float32 op order."""
        m = comp.shape[0]
        x = np.empty((self.f, m), np.float32)
        tmp = np.empty(m, np.float32)
        for i in range(self.f):
            np.take(self.mui[i], comp, out=x[i], mode="clip")
            for j in range(i + 1):
                np.take(self.Lij[i, j], comp, out=tmp, mode="clip")
                np.multiply(tmp, z[j], out=tmp)
                np.add(x[i], tmp, out=x[i])
            np.clip(x[i], self.lo[i], self.hi[i], out=x[i])
        return x

    def block(self, seed: int, b: int) -> np.ndarray:
        """Block ``b``: (GEN_BLOCK, F) float32."""
        return np.ascontiguousarray(self._mix(*self._draw(seed, b)).T)

    def rows(self, seed: int, ix: np.ndarray, threads: int) -> np.ndarray:
        """Rows ``ix`` (any order) as float32, without generating whole blocks of features."""
        ix = np.asarray(ix, np.int64)
        out = np.empty((len(ix), self.f), np.float32)
        blocks = np.unique(ix // GEN_BLOCK)

        def work(b):
            sel = np.flatnonzero(ix // GEN_BLOCK == b)
            comp, z = self._draw(seed, int(b))
            j = ix[sel] - b * GEN_BLOCK
            out[sel] = self._mix(comp[j], np.ascontiguousarray(z[:, j])).T

        with concurrent.futures.ThreadPoolExecutor(threads) as pool:
            list(pool.map(work, blocks))
        return out


class OwnedRows:
    """Callable source over N generated rows; this rank holds only the chunks it owns."""

    def __init__(self, gmm: GMM, n: int, chunk: int, seed: int, rank: int, size: int, threads: int):
        if chunk % GEN_BLOCK:
            raise ValueError(f"chunk must be a multiple of {GEN_BLOCK}")
        self.gmm, self.n, self.chunk, self.seed, self.threads = gmm, n, chunk, seed, threads
        self.rank = rank
        self.n_feat = gmm.f
        self.n_chunks = -(-n // chunk)
        self.owned = [i for i in range(self.n_chunks) if i % size == rank]
        self.data: dict[int, np.ndarray] = {}
        self.rows_seconds = 0.0
        t = time.perf_counter()
        per = chunk // GEN_BLOCK
        jobs = [(i, k) for i in self.owned for k in range(per) if (i * per + k) * GEN_BLOCK < n]
        for i in self.owned:
            self.data[i] = np.empty((min(chunk, n - i * chunk), self.n_feat), np.float32)

        def work(job):
            i, k = job
            lo = k * GEN_BLOCK
            hi = min(lo + GEN_BLOCK, self.data[i].shape[0])
            self.data[i][lo:hi] = gmm.block(seed, i * per + k)[: hi - lo]

        with concurrent.futures.ThreadPoolExecutor(threads) as pool:
            list(pool.map(work, jobs))
        self.generate_seconds = time.perf_counter() - t
        self.owned_rows = int(sum(a.shape[0] for a in self.data.values()))

    def _read(self, i: int) -> np.ndarray:
        if i not in self.data:
            raise RuntimeError(f"chunk {i} is not owned by this rank")
        return self.data[i]

    def lazy_chunks(self):
        for i in range(self.n_chunks):
            yield min(self.chunk, self.n - i * self.chunk), (lambda i=i: self._read(i))

    def __call__(self):
        for _, read in self.lazy_chunks():
            yield read()

    def rows(self, ix):
        if self.rank != 0:  # train_batch_stream broadcasts rank 0's rows
            return np.zeros((len(ix), self.n_feat), np.float32)
        t = time.perf_counter()
        out = self.gmm.rows(self.seed, ix, self.threads)
        self.rows_seconds += time.perf_counter() - t
        return out


# ------------------------------------------------------------------ environment

def lscpu() -> dict:
    try:
        text = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    keep = ("Architecture", "Model name", "CPU(s)", "Socket(s)", "Core(s) per socket", "Thread(s) per core",
            "NUMA node(s)", "CPU max MHz", "L3 cache")
    out = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = (s.strip() for s in line.split(":", 1))
            if k in keep or k.startswith("NUMA node") and "CPU(s)" in k:
                out[k] = v
    return out


def peak_rss() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def source_info() -> dict:
    import aweSOM

    pkg = Path(aweSOM.__file__).resolve().parent
    manifest = KIT / "kit-manifest.json"
    info = dict(package_dir=str(pkg), version=aweSOM.__version__)
    if manifest.exists():
        m = json.loads(manifest.read_text())
        info.update(commit=m.get("commit"), branch=m.get("branch"), manifest_tree_sha256=m.get("tree_sha256"))
        files = {str(p.relative_to(pkg)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(pkg.rglob("*.py"))}
        info["package_matches_manifest"] = files == m.get("files")
    else:
        try:
            info["commit"] = subprocess.run(["git", "-C", str(pkg), "rev-parse", "HEAD"], capture_output=True,
                                            text=True, timeout=10).stdout.strip()
        except OSError:
            info["commit"] = None
    return info


def affinity_check(threads: int, rank: int, comm=None) -> int:
    """CPU count of this rank. Raises on every rank if any rank holds fewer than ``threads``
    CPUs (gathered, so all ranks stop together, before any data or training).

    Job 8413: Open MPI rebound every rank of a 1-rank-per-node srun to one CPU, and the
    32 numba threads ran 10x slow on it. Skipped where ``os.sched_getaffinity`` is missing.
    """
    if not hasattr(os, "sched_getaffinity"):
        return -1
    mine = (rank, platform.node(), len(os.sched_getaffinity(0)))
    every = comm.allgather(mine) if comm is not None else [mine]
    bad = [f"rank {r} on {h}: affinity {n} CPUs < {threads} threads" for r, h, n in every if n < threads]
    if bad:
        raise RuntimeError("CPU affinity too small, refusing to run (srun needs --cpu-bind=cores and "
                           "OMPI_MCA_hwloc_base_binding_policy=none, see README): " + "; ".join(bad))
    return mine[2]


# ------------------------------------------------------------------ run

def parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--mpi", action="store_true", help="use MPI.COMM_WORLD (launch with python -m mpi4py)")
    p.add_argument("--n", type=float, default=1e8)
    p.add_argument("--lattice", default="91x64")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--r-min", type=float, default=None, help="default 8 for >= 91 columns, else 5")
    p.add_argument("--chunk", type=int, default=2**21)
    p.add_argument("--groups", type=int, default=64)
    p.add_argument("--bmu", default="patch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-seed", type=int, default=20261103)
    p.add_argument("--resume", action="store_true", help="continue from OUT/checkpoint.npz")
    p.add_argument("--warmup", type=int, default=131072, help="rows of the JIT warm-up (0: none)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    n = int(args.n)
    xdim, ydim = (int(v) for v in args.lattice.lower().split("x"))
    r_min = args.r_min if args.r_min is not None else (8.0 if xdim >= 91 else 5.0)
    comm = None
    if args.mpi:
        from aweSOM.distributed import get_comm

        comm = get_comm()
    rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    affinity_check(args.threads, rank, comm)
    out = args.out
    if rank == 0:
        out.mkdir(parents=True, exist_ok=args.resume)
    if comm is not None:
        comm.Barrier()
    import numba

    numba.set_num_threads(args.threads)
    from aweSOM.stream import train_batch_stream

    log = open(out / "progress.log", "a", buffering=1) if rank == 0 else None

    def say(msg):
        if log is not None:
            text = f"{utc()}  {msg}"
            log.write(text + "\n")
            print(text, flush=True)

    settings = dict(tag=args.tag, n=n, lattice=f"{xdim}x{ydim}", epochs=args.epochs, threads=args.threads,
                    ranks=size, r_min=r_min, chunk=args.chunk, groups=args.groups, bmu=args.bmu, seed=args.seed,
                    data_seed=args.data_seed, final_pass="none", sample_fraction=1.0, dtype="float32",
                    source="per-rank generated owned chunks (round-robin by chunk index)")
    say(f"start {json.dumps(settings)}")
    gmm = GMM(KIT / "gmm_bbh7.npz")
    block0 = hashlib.sha256(gmm.block(args.data_seed, 0).tobytes()).hexdigest()
    src = OwnedRows(gmm, n, args.chunk, args.data_seed, rank, size, args.threads)
    rss_after_generate = peak_rss()
    say(f"rank 0 generated {src.owned_rows} rows ({len(src.owned)} chunks) in {src.generate_seconds:.1f} s")

    t = time.perf_counter()
    if args.warmup:
        wx = gmm.block(args.data_seed + 1, 0)[: args.warmup]
        train_batch_stream(wx, xdim, ydim, epochs=2, chunk=65536, bmu=args.bmu, seed=args.seed + 1,
                           groups=args.groups, r_min=r_min, final_pass="none")
    warmup = time.perf_counter() - t
    say(f"jit warm-up {warmup:.1f} s, threading layer {numba.threading_layer()}")

    epoch_wall = []
    clock = {"t": None}

    def callback(e, w):
        now = time.perf_counter()
        epoch_wall.append(now - clock["t"])
        say(f"epoch {e + 1}/{args.epochs}: {epoch_wall[-1]:.1f} s since previous")
        clock["t"] = now

    if comm is not None:
        comm.Barrier()
    t0 = time.perf_counter()
    clock["t"] = t0
    w, info = train_batch_stream(src, xdim, ydim, epochs=args.epochs, chunk=args.chunk, bmu=args.bmu,
                                 seed=args.seed, groups=args.groups, r_min=r_min, final_pass="none",
                                 n_samples=n, n_threads=args.threads, comm=comm, callback=callback,
                                 checkpoint=str(out / "checkpoint.npz"), resume=args.resume, verbose=rank == 0)
    wall = time.perf_counter() - t0
    if w.dtype != np.float64 or not np.isfinite(w).all():
        raise RuntimeError("training returned invalid weights")

    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    mine = dict(rank=rank, hostname=platform.node(), pid=os.getpid(), affinity=affinity,
                numba_threads=numba.get_num_threads(), threading_layer=numba.threading_layer(),
                owned_chunks=len(src.owned), owned_rows=src.owned_rows, generate_seconds=src.generate_seconds,
                init_rows_seconds=src.rows_seconds, warmup_seconds=warmup, training_wall_seconds=wall,
                peak_rss_bytes=peak_rss(), rss_after_generate_bytes=rss_after_generate,
                bmu_kind=info["bmu_kind"], io_wait=info["io_wait"], bmu_time=info["bmu_time"],
                acc_time=info["acc_time"], weights_sha256=hashlib.sha256(w.tobytes()).hexdigest())
    (out / f"rank{rank}.json").write_text(json.dumps(plain(mine), indent=1) + "\n")
    ranks = comm.gather(mine, root=0) if comm is not None else [mine]
    if rank != 0:
        return 0

    rows_done = int(np.sum(info["rows_per_epoch"][info["resumed_from"]:]))
    later = info["epoch_time"][1:]
    rec = dict(
        schema="scale_out/1", finished_utc=utc(), settings=settings, source=source_info(),
        block0_sha256=block0, environment=dict(python=sys.version.split()[0], numpy=np.__version__,
                                               numba=numba.__version__, platform=platform.platform(),
                                               lscpu=lscpu(), slurm={k: v for k, v in os.environ.items()
                                                                     if k.startswith("SLURM_")}),
        nodes=sorted({r["hostname"] for r in ranks}), n_nodes=len({r["hostname"] for r in ranks}),
        ranks=size, threads_per_rank=args.threads, total_threads=size * args.threads,
        training_wall_seconds=wall, rows_processed=rows_done, rows_per_second=rows_done / wall,
        first_epoch_rows_per_second=float(info["rows_per_epoch"][0] / info["epoch_time"][0]),
        later_epoch_rows_per_second=float(np.sum(info["rows_per_epoch"][1:]) / np.sum(later)) if len(later) else None,
        reduce_seconds_total=float(np.sum(info["reduce_time"])), io_wait_total=float(np.sum(info["io_wait"])),
        checkpoint_seconds_total=float(np.sum(info["checkpoint_time"])),
        peak_rss_bytes_per_rank=[r["peak_rss_bytes"] for r in ranks], resumed_from=info["resumed_from"],
        weights_sha256=mine["weights_sha256"],
        epochs=[dict(epoch=e + 1, radius=float(info["radius"][e]), seconds=float(info["epoch_time"][e]),
                     rows=int(info["rows_per_epoch"][e]),
                     rows_per_second=float(info["rows_per_epoch"][e] / info["epoch_time"][e]),
                     io_wait=float(info["io_wait"][e]), bmu=float(info["bmu_time"][e]),
                     acc=float(info["acc_time"][e]), update=float(info["update_time"][e]),
                     reduce=float(info["reduce_time"][e]), checkpoint=float(info["checkpoint_time"][e]),
                     rank_rows=info["rank_rows"][e], rank_pass_seconds=info["rank_pass_time"][e],
                     rank_bmu_seconds=info["rank_bmu"][e], fallbacks=int(info["fallbacks"][e]),
                     qe_entering=float(info["qe"][e - 1]) if e else float(info["qe_initial"]))
                for e in range(args.epochs)],
        per_rank=ranks)
    np.save(out / "weights.npy", w)
    (out / "run.json").write_text(json.dumps(plain(rec), indent=1) + "\n")
    say(f"done: {wall:.1f} s, {rec['rows_per_second'] / 1e6:.2f} Mrows/s, ranks {size}, "
        f"nodes {rec['nodes']}, weights {mine['weights_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
