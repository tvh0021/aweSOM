"""Node probes for the scale-out kit: Python environment and file read/write rates.

  python probe.py env --json PATH
      Python, numpy, numba (threading layers), mpi4py import and MPI library
      version, CPU count, affinity, NUMA nodes, memory.
  python probe.py write --path FILE --gb G --json PATH
      Write G GB of random bytes in 64 MiB blocks, fsync, then
      posix_fadvise(DONTNEED) so this node's page cache drops the file where the
      file system honours it.
  python probe.py read --path FILE --test NAME --role ROLE --json PATH
      Read FILE sequentially in 64 MiB blocks; record the start and end times
      (UTC epoch seconds) so concurrent readers can be lined up.

The page cache cannot be flushed on the cluster. A cold read is a read of a
freshly written file on a node other than the writer.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

BLOCK = 64 << 20


def node_name():
    """Node name from Slurm (SLURMD_NODENAME), or "local" outside Slurm; no system call."""
    return os.environ.get("SLURMD_NODENAME") or "local"


def refuse_on_tigris(*extra_paths):
    """GP-only survey: stop on RIT RC (Tigris or SPORC; user rule 2026-11-06, extended 2026-11-12)."""
    raw = (os.getcwd(), *(os.environ.get(k, "") for k in ("PWD", "SLURM_SUBMIT_DIR", "KIT", "OUT")), *extra_paths)
    checked = [os.path.normpath(str(p)) + "/" for p in raw if p] + [os.path.abspath(p) + "/" for p in raw if p]
    if os.environ.get("SLURM_CLUSTER_NAME", "").lower() in {"tigris", "sporc"} or any(
        m in c for c in checked for m in ("/shared/rc/", "/home/thccrg/")
    ):
        raise SystemExit("GP-only probe: refusing to run on RIT RC")


def env():
    import numpy as np

    refuse_on_tigris()
    out = dict(host=node_name(), python=sys.version.split()[0], executable=sys.executable,
               platform=platform.platform(), machine=platform.machine(), numpy=np.__version__,  # sysinfo-ok: GP-only probe, refuses on RIT RC
               cpu_count=os.cpu_count(),
               affinity=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
    try:
        import numba

        out["numba"] = numba.__version__

        @numba.njit(parallel=True)
        def f(n):
            s = 0.0
            for i in numba.prange(n):
                s += i
            return s

        f(10)
        out["numba_threading_layer"] = numba.threading_layer()
        out["numba_num_threads"] = numba.get_num_threads()
    except ImportError as exc:
        out["numba"] = repr(exc)
    try:
        from mpi4py import MPI

        out["mpi4py"] = dict(ok=True, file=__import__("mpi4py").__file__,
                             library=MPI.Get_library_version().strip()[:300],
                             vendor=list(MPI.get_vendor()))
    except Exception as exc:  # noqa: BLE001 - recorded
        out["mpi4py"] = dict(ok=False, error=repr(exc)[:300])
    try:
        rows = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())  # sysinfo-ok: GP-only probe, refuses on RIT RC
        out["mem_total_gb"] = int(rows["MemTotal"].split()[0]) / 2**20
        out["mem_available_gb"] = int(rows["MemAvailable"].split()[0]) / 2**20
    except OSError:
        pass
    numa = sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))  # sysinfo-ok: GP-only probe, refuses on RIT RC
    out["numa_nodes"] = {p.name: (p / "cpulist").read_text().strip() for p in numa}
    return out


def write(path, gb):
    import numpy as np

    rng = np.random.default_rng(int(time.time()))
    n = int(gb * (1 << 30)) // BLOCK
    gen = 0.0
    t0 = time.perf_counter()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        for _ in range(n):
            t = time.perf_counter()
            buf = rng.bytes(BLOCK)
            gen += time.perf_counter() - t
            os.write(fd, buf)
        os.fsync(fd)
        advised = hasattr(os, "posix_fadvise")
        if advised:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    secs = time.perf_counter() - t0
    size = n * BLOCK / (1 << 30)
    return dict(test="write", role="writer", path=str(path), gb=size, seconds=secs, generate_seconds=gen,
                gb_per_s=size / max(secs - gen, 1e-9), fadvise_dontneed=advised,
                note="rate excludes random-byte generation")


def read(path, test, role):
    start = time.time()
    t0 = time.perf_counter()
    total = 0
    with open(path, "rb", buffering=0) as fh:
        buf = bytearray(BLOCK)
        view = memoryview(buf)
        while True:
            k = fh.readinto(view)
            if not k:
                break
            total += k
    secs = time.perf_counter() - t0
    gb = total / (1 << 30)
    return dict(test=test, role=role, path=str(path), gb=gb, seconds=secs, gb_per_s=gb / secs, start=start,
                end=time.time())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("env", "write", "read"))
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--path", type=Path)
    p.add_argument("--gb", type=float, default=20.0)
    p.add_argument("--test", default="read")
    p.add_argument("--role", default="reader")
    a = p.parse_args(argv)
    if a.what == "env":
        refuse_on_tigris(a.json)
        res = env()
    elif a.what == "write":
        res = write(a.path, a.gb)
    else:
        res = read(a.path, a.test, a.role)
    res.setdefault("host", node_name())
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "numa_nodes"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
