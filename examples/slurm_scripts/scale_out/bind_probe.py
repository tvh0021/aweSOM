"""CPU affinity of one rank around MPI_Init, for bind_probe.sbatch. Prints one line:

  BIND case=A rank=0 size=1 host=gp-n0015 online=32 cgroup=32 policy_env=none
       before=32 after_load=32 after_init=1 cpus_after=0 threads=4:1,32,32,32

before: CPUs of this thread at start; after_load: after ``from mpi4py import MPI`` with
``rc.initialize = False`` (the MPI library is loaded, MPI_Init has not run); after_init:
after ``MPI.Init_thread(THREAD_MULTIPLE)`` (what ``from mpi4py import MPI`` does by default).
threads: CPU count of every thread of the process after MPI_Init (main thread first), which
tells a per-thread ``sched_setaffinity(0)`` (other threads keep their set) from a
process-wide hwloc binding (every thread changes). No restore unless ``--restore`` (then
``restored=<CPUs after setting the saved set back>``, 0 if the set did not shrink).
``--no-mpi`` prints only the start values (Python under the same srun, no MPI).
"""
import os
import sys


def refuse_on_tigris():
    """GP-only survey: stop on Tigris (no host-state reads there, user rule 2026-11-06)."""
    paths = (os.getcwd(), os.environ.get("SLURM_SUBMIT_DIR", ""), os.environ.get("KIT", ""), os.environ.get("OUT", ""))
    if os.environ.get("SLURM_CLUSTER_NAME", "").lower() == "tigris" or any("/shared/rc/" in p for p in paths):
        raise SystemExit("GP-only probe: refusing to run on Tigris")


refuse_on_tigris()
SAVED = frozenset(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def ncpu(pid=0):
    return len(os.sched_getaffinity(pid)) if hasattr(os, "sched_getaffinity") else -1


def ranges(cpus):
    out, run = [], []
    for c in sorted(cpus):
        if run and c == run[-1] + 1:
            run.append(c)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [c]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out) or "-"


def cgroup_cpus():
    """CPU count the step's cgroup allows (cgroup v2 or v1 cpuset), or '?'."""
    try:
        lines = open("/proc/self/cgroup").read().splitlines()  # sysinfo-ok: GP-only probe, refuses on Tigris
    except OSError:
        return "?"
    paths = []
    for line in lines:
        _, ctrl, path = line.split(":", 2)
        if ctrl == "":
            paths.append(f"/sys/fs/cgroup{path}/cpuset.cpus.effective")  # sysinfo-ok: GP-only probe, refuses on Tigris
        elif "cpuset" in ctrl.split(","):
            paths += [f"/sys/fs/cgroup/cpuset{path}/cpuset.effective_cpus", f"/sys/fs/cgroup/cpuset{path}/cpuset.cpus"]  # sysinfo-ok: GP-only probe, refuses on Tigris
    for p in paths:
        try:
            text = open(p).read().strip()
        except OSError:
            continue
        n = 0
        for part in text.split(","):
            if part:
                a, _, b = part.partition("-")
                n += int(b or a) - int(a) + 1
        return str(n)
    return "?"


def thread_cpus():
    try:
        tids = sorted(int(t) for t in os.listdir("/proc/self/task"))
    except OSError:
        return "?"
    main = os.getpid()
    tids.sort(key=lambda t: t != main)
    counts = []
    for t in tids:
        try:
            counts.append(str(ncpu(t)))
        except OSError:
            counts.append("x")
    return f"{len(tids)}:{','.join(counts)}"


def main(argv):
    case = argv[argv.index("--case") + 1] if "--case" in argv else "-"
    head = (f"BIND case={case} host={os.environ.get('SLURMD_NODENAME') or 'local'} online={os.cpu_count()} cgroup={cgroup_cpus()} "
            f"policy_env={os.environ.get('OMPI_MCA_hwloc_base_binding_policy', '(unset)')}")
    before = len(SAVED) if SAVED is not None else -1
    if "--no-mpi" in argv:
        print(f"{head} rank=- size=- before={before} after_load=- after_init=- "
              f"cpus_before={ranges(SAVED or ())}", flush=True)
        return 0
    import mpi4py

    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI

    after_load = ncpu()
    MPI.Init_thread(MPI.THREAD_MULTIPLE)
    now = os.sched_getaffinity(0) if SAVED is not None else set()
    tail = ""
    if "--restore" in argv:
        restored = 0
        if SAVED is not None and len(now) < len(SAVED):
            os.sched_setaffinity(0, SAVED)
            restored = ncpu()
        tail = f" restored={restored}"
    c = MPI.COMM_WORLD
    print(f"{head} rank={c.Get_rank()} size={c.Get_size()} before={before} after_load={after_load} "
          f"after_init={len(now) if SAVED is not None else -1}{tail} cpus_after={ranges(now)} threads={thread_cpus()}", flush=True)
    MPI.Finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
