# Shared functions of the scale-out sbatch files. Source it, then call kit_init NAME.
# Paths: KIT (this directory, default $SLURM_SUBMIT_DIR), VENV (python with numpy/numba/h5py),
# OUT (fresh output dir, default $KIT/runs/NAME-JOBID), OUT_ROOTS (allowed parents).
# MPI: $KIT/mpi-site (pip --target dir holding mpi4py, fixed, not overridable; PYTHONPATH),
# LAUNCH (auto | srun-pmix | srun-pmi2 | srun-pmix_v4 | mpirun), MPIRUN (path, mpirun mode),
# MPI_BUILD (probe only: wheel = binary mpi4py wheel, default; source = compile, needs a C compiler).

# True on Tigris (cluster name, or the /shared/rc/ paths): GP-only surveys skip or refuse there
# (no host-state reads on Tigris, user rule 2026-11-06).
on_tigris() {
  case "$(printf '%s' "${SLURM_CLUSTER_NAME:-}" | tr '[:upper:]' '[:lower:]')|$PWD|${SLURM_SUBMIT_DIR:-}|${KIT:-}|${OUT:-}" in
    tigris\|*|*/shared/rc/*) return 0;;
  esac
  return 1
}

kit_modules() {
  # Make `module` available in a batch shell; load the MPI module the probe recorded.
  if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
      [ -f "$f" ] && { . "$f"; break; }
    done
  fi
  if [ -f "$MPI_SITE/MODULE" ] && [ -s "$MPI_SITE/MODULE" ]; then
    module load "$(cat "$MPI_SITE/MODULE")" || echo "module load $(cat "$MPI_SITE/MODULE") failed" >&2
  fi
  # The binary mpi4py wheel dlopens the MPI library named here (srun and mpirun -x pass it on).
  if [ -s "$MPI_SITE/LIBMPI" ]; then MPI4PY_LIBMPI=$(cat "$MPI_SITE/LIBMPI"); export MPI4PY_LIBMPI; fi
}

kit_init() {
  NAME=$1
  KIT=${KIT:-$SLURM_SUBMIT_DIR}
  OUT=${OUT:-$KIT/runs/$NAME-${SLURM_JOB_ID:-local}}
  OUT_ROOTS=${OUT_ROOTS:-$HOME:/lagoon/trungha}
  VENV=${VENV:-/lagoon/trungha/bbh-output/for-awesom-2026-10-28/check7-venv}
  # The only pip target is inside the kit; an inherited MPI_SITE (--export=ALL) is ignored.
  MPI_SITE=$KIT/mpi-site
  [ ! -L "$MPI_SITE" ] || { echo "mpi-site must not be a symlink: $MPI_SITE" >&2; exit 2; }
  # Node-local disk for the local I/O test: the scheduler's scratch or /tmp, never an
  # inherited TMPDIR (which may point at HOME). Each node gets a fresh per-job directory.
  LOCAL_SCRATCH=${SLURM_TMPDIR:-/tmp}
  [ -f "$KIT/scale_out.py" ] || { echo "KIT has no scale_out.py: $KIT" >&2; exit 2; }
  [ -d "$KIT/aweSOM-src/src/aweSOM" ] || { echo "KIT has no aweSOM-src/src/aweSOM (run make_kit.sh)" >&2; exit 2; }
  [ -x "$VENV/bin/python" ] || { echo "VENV has no bin/python: $VENV" >&2; exit 2; }
  [ -n "$SLURM_CPUS_PER_TASK" ] || { echo "SLURM_CPUS_PER_TASK is not set" >&2; exit 2; }
  # Validate OUT before creating anything. Refuse an existing OUT, even a failed run.
  case "$OUT" in /*) ;; *) echo 'OUT must be absolute' >&2; exit 2;; esac
  case "/$OUT/" in */../*|*/./*) echo 'OUT contains dot components' >&2; exit 2;; esac
  local p=$OUT anc resolved inside=no r
  while [ "$p" != / ]; do
    [ ! -L "$p" ] || { echo "symlink in OUT: $p" >&2; exit 2; }
    p=$(dirname "$p")
  done
  anc=$OUT; while [ ! -d "$anc" ]; do anc=$(dirname "$anc"); done
  resolved=$(cd "$anc" && pwd -P)
  IFS=: read -r -a roots <<< "$OUT_ROOTS"
  for r in "${roots[@]}"; do
    [ -n "$r" ] || continue
    case "$resolved/" in "${r%/}/"*) inside=yes;; esac
  done
  [ "$inside" = yes ] || { echo "OUT must resolve under one of: $OUT_ROOTS" >&2; exit 2; }
  [ ! -e "$OUT" ] || { echo "OUT exists; use a fresh output directory: $OUT" >&2; exit 2; }
  mkdir -p "$(dirname "$OUT")"
  mkdir "$OUT"
  OUT=$(cd "$OUT" && pwd -P)
  STATUS_DIR="$OUT/status/${SLURM_JOB_ID:-local}-${SLURM_RESTART_COUNT:-0}"
  completed=no
  trap 'kit_finish "$?"' EXIT; trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP
  mkdir -p "$STATUS_DIR" "$OUT/tmp" "$OUT/mpl" "$OUT/probe"
  printf 'STARTED\t%s\t%s\n' "$(date -u +%FT%TZ)" "${SLURMD_NODENAME:-local}" > "$STATUS_DIR/start.tsv"
  cd "$OUT"
  PY="$VENV/bin/python"
  export PYTHONPATH="$KIT/aweSOM-src/src${MPI_SITE:+:$MPI_SITE}"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
  export MPLCONFIGDIR="$OUT/mpl" TMPDIR="$OUT/tmp" SCALE_CACHE_ROOT="$OUT/numba-cache"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
  NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
  kit_modules
  echo "$NAME: OUT=$OUT nodes=${NODES[*]} cpus/task=$SLURM_CPUS_PER_TASK"
}

kit_finish() {
  local rc=$1 state=FAILED
  trap - EXIT TERM INT HUP; set +e
  # The I/O test's large files live only in the fresh $OUT/io; remove them on every exit path.
  [ -d "$OUT/io" ] && rm -f "$OUT/io/a.bin" "$OUT/io/b.bin"
  mkdir -p "$STATUS_DIR"
  if [ "$rc" -eq 0 ] && [ "$completed" = yes ]; then state=VALIDATED; elif [ "$rc" -eq 0 ]; then rc=1; fi
  printf '%s\t%s\t%s\n' "$state" "$rc" "$(date -u +%FT%TZ)" > "$STATUS_DIR/final.tmp"
  mv "$STATUS_DIR/final.tmp" "$STATUS_DIR/final.tsv"
  echo "$NAME: $state exit=$rc"
  exit "$rc"
}

# Light environment probe on every node of the job (brief item e, without the I/O test).
env_probe() {
  local d="$OUT/probe"
  if on_tigris; then echo "env probe skipped: GP-only survey, not run on Tigris"; return 0; fi
  { sinfo -Nel -n "$SLURM_JOB_NODELIST"; echo; sinfo -p "$SLURM_JOB_PARTITION" -o '%P %D %c %m %l %N'; } > "$d/sinfo.txt" 2>&1 || true  # sysinfo-ok: GP-only survey, skipped on Tigris
  # Cluster-wide: every partition (availability, nodes, cpus, mem, limit, gres) and every node
  # (gres and features), to see whether any partition has GPUs and how big gp-default is.
  { echo; echo '== all partitions'; sinfo -a -o '%P %a %D %c %m %l %G %N';  # sysinfo-ok: GP-only survey, skipped on Tigris
    echo; echo '== all nodes'; sinfo -a -N -o '%N %P %c %m %G %f'; } >> "$d/sinfo.txt" 2>&1 || true  # sysinfo-ok: GP-only survey, skipped on Tigris
  { module -t avail 2>&1 | grep -iE 'mpi|cuda|pmix|ucx' ; } > "$d/modules.txt" 2>&1 || true  # sysinfo-ok: GP-only survey, skipped on Tigris
  srun --mpi=list > "$d/srun-mpi-list.txt" 2>&1 || true  # sysinfo-ok: GP-only survey, skipped on Tigris
  srun --nodes="${#NODES[@]}" --ntasks="${#NODES[@]}" --ntasks-per-node=1 --cpus-per-task=1 --mem=4G \
    bash -c 'h=${SLURMD_NODENAME:-local}; h=${h%%.*}; { echo "== $h"; lscpu; echo; free -g; echo; df -h "$0" /tmp "$1"; echo;  # sysinfo-ok: GP-only survey, skipped on Tigris
             numactl -H 2>&1 | head -20; echo; ulimit -a; } > "$2/node-$h.txt" 2>&1;  # sysinfo-ok: GP-only survey, skipped on Tigris
             "$3" "$4/probe.py" env --json "$2/env-$h.json" > /dev/null 2>&1 || echo "env probe failed on $h"' \
    "$LOCAL_SCRATCH" "$OUT" "$d" "$PY" "$KIT"
  $PY -c "import mpi4py, sys; print('mpi4py', mpi4py.__version__, mpi4py.__file__)" > "$d/mpi4py-import.txt" 2>&1 \
    && MPI4PY=yes || MPI4PY=no
  echo "env probe: mpi4py import $MPI4PY; files in $d"
}

# Lagoon read test: writer = first node; 1 cold reader (second node); then every node reads a
# second fresh file at the same time. Local disk: write + read on each node. GB=0 skips.
io_probe() {
  local gb=${1:-20} d="$OUT/probe" io="$OUT/io" n=${#NODES[@]}
  [ "$gb" != 0 ] || return 0
  mkdir -p "$io"
  local one="--nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G"
  srun $one -w "${NODES[0]}" $PY "$KIT/probe.py" write --path "$io/a.bin" --gb "$gb" --json "$d/io-write-a.json"
  srun $one -w "${NODES[0]}" $PY "$KIT/probe.py" write --path "$io/b.bin" --gb "$gb" --json "$d/io-write-b.json"
  local cold=${NODES[$((n > 1 ? 1 : 0))]}
  srun $one -w "$cold" $PY "$KIT/probe.py" read --path "$io/a.bin" --test lagoon-1-reader \
    --role "$([ "$n" -gt 1 ] && echo cold || echo writer-node)" --json "$d/io-lagoon-1-reader.json"
  srun $one -w "$cold" $PY "$KIT/probe.py" read --path "$io/a.bin" --test lagoon-reread \
    --role warm --json "$d/io-lagoon-reread.json"
  srun --nodes="$n" --ntasks="$n" --ntasks-per-node=1 --cpus-per-task=1 --mem=8G \
    bash -c 'h=${SLURMD_NODENAME:-local}; h=${h%%.*}; r=reader; [ "$h" = "$3" ] && r=writer-node;
             "$0" "$1/probe.py" read --path "$2/b.bin" --test lagoon-'"$n"'-concurrent --role "$r" \
               --json "$4/io-lagoon-concurrent-$h.json"' "$PY" "$KIT" "$io" "${NODES[0]%%.*}" "$d"
  rm -f "$io/a.bin" "$io/b.bin"
  local lgb; lgb=$(awk -v g="$gb" 'BEGIN { print (g < 10 ? g : 10) }')
  srun --nodes="$n" --ntasks="$n" --ntasks-per-node=1 --cpus-per-task=1 --mem=8G \
    bash -c 'h=${SLURMD_NODENAME:-local}; h=${h%%.*}; d="$3/awesom-so-$SLURM_JOB_ID-$h"; f="$d/io.bin";
             mkdir "$d" || exit 1; trap '"'"'rm -rf "$d"'"'"' EXIT;
             "$0" "$1/probe.py" write --path "$f" --gb "$2" --json "$4/io-local-write-$h.json" &&
             "$0" "$1/probe.py" read --path "$f" --test local-disk --role after-fadvise --json "$4/io-local-read-$h.json"' \
      "$PY" "$KIT" "$lgb" "$LOCAL_SCRATCH" "$d" || echo "local disk test failed"
}

# Find a launcher that starts NRANKS ranks over NNODES nodes and reports the right size.
pick_launcher() {
  local nn=$1 nr=$2 cand out ok
  LAUNCH=${LAUNCH:-auto}
  MPIRUN=${MPIRUN:-$(command -v mpirun || find "$MPI_SITE" -path '*bin/mpirun' -type f 2>/dev/null | head -1)}
  if [ "$LAUNCH" = auto ]; then cand="srun-pmix srun-pmix_v4 srun-pmi2 mpirun"; else cand=$LAUNCH; fi
  for l in $cand; do
    out="$OUT/probe/hello-$l-${nn}x${nr}.txt"
    case $l in
      srun-*) timeout 300 srun --mpi="${l#srun-}" --nodes="$nn" --ntasks="$nr" --cpus-per-task=1 --mem=4G \
                $PY -m mpi4py.bench helloworld > "$out" 2>&1 || true;;
      mpirun) [ -n "$MPIRUN" ] || continue
              timeout 300 "$MPIRUN" -np "$nr" --host "$(hostlist "$nn" $((nr / nn)))" \
                -x PYTHONPATH -x PATH -x LD_LIBRARY_PATH ${MPI4PY_LIBMPI:+-x MPI4PY_LIBMPI} \
                $PY -m mpi4py.bench helloworld > "$out" 2>&1 || true;;
    esac
    ok=$(grep -c "of $nr on" "$out" 2>/dev/null || true)
    if [ "$ok" = "$nr" ]; then LAUNCH=$l; echo "$l" > "$OUT/launcher.txt"; echo "launcher: $l"; return 0; fi
  done
  echo "no MPI launcher started $nr ranks on $nn nodes (see $OUT/probe/hello-*)" >&2
  return 1
}

hostlist() {  # first $1 nodes, $2 slots each: h1:2,h2:2
  local i s=""
  for ((i = 0; i < $1; i++)); do s="$s${s:+,}${NODES[$i]}:$2"; done
  echo "$s"
}

# launch_cmd NNODES RANKS_PER_NODE CPUS_PER_RANK BIND: set the array CMD to the multi-rank
# launch (environment + launcher + flags) that run_one and affinity_preflight both use.
# srun: Slurm binds each rank to its --cpus-per-task cores (--cpu-bind=verbose,cores, ranks
# packed block:block, so 2 ranks x 16 on a node land one per socket).
# NO_AFFINITY: GP nodes have Intel Omni-Path (hfi1_0). In MPI_Init, Open MPI's btl ofi opens
# libfabric's psmx2 provider, and libpsm2's hfi_userinit pins a process whose mask is the
# whole node to one CPU (jobs 8413, 8444: 32 -> 1). Half-node masks (2 ranks x 16) are left
# alone. HFI_NO_CPUAFFINITY=1 turns this off; the PSM3/IPATH names cover the other PSM
# libraries. Evidence: job 8448 (bind_probe.sbatch), whose strace names libpsm2 as the
# caller of sched_setaffinity(0, [0]), and whose cases with Open MPI's binding policy set
# lose the CPU all the same, so that policy is irrelevant (affinity-round2.md).
# --mpi=pmi2 and pml ucx abort in MPI_Init on GP. The 1-rank path of run_one (plain srun, no
# MPI) is not affected. The restore in scale_out.py and the pre-flight stay as the guard.
# BIND (socket or none) only changes mpirun's --map-by. mpirun binds each rank to
# CPUS_PER_RANK cores itself (--map-by ...:pe=N --bind-to core).
NO_AFFINITY=(HFI_NO_CPUAFFINITY=1 PSM3_NO_CPUAFFINITY=1 IPATH_NO_CPUAFFINITY=1)
launch_cmd() {
  local nn=$1 rpn=$2 cpus=$3 bind=$4 nr=$(($1 * $2)) mb="ppr:$2:node:pe=$3"
  [ "$bind" = socket ] && mb="ppr:1:package:pe=$cpus"
  case $LAUNCH in
    srun-*) CMD=(env NUMBA_NUM_THREADS="$cpus" "${NO_AFFINITY[@]}"
                 srun --mpi="${LAUNCH#srun-}" --nodes="$nn" --ntasks="$nr" --ntasks-per-node="$rpn"
                 --cpus-per-task="$cpus" --distribution=block:block --cpu-bind=verbose,cores);;
    mpirun) CMD=(env NUMBA_NUM_THREADS="$cpus" "${NO_AFFINITY[@]}"
                 "$MPIRUN" -np "$nr" --host "$(hostlist "$nn" "$rpn")" --map-by "$mb" --bind-to core
                 --report-bindings -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NUMBA_NUM_THREADS
                 -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x SCALE_CACHE_ROOT
                 -x HDF5_USE_FILE_LOCKING -x PYTHONUNBUFFERED -x PYTHONDONTWRITEBYTECODE -x TMPDIR
                 -x MPLCONFIGDIR -x HFI_NO_CPUAFFINITY -x PSM3_NO_CPUAFFINITY -x IPATH_NO_CPUAFFINITY
                 ${MPI4PY_LIBMPI:+-x MPI4PY_LIBMPI});;
    *) echo "launch_cmd: no launcher (LAUNCH=$LAUNCH)" >&2; return 2;;
  esac
}

# run_one DIR NNODES RANKS_PER_NODE CPUS_PER_RANK BIND -- scale_out.py args
# One rank: plain srun, no MPI. More: launch_cmd with the launcher from pick_launcher.
run_one() {
  local dir=$1 nn=$2 rpn=$3 cpus=$4 bind=$5; shift 5
  [ "$1" = -- ] && shift
  local nr=$((nn * rpn)) log="$OUT/logs/$(basename "$dir").log"
  mkdir -p "$OUT/logs" "$(dirname "$dir")"
  echo "$(date -u +%FT%TZ) run $(basename "$dir"): $nn node(s) x $rpn rank(s) x $cpus threads $*"
  if [ "$nr" -eq 1 ]; then
    NUMBA_NUM_THREADS=$cpus srun --nodes=1 --ntasks=1 --cpus-per-task="$cpus" \
      $PY "$KIT/scale_out.py" --out "$dir" --threads "$cpus" "$@" > "$log" 2>&1
    return
  fi
  launch_cmd "$nn" "$rpn" "$cpus" "$bind" || return 2
  "${CMD[@]}" $PY -m mpi4py "$KIT/scale_out.py" --mpi --out "$dir" --threads "$cpus" "$@" > "$log" 2>&1
}

# affinity_verdict FILE NRANKS CPUS: read the AFFINITY lines of a pre-flight; print one line
# per problem (nothing if every rank is fine). A rank is fine if it holds >= CPUS CPUs after
# MPI_Init, or after the restore when the restore fired (restored=<CPUs after the restore>;
# restored=0 or absent: the set did not shrink). The last line, "RESTORED n", counts the
# ranks where the restore fired.
affinity_verdict() {
  awk -v nr="$2" -v cpus="$3" '
    /^AFFINITY / { n++; delete v; for (i = 2; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
                   got = v["after_init"] + 0
                   if (v["restored"] + 0 > 0) { got = v["restored"] + 0; fired++ }
                   if (got < cpus) printf "rank %s on %s: %d CPUs after MPI_Init%s (%s before), needs %d\n",
                                          v["rank"], v["host"], got, (v["restored"] + 0 > 0 ? " and the restore" : ""),
                                          v["before_init"], cpus }
    END { if (n != nr) printf "%d of %d ranks reported\n", n, nr; printf "RESTORED %d\n", fired + 0 }' "$1"
}

# affinity_report CASE FILE NRANKS CPUS: the pre-flight's verdict on FILE. Fails (stderr) if a
# rank is missing or short even after the restore. With NO_AFFINITY in launch_cmd the restore
# should never fire; if it did, print a WARNING line (the run goes on) so that a change in
# PSM or the MPI stack that pins ranks again shows up in the job log.
affinity_report() {
  local case=$1 f=$2 nr=$3 cpus=$4 verdict bad fired
  verdict=$(affinity_verdict "$f" "$nr" "$cpus")
  fired=$(echo "$verdict" | sed -n 's/^RESTORED //p')
  bad=$(echo "$verdict" | grep -v '^RESTORED ' || true)
  if [ -n "$bad" ]; then
    { echo "affinity pre-flight $case FAILED ($f):"; echo "$bad"; } >&2
    return 1
  fi
  echo "affinity pre-flight $case: $nr rank(s), each >= $cpus CPUs after MPI_Init"
  if [ "${fired:-0}" -gt 0 ]; then
    echo "WARNING: affinity pre-flight $case: MPI_Init shrank the CPU set on $fired rank(s) despite" \
         "${NO_AFFINITY[*]}; the restore gave the CPUs back ($f)"
  fi
}

# affinity_preflight CASE NNODES RANKS_PER_NODE CPUS_PER_RANK BIND: start the launch of
# run_one (same CMD) with a few lines of python. Each rank writes its host and CPU count
# before and after MPI_Init, restores the saved CPU set if MPI_Init shrank it (as
# scale_out.py does), and writes restored=<CPUs after the restore> (0: not needed) to
# $OUT/probe/affinity-CASE.txt. affinity_report then fails, in seconds, if a rank is missing
# or holds fewer than CPUS_PER_RANK CPUs even after the restore (scale_out.py would run 10x
# slow), and warns if the restore had to fire.
affinity_preflight() {
  local case=$1 nn=$2 rpn=$3 cpus=$4 bind=$5 f="$OUT/probe/affinity-$1.txt"
  launch_cmd "$nn" "$rpn" "$cpus" "$bind" || return 2
  timeout 120 "${CMD[@]}" $PY -c '
import os
saved = os.sched_getaffinity(0)
from mpi4py import MPI
c = MPI.COMM_WORLD
after = len(os.sched_getaffinity(0))
restored = 0
if after < len(saved):
    os.sched_setaffinity(0, saved)
    restored = len(os.sched_getaffinity(0))
host = os.environ.get("SLURMD_NODENAME") or "local"
print(f"AFFINITY rank={c.Get_rank()} size={c.Get_size()} host={host} "
      f"before_init={len(saved)} after_init={after} restored={restored}", flush=True)
' > "$f" 2>&1 || true
  affinity_report "$case" "$f" $((nn * rpn)) "$cpus"
}

r_min_for() { case $1 in 160x100) echo 8;; *x*) [ "${1%%x*}" -ge 91 ] && echo 8 || echo 5;; esac; }

# Write mpi4py's version, the MPI extension file, MPI4PY_LIBMPI and the MPI library version
# to $1. Fails if `from mpi4py import MPI` fails, if the binary wheel picked its MPICH
# flavour, or (when $2 is given) if the library is not Open MPI v$2.
mpi_check() {
  local f=$1 ver=$2
  $PY - > "$f" 2>&1 <<'PY' || return 1
import os, mpi4py
from mpi4py import MPI
print("mpi4py", mpi4py.__version__, mpi4py.__file__)
print("MPI.__file__", MPI.__file__)
print("MPI4PY_LIBMPI", os.environ.get("MPI4PY_LIBMPI", "(unset)"))
print(MPI.Get_library_version()[:200].strip())
if ".mpich." in os.path.basename(MPI.__file__):
    raise SystemExit("mpi4py picked the MPICH flavour")
PY
  [ -z "$ver" ] || grep -q "Open MPI v$ver" "$f"
}

# Install mpi4py into MPI_SITE if the venv cannot import it. Runs once, on the batch node,
# before any srun, so the nodes of a multi-node job never race on MPI_SITE.
# GP compute nodes have no internet and no C headers (probes 8401, 8407), so the default
# (MPI_BUILD=wheel) installs the binary PyPI wheel from $KIT/wheels. That wheel dlopens the
# MPI library at import: the path to the module's libmpi.so.40 goes into MPI_SITE/LIBMPI,
# which kit_modules exports as MPI4PY_LIBMPI. MPI_BUILD=source compiles mpi4py from the sdist
# in $KIT/wheels against the module's mpicc (for sites with a compiler). Either way the
# module goes into MPI_SITE/MODULE and MPI.Get_library_version() must name its version.
# Online fallback, only if $KIT/wheels is absent: the same install from PyPI, then the PyPI
# mpi4py + openmpi wheels. Writes probe/mpi-install.txt and probe/mpi4py-version.txt.
mpi_install() {
  local log="$OUT/probe/mpi-install.txt" vfile="$OUT/probe/mpi4py-version.txt" mod ver
  local wheels="$KIT/wheels" build=${MPI_BUILD:-wheel} rc=0
  case $build in wheel|source) ;; *) echo "MPI_BUILD must be wheel or source, not $build" | tee "$log"; return 2;; esac
  if on_tigris; then echo "mpi_install is GP-only (module survey); refusing on Tigris" | tee "$log"; return 2; fi
  mod=$(module -t avail 2>&1 | sed 's/(.*)//; s/ *$//' | grep -iE '^(openmpi|mpi/openmpi)[^ ]*$' | sort -V | tail -1)  # sysinfo-ok: GP-only, refuses on Tigris
  ver=${mod##*/}
  # Skip only if MPI itself loads and names the module's Open MPI version (kit_init has
  # already loaded MPI_SITE/MODULE and exported MPI4PY_LIBMPI); a bare `import mpi4py`
  # would pass on a half-installed site.
  if mpi_check "$vfile" "$ver"; then
    { echo "mpi4py already works with ${mod:-the default MPI}; not reinstalling"; cat "$vfile"; } | tee "$log"
    return 0
  fi
  : > "$log"
  if [ -d "$MPI_SITE" ] && [ -n "$(ls -A "$MPI_SITE")" ]; then
    mv "$MPI_SITE" "$MPI_SITE.stale-${SLURM_JOB_ID:-local}-$(date +%s)"
    echo "moved a non-working mpi-site to $MPI_SITE.stale-*; reinstalling" >> "$log"
  fi
  mkdir -p "$MPI_SITE"
  # Subshell: its exit status is the result; the module load here does not leak (kit_modules loads it).
  (
    echo "openmpi module: ${mod:-none}; MPI_BUILD=$build"
    if [ -n "$mod" ] && module load "$mod" && command -v mpicc; then
      lib=$(dirname "$(dirname "$(command -v mpicc)")")/lib/libmpi.so.40
      [ -f "$lib" ] || [ ! -f "${lib%/*}/libmpi.40.dylib" ] || lib=${lib%/*}/libmpi.40.dylib  # macOS name
      [ -f "$lib" ] || { echo "no MPI library at $lib"; echo "mpi4py install FAILED"; exit 1; }
      export MPI4PY_LIBMPI=$lib
      if [ -d "$wheels" ]; then
        src=(--no-index --find-links "$wheels"); echo "offline install from $wheels:"; ls -l "$wheels"
      else
        src=(); echo "FALLBACK (online, needs network from the node): no $wheels, installing from PyPI"
      fi
      if [ "$build" = wheel ]; then
        $PY -m pip install --no-cache-dir "${src[@]}" --target "$MPI_SITE" --only-binary mpi4py mpi4py
      else
        MPICC=$(command -v mpicc) $PY -m pip install --no-cache-dir "${src[@]}" --target "$MPI_SITE" --no-binary mpi4py mpi4py
      fi && echo "$mod" > "$MPI_SITE/MODULE" && echo "$lib" > "$MPI_SITE/LIBMPI" \
        && echo "installed mpi4py ($build) for $mod, MPI4PY_LIBMPI=$lib"
    fi
    if ! $PY -c 'import mpi4py' > /dev/null 2>&1 && [ ! -d "$wheels" ]; then
      echo "FALLBACK (online, needs network from the node): PyPI mpi4py + openmpi wheels"
      $PY -m pip install --no-cache-dir --target "$MPI_SITE" mpi4py openmpi && : > "$MPI_SITE/MODULE"
    fi
    [ -s "$MPI_SITE/MODULE" ] || ver=""
    mpi_check "$vfile" "$ver" || rc=1
    cat "$vfile"
    [ "$rc" -eq 0 ] || echo "check failed: MPI must import, not as the MPICH flavour, and name Open MPI v$ver"
    [ "$rc" -eq 0 ] && echo "mpi4py OK" || echo "mpi4py install FAILED"
    exit "$rc"
  ) >> "$log" 2>&1 < /dev/null
  rc=$?
  if [ "$rc" -ne 0 ] && [ -d "$MPI_SITE" ]; then
    # Keep a failed or mislinked build out of PYTHONPATH; the next probe starts clean.
    mv "$MPI_SITE" "$MPI_SITE.failed-${SLURM_JOB_ID:-local}-$(date +%s)" \
      && echo "moved the failed install to $MPI_SITE.failed-*" >> "$log"
  fi
  cat "$log"
  return $rc
}

# try_run: run_one that records a failure in FAILS instead of ending the job.
FAILS=0
try_run() {
  local rc=0
  run_one "$@" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "run $(basename "$1") failed (exit $rc), log $OUT/logs/$(basename "$1").log" >&2
    FAILS=$((FAILS + 1))
  fi
}

# Write the CPU list of every rank of a run and the NUMA node(s) it covers.
check_pinning() {
  $PY - "$1" "$OUT/probe" <<'PY' | tee "$1/pinning.txt"
import json, sys
from pathlib import Path
run, probe = Path(sys.argv[1]), Path(sys.argv[2])
def cpus(s):
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out
numa = {}
for p in probe.glob("env-*.json"):
    numa[json.loads(p.read_text())["host"].split(".")[0]] = {k: cpus(v) for k, v in
                                                            json.loads(p.read_text()).get("numa_nodes", {}).items()}
for p in sorted(run.glob("rank*.json")):
    r = json.loads(p.read_text())
    aff = set(r.get("affinity") or [])
    nodes = numa.get(r["hostname"].split(".")[0], {})
    on = sorted(k for k, v in nodes.items() if aff & v)
    print(f"rank {r['rank']} on {r['hostname']}: {len(aff)} cpus, NUMA {on}")
PY
}

# Summary of every run of this job (plus any extra roots); never fails the job.
summarize_job() {
  $PY "$KIT/summarize.py" --out "$OUT/summary" "$OUT" "$@" || echo "summarize.py failed" >&2
}

# End of a sweep: completed=yes only if every run succeeded.
end_sweep() {
  if [ "$FAILS" -eq 0 ]; then completed=yes; else echo "$FAILS run(s) failed" >&2; exit 1; fi
}
