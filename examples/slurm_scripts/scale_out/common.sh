# Shared functions of the scale-out sbatch files. Source it, then call kit_init NAME.
# Paths: KIT (this directory, default $SLURM_SUBMIT_DIR), VENV (python with numpy/numba/h5py),
# OUT (fresh output dir, default $KIT/runs/NAME-JOBID), OUT_ROOTS (allowed parents).
# MPI: MPI_SITE (pip --target dir holding mpi4py, default $KIT/mpi-site; PYTHONPATH),
# LAUNCH (auto | srun-pmix | srun-pmi2 | srun-pmix_v4 | mpirun), MPIRUN (path, mpirun mode).

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
}

kit_init() {
  NAME=$1
  KIT=${KIT:-$SLURM_SUBMIT_DIR}
  OUT=${OUT:-$KIT/runs/$NAME-${SLURM_JOB_ID:-local}}
  OUT_ROOTS=${OUT_ROOTS:-$HOME:/lagoon/trungha}
  VENV=${VENV:-/lagoon/trungha/bbh-output/for-awesom-2026-10-28/check7-venv}
  MPI_SITE=${MPI_SITE:-$KIT/mpi-site}
  LOCAL_SCRATCH=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
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
  printf 'STARTED\t%s\t%s\n' "$(date -u +%FT%TZ)" "$(hostname)" > "$STATUS_DIR/start.tsv"
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
  { sinfo -Nel -n "$SLURM_JOB_NODELIST"; echo; sinfo -p "$SLURM_JOB_PARTITION" -o '%P %D %c %m %l %N'; } \
    > "$d/sinfo.txt" 2>&1 || true
  { module -t avail 2>&1 | grep -iE 'mpi|cuda|pmix|ucx' ; } > "$d/modules.txt" 2>&1 || true
  srun --mpi=list > "$d/srun-mpi-list.txt" 2>&1 || true
  srun --nodes="${#NODES[@]}" --ntasks="${#NODES[@]}" --ntasks-per-node=1 --cpus-per-task=1 --mem=4G \
    bash -c 'h=$(hostname -s); { echo "== $h"; lscpu; echo; free -g; echo; df -h "$0" /tmp "$1"; echo;
             numactl -H 2>&1 | head -20; echo; ulimit -a; } > "$2/node-$h.txt" 2>&1;
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
    bash -c 'h=$(hostname -s); r=reader; [ "$h" = "$3" ] && r=writer-node;
             "$0" "$1/probe.py" read --path "$2/b.bin" --test lagoon-'"$n"'-concurrent --role "$r" \
               --json "$4/io-lagoon-concurrent-$h.json"' "$PY" "$KIT" "$io" "${NODES[0]%%.*}" "$d"
  rm -f "$io/a.bin" "$io/b.bin"
  local lgb; lgb=$(awk -v g="$gb" 'BEGIN { print (g < 10 ? g : 10) }')
  srun --nodes="$n" --ntasks="$n" --ntasks-per-node=1 --cpus-per-task=1 --mem=8G \
    bash -c 'h=$(hostname -s); f="$3/awesom-io-$SLURM_JOB_ID-$h.bin";
             "$0" "$1/probe.py" write --path "$f" --gb "$2" --json "$4/io-local-write-$h.json" &&
             "$0" "$1/probe.py" read --path "$f" --test local-disk --role after-fadvise --json "$4/io-local-read-$h.json";
             rm -f "$f"' "$PY" "$KIT" "$lgb" "$LOCAL_SCRATCH" "$d" || echo "local disk test failed"
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
                -x PYTHONPATH -x PATH -x LD_LIBRARY_PATH $PY -m mpi4py.bench helloworld > "$out" 2>&1 || true;;
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

# run_one DIR NNODES RANKS_PER_NODE CPUS_PER_RANK [socket] -- scale_out.py args
# One rank: plain srun, no MPI. More: the launcher from pick_launcher, python -m mpi4py.
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
  local sb="" mb="ppr:$rpn:node:pe=$cpus"
  if [ "$bind" = socket ]; then sb="--distribution=block:block --cpu-bind=cores"; mb="ppr:1:package:pe=$cpus"; fi
  case $LAUNCH in
    srun-*) NUMBA_NUM_THREADS=$cpus srun --mpi="${LAUNCH#srun-}" --nodes="$nn" --ntasks="$nr" \
              --ntasks-per-node="$rpn" --cpus-per-task="$cpus" $sb \
              $PY -m mpi4py "$KIT/scale_out.py" --mpi --out "$dir" --threads "$cpus" "$@" > "$log" 2>&1;;
    mpirun) NUMBA_NUM_THREADS=$cpus "$MPIRUN" -np "$nr" --host "$(hostlist "$nn" "$rpn")" --map-by "$mb" \
              --bind-to core -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NUMBA_NUM_THREADS -x OMP_NUM_THREADS \
              -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x SCALE_CACHE_ROOT -x HDF5_USE_FILE_LOCKING \
              -x PYTHONUNBUFFERED -x PYTHONDONTWRITEBYTECODE -x TMPDIR -x MPLCONFIGDIR \
              $PY -m mpi4py "$KIT/scale_out.py" --mpi --out "$dir" --threads "$cpus" "$@" > "$log" 2>&1;;
    *) echo "run_one: no launcher (LAUNCH=$LAUNCH)" >&2; return 2;;
  esac
}

r_min_for() { case $1 in 160x100) echo 8;; *x*) [ "${1%%x*}" -ge 91 ] && echo 8 || echo 5;; esac; }

# Install mpi4py into MPI_SITE if the venv cannot import it. First choice: build against
# the newest openmpi module (records the module in MPI_SITE/MODULE). Fallback: PyPI
# wheels of mpi4py + openmpi (needs network from the node). Writes probe/mpi-install.txt.
mpi_install() {
  local log="$OUT/probe/mpi-install.txt" mod
  if $PY -c 'import mpi4py' > /dev/null 2>&1; then echo "mpi4py already importable" | tee "$log"; return 0; fi
  mkdir -p "$MPI_SITE"
  mod=$(module -t avail 2>&1 | sed 's/(.*)//; s/ *$//' | grep -iE '^(openmpi|mpi/openmpi)[^ ]*$' | sort -V | tail -1)
  {
    echo "openmpi module: ${mod:-none}"
    if [ -n "$mod" ] && module load "$mod" && command -v mpicc; then
      MPICC=$(command -v mpicc) $PY -m pip install --no-cache-dir --target "$MPI_SITE" --no-binary mpi4py mpi4py \
        && echo "$mod" > "$MPI_SITE/MODULE" && echo "built mpi4py against $mod"
    fi
    if ! $PY -c 'import mpi4py' > /dev/null 2>&1; then
      echo "falling back to PyPI mpi4py + openmpi wheels"
      $PY -m pip install --no-cache-dir --target "$MPI_SITE" mpi4py openmpi && : > "$MPI_SITE/MODULE"
    fi
    $PY -c 'import mpi4py; from mpi4py import MPI; print(mpi4py.__file__, MPI.Get_library_version()[:200])'
  } > "$log" 2>&1
  local rc=$?
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
