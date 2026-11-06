#!/bin/bash
#SBATCH --job-name=so-sweep1
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thccrg@rit.edu
# Where does a rank lose its CPUs in MPI_Init (jobs 8413, 8444)? One node, one srun step per
# case, each with a 60 s timeout, raw behaviour (no restore except case H). Each case writes
# probe/bind-<case>.txt (command, full stdout+stderr, exit code); the job log gets one line
# per case: before / after loading libmpi / after MPI_Init. --job-name=so-sweep1 so that
# --dependency=singleton serializes it with the sweeps.
# The 8413/8444 condition is a task mask that covers the whole node. On lm-n0001 (debug, 48
# CPUs) a 32-CPU task covers 2/3 of it, so the job prints a warning; to reproduce the
# condition submit with --partition=gp-default (32 CPUs = one whole gp node).
# Cases: A launch_cmd as run_one uses it (pmix, --cpu-bind=verbose,cores, MCA policy none);
# B A + ess/hwloc/pmix/pml/mtl verbose; C --mpi=pmi2; D --cpu-bind=verbose,none;
# E A + OMPI_MCA_orte_bound_at_launch=1; F A with policy core; G 2 ranks x half the CPUs;
# H A + the restore of scale_out.py; I env and mask of the task without MPI;
# J A + HFI/PSM3/IPATH_NO_CPUAFFINITY=1; K A with pml ob1 + btl self,vader,tcp (no fabric
# library); L A with pml ucx + btl self,vader; M A under strace (sched_setaffinity callers).
set -o pipefail  # not -e: a failing case must not stop the others
. "${KIT:-$SLURM_SUBMIT_DIR}/common.sh"
kit_init bindprobe
C=$SLURM_CPUS_PER_TASK
P="$OUT/probe"
BP="$KIT/bind_probe.py"
ONLINE=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc --all)
if [ "$C" -lt "$ONLINE" ]; then
  echo "WARNING: $(hostname -s) has $ONLINE CPUs and each 1-rank case gets $C, so the task mask is not the whole node" \
       "(the 8413/8444 condition); resubmit with --partition=gp-default to reproduce it"
fi

# Fabric and MPI inventory: which transports MPI_Init can load on this node.
{ echo "== host $(hostname) online=$ONLINE cpus/task=$C"; srun --version
  echo "== batch shell"; grep Cpus_allowed_list /proc/self/status
  echo "== /sys/class/infiniband"; ls -l /sys/class/infiniband/ 2>&1
  echo "== /dev/hfi1*"; ls -l /dev/hfi1* 2>&1
  echo "== ompi_info components (pml mtl btl ess pmix hwloc osc)"
  ompi_info 2>&1 | grep -E 'MCA (pml|mtl|btl|ess|pmix|hwloc|osc):'
  echo "== ompi_info build"; ompi_info 2>&1 | grep -iE 'Configure command|psm|ofi|ucx' | head -20
  echo "== fi_info -l"; command -v fi_info >/dev/null && fi_info -l 2>&1 | head -40 || echo "no fi_info"
  echo "== ucx_info -v"; command -v ucx_info >/dev/null && ucx_info -v 2>&1 || echo "no ucx_info"
  echo "== fabric env"; env | grep -E '^(OMPI_|PMIX_|UCX_|HFI_|PSM2?_|PSM3_|IPATH_|FI_)' | sort
  echo "== modules"; module list 2>&1
} > "$P/fabric.txt" 2>&1

# run_case NAME DESCRIPTION -- COMMAND...: one srun step, never fails the job.
run_case() {
  local name=$1 desc=$2; shift 2; [ "$1" = -- ] && shift
  local f="$P/bind-$name.txt" rc=0 s
  { echo "== case $name: $desc"; printf '== command:'; printf ' %q' "$@"; echo; } > "$f"
  timeout -k 10 60 "$@" >> "$f" 2>&1 || rc=$?
  echo "== exit $rc" >> "$f"
  s=$(awk '/^BIND / { delete v; for (i = 2; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
                     printf "%sr%s before=%s load=%s init=%s%s cpus=%s threads=%s", sep, v["rank"], v["before"],
                            v["after_load"], v["after_init"], ("restored" in v ? " restored=" v["restored"] : ""),
                            ("cpus_after" in v ? v["cpus_after"] : v["cpus_before"]), v["threads"]; sep = "; " }' "$f")
  printf 'CASE %s rc=%s %s | %s\n' "$name" "$rc" "${s:-no BIND line}" "$desc"
}

# Edit CMD (from launch_cmd): add VAR=VAL just before srun (they override launch_cmd's), or
# replace one element.
cmd_env() {
  local out=() x put=no
  for x in "${CMD[@]}"; do
    if [ "$put" = no ] && [ "$x" = srun ]; then out+=("$@"); put=yes; fi
    out+=("$x")
  done
  CMD=("${out[@]}")
}
cmd_sub() {
  local i
  for i in "${!CMD[@]}"; do [ "${CMD[$i]}" = "$1" ] && CMD[$i]=$2; done
}
base() { LAUNCH=${1:-srun-pmix}; launch_cmd 1 1 "$C" none; }

base; run_case A "launch_cmd: pmix, cpu-bind cores, policy none" -- "${CMD[@]}" $PY "$BP" --case A

base; cmd_env OMPI_MCA_ess_base_verbose=10 OMPI_MCA_hwloc_base_verbose=10 OMPI_MCA_pmix_base_verbose=10 \
  OMPI_MCA_pml_base_verbose=10 OMPI_MCA_mtl_base_verbose=10
run_case B "A + ess/hwloc/pmix/pml/mtl verbose 10" -- "${CMD[@]}" $PY "$BP" --case B
grep -v '^== ' "$P/bind-B.txt" | grep -E 'Not bound at launch|externally bound|Binding not supported|binding|bound' | head -8 \
  | sed 's/^/  B: /'

base srun-pmi2; run_case C "A with --mpi=pmi2" -- "${CMD[@]}" $PY "$BP" --case C

base; cmd_sub --cpu-bind=verbose,cores --cpu-bind=verbose,none
run_case D "A with --cpu-bind=verbose,none" -- "${CMD[@]}" $PY "$BP" --case D

base; cmd_env OMPI_MCA_orte_bound_at_launch=1
run_case E "A + OMPI_MCA_orte_bound_at_launch=1" -- "${CMD[@]}" $PY "$BP" --case E

base; cmd_sub OMPI_MCA_hwloc_base_binding_policy=none OMPI_MCA_hwloc_base_binding_policy=core
run_case F "A with OMPI_MCA_hwloc_base_binding_policy=core" -- "${CMD[@]}" $PY "$BP" --case F

LAUNCH=srun-pmix; launch_cmd 1 2 $((C / 2)) none
run_case G "2 ranks x $((C / 2)) CPUs, one node (8409 shape)" -- "${CMD[@]}" $PY "$BP" --case G

base; run_case H "A + restore of the saved CPU set (the workaround)" -- "${CMD[@]}" $PY "$BP" --case H --restore

base; run_case I "A's srun, no MPI: env, mask, python" -- "${CMD[@]}" bash -c \
  'echo "policy_env_count=$(env | grep -c "^OMPI_MCA_hwloc_base_binding_policy=")"
   env | grep "^OMPI_MCA_hwloc_base_binding_policy="; grep Cpus_allowed_list /proc/self/status; taskset -p $$
   "$0" "$1" --case I --no-mpi' "$PY" "$BP"

base; cmd_env HFI_NO_CPUAFFINITY=1 PSM3_NO_CPUAFFINITY=1 IPATH_NO_CPUAFFINITY=1
run_case J "A + HFI/PSM3/IPATH_NO_CPUAFFINITY=1 (PSM affinity off)" -- "${CMD[@]}" $PY "$BP" --case J

base; cmd_env OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
run_case K "A with pml ob1, btl self,vader,tcp (no fabric library)" -- "${CMD[@]}" $PY "$BP" --case K

base; cmd_env OMPI_MCA_pml=ucx OMPI_MCA_btl=self,vader
run_case L "A with pml ucx, btl self,vader (UCX only)" -- "${CMD[@]}" $PY "$BP" --case L

if command -v strace >/dev/null; then
  K_FLAG=-k; strace -k -e trace=none true > /dev/null 2>&1 || K_FLAG=
  base; run_case M "A under strace -f $K_FLAG -e sched_setaffinity (trace in bind-M-strace.txt)" -- \
    "${CMD[@]}" strace -f $K_FLAG -e trace=sched_setaffinity -o "$P/bind-M-strace.txt" $PY "$BP" --case M
  grep -c sched_setaffinity "$P/bind-M-strace.txt" 2>/dev/null | sed 's/^/  M: sched_setaffinity calls: /'
else
  echo "CASE M skipped: no strace on $(hostname -s)"
fi

completed=yes
