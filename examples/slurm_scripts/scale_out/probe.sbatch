#!/bin/bash
#SBATCH --job-name=so-probe
#SBATCH --partition=debug
#SBATCH --nodes=1-2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thccrg@rit.edu
# Environment probe of the scale-out kit: sinfo, modules, lscpu, memory, disks, NUMA,
# mpi4py (installed into $KIT/mpi-site if missing, offline from the binary wheel in
# $KIT/wheels because compute nodes have no internet or compiler; MPI_BUILD=source compiles
# instead; make_kit.sh copies the wheels in, and README "Offline wheels" gives the pip
# download commands for a machine with internet), MPI launchers (1 and 2 ranks per
# node), and the lagoon read test (IO_TEST_GB, default 20; cold read on the second node
# when the job gets two). Submit this first; the other jobs use what it installs.
# debug has one node (lm-n0001); submit with --partition=gp-default --nodes=2 (README).
set -eo pipefail
. "${KIT:-$SLURM_SUBMIT_DIR}/common.sh"
kit_init probe
mpi_install || echo "mpi4py install failed; see $OUT/probe/mpi-install.txt" >&2
kit_modules
env_probe
nn=${#NODES[@]}
pick_launcher "$nn" "$nn" || true
LAUNCH=auto
pick_launcher "$nn" $((2 * nn)) || true
io_probe "${IO_TEST_GB:-20}"
summarize_job
completed=yes
