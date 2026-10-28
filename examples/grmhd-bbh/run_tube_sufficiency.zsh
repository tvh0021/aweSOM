#!/bin/zsh
set -euo pipefail
set -o noclobber
it=${1:?Pass iteration}
repo=/Users/tvh0021/git_repos/aweSOM
python=/Users/tvh0021/git_repos/aweSOM-research/.venv/bin/python
base=/Volumes/ExternalNVMe/awesom-results
if [[ $it == 405504 ]]; then
 data=$base/cluster/HHR_flux_rl5-8
 raw=$base/cluster/HHR_flux/raw_planes_it405504.npz
 run=$base/2026-10-28/flux-tubes-hhr/rl5-8
else
 data=$base/cluster/HHR_flux_it${it}_rl5-8
 raw=$data/raw_planes_it${it}.npz
 run=$base/2026-10-28/flux-tubes-hhr/it${it}-rl5-8
fi
model=$base/2026-10-28/hhr-flux-405504/models/som_model_HHR_flux_405504_91-64
[[ -f $data/grid_HHR_flux_${it}.h5 && -f $raw && ! -e $run && ! -e ${run}.log ]]
export PYTHONPATH=$repo/src NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
nice -n 10 "$python" "$repo/examples/grmhd-bbh/tube_sufficiency_hhr.py" --grid "$data/grid_HHR_flux_${it}.h5" --raw "$raw" --model005 "${model}_mc0.05.h5" --model01 "${model}_mc0.1.h5" --out-dir "$run" > "${run}.log" 2>&1
nice -n 10 "$python" "$repo/examples/grmhd-bbh/tube_sufficiency_report.py" "$run" >> "${run}.log" 2>&1
