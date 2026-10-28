# Comparing current sheets across GRMHD BBH runs

This page covers the three-run comparison of Michi Chabanov's binary black hole
runs `LR`, `LR_flux` and `LR_flux_CBD_2` at one iteration (180224 by default).
The current sheets are found with a SOM trained on `LR_flux`. Their statistics
come from `aweSOM.sheet_stats`, and the scripts are in `examples/grmhd-bbh/`:

| script | what it does |
|---|---|
| `build_model.py` | freezes the trained `Lattice` pickle into a `SOMModel` file and picks the sheet clusters |
| `compare_runs.py` | predicts labels on each run's full grid, computes the sheet statistics, and writes the comparison table and figures |
| `check_labels.py` | checks the predicted labels of the training run against the labels saved at training time and the training-sample fractions |

All commands below use the research virtual environment with this checkout on
`PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/aweSOM/src
PY=/Users/tvh0021/git_repos/aweSOM-research/.venv/bin/python
EX=/path/to/aweSOM/examples/grmhd-bbh
OUT=/Volumes/ExternalNVMe/awesom-results/2026-10-28/sheet-stats
```

## Inputs

Each run needs a grid file written by `etkplot som-features --grid-out`. A grid
file holds every owned point of levels 5 to 7 (`xyz`, `level`, `features`,
`names`) and the horizon attributes (`horizon_centres`, `horizon_mass`,
`horizon_radius`, `iteration`, `time`). The cluster job `som_run.sbatch` writes
these files. After the rsync they are at:

```
/Volumes/ExternalNVMe/awesom-results/cluster/<TAG>/grid_<TAG>_<IT>.h5
/Volumes/ExternalNVMe/awesom-results/cluster/<TAG>/features_<TAG>_<IT>.h5   (training sample, 10^7 rows)
```

where `TAG` is `LR`, `LR_flux_CBD_2` or `LR_flux`, and `IT` is 180224 unless
that iteration was missing from a run. In that case all three runs have to be
resubmitted at the same `IT` (see `cluster/kit/README.md`). The local `LR_flux`
grid at 180224 is
`/Volumes/ExternalNVMe/awesom-results/2026-10-28/lr-flux-180224/grid_LR_flux_180224.h5`.

Before you start, check that each job's `.out` log ends with `etkplot job: done`
and that its `<TAG>/status/<jobid>-0/final.tsv` starts with `VALIDATED`.

## 1. Freeze the model (done once)

```bash
nice -n 10 $PY $EX/build_model.py \
  --lattice /Volumes/ExternalNVMe/etk-analysis/som_results/som_object.bbh_0003_180224-91-64-0.5-10000000-1u.pkl \
  --training-file /Volumes/ExternalNVMe/awesom-results/2026-10-28/lr-flux-180224/features_bbh_0003-180224-N10000000.h5 \
  --merge-cost 0.05 \
  --training-labels /Volumes/ExternalNVMe/etk-analysis/som_results/labels.bbh_0003_180224-91-64-0.5-10000000-1u-mc0.05.npy \
  -o $OUT/som_model_bbh_0003_180224_N1e7_mc0.05.h5
```

At merge cost 0.05 the lattice has 19 clusters. The rule
`log_chi_B>0.3,log_beta_inv<0.5` (each cluster's mean over the training rows)
selects clusters 3, 5, 7 and 16 as the sheet. These are stored in the model
metadata, and `compare_runs.py` reads them from there. Cluster 5
(`log_chi_B` 0.88, `log_beta_inv` -0.62) is the sheet core and holds 0.207% of
the training volume. It is the same set of nodes as cluster 9 at merge cost
0.1. At merge cost 0.2 no cluster passes the rule. Always state the merge cost
and cluster ids next to any sheet number.

The script prints `labels ... agree with ...: 1.000000` when re-assigning the
lattice reproduces the saved training labels. It also prints the largest
difference between the frozen scaler applied to the training features and
`lattice.data_array`, which should be about 2e-7.

### The scaler

`train_som.py` scales the features with `run_som.manual_scaling`. That function
computes the mean and standard deviation in float32 along axis 0, so they
differ from the float64 `scaler/mean` and `scaler/sigma` stored in the feature
file: the offsets differ by up to 0.015 and the scales by about 1%.

- `--scaler training-float32` (the default) recomputes the float32 statistics
  from the training file. This reproduces `data_array` to 2e-7 and gives the
  saved labels on all 10^7 training rows.
- `--scaler file` uses the stored float64 values. Only 83% of best-matching
  nodes stay the same, and 1.1% of the labels change at merge cost 0.05 (0.6%
  at 0.1, 0.1% at 0.2).

Use the file scaler only as a sensitivity check.

## 2. Run the comparison

To run all three:

```bash
nice -n 10 $PY $EX/compare_runs.py --model $OUT/som_model_bbh_0003_180224_N1e7_mc0.05.h5 --out $OUT \
  --run LR_flux=/Volumes/ExternalNVMe/awesom-results/2026-10-28/lr-flux-180224/grid_LR_flux_180224.h5 \
  --run LR=/Volumes/ExternalNVMe/awesom-results/cluster/LR/grid_LR_180224.h5 \
  --run LR_flux_CBD_2=/Volumes/ExternalNVMe/awesom-results/cluster/LR_flux_CBD_2/grid_LR_flux_CBD_2_180224.h5 \
  --reuse 2>&1 | tee $OUT/compare_runs_3runs.log
```

`--reuse` skips the prediction and statistics of any run whose outputs already
exist (here `LR_flux`), so only the new runs are processed. The table and
figures are always rebuilt from the per-run files. Runs appear in the order
given. Run colours are assigned in that order, so keep the order the same
between calls.

Per run (about 68 million points for `LR_flux`), the cost is:

- prediction: about 6 microseconds per point on a loaded machine;
- statistics: about 3 minutes.

Peak memory is 6.5 GB. Rows are predicted and read in chunks
(`--chunk-rows`, default 4,000,000), each level is labelled on its own lattice
box, and the distance transform runs on slabs of 128 cells along x
(`SheetConfig.edt_block`).

Outputs in `$OUT`:

| path | content |
|---|---|
| `labels_<NAME>_<IT>.npy` | uint8 label per grid row (255 = non-finite features) |
| `runs/<NAME>/<NAME>_sheet_stats.json` | all statistics of one run; `run.label_fractions` has the point and volume fraction of every label, per level and in total, and `run.predict` the prediction timing |
| `runs/<NAME>/<NAME>_components.csv` | one row per connected sheet component, largest first |
| `runs/<NAME>/<NAME>_planes.npz` | z=0 slices of the sheet mask and of `log_beta_inv` per level |
| `comparison.md`, `comparison.json` | the comparison table |
| `figures/*.png` | component volume, thickness, features inside and outside the sheets, radial profiles, z=0 slices |

To pass labels that already exist, use `NAME=GRID.h5:LABELS.npy`.

To get statistics for a subset of the sheet clusters, use a separate output
directory so that `comparison.md` is not overwritten. For example, for the core
cluster only:

```bash
nice -n 10 $PY $EX/compare_runs.py --out $OUT/core --sheet-clusters 5 \
  --run LR_flux=<grid>:$OUT/labels_LR_flux_180224.npy \
  --run LR=<grid>:$OUT/labels_LR_180224.npy \
  --run LR_flux_CBD_2=<grid>:$OUT/labels_LR_flux_CBD_2_180224.npy --reuse
```

## 3. Check the labels of the training run

```bash
nice -n 10 $PY $EX/check_labels.py \
  --grid /Volumes/ExternalNVMe/awesom-results/2026-10-28/lr-flux-180224/grid_LR_flux_180224.h5 \
  --labels $OUT/labels_LR_flux_180224.npy \
  --training-file /Volumes/ExternalNVMe/awesom-results/2026-10-28/lr-flux-180224/features_bbh_0003-180224-N10000000.h5 \
  --training-labels /Volumes/ExternalNVMe/etk-analysis/som_results/labels.bbh_0003_180224-91-64-0.5-10000000-1u-mc0.05.npy \
  --sheet-clusters 3,5,7,16 -o $OUT/label_check.json
```

The training rows are a subset of the grid points. At each training row the
grid label must equal the saved training label, and the grid volume fractions
must match the training fractions weighted by `weight_volume`. The other two
runs have no labels saved at training time, so this check applies to `LR_flux`
only.

## Caveat: one SOM applied to three runs

The SOM, its merge cost, the sheet clusters and the feature scaler all come
from `LR_flux` and are frozen. On `LR` and `LR_flux_CBD_2` each point is
scaled with the `LR_flux` mean and spread, and gets the label of the nearest
`LR_flux` node. This makes the comparison consistent: a "sheet" is the same
region of feature space in every run. It also has limits:

- If a run's feature distribution is shifted (for example a different
  magnetisation without the flux tube, or a different disk density with the
  CBD), its sheets may sit in a part of feature space that the `LR_flux` SOM
  covers poorly. Points there are assigned to the nearest node anyway.
- Watch the distance to the best-matching node, and the fraction of points
  whose features fall outside the training range.
- Feature structures that exist only in `LR` or `LR_flux_CBD_2` cannot become
  their own cluster.

The alternative is to retrain per run:

1. Train a SOM on each run's own training file with the same settings
   (`etk-analysis/scripts/train_som.py --input features_<TAG>_<IT>.h5
   --n-samples 10000000`, 91x64 lattice, alpha 0.5).
2. Freeze each SOM with `build_model.py` and the same `--sheet-rule`.
3. Run `compare_runs.py` once per model, each with its own `--out`.

The sheets are then defined by the same physical rule but by different
clusterings. Cluster ids and boundaries differ between runs, and part of any
difference in the statistics comes from the clustering itself. Training costs
hours per run on this machine.

A reasonable plan is to use the frozen `LR_flux` SOM for the headline
comparison, and retrain on one other run to see how much the sheet volume
fraction and thickness depend on the clustering.
