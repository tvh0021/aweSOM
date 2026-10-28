# Physicality study (2026-10-28)

Scripts behind `/Volumes/ExternalNVMe/awesom-results/2026-10-28/physical/report.md`.
That report has the results and the recommendation. This file covers how to
rerun the study and which settings to use.

Package code added for the study:

- `src/aweSOM/physics_score.py`: the scoreboard. It covers per-cluster volume
  fraction, feature quantiles, neighbour agreement (kappa), connected
  components, overlap with a reference mask, ridge enrichment, best-match
  stability and the scalar `physicality`.
- `src/aweSOM/postprocess.py`: optional spatial clean-up (`clean_labels`),
  off by default.

## Recommended settings

- **Post-processing:** `clean_labels(labels, grid=grid, min_size=27)` (or
  `xyz=`, `spacing=`, `groups=level`). Components smaller than 27 cells (one
  3^3 stencil) are dissolved into the nearest surviving label.
- **Avoid the majority filter** (`majority_iters`) for current sheets: it
  erodes structures one or two cells thick.
- **Features:** see the report.
  - Dropping `asinh_div_v` is the robust improvement.
  - Swapping it for `log_one_minus_C_B` scores highest, but it is partly
    circular with the sheet metric.

## Pipeline

All scripts read paths from `common.py`.

Python: `PYTHONPATH=src /Users/tvh0021/git_repos/aweSOM-research/.venv/bin/python`.

1. `extra_fields.py`: derived fields per level. Run with the etk-analysis venv
   and `PYTHONPATH=/Volumes/ExternalNVMe/etk-analysis`. It writes
   `log_one_minus_C_B` and the sheet metric (`ratio_m`, `th3`, `b2`).
2. `prep.py IT` (via `run_prep.sh`): the column store `data/cols/`. It holds
   features, level, the sheet masks at T=1 and T=3, `chi_contrast`, the chi_B
   ridge, the 3M-row stats subsample, the sample-to-grid map and the
   `VoxelGrid`.
3. `train.py`: one lattice per call. Use `run_train.sh JOBS P` with job files
   `stage*.txt`. `import_prod.py` converts the production 91x64 lattice, using
   the float32 training scaler.
4. `score.py --it IT` (via `run_score.sh`, serialised by a lock dir): full-grid
   BMUs and the scoreboard at all merge costs. It writes
   `scores/<tag>__it<IT>.json`.
5. `aggregate.py`: builds `tables/scoreboard.csv` (one row per model and
   merge cost) and `tables/seeds.csv` (cross-seed stability S and P).
   `show.py` prints both as Markdown.
6. `postproc_study.py` (via `run_postproc.sh`): cleans labels at several
   settings and scores them before and after.
7. `figures.py` (via `make_figs.sh`): the report figures.

Memory: about 8 GB per scoring or post-processing process on the 68M-point
grids, and about 2 GB per 10^7-row training.
