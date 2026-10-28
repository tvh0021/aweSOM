# Checklist, session of 2026-10-28

Living copy of the user's to-do list. Each tick names the evidence (file, figure, or commit).
Results directory: `/Volumes/ExternalNVMe/awesom-results/2026-10-28/`. Nothing is pushed; all commits are local.

## Decisions settled in the opening session

- GitHub stays untouched. `dev` was fast-forwarded to `research/grmhd-scale` (18b63bc) and the four `opt/*` branches were merged locally (0a990ea, 24d86eb, 3a635d9). The `opt/*` and `research/*` branches themselves are left exactly as they were.
- Training infrastructure: Green Prairies `lm-n0001` (48 cores, 770 GB, no GPU, `debug` 12 h / `lm-largemem` 24 h). Target: train on N = 10^10 in under 1 h on that node; demonstrated with synthetic data here, not run at full scale this session.
- Runs: LR, LR_flux (local copy in `/Volumes/ExternalNVMe/simulation-data/michi-bbh`), LR_flux_CBD_2. Anchor iteration 180224 of output-0003, else the latest iteration common to all three. HHR and HHR_flux are the "extreme resolution" runs.
- Cluster data comes from the user running `etk-analysis` jobs; the kit is in `/Volumes/ExternalNVMe/awesom-results/cluster/` and mirrored to the user's own `/lagoon/trungha/bbh-output/for-awesom-2026-10-28/` on the cluster.
- "Send to Manuela" and "chat with Michi" are the user's own actions; the paper identification is recorded here.

## Things to do

- [x] Google "learned simulator" paper, 2D or 3D. It is Stachenfeld et al. 2022, "Learned Coarse Models for Efficient Turbulence Simulation" (DeepMind, ICLR 2022, arXiv:2112.15275, poster title "Learned Simulators for Turbulence"). Its headline turbulence boxes are 3D: decaying compressible turbulence in a periodic cube and a radiatively cooling mixing layer, Athena++ at 128^3 learned at 32^3; it also has 1D Kuramoto-Sivashinsky and 2D incompressible cases. The runner-up, Kochkov et al. 2021 PNAS (Google Research, learned interpolation), is 2D only. Sending it to Manuela is the user's action.
- [ ] Test speed optimizations against baseline on real data, then adopt
- [ ] Scale aweSOM to 10^10 points per node (training < 1 h)
- [ ] More physical clusters (training and cluster finding)
- [ ] Several realizations stacked for a robust clustering
- [ ] Current-sheet statistics: LR vs LR_flux vs LR_flux_CBD_2. Pipeline done and merged (bf3bd63: `src/aweSOM/sheet_stats.py`, `examples/grmhd-bbh/{build_model,compare_runs,check_labels}.py`, 14 tests). LR_flux it 180224 done: sheet = merge_cost 0.05 clusters 3,5,7,16, 1.23 % of the volume, one cell thick on every level (unresolved), mostly one network. LR and LR_flux_CBD_2 wait for the cluster grids; then one `compare_runs.py --reuse` call (5 min per run). Report: `/Volumes/ExternalNVMe/awesom-results/2026-10-28/sheet-stats/report.md`.
- [x] Flux tubes: 3D grouping step (`src/aweSOM/tubes.py`, 15 tests; merged 5aa46db). Synthetic: 5/5 tubes recovered, no false positives. LR_flux it 180224: 51 field-aligned tubes from merge_cost 0.05 cluster 0, but |B| inside is only 1.007x its surroundings, so whether they are Michi's flux tubes is open. Matching interface `tubes.match_to_reference` accepts his labels, masks or centre+radius tracks; the test against his tubes waits on his data. Report: `/Volumes/ExternalNVMe/awesom-results/2026-10-28/flux-tubes/report.md`.
- [x] Low-res AsterX test run: readers for CarpetX Silo (verified on a real 2D FM torus run), openPMD-HDF5 (synthetic only) and TSV, plus a native Carpet reader (`src/aweSOM/io/`, `src/aweSOM/features_grmhd.py`, 29 tests, merged). The feature port is bitwise equal to etkplot on a real block. Michi's directory has not arrived; the open questions (format, variable names, b2small units, openPMD box order) are listed in `docs/source/asterx-io.md`. Report: `/Volumes/ExternalNVMe/awesom-results/2026-10-28/asterx-chunked/report.md`.
- [x] Proposal figure: `/Volumes/ExternalNVMe/awesom-results/2026-10-28/proposal-figure/fig_current_sheet_between_jets_it180224.{pdf,png}` (plus it 135168), caption in `caption.md`, pipeline in `examples/grmhd-bbh/` (merged 649348c). Sheet cluster c9 at merge_cost 0.1 (none exists at 0.2): 0.21 % of the owned volume. Caveat: the stored float64 scaler in the training file is not the float32 one the lattice was trained with (up to 1 % off); the figure uses the float32 statistics, saved in `flux-tubes/training_scaler_float32.npz`.
- [x] Chunked analysis as data arrives: `examples/grmhd-bbh/analyze_chunks.py` (frozen model, per-block read/features/predict/write, resume after interruption, `--watch` for a run still writing). On LR_flux it 180224 its labels match the grid-file labels on all 68,028,213 rows; 2.6e5 cells/s, peak 4.5 GB on the loaded laptop. Model merge_cost 0.1 is provisional. HHR data itself is not here yet; its size comes from the `hhr_dryrun` cluster job.
- [ ] Chat with Michi (user's action)

## Waiting on others (unchanged, user's list)

- Michi: AsterX low-res test directory; Frontier allocation and HHR submission; MHD figure; flux-tube velocities and centroids.
- Manuela: proposal duration; extra compute; diffusion-model add-on decision.

## Review findings to fix (Sol, 2026-10-28; reports in `/Volumes/ExternalNVMe/awesom-results/2026-10-28/reviews/`)

- [ ] tubes.py:248/150 (P1): cross-level stitching can bridge a non-selected fine cell or join cubes that do not touch; check intervening ownership and use true cube adjacency.
- [ ] sheet_stats.py:496 (P1): floor+ceil interface rule connects coarse cells that do not contain the neighbouring point; use the nearest coarse candidate, expand only exact ties.
- [ ] sheet_stats.py:543 (P1): composite EDT mask ORs resampled levels without ownership precedence, so coarse sheet overwrites fine background; keep the finest owner's background.
- [ ] sheet_stats.py:479 (P2): connectivity 18/26 ignored across levels.
- [ ] sheet_stats.py:697 (P2): EDT thickness alternates n±0.5 cells; the "one cell thick" claim rests on the inertia estimator, state which one is reported.
- [ ] tubes.py:378 (P2): length_span ignores the projected cell half-widths.
- Scale: dense bounding boxes (tubes.py:228, sheet_stats.py:623) and per-component axes would need ~49 GB at 6.8e8 selected points; not a 10^10 path yet.
- [ ] freeze_model.py:77 (P1): frozen scaler uses float64 statistics, not the float32 reductions of manual_scaling (0.8 to 1.15 % off); use the float32 ones (saved in `flux-tubes/training_scaler_float32.npz`) and say so in the model metadata.
- [ ] make_proposal_figures.sh:37 + proposal_figure.py:131 (P1): it 135168 gets fresh thresholds (b2_min, s_v, winsor) but is predicted with the 180224 lattice; reuse the 180224 thresholds for other snapshots.
- [ ] analyze_chunks.py:300 (P2): resuming with a different slab width silently drops rows (657 in the probe); store the slab config in the file and refuse a mismatch.
- [ ] analyze_chunks.py:379/340 (P1): watch mode marks a partial snapshot complete when a level or Silo shard appears later; require all referenced shards and a stable layout before marking done.
