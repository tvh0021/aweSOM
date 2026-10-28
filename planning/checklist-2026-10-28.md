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
- [ ] Current-sheet statistics: LR vs LR_flux vs LR_flux_CBD_2
- [x] Flux tubes: 3D grouping step (`src/aweSOM/tubes.py`, 15 tests; merged 5aa46db). Synthetic: 5/5 tubes recovered, no false positives. LR_flux it 180224: 51 field-aligned tubes from merge_cost 0.05 cluster 0, but |B| inside is only 1.007x its surroundings, so whether they are Michi's flux tubes is open. Matching interface `tubes.match_to_reference` accepts his labels, masks or centre+radius tracks; the test against his tubes waits on his data. Report: `/Volumes/ExternalNVMe/awesom-results/2026-10-28/flux-tubes/report.md`.
- [ ] Low-res AsterX test run: reader for its grid structure and output format
- [x] Proposal figure: `/Volumes/ExternalNVMe/awesom-results/2026-10-28/proposal-figure/fig_current_sheet_between_jets_it180224.{pdf,png}` (plus it 135168), caption in `caption.md`, pipeline in `examples/grmhd-bbh/` (merged 649348c). Sheet cluster c9 at merge_cost 0.1 (none exists at 0.2): 0.21 % of the owned volume. Caveat: the stored float64 scaler in the training file is not the float32 one the lattice was trained with (up to 1 % off); the figure uses the float32 statistics, saved in `flux-tubes/training_scaler_float32.npz`.
- [ ] Chunked analysis of extreme-resolution data as it arrives
- [ ] Chat with Michi (user's action)

## Waiting on others (unchanged, user's list)

- Michi: AsterX low-res test directory; Frontier allocation and HHR submission; MHD figure; flux-tube velocities and centroids.
- Manuela: proposal duration; extra compute; diffusion-model add-on decision.
