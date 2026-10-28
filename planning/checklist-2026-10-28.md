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
- [ ] Flux tubes: 3D grouping step and test against Michi's tubes
- [ ] Low-res AsterX test run: reader for its grid structure and output format
- [ ] Proposal figure: current sheet between the two jets
- [ ] Chunked analysis of extreme-resolution data as it arrives
- [ ] Chat with Michi (user's action)

## Waiting on others (unchanged, user's list)

- Michi: AsterX low-res test directory; Frontier allocation and HHR submission; MHD figure; flux-tube velocities and centroids.
- Manuela: proposal duration; extra compute; diffusion-model add-on decision.
