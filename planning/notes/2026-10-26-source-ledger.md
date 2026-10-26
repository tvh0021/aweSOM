# Source ledger (2026-10-26 session)

Private working notes. Each entry: claim -> source. "Delegate" entries name
the model that gathered the fact; the primary source is listed alongside.

## A. AsterX runtime constraints (Gemini 3.8 Flash index sweep of /Users/tvh0021/git_repos/asterx)

- Diagnostic thorns loop with `grid.loop_int_device<1,1,1>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc &p) ...)`
  for cell-centred `{ccc}` grid functions. Source: `AsterAnalysis/src/analysis_mhd.cxx:33-35`,
  `AsterMasks/src/Masks.cxx:12-26`, CarpetX `Loop/src/loop_device.hxx:136-141`.
- Analysis grid functions are declared `CCTK_REAL ... TYPE=gf CENTERING={ccc} tags='checkpoint="no"'`
  and scheduled `AT analysis` with `READS: ...(everywhere)` / `WRITES: ...(interior)`.
  Source: `AsterAnalysis/interface.ccl:9-17`, `AsterAnalysis/schedule.ccl:3-43`.
- External table ingestion precedent: EOSX reads an HDF5 table named by a `CCTK_STRING` parameter
  (`EOSX/param.ccl:116-119`) once, globally, `AT CCTK_WRAGH` (`EOSX/schedule.ccl:9-13`); rank 0 reads
  with `H5Dread` and `MPI_Bcast`s; device copy via `amrex::The_Managed_Arena()->alloc` +
  `amrex::Gpu::copy(hostToDevice, ...)` (`EOSX/src/eos_3p_tabulated3d/eos_readtable_compose.hxx:131-138,414-432`);
  kernels capture the managed pointer by value (`AsterX/src/fluxes.cxx:1290`).
- No current density J and no curl B grid function exists in AsterX/AsterUtils/AsterAnalysis; the only curl is
  B = curl A onto staggered faces (`AsterX/src/computeBfromA.cxx:33-68`). A runtime thorn must compute a
  curl-B proxy itself from `HydroBaseX::Bvec` (cell-centred) with `p.DI`/`p.DX` offsets.
- A new thorn needs `REQUIRES Loop AsterUtils` (+ `AsterX`), `INHERITS: HydroBaseX ADMBaseX AsterX`,
  `USES INCLUDE HEADER: loop_device.hxx aster_fd.hxx aster_interp.hxx aster_utils.hxx`
  (`AsterAnalysis/configuration.ccl:3-6`, `AsterAnalysis/interface.ccl:3-7`).
- Output goes through `CarpetX::out_silo_vars` / `out_openpmd_vars` / `out_tsv_vars` / `out_norm_vars`
  with `"Thorn::group"` names (`AsterX/par/magTOV_Z4c_AMR.par:140-165`).

## B. User-authored contract for the aweSOM<->AsterX link

Source: `/Users/tvh0021/git_repos/asterx/audit/ideas.md` §4 (lines 623-740), "Origin: User-suggested".
Verbatim requirements that bind this session's design:

- "Keep training, ensemble construction, and model selection offline in Python. Export a
  language-neutral frozen artifact containing the feature specification, means and scales, SOM
  lattice weights, node-to-semantic-cluster map, uncertainty thresholds, and source provenance; do
  not make Python pickle the production interface."
- "Implement only feature evaluation and best-matching-unit inference in a separate analysis thorn,
  provisionally `AsterCurrentSheets`, using the same CPU/CUDA/ROCm-portable device conventions as
  AsterX. Run at a configurable analysis cadence after synchronized physical fields are available."
- "The first runtime version may write label or confidence grid functions ... but it must not change
  resistivity, evolution variables, refinement, or particle injection."
- "Require offline/runtime statistical equivalence and measured overhead before extending."
- "Do not train independent models or normalize each snapshot separately."
- "A current-sheet label must not be called a reconnection event."
- Stage 1 (snapshot analysis) must come before Stage 2 (runtime); tracking across time, connected
  components, AMR de-duplication are "Year-2 tasks".

Consequence: the thorn name is `AsterCurrentSheets`, not "AsterSOM"; the artifact must carry
uncertainty thresholds and provenance; aweSOM must offer a frozen-scaler, pickle-free model file,
and a reference BMU classifier that the thorn can reproduce exactly.

## C. JAX multi-device batch SOM (Opus 5.5 research; verified by running on JAX 0.12.0.1 CPU, 4 fake devices)

- Batch map: w_i = sum_j h(c_j,i) x_j / sum_j h(c_j,i); alpha cancels. Fort, Letremy & Cottrell, ESANN 2002,
  https://www.esann.org/sites/default/files/proceedings/legacy/es2002-71.pdf . Shares stationary points
  with online SOM; trajectories differ; equivalence is statistical.
- Factorised form: per-node sums S_c (nodes x F) and counts n_c via `jax.ops.segment_sum`, then
  num = H^T S, den = H^T n with H[c,i] = h(c,i) (nodes x nodes). Somoclu accumulates the same sums
  across MPI ranks (`src/denseCpuKernels.cpp`).
- JAX 0.12: `jax.pmap` is in maintenance mode, use `jax.shard_map` + `jax.make_mesh` + `NamedSharding`
  + `lax.psum` (changelog 0.8.0, 0.10.0: https://docs.jax.dev/en/latest/changelog.html).
  Gotcha: scan carries inside shard_map need `jax.lax.pcast(..., to="varying")` under check_vma
  (https://docs.jax.dev/en/latest/201/shard-map.html#scan-vma).
- Memory: N_chunk x nodes distance block; 2^16 x 2368 fp32 = 620 MB; chunk with `lax.scan`.
- Multi-host: `jax.distributed.initialize()` auto-detects SLURM (`jax/_src/clusters/slurm_cluster.py`),
  one process per GPU recommended (https://docs.jax.dev/en/latest/multi_process.html).
- CPU test harness: `jax.config.update('jax_num_cpu_devices', N)` or
  `XLA_FLAGS=--xla_force_host_platform_device_count=N` before backend init
  (https://docs.jax.dev/en/latest/parallel.html).
- Determinism: scatter-add is atomic on GPU; `--xla_gpu_exclude_nondeterministic_ops=true`
  (https://openxla.org/xla/determinism). TF32 may be used for fp32 matmul: use
  `precision=jax.lax.Precision.HIGHEST` for distances.
- jax-metal last released 0.1.1 (2024-10) for jax 0.4.34; incompatible with 0.12 API. Do not support Metal.

## D. Local facts (verified in this session)

- Baseline tests: 27 pass, 1 fail (`newshape=`), fixed. After bug fixes: 28 + 6 regression pass.
- Reproducibility bug confirmed by running training twice with identical input (different lattices).
- `train=20` -> ZeroDivisionError; single-centroid umat -> IndexError; `"labels.x-1p.npy".strip(".npy")`
  -> `labels.x-1`; `compute_SQ` on two empty masks -> RuntimeWarning invalid division.

## E. Math check (GPT-6.1 Sol, reasoning high; scripts run under /tmp with this checkout's venv)

- Online zero-drift condition w_i = B_i(W)/A_i(W) with A_i = E[H_{c(X),i}], B_i = E[H_{c(X),i} X];
  the batch map is the empirical fixed-point iteration of that condition (Kohonen,
  https://www.cis.hut.fi/somtoolbox/theory/somalgorithm.shtml). Not equivalent: frozen BMUs per epoch,
  order independence, alpha acts as averaging weight online and cancels in batch.
- Recommended starting budget: 40 batch epochs, radius r_max for epochs 1-20, linear to r_min over
  21-30, r_min for 31-40 (same proportions as the online schedule), UNCERTAIN until measured with
  20/40/80 epochs.
- Degenerate node (sum_j H = 0): keep the old weight; consistent with the online rule (all increments 0).
- BMU numerics, 2368x6 random lattice, 1e6 points, seed 20261026, NumPy 2.5.4, vs float64 direct:
  f32 direct 0 disagreements; f64 expansion 0; f32 expansion 11 (rate 1.1e-5, min relative gap 5.8e-7).
  Bounds: direct relative gamma_{F+2} ~ 4.8e-7 (F=6); expansion absolute ~ 1.1e-5 for |x|,|w| <= 1.
  Contract: reference-exact path = float64 direct arithmetic, sequential feature sum, strict '<' scan
  ascending over node index (matches np.argmin first occurrence); approximate fast path = f32 direct.
- Merge cost: invariant to global U-matrix scaling (cubic interpolation is linear, cost ~ q^2, ratio
  cancels) so the 1/(xdim*ydim) factor in compute_umat is irrelevant to merges; transfer of the
  threshold to a 2x larger lattice is NOT guaranteed (relative r_min halves, geometry changes).
- The recursive merge is NOT equivalent to a single frozen-normaliser greedy pass (counter-example
  AB=1, AC=4, AD=5, BC=10, BD=9, CD=6, thr 0.5: recursive merges only AB; frozen also merges AC).
  Equivalent non-recursive algorithm: loop {recompute reps/counts; costs from an ordered-pair cache;
  stable sort; M = max; stop if min/M >= thr; replace smaller-count rep}. Cubic interpolation can go
  negative (min -0.136 on a delta); do not clip.
- QE = mean ||x - w_bmu||, TE = fraction with d_cheb(c1,c2) > 1 (8-neighbourhood, no wrap).
  Acceptance: 5 paired seeds, upper bound mean + 2.776 sd/sqrt(5) of QE ratio-1 <= 0.03 and of TE
  difference <= 0.01 (t_{0.975,4} = 2.7764).
