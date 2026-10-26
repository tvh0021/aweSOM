# Self-grill: aweSOM at GRMHD scale (2026-10-26, autonomous session)

Working notes, not a deliverable. The user is asleep, so every ➡️ answer is my
own decision under stated assumptions. All of them are flagged in the final
report as assumptions the user can overturn.

Facts gathered so far (verified in the checkout `aweSOM-research`, branch
`research/grmhd-scale`, from `dev` @ 1069cff):

- Baseline test suite: 27 pass, 1 fail (`np.reshape(..., newshape=)` removed in
  NumPy 2.x, `make_sce_clusters.py:216`). Fixed in this checkout.
- Local hardware: Apple M1 Max, 10 cores, 64 GB. No CUDA. JAX 0.12 CPU only.
- `planning/optimization-plan.md` already states the goal: inference per
  snapshot is the priority, training once on a sample is secondary.
- ISSUES.md lists 2 open bugs (CLI `neurons_path`; plasma example normalizes
  only jz).

## Round 1 (frontier: top-level scope)

❓ **Q1 - What does "parallelized GPUs training" buy, given inference is the
stated priority?** Training runs once on a ~2M-point sample today (~5 min
per realization at 186 µs/step). The SCE workflow needs 30-100 realizations,
which today are run as independent processes (`run_plasma_som.py`,
multiprocessing). So "parallel training" can mean (a) one lattice trained
faster on many GPUs (batch SOM, data parallel), or (b) many realizations
trained concurrently, one per device. Which?

➡️ Both, but (b) is nearly free and (a) is the real feature. Implement a batch
SOM trainer in JAX that shards the data over all visible devices (data
parallel, psum of neighbourhood-weighted sums). Running several realizations
is then a loop over independent processes, one per device, which the
existing multiprocessing script already does. Keep the numba online trainer
as the reference and CPU fallback; the batch trainer must be validated
statistically (quantization error, topographic error, cluster count) against
it, never bitwise.

---

❓ **Q2 - Does the batch SOM have to reproduce the online SOM's learning-rate
and radius schedule?** The online trainer decays alpha by 0.75 every
train//25 steps and shrinks the radius linearly over the third quarter.
Batch SOM has no learning rate; only a radius schedule.

➡️ No. Batch SOM is a different estimator. Map the radius schedule by
fraction of training: radius = nsize_max for the first half of epochs,
linear to nsize_min over the third quarter, nsize_min for the last quarter,
same as online. Record the schedule in the saved model so it is reproducible.
Have the Sol math agent check the convergence claim and the mapping.

---

❓ **Q3 - What is the on-disk "table" that an AsterX run consumes?** The
simulation is C++ (CarpetX, CUDA/ROCm). It cannot unpickle a Python object.

➡️ A single HDF5 file (h5py already a dependency; Cactus builds link HDF5)
plus an optional plain-text twin for codes without HDF5. Contents: lattice
weights (nodes x F, float32 and float64), lattice shape, node-to-cluster id
table (nodes, int32), feature names, scaler kind + mean + scale per feature,
training metadata (alpha, train, merge threshold, seed, package version),
and a format version. Loader in Python (`aweSOM.model.SOMModel.load`) and a
header-only C++ reference reader + classifier under `cpp/` with a test that
reproduces the Python labels exactly on a saved fixture. The AsterX thorn
itself lives in the asterx repo and is out of scope for edits here; a design
document for it goes in `docs/`.

---

❓ **Q4 - Frozen scaling: is `manual_scaling` (mean, 2*std) the only scaler
to freeze?** `run_som.py` hard-codes `scale_method = "manual"`; MinMaxScaler
is dead code.

➡️ Freeze both (kind="manual": mean, 2*std*bulk_range; kind="minmax": min,
max). The model stores `x_scaled = (x - offset) / scale` so the C++ side has
one formula regardless of kind. Fitting happens once on the training sample
and is stored in the model; `predict` applies it. The old path (user scales
by hand, then trains) keeps working: scaler is optional.

---

❓ **Q5 - What "features" does the runtime classifier compute?** The plasma
features in `obtain_features.py` include convolutions (smoothed |J|,
derived j_par, j_perp, E.J). An in-situ thorn cannot run scipy.

➡️ The table carries only feature *names*; feature computation is the
thorn's job and is out of scope for aweSOM. aweSOM guarantees: given the
same F-vector per cell (in physical units) it produces the same label. The
C++ reference therefore takes a raw F-vector, applies the stored scaler,
finds the BMU, looks up the cluster. Index agent on asterx reports what J is
already available as a grid function; that goes into the design doc.

---

❓ **Q6 - Inference contract: exact match to current `best_match` +
`assign_cluster_to_data`, or approximate?** The plan's open question 4.

➡️ Exact on the chosen node for non-tied points, with ties broken by lowest
node index (what `np.argmin` does). float32 inputs may flip genuine ties;
the test checks label equality on float64 and ≥99.99% on float32, and
reports tie counts. KD-tree / early-exit search are out of scope for this
session (phase 4) because brute force at nodes≈2400, F≤10 is already
memory-bound not compute-bound after the allocation fix.

---

❓ **Q7 - Scope of bug fixing beyond ISSUES.md?** Reading `som.py` turned up
more: unseeded `default_rng()` defeats the module seed (non-reproducible
runs despite `seed=42`); `merge_similar_centroids` crashes with one centroid
(`sorted_cost[-1]` on empty list) and is O(k^3) recursive; `fast_som`
divides by zero for train < 25; `sce.py` uses `str.strip(".npy")` which
strips characters not a suffix; `compute_SQ` divides by zero before its
empty-mask guard; README snippet uses undefined names.

➡️ Fix every one that has a deterministic failing input, add a regression
test for each, leave behaviour-changing "improvements" (e.g. a different
merge order) alone. Reproducibility fix: seed the Generator from `self.seed`
(behaviour change: runs become reproducible; previously they were not, so no
consumer can depend on the old sequence).

---

❓ **Q8 - Where does the GPU path run when there is no GPU here?**

➡️ JAX CPU with `XLA_FLAGS=--xla_force_host_platform_device_count=4` in
tests to exercise sharding across 4 virtual devices. Real multi-GPU numbers
cannot be measured in this session; the report says so and ships a SLURM
example instead.

## Round 2 (after index reports)

(filled in below as facts arrive)

Facts now in hand: AsterX index (ledger A), user-authored aweSOM<->AsterX
contract in asterx/audit/ideas.md §4 (ledger B), JAX research (C), math
check (E). Still pending: aweSOM API/consumer sweep.

❓ **Q9 - The asterx audit names the thorn `AsterCurrentSheets` and demands
a language-neutral artifact with feature spec, means/scales, weights,
node->cluster map, uncertainty thresholds and provenance. Does that change
Q3?**

➡️ Yes, in two ways. (1) Add uncertainty thresholds: per node, the 99th
percentile of the BMU distance seen in training (`node_dist_q99`), plus a
global `reject_distance` (99.9th percentile over all training points); the
runtime writes the BMU distance as a second grid function so a cell can be
flagged "unlike anything in training". (2) Provenance fields: aweSOM
version, git commit, training data description string, creation time, the
exact scaler and the merge threshold. The thorn design doc uses the name
`AsterCurrentSheets`.

---

❓ **Q10 - Exactness contract for inference (closes Q6 with the numbers)?**
Sol: f32 direct form gave 0/1e6 disagreements vs f64 direct; f32 expansion
form 11/1e6. Expansion is what makes the GPU matmul fast.

➡️ Two paths, both documented. Reference path: float64, direct differences,
sequential feature sum, strict '<' scan over ascending node index. This is
what `Lattice.best_match` (numba) computes today and what `predict()` and
the C++ reference implement; a test checks bitwise label equality between
them on a saved fixture. Fast path (JAX, GPU): f32 expansion with
`Precision.HIGHEST`; test requires >= 99.99% label agreement and reports
the disagreement count. The thorn uses the direct form (per-cell loop over
~2400 nodes x F, no matmul needed), so it sits on the reference path.

---

❓ **Q11 - Batch trainer schedule and acceptance (closes Q2)?**

➡️ Defaults: 40 epochs, radius r_max for the first 50%, linear to r_min over
the next 25%, r_min for the last 25%; degenerate nodes keep their weight;
initial lattice = same sampling init as online (seeded). Acceptance test
(marked slow): 5 paired seeds on a synthetic 3-cluster dataset, QE ratio
upper bound <= 1.03 and TE difference upper bound <= 0.01 vs the online
trainer. If the batch trainer fails this on the synthetic set, the feature
ships as "experimental" with the measured numbers in the report rather than
being tuned until it passes.

---

❓ **Q12 - Replace the recursive merge with the equivalent loop?** Sol gave
a counter-example showing a frozen-normaliser greedy pass is NOT equivalent,
and an equivalent cached loop.

➡️ Yes, implement the equivalent loop with an ordered-pair cost cache. It
removes the recursion-depth risk and the O(k^3) recomputation while
producing identical merges. Test: on 20 random trained maps, the new merge
must give identical `lattice_assigned_clusters` to the old recursive code
(kept in the test as a reference copy).

---

❓ **Q13 - Where do the GPU/multi-device knobs live?**

➡️ `aweSOM.batch` module: `train_batch(data, lattice_shape, ..., epochs,
radius schedule, chunk, devices=None)`. It picks `jax.devices()` by
default; a `mesh` argument allows a user-defined mesh; multi-host via
`jax.distributed.initialize()` is the caller's job (documented SLURM
example in `examples/slurm_scripts/`). JAX stays optional: importing
`aweSOM` must not import JAX; `aweSOM.batch` raises a clear ImportError.

---

❓ **Q14 - Do we touch `fast_som` (online trainer) performance?** The plan's
phase 2 wants the loop compiled. 30% of time is U-matrix snapshots.

➡️ Minimal, behaviour-preserving: (a) make the U-matrix/lattice history
snapshot frequency configurable (`history=True` default keeps today's
behaviour for the tests that read `lattice_history`), (b) compile the
per-step update (BMU + neighbourhood update) into one numba kernel that
updates only nodes within the radius, with the same arithmetic. Validate
by bitwise-equal lattice vs the current loop on a seeded run (same RNG
stream, same operations in the same order: this is achievable because the
update is `lattice -= diff * gamma` elementwise). If bitwise equality is
not achieved, keep the old loop and report.

---

❓ **Q15 - Scope cut: what is explicitly NOT done this session?**

➡️ No AsterX thorn code in the asterx repo (not my checkout). No KD-tree.
No connected-component tracking. No change to SCE. No README rewrite
beyond fixing the broken snippet names. No publishing, no push, no PR.
