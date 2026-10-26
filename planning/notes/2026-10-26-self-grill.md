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
