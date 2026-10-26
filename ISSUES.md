# Issue tracker

Bugs found while preparing GRMHD features for aweSOM (etk-analysis project). Each entry gives the location, what goes wrong, and a suggested fix. Fixed entries are in the "Fixed" section at the bottom.

## Open

### A. Final `umat` can lag the final lattice by up to `save_frequency - 1` steps (pre-existing)

- **Where:** `src/aweSOM/som.py` (`fast_som`, end of training)
- **What:** with `history=True` (the default, and the only behaviour before this branch) `self.umat` is computed from the last snapshot before `self.lattice` is set to the final lattice. With `history=False` there are no snapshots, so the U-matrix comes from the final lattice. The two settings can therefore give slightly different `umat` and merges.
- **Status:** deliberately kept for bitwise compatibility with the old loop. Suggested fix: compute `umat` from the final lattice in both branches (changes results of every existing notebook by a few steps' worth of drift).

### B. `tests/run_som_test.py` writes into the current directory (pre-existing)

- **Where:** `tests/run_som_test.py:68-85`
- **What:** `som_object.test-*.pkl` and `labels.test-*.npy` are written to cwd and removed afterwards; a failing test leaves them behind.
- **Fix:** use the `tmp_path` fixture.

### C. Plasma example's merge threshold and stored clusters do not match the paper (pre-existing, documentation)

- **Where:** `examples/plasma-turbulence/run_plasma_som.py:95` (`merge_threshold = 0.2`), `examples/plasma-turbulence/som_object.fiducial-5000-64-37-0.1-2097152-1u.pkl` (`clusters` attribute).
- **What:** Ha et al. (2024, Appendix A) merge at 0.25, which on the shipped fiducial lattice reproduces the paper's seven clusters and volume fractions exactly. The example script hard-codes 0.2 (nine clusters on the same lattice), and the `clusters` array stored in the pickle (fractions 6.5, 1.7, 1.1, 1.5, 69.8, 17.9, 1.5 %) matches none of 0.2, 0.25 or 0.3. Appendix B also states initial weights in [-1, 1] while `Lattice` draws them in [0, 1).
- **Status:** reported in `planning/reports/2026-10-26-plasma-metal-results.html`, section 1; not changed because each fix alters published example output.

### D. `tests/sce_test.py` under a Metal/GPU JAX (fixed on `research/grmhd-scale`, commit f11f9f2)

- **What:** `test_jax_cpu_available` patched `jax.default_backend` without re-importing `aweSOM.sce`, so it only passed on CPU-only machines; `test_compute_SQ` required a Python `float` and a NumPy array, which JAX device arrays are not.
- **Fix:** reload the module under the patch; compare by value and shape.

## Fixed on `research/grmhd-scale` (2026-10-26)

### 1. `run_som.py` CLI crashes on an undefined argument

- **Where:** `src/aweSOM/run_som.py:273`
- **What:** `__main__` reads `args.neurons_path`, but `parse_args()` defines no `--neurons_path` option (the closest is `--lattice_path`). Any CLI run stops with `AttributeError: 'Namespace' object has no attribute 'neurons_path'`.
- **Also:** `__main__` uses a relative import (`from .som import Lattice`, line 254), so `python src/aweSOM/run_som.py` fails before that; only `python -m aweSOM.run_som` gets as far as line 273.
- **Fix:** read `args.lattice_path` (or add the missing option), and use an absolute import.

### 2. Plasma example normalizes only one current and no electric-field component

- **Where:** `examples/plasma-turbulence/obtain_features.py:74-104` (`get_normalization`); the same pattern is in `examples/plasma-turbulence/read_turbulence_data.py` (`get_normalization`, from line 105).
- **What:** only `"jz"` gets the current normalization `qe * n0 * cfl**2`. `"jx"`, `"jy"`, `"ex"`, `"ey"` and `"ez"` match no branch and keep `norm = 1.0`, although lines 317-325 divide them by `get_normalization(...)` as if they were normalized. jz then has different units from jx and jy, which skews every quantity built from the vector: `j_mag`, j∥ = (B·J)/|B|, j⊥, e∥, E·J.
- **Fix:** give `jx`/`jy` the same branch as `jz`, and the E components the `norm_E` factor used in the `"je"` branch. Then check that the published features were not computed with the unnormalized components, or that the factor happens to be 1 for the fiducial run.

### 3. `make_sce_clusters.py` uses the removed `np.reshape(..., newshape=)` keyword

- **Where:** `src/aweSOM/make_sce_clusters.py:216`
- **What:** NumPy 2.x removed the `newshape` keyword, so `test_combine_separated_clusters` failed with `TypeError`.
- **Fix:** pass the shape positionally.

### 4. Training was not reproducible despite `seed = 42`

- **Where:** `src/aweSOM/som.py` (`fast_som`)
- **What:** the sample order came from an unseeded `np.random.default_rng()` and the initial lattice from the global NumPy stream, so two identical calls gave different lattices.
- **Fix:** one `default_rng(self.seed)` drives both the initial lattice and the shuffle; the restart lattice is copied instead of modified in place.

### 5. `fast_som` divided by zero for `train < 25`

- **Where:** `src/aweSOM/som.py` (`alpha_freq`, `nsize_freq`, progress print)
- **Fix:** every period is clamped to at least 1.

### 6. `merge_similar_centroids` crashed with a single centroid

- **Where:** `src/aweSOM/som.py` (`sorted_cost[-1]` on an empty list)
- **Fix:** return early when there is nothing to merge.

### 7. `sce.py` stripped characters instead of the `.npy` suffix

- **Where:** `src/aweSOM/sce.py:140,191` (`str.strip(".npy")`)
- **What:** a run name ending in `n`, `p` or `y` lost its last letter in `multimap_mappings.txt` and in the mask file name.
- **Fix:** `str.removesuffix(".npy")`.

### 8. `compute_SQ` divided by zero before its empty-mask guard

- **Where:** `src/aweSOM/sce.py` (`compute_SQ`)
- **Fix:** the guard now runs before `S` is computed.
