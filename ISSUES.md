# Issue tracker

Bugs found while preparing GRMHD features for aweSOM (etk-analysis project). Each entry gives the location, what goes wrong, and a suggested fix. None of them has been fixed yet.

## Open

### 1. `run_som.py` CLI crashes on an undefined argument

- **Where:** `src/aweSOM/run_som.py:273`
- **What:** `__main__` reads `args.neurons_path`, but `parse_args()` defines no `--neurons_path` option (the closest is `--lattice_path`). Any CLI run stops with `AttributeError: 'Namespace' object has no attribute 'neurons_path'`.
- **Also:** `__main__` uses a relative import (`from .som import Lattice`, line 254), so `python src/aweSOM/run_som.py` fails before that; only `python -m aweSOM.run_som` gets as far as line 273.
- **Fix:** read `args.lattice_path` (or add the missing option), and use an absolute import.

### 2. Plasma example normalizes only one current and no electric-field component

- **Where:** `examples/plasma-turbulence/obtain_features.py:74-104` (`get_normalization`); the same pattern is in `examples/plasma-turbulence/read_turbulence_data.py` (`get_normalization`, from line 105).
- **What:** only `"jz"` gets the current normalization `qe * n0 * cfl**2`. `"jx"`, `"jy"`, `"ex"`, `"ey"` and `"ez"` match no branch and keep `norm = 1.0`, although lines 317-325 divide them by `get_normalization(...)` as if they were normalized. jz then has different units from jx and jy, which skews every quantity built from the vector: `j_mag`, j∥ = (B·J)/|B|, j⊥, e∥, E·J.
- **Fix:** give `jx`/`jy` the same branch as `jz`, and the E components the `norm_E` factor used in the `"je"` branch. Then check that the published features were not computed with the unnormalized components, or that the factor happens to be 1 for the fiducial run.
