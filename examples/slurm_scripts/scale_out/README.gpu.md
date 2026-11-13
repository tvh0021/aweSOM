# CUDA probe

The GPU implementation is selected explicitly with `bmu="cuda_screen"` or
`backend="cuda_screen"`. Install `aweSOM[cuda]` in a CUDA 12 environment.
`auto` continues to choose between the existing CPU backends.

The prepared kit is at
`/Volumes/ExternalNVMe/awesom-results/planning/2026-11-05/scale-out/gpu/kit/`.
It contains these scripts, a source snapshot, the original `scale_1e9.py`
generator, `gmm_bbh7.npz`, and `trained-8238.npy` from job 8238.
Copy the complete kit to the cluster. The laptop paths are provenance only.

From the copied kit, submit with a fresh absolute output directory:

```sh
KIT="$PWD" OUT="$PWD/run-a100-01" sbatch gpu_probe.sbatch
```

The batch script requests one A100-80GB, eight CPUs, 64 GB RAM and 30 minutes.
Use the commented constraint variant for A100-40GB. It creates a fresh venv
and records resolved wheel URLs and hashes in `install.json`.
`REPO` can override the bundled source directory. `VENV` must be a fresh path.
The batch script refuses to overwrite `OUT` or reuse an existing venv.

For GH200, use the site's scheduler allocation and run the same Python script
with its CuPy environment and the same kit assets:

```sh
export NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR="$PWD/numba-cache/rank-0"
python gpu_probe.py --kit "$PWD" --out "$PWD/results-gh200-01"
```

`results/probe.json` records exact checks against both the trained and sampled
lattices, rates for six feature/lattice combinations with serial and overlapping
transfers, pinned and pageable H2D bandwidth, five training epochs and peak RSS.
`hardware.log` records the GPU, topology, driver, host memory, scratch and MPI
availability. `pytest.log` contains the GPU edge-case and training tests.
A failed assertion leaves `probe.json` with `status="failed"`.

Search calls complete synchronously. Two streams overlap subchunks inside each
call. A call with at most the internal chunk size uses one subchunk, so its
transfers cannot overlap its own compute. The default trainer chunk is larger
than the internal CUDA chunk. For throughput probes, keep multiple internal
subchunks per call. No asynchronous work survives a call or epoch boundary.

`cuda_screen` was verified on GH200 (Tigris jobs 961938 and 995672: labels and
training bitwise equal to the CPU float64 reference). Run the kit on each new
target GPU before production use.
