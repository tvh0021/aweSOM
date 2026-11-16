# CUDA BMU backend

`cuda_screen` is an optional exact CUDA backend for BMU search, streaming inference, and batch SOM training. `auto` continues to select between the CPU `screen` and `patch` backends, so callers enable CUDA explicitly with `backend="cuda_screen"` or `bmu="cuda_screen"`.

## Installation

Install the `cuda` extra in a CUDA 12 environment:

```sh
pip install 'aweSOM[cuda]'
```

In `setup.py`, `aweSOM[cuda]` installs `cupy-cuda12x`. `aweSOM.stream.cuda_available()` imports CuPy on demand and returns `True` when `cupy.cuda.runtime.getDeviceCount() >= 1`, returning `False` instead of raising when CuPy or a GPU is absent. Importing `aweSOM.stream` does not import `aweSOM._gpu_screen` until `cuda_available()` or `"cuda_screen"` is called.

## Search kernel and exactness

`aweSOM._gpu_screen.GpuScreenSearcher` supports 1 through 32 features, 2D `float32` or `float64` input blocks, `1 <= chunk <= 2**20`, `1 <= block <= 1024`, and `tile >= 1`. It stores the lattice weights in `float64`, compiles the `bmu_screen_f32` and `bmu_screen_f64` `RawKernel` pair with `--fmad=false --ftz=false`, and allocates pinned host staging buffers and device buffers on `device`, defaulting to the current CUDA device at construction.

Each call screens all lattice nodes in `float32` using `__fmaf_rn` on `w32 - x32` and tracks the best and second-best values with lowest-index tie-breaking. When the lattice weights are finite with `max(|w|) <= 1e15` and the `float64` error bound separates the `float32` winner from a finite runner-up, the kernel evaluates one exact `float64` squared distance for that node. Near-ties, non-finite inputs, `float32` overflow, and unsafe weights rescan the full lattice in `float64` on the device. Returned `int32` node indices and `float64` squared distances match `aweSOM.inference.best_match_nodes` bit for bit.

Calling `searcher(block, idx, d2)` is synchronous and leaves no pending device work when it returns, including after an error. `submit(block, out, d2=None, clusters=None)` and `wait(token=None)` alternate subchunks of at most `chunk` rows across two non-blocking CUDA streams when `overlap=True`, overlapping host staging and transfers with kernel execution. `aweSOM._gpu_screen.best_match_nodes_cuda_screen`, `aweSOM.inference.predict(..., backend="cuda_screen")`, and `aweSOM.stream.BMUSearcher(weights, backend="cuda_screen", device=None)` wrap `GpuScreenSearcher`.

## Streaming inference and training

`aweSOM.stream.predict_stream(weights, node_cluster, data_source, ..., backend="cuda_screen", comm=None, device=None)` queues chunks through `GpuScreenSearcher.submit` so chunk `k` writes to `out` while chunk `k + 1` runs on the GPU:

```python
from aweSOM.stream import predict_stream, train_batch_stream

weights, info = train_batch_stream(data_source, xdim=91, ydim=64, bmu="cuda_screen")
labels, stats = predict_stream(weights, node_cluster, data_source, backend="cuda_screen")
```

Unless `out_node` is given, `GpuScreenSearcher.cluster_map` uploads the node-to-cluster table and gathers cluster IDs on the GPU in an integer dtype up to 4 bytes; squared distances are copied off the device only when `out_distance` is given. With an `mpi4py` communicator `comm`, each rank labels its contiguous slice `aweSOM.distributed.row_range(N, rank, size)`, and `device=None` selects `aweSOM.distributed.gpu_device(comm)`, which maps the node-local rank modulo the visible CUDA device count via a collective shared-memory split. When ranks label local blocks without passing `comm` on multi-GPU nodes, pass `device=aweSOM.distributed.gpu_device(mpi_comm)` explicitly.

`train_batch_stream(..., bmu="cuda_screen", comm=None)` runs exact BMU search on each rank's current CUDA device and takes no `device` parameter; set the active device or `CUDA_VISIBLE_DEVICES` per rank before calling. Per-group `float64` accumulation and `neighbourhood_update` run on the CPU in Numba. Trained weights and quantization error match the CPU BMU backends bit for bit. `info["fallbacks"]` is always `0` with `"cuda_screen"` because the kernel keeps no device counter.

## Validation

`cuda_screen` was validated on NVIDIA GH200, pairing an `aarch64` host with an H100-class GPU, where labels, distances, and trained weights matched the CPU `float64` reference bit for bit. `x86_64` with A100 is not validated for this backend.
