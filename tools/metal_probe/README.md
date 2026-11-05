# Metal probe of the CUDA BMU search

A development probe, not a backend. `metal_bmu.py` reimplements the CUDA BMU
search in `src/aweSOM/_gpu.py` as an MLX custom Metal kernel so its structure
can run on an Apple GPU. No CUDA GPU is available to us yet: the GP cluster has
none and Tigris access is pending.

Metal has no float64. The probe computes in float32 and checks the float32
result bit for bit against a CPU reference in the same operation order. It
cannot check the CUDA kernel's float64 contract.

## What it checks

On an M1 Max (24 GPU cores), against `reference.py` (numba, float32, kernel
order, strict `<`):

- Kernel structure: one thread per row, a threadgroup tile of weights loaded
  cooperatively, a partial last tile, inactive threads reaching both barriers,
  and rows that are not a multiple of the block. Tested for 1 to 32 features,
  1/37/83/100/5824 nodes, tiles from 1 to 1000 (larger than the node count),
  and blocks from 1 to 1024.
- Feature-order subtract, multiply and add with no FMA contraction. MLX
  compiles custom kernels with contraction on inside a statement:
  `acc = acc + d * d` became `fma(d, d, acc)` and changed 23% of 2^20 random
  sums. The kernel puts `#pragma clang fp contract(off)` at the top of the body
  and squares in a separate statement. `test_no_fma_contraction` builds 64
  two-node cases that tie only without FMA. The guarded kernel keeps every tie.
  The unguarded source, compiled the same way, reproduces every fused answer,
  which shows the test can detect contraction.
- First-minimum tie rule: rows exactly equidistant (integer arithmetic, exact in
  float32) to two nodes shuffled across tile boundaries in either order, plus
  repeated copies of the same node. The lowest index wins for tiles 1, 5 and 32.
- Staging and subchunks: a chunk smaller than the row count (several
  subchunks, odd last one), one and two slots, strided and Fortran-order input,
  float32 and float64 input blocks, int32 `idx` and float64 `d2` outputs, slot
  reuse on smaller calls, reallocation on a dtype change, empty blocks, and the
  same validation errors as `GpuSearcher`.
- Training on real GPU hardware: `aweSOM._gpu.GpuSearcher` is monkeypatched
  with `MetalSearcher` and `train_batch_stream(..., bmu='cuda')` is run. The
  weights depend only on `idx`, so they match the CPU `bmu='numba'` run bit for
  bit whenever the float32 argmin equals the float64 argmin. A wrapper checks
  that equality on every row of every call before the weights are compared, and
  `reference.certified_rows` proves it from a rounding-error bound. Two cases
  are fully certified: the `scale_out_cpu` fixture with sampled init (12x8
  nodes, 3 epochs, also equal to the stored `expected_sampled`), and 200k rows
  in 8 separated clusters (4x2 nodes, 5 epochs, 7001-row Metal subchunks).
  In the given-init fixture run, 1 of 12,288 row searches falls inside the
  bound. It still agrees, so that case counts as observed equality, not
  guaranteed.

## What it cannot check

- Float64 arithmetic: `__dsub_rn/__dmul_rn/__dadd_rn`, last-ULP float64
  distances, float64 ties, float64 subnormals, and the in-kernel float32 to
  float64 conversion of input rows. Only `tests/gpu_test.py` on a CUDA device
  covers these.
- NVRTC compilation and its options (`--fmad=false`, no `#include`, the
  `AWESOM_F` macro).
- CUDA streams, pinned host memory, asynchronous copies and their overlap. MLX
  uses unified memory, so the probe's two-slot loop runs the same bookkeeping
  but measures no PCIe or NVLink transfer.
- Subnormals: Metal flushes float32 subnormal results to zero, where the CPU
  keeps them. `test_subnormals_flush_to_zero` records this.
- Large magnitudes: float32 overflows at about 1.8e19 in `d`, far below float64.

## Deviations from the CUDA kernel

| CUDA (`_gpu.py`) | Metal probe | Why |
|---|---|---|
| float64 weights, rows converted to double in the kernel | float32 weights; float64 rows rounded to float32 on the host while staging | no float64 in Metal |
| float64 `d2` from the kernel | float32 `d2`, upcast exactly on the host | same |
| `extern __shared__` sized at launch (`tile * F * 8`) | `threadgroup float sw[TILE * F]`, TILE and F template constants; `tile` is also passed at run time and must equal TILE | MLX exposes no dynamic threadgroup memory |
| `AWESOM_F` macro, `#pragma unroll` | template parameter `F`; the compiler decides unrolling | MLX template mechanism |
| `--fmad=false`, `__d*_rn` intrinsics | `#pragma clang fp contract(off)` and a separate statement for the square | MLX exposes no compile flags; Metal has no `_rn` intrinsics |
| `__longlong_as_double(0x7ff0...)` | `as_type<float>(0x7f800000u)` | same +inf bit-pattern idea |
| two CUDA streams, pinned slots, D2H into pinned outputs | two MLX GPU streams, `mx.async_eval`, outputs copied out of MLX arrays | unified memory |
| shared-memory limit read from the device | fixed 32 KiB threadgroup limit | MLX does not expose it |

The loop order, tile load, barriers, tie rule, launch geometry
(`ceil(rows / block)` full threadgroups) and host subchunk loop follow the CUDA
code with the same control structure. Transfer handling differs as the
table says.

## Measured rates (2026-11-05, M1 Max, 91x64 nodes, 7 features, float32 input)

`test_timing` writes `results-<date>.json`. Each figure is the best of 3 runs
after a warm-up, except numba brute force (one timed run) and kernel only
(best of 5, on the first 2^20 rows or the whole input if smaller). Numba used 8 threads.
The laptop was not idle: other jobs held the load average near 8 (10 cores),
so the CPU figures are low and noisy. An earlier single-run measurement gave
the screen 0.62 to 0.64 Mrows/s.

| rows | Metal, 2 slots | Metal, 1 slot | Metal kernel only | numba float64 brute force | numba `screen` |
|---|---|---|---|---|---|
| 1e6 | 18.3 Mrows/s | 18.7 | 19.4 | 0.63 Mrows/s | 1.68 Mrows/s |
| 1e7 | 19.2 Mrows/s | 18.6 | 19.4 | 0.53 Mrows/s | 1.23 Mrows/s |

At 1e7 rows, the Metal search is about 37 times the numba brute-force search
and 16 times the numba screen. Kernel-only and end-to-end rates differ by under
6%, so host copies hardly matter on unified memory, and the second slot gains
under 5%. 19.4 Mrows/s is about 1.1e11 row-node pairs/s, or 2.4 TFLOP/s
counting 21 operations per pair, about a quarter of the M1 Max float32 peak. A
2^21-row sweep over tile 16 to 256 and block 64 to 512 gave 13.5 to
19.3 Mrows/s. The defaults (tile 32, block 128) reached 19.0, so they need no
tuning here. None of these numbers predicts
float64 CUDA rates.

## Running it

```sh
cd <worktree>
/opt/homebrew/bin/python3.12 -m venv .venv-metal
.venv-metal/bin/pip install -e . mlx pytest
.venv-metal/bin/python -m pytest tools/metal_probe -q            # about 75 s with timing
METAL_PROBE_TIMING=0 .venv-metal/bin/python -m pytest tools/metal_probe -q   # skip timing
.venv-metal/bin/python -m pytest tests/gpu_glue_test.py tests/gpu_test.py -q -rs
```

The tests skip unless `mlx` imports and `mx.metal.is_available()` is true.
Measured with Python 3.12.7, MLX 0.32.4, NumPy 2.5.4 and Numba 0.68.1.
