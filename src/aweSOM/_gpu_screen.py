"""Optional exact CUDA float32 screen with float64 certification and fallback.

The candidate set is a singleton only when the CPU screen's error bound
separates it from every other node. Otherwise it is the entire lattice.
This avoids a fixed candidate buffer and cannot overflow on duplicate nodes.
"""
from functools import lru_cache

import numpy as np

from ._gpu import GpuSearcher
from .inference import _check_inputs, _screen_kernel, SCREEN_TILE


def _screen_parameters(w):
    safe = bool(np.all(np.isfinite(w)) and np.max(np.abs(w)) <= 1e15)
    wmax = float(np.sqrt((w * w).sum(axis=1)).max()) if safe else 0.0
    return safe, wmax


def _cuda_screen_reference(weights, data, chunk=2**20, return_fallbacks=False):
    """CPU emulation of the CUDA candidate rule, including full-scan fallback.

    Both implementations use the top two float32 distances and the bound in
    :func:`aweSOM.inference.best_match_nodes_screen`. The CPU implementation
    shares that kernel, whose explicit float32 operations do not contract.
    ``return_fallbacks`` also returns the number of full-lattice candidate sets.
    """
    from .inference import _bmu_kernel

    w, x = _check_inputs(weights, data)
    if not 1 <= w.shape[1] <= 32:
        raise ValueError('CUDA search supports 1 through 32 features')
    if chunk < 1:
        raise ValueError('chunk must be positive')
    safe, wmax = _screen_parameters(w)
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x), np.float64)
    w32 = w.astype(np.float32) if safe else None
    fallbacks = 0
    for start in range(0, len(x), chunk):
        stop = min(start + chunk, len(x))
        block = np.ascontiguousarray(x[start:stop])
        if safe:
            fallbacks += int(_screen_kernel(w, w32, wmax, block, idx[start:stop],
                                            d2[start:stop], SCREEN_TILE))
        else:
            _bmu_kernel(w, block, idx[start:stop], d2[start:stop])
            fallbacks += len(block)
    return (idx, d2, fallbacks) if return_fallbacks else (idx, d2)


@lru_cache(maxsize=32)
def _kernels(features, safe, wmax):
    import cupy as cp

    source = ('#define AWESOM_F %d\n#define AWESOM_SAFE %d\n'
              '#define AWESOM_WMAX %.17e\n' % (features, safe, wmax)) + r'''
    __device__ double exact(const double* w, const double* row) {
        double acc = 0.0;
        #pragma unroll
        for (int f = 0; f < AWESOM_F; ++f) {
            double d = __dsub_rn(w[f], row[f]);
            acc = __dadd_rn(acc, __dmul_rn(d, d));
        }
        return acc;
    }
    template <typename T>
    __device__ void search(const double* w, const T* x, int* idx, double* d2,
                          int rows, int nodes, int tile, double* sw) {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        const double inf = __longlong_as_double(0x7ff0000000000000LL);
        double row[AWESOM_F];
        float row32[AWESOM_F];
        if (p < rows) {
            #pragma unroll
            for (int f = 0; f < AWESOM_F; ++f) {
                row[f] = (double)x[p * AWESOM_F + f];
                row32[f] = __double2float_rn(row[f]);
            }
        }
        float s1 = (float)inf, s2 = (float)inf;
        int j1 = 0;
        float* sf = (float*)sw;
        if (AWESOM_SAFE) {
            for (int base = 0; base < nodes; base += tile) {
                int count = min(tile, nodes - base);
                for (int k = threadIdx.x; k < count * AWESOM_F; k += blockDim.x)
                    sf[k] = __double2float_rn(w[base * AWESOM_F + k]);
                __syncthreads();
                if (p < rows) {
                    for (int j = 0; j < count; ++j) {
                        float acc = 0.0f;
                        #pragma unroll
                        for (int f = 0; f < AWESOM_F; ++f) {
                            float d = __fsub_rn(sf[j * AWESOM_F + f], row32[f]);
                            acc = __fadd_rn(acc, __fmul_rn(d, d));
                        }
                        if (acc < s1) { s2 = s1; s1 = acc; j1 = base + j; }
                        else if (acc < s2) s2 = acc;
                    }
                }
                __syncthreads();
            }
        }
        bool certified = false;
        if (p < rows && AWESOM_SAFE) {
            double xn2 = 0.0;
            #pragma unroll
            for (int f = 0; f < AWESOM_F; ++f)
                xn2 = __dadd_rn(xn2, __dmul_rn(row[f], row[f]));
            const double u = 5.9604644775390625e-8;
            double g = (AWESOM_F + 2) * u * 1.01;
            double e = 2.0 * u * (AWESOM_WMAX + __dsqrt_rn(xn2)) * 1.01 + 1e-18;
            double lo2 = __dsqrt_rn(__ddiv_rn((double)s2, 1.0 + g)) - e;
            double hi1 = (__dsqrt_rn(__ddiv_rn((double)s1, 1.0 - g)) + e) * (1.0 + 1e-12);
            certified = lo2 > hi1;
        }
        double best = inf;
        int best_i = 0;
        if (p < rows && certified) {
            best = exact(w + j1 * AWESOM_F, row);
            best_i = j1;
        }
        // All threads must visit barriers, including padding and certified rows.
        // Skip the second lattice load entirely when this block has no fallback.
        if (__syncthreads_or(p < rows && !certified)) {
            for (int base = 0; base < nodes; base += tile) {
                int count = min(tile, nodes - base);
                for (int k = threadIdx.x; k < count * AWESOM_F; k += blockDim.x)
                    sw[k] = w[base * AWESOM_F + k];
                __syncthreads();
                if (p < rows && !certified) {
                    for (int j = 0; j < count; ++j) {
                        double acc = exact(sw + j * AWESOM_F, row);
                        if (acc < best) { best = acc; best_i = base + j; }
                    }
                }
                __syncthreads();
            }
        }
        if (p < rows) { idx[p] = best_i; d2[p] = best; }
    }
    extern "C" __global__ void bmu_screen_f32(const double* w, const float* x,
        int* idx, double* d2, int rows, int nodes, int tile) {
        extern __shared__ double sw[];
        search(w, x, idx, d2, rows, nodes, tile, sw);
    }
    extern "C" __global__ void bmu_screen_f64(const double* w, const double* x,
        int* idx, double* d2, int rows, int nodes, int tile) {
        extern __shared__ double sw[];
        search(w, x, idx, d2, rows, nodes, tile, sw);
    }
    '''
    return tuple(cp.RawKernel(source, name, options=('--fmad=false', '--ftz=false', '--std=c++11'))
                 for name in ('bmu_screen_f32', 'bmu_screen_f64'))


class GpuScreenSearcher(GpuSearcher):
    """Reuse CUDA staging and streams with a certified float32 screening kernel.

    Chunk, tile, block and overlap options match :class:`GpuSearcher`.
    As with that class, the return value is zero (no fallback telemetry).
    """

    def __init__(self, weights, xdim, ydim, **kwargs):
        super().__init__(weights, xdim, ydim, **kwargs)
        w, _ = _check_inputs(weights, np.empty((0, self.features)))
        self.kernels = _kernels(self.features, *_screen_parameters(w))


def best_match_nodes_cuda_screen(weights, data, chunk=2**20):
    """Return exact int32 BMUs and float64 squared distances using CUDA screening.

    Requires optional CuPy and a CUDA device. Near-ties, non-finite values,
    overflow and unsafe weights use the full reference float64 scan.
    ``chunk`` bounds host/device staging and is capped at 2**20 rows.
    """
    w, x = _check_inputs(weights, data)
    search = GpuScreenSearcher(w, len(w), 1, chunk=min(int(chunk), 2**20))
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x), np.float64)
    search(x, idx, d2)
    return idx, d2
