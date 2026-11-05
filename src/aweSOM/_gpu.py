"""Optional exact CUDA BMU search. CuPy is imported only on construction."""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def _kernels(features):
    import cupy as cp

    # Explicit round-to-nearest operations preserve the CPU feature order,
    # signed zero and float64 subnormals. No fast math or FMA contraction.
    source = '#define F %d\n' % features + r'''
    #include <math.h>
    template <typename T>
    __device__ void search(const double* w, const T* x, int* idx, double* d2,
                          int rows, int nodes, int tile, double* sw) {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        double row[F];
        if (p < rows) {
            #pragma unroll
            for (int f = 0; f < F; ++f) row[f] = (double)x[p * F + f];
        }
        double best = INFINITY;
        int best_i = 0;
        for (int base = 0; base < nodes; base += tile) {
            int count = min(tile, nodes - base);
            for (int k = threadIdx.x; k < count * F; k += blockDim.x)
                sw[k] = w[base * F + k];
            __syncthreads();
            if (p < rows) {
                for (int j = 0; j < count; ++j) {
                    double acc = 0.0;
                    #pragma unroll
                    for (int f = 0; f < F; ++f) {
                        double d = __dsub_rn(sw[j * F + f], row[f]);
                        acc = __dadd_rn(acc, __dmul_rn(d, d));
                    }
                    if (acc < best) { best = acc; best_i = base + j; }
                }
            }
            __syncthreads();
        }
        if (p < rows) { idx[p] = best_i; d2[p] = best; }
    }
    extern "C" __global__ void bmu_f32(const double* w, const float* x,
        int* idx, double* d2, int rows, int nodes, int tile) {
        extern __shared__ double sw[];
        search(w, x, idx, d2, rows, nodes, tile, sw);
    }
    extern "C" __global__ void bmu_f64(const double* w, const double* x,
        int* idx, double* d2, int rows, int nodes, int tile) {
        extern __shared__ double sw[];
        search(w, x, idx, d2, rows, nodes, tile, sw);
    }
    '''
    return tuple(cp.RawKernel(source, name, options=('--fmad=false', '--std=c++11'))
                 for name in ('bmu_f32', 'bmu_f64'))


def _pinned(cp, shape, dtype):
    owner = cp.cuda.alloc_pinned_memory(int(np.prod(shape)) * np.dtype(dtype).itemsize)
    return owner, np.ndarray(shape, dtype=dtype, buffer=owner)


class GpuSearcher:
    """Search fixed weights with bounded device memory and host staging.

    Calls are synchronous and not thread-safe. Within a call, alternating
    subchunks on two streams overlap H2D with the previous subchunk's search.
    ``overlap=False`` uses one slot for the probe's serial comparison.
    No work remains pending when a call returns, including on an error.
    """

    def __init__(self, weights, xdim, ydim, *, chunk=2**20, tile=32,
                 block=128, overlap=True):
        from .inference import _check_inputs

        w, _ = _check_inputs(weights, np.empty((0, np.shape(weights)[-1])))
        if not 1 <= w.shape[1] <= 32:
            raise ValueError('CUDA search supports 1 through 32 features')
        if xdim * ydim != len(w):
            raise ValueError('lattice dimensions do not match weights')
        if not 1 <= chunk <= 2**20 or not 1 <= block <= 1024 or tile < 1:
            raise ValueError('need 1 <= chunk <= 2**20, 1 <= block <= 1024 and tile > 0')
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError('CUDA BMU search needs CuPy; install aweSOM[cuda] (cupy-cuda12x)') from exc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('CUDA BMU search needs a CUDA device')
        self.cp = cp
        self.w = cp.asarray(w)
        cp.cuda.get_current_stream().synchronize()
        self.nodes, self.features = w.shape
        self.chunk, self.tile, self.block = int(chunk), int(tile), int(block)
        self.shared = self.tile * self.features * 8
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        if self.shared > props['sharedMemPerBlock'] or block > props['maxThreadsPerBlock']:
            raise ValueError('tile or block exceeds the device limit')
        self.kernels = _kernels(self.features)
        self.streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2 if overlap else 1)]
        self.slots = []
        self.capacity = 0
        self.dtype = None

    def _allocate(self, rows, dtype):
        capacity = min(rows, self.chunk)
        if dtype == self.dtype and capacity <= self.capacity:
            return
        self.slots = []
        for stream in self.streams:
            owners, arrays = zip(*(_pinned(self.cp, shape, dt) for shape, dt in
                                  [((capacity, self.features), dtype),
                                   ((capacity,), np.int32), ((capacity,), np.float64)]))
            with stream:
                device = [self.cp.empty(a.shape, a.dtype) for a in arrays]
            self.slots.append(dict(owners=owners, host=arrays, device=device, pending=None))
        self.capacity, self.dtype = capacity, dtype

    def __call__(self, block, idx, d2):
        x = np.asarray(block)
        if x.ndim != 2 or x.shape[1] != self.features or x.dtype not in (np.float32, np.float64):
            raise ValueError('block must be float32 or float64 with matching features')
        if idx.shape != (len(x),) or idx.dtype != np.int32 or d2.shape != (len(x),) or d2.dtype != np.float64:
            raise ValueError('outputs must be int32 idx and float64 d2, one per row')
        if not len(x):
            return 0
        self._allocate(len(x), x.dtype)

        def finish(slot, stream):
            stream.synchronize()
            if slot['pending'] is not None:
                start, stop = slot['pending']
                n = stop - start
                idx[start:stop] = slot['host'][1][:n]
                d2[start:stop] = slot['host'][2][:n]
                slot['pending'] = None

        try:
            for k, start in enumerate(range(0, len(x), self.capacity)):
                slot = self.slots[k % len(self.slots)]
                stream = self.streams[k % len(self.slots)]
                finish(slot, stream)
                stop = min(start + self.capacity, len(x))
                n = stop - start
                host, dev = slot['host'], slot['device']
                host[0][:n] = x[start:stop]
                with stream:
                    dev[0][:n].set(host[0][:n], stream=stream)
                    kernel = self.kernels[int(x.dtype == np.float64)]
                    kernel(((n + self.block - 1) // self.block,), (self.block,),
                           (self.w, dev[0], dev[1], dev[2], np.int32(n),
                            np.int32(self.nodes), np.int32(self.tile)),
                           shared_mem=self.shared, stream=stream)
                    dev[1][:n].get(out=host[1][:n], stream=stream, blocking=False)
                    dev[2][:n].get(out=host[2][:n], stream=stream, blocking=False)
                slot['pending'] = start, stop
            for slot, stream in zip(self.slots, self.streams):
                finish(slot, stream)
        finally:
            for stream in self.streams:
                stream.synchronize()
            for slot in self.slots:
                slot['pending'] = None
        return 0


def best_match_nodes_cuda(weights, data, chunk=2**20):
    """Return exact int32 BMU indices and float64 squared distances."""
    from .inference import _check_inputs

    w, x = _check_inputs(weights, data)
    search = GpuSearcher(w, len(w), 1, chunk=chunk)
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x), np.float64)
    search(x, idx, d2)
    return idx, d2
