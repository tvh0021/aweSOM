"""Optional exact CUDA BMU search. CuPy is imported only on construction."""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def _kernels(features):
    import cupy as cp

    # Explicit round-to-nearest operations preserve the CPU feature order,
    # signed zero and float64 subnormals. No fast math or FMA contraction.
    # No #include: NVRTC does not search system headers. +inf is built from
    # its bit pattern instead of INFINITY, and the feature count macro has a
    # prefixed name so it cannot rewrite anything else.
    source = '#define AWESOM_F %d\n' % features + r'''
    template <typename T>
    __device__ void search(const double* w, const T* x, int* idx, double* d2,
                          int rows, int nodes, int tile, double* sw) {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        double row[AWESOM_F];
        if (p < rows) {
            #pragma unroll
            for (int f = 0; f < AWESOM_F; ++f) row[f] = (double)x[p * AWESOM_F + f];
        }
        double best = __longlong_as_double(0x7ff0000000000000LL);
        int best_i = 0;
        for (int base = 0; base < nodes; base += tile) {
            int count = min(tile, nodes - base);
            for (int k = threadIdx.x; k < count * AWESOM_F; k += blockDim.x)
                sw[k] = w[base * AWESOM_F + k];
            __syncthreads();
            if (p < rows) {
                for (int j = 0; j < count; ++j) {
                    double acc = 0.0;
                    #pragma unroll
                    for (int f = 0; f < AWESOM_F; ++f) {
                        double d = __dsub_rn(sw[j * AWESOM_F + f], row[f]);
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

    ``__call__`` is synchronous and not thread-safe: no work remains pending
    when it returns, including on an error. ``submit`` queues a block and
    returns while the last subchunks still run, so a caller such as
    :func:`aweSOM.stream.predict_stream` keeps the device busy between its
    chunks; ``wait`` completes the queued work. Subchunks alternate between
    two streams, overlapping one subchunk's staging and copies with the
    previous one's search, also across ``submit`` calls.
    ``overlap=False`` uses one slot for the probe's serial comparison.
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
        self.rows_per_thread = 1  # rows one kernel thread searches; the grid covers n rows
        self.args = ()  # extra kernel arguments after (w, x, idx, d2, rows, nodes, tile)
        self.streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2 if overlap else 1)]
        self.slots = []
        self.capacity = 0
        self.dtype = None
        self.turn = 0  # next slot; persists across submits so the oldest slot is reused
        self.token = 0

    def _allocate(self, rows, dtype):
        capacity = min(rows, self.chunk)
        if dtype == self.dtype and capacity <= self.capacity:
            return
        self.wait()  # queued results live in the slots about to be replaced
        # Reset first, so a failed allocation leaves a state that reallocates.
        self.slots, self.capacity, self.dtype, self.turn = [], 0, None, 0
        for stream in self.streams:
            owners, arrays = zip(*(_pinned(self.cp, shape, dt) for shape, dt in
                                  [((capacity, self.features), dtype),
                                   ((capacity,), np.int32), ((capacity,), np.float64)]))
            with stream:
                device = [self.cp.empty(a.shape, a.dtype) for a in arrays]
            self.slots.append(dict(owners=owners, host=arrays, device=device,
                                   labels=None, pending=None))
        self.capacity, self.dtype = capacity, dtype

    def cluster_map(self, node_cluster, dtype=np.int32):
        """Upload a node -> label table for :meth:`submit`.

        ``dtype`` is the label dtype, at most 4 bytes, used in native byte
        order; the caller chooses one that holds every entry, so the gather
        equals ``node_cluster[idx]``.
        """
        table = np.asarray(node_cluster).astype(np.dtype(dtype).newbyteorder('=')).ravel()
        if len(table) != self.nodes or table.dtype.itemsize > 4:
            raise ValueError('need one label per node in a dtype of at most 4 bytes')
        table = self.cp.asarray(table)
        # The upload runs on the current stream; the searches run on
        # non-blocking streams that do not order against it.
        self.cp.cuda.get_current_stream().synchronize()
        return table

    def _finish(self, slot, stream):
        stream.synchronize()
        pending, slot['pending'] = slot['pending'], None
        if pending is not None:
            _, out, host, d2, host_d2, _ = pending
            out[...] = host
            if d2 is not None:
                d2[...] = host_d2

    def cancel(self):
        """Wait for all device work and drop results that were not written yet."""
        try:
            for stream in self.streams:
                stream.synchronize()
        finally:
            for slot in self.slots:
                slot['pending'] = None

    def wait(self, token=None):
        """Write the results of every ``submit`` up to ``token`` (all when None)."""
        try:
            for i in range(len(self.slots)):  # oldest first: the next slot to reuse
                k = (self.turn + i) % len(self.slots)
                slot, stream = self.slots[k], self.streams[k]
                pending = slot['pending']
                if pending is not None and (token is None or pending[0] <= token):
                    self._finish(slot, stream)
        except BaseException:
            self.cancel()
            raise

    def submit(self, block, out, d2=None, clusters=None):
        """Queue the search of ``block`` and return a token for :meth:`wait`.

        ``out`` receives int32 BMU indices or, with a ``clusters`` table from
        :meth:`cluster_map`, ``clusters[idx]`` gathered on the device in the
        table's dtype. ``d2`` (optional, float64) receives squared distances;
        without it they are not copied off the device. Results are written on
        the host by a later ``submit`` that reuses their slot, or by ``wait``;
        ``block`` may be reused as soon as this returns. An error after the
        argument checks drops all queued work; a rejected argument leaves
        earlier submits queued.
        """
        x = np.asarray(block)
        if x.ndim != 2 or x.shape[1] != self.features or x.dtype not in (np.float32, np.float64):
            raise ValueError('block must be float32 or float64 with matching features')
        if len(out) != len(x) or (d2 is not None and len(d2) != len(x)):
            raise ValueError('outputs need one entry per row')
        if not len(x):
            return self.token
        self.token += 1
        cp = self.cp
        kernel = self.kernels[int(x.dtype == np.float64)]
        try:
            self._allocate(len(x), x.dtype)
            for start in range(0, len(x), self.capacity):
                k = self.turn
                self.turn = (k + 1) % len(self.slots)
                slot, stream = self.slots[k], self.streams[k]
                self._finish(slot, stream)
                stop = min(start + self.capacity, len(x))
                n = stop - start
                host, dev = slot['host'], slot['device']
                host[0][:n] = x[start:stop]
                with stream:
                    if clusters is not None and (slot['labels'] is None
                                                 or slot['labels'].dtype != clusters.dtype):
                        slot['labels'] = cp.empty(self.capacity, clusters.dtype)
                    dev[0][:n].set(host[0][:n], stream=stream)
                    per = self.block * self.rows_per_thread
                    kernel(((n + per - 1) // per,), (self.block,),
                           (self.w, dev[0], dev[1], dev[2], np.int32(n),
                            np.int32(self.nodes), np.int32(self.tile)) + self.args,
                           shared_mem=self.shared, stream=stream)
                    if clusters is None:
                        result, host_out = dev[1][:n], host[1][:n]
                    else:
                        result = slot['labels'][:n]
                        cp.take(clusters, dev[1][:n], out=result)
                        host_out = host[1].view(clusters.dtype)[:n]
                    result.get(out=host_out, stream=stream, blocking=False)
                    if d2 is not None:
                        dev[2][:n].get(out=host[2][:n], stream=stream, blocking=False)
                # The table stays referenced until the queued gather has run.
                slot['pending'] = (self.token, out[start:stop], host_out,
                                   None if d2 is None else d2[start:stop], host[2][:n], clusters)
        except BaseException:
            self.cancel()
            raise
        return self.token

    def __call__(self, block, idx, d2):
        x = np.asarray(block)
        if x.ndim != 2 or x.shape[1] != self.features or x.dtype not in (np.float32, np.float64):
            raise ValueError('block must be float32 or float64 with matching features')
        if idx.shape != (len(x),) or idx.dtype != np.int32 or d2.shape != (len(x),) or d2.dtype != np.float64:
            raise ValueError('outputs must be int32 idx and float64 d2, one per row')
        if len(x):
            self.submit(x, idx, d2)
            self.wait()
        return 0


def best_match_nodes_cuda(weights, data, chunk=2**20):
    """Return exact int32 BMU indices and float64 squared distances.

    ``chunk`` only bounds the staging buffers; larger values are capped at
    2**20 rows, as the other backends accept any chunk size.
    """
    from .inference import _check_inputs

    w, x = _check_inputs(weights, data)
    search = GpuSearcher(w, len(w), 1, chunk=min(int(chunk), 2**20))
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x), np.float64)
    search(x, idx, d2)
    return idx, d2
