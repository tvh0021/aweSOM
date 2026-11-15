"""Optional exact CUDA float32 screen with float64 certification and fallback.

The candidate set is a singleton only when the CPU screen's error bound
separates it from every other node. Otherwise it is the entire lattice.
This avoids a fixed candidate buffer and cannot overflow on duplicate nodes.

The float32 screen value of node j is computed in the difference form
``acc = fma(d_f, d_f, acc)`` with ``d_f = w32_jf - x32_f``, f = 0..F-1, each
operation rounded to nearest. Why the CPU screen's bound still holds, with
u = 2**-24 and S = sum_f (w32_jf - x32_f)**2 computed exactly:

* ``d_f = (w32_jf - x32_f)(1 + a_f)``, |a_f| <= u (a subtraction is exact
  when its result is subnormal, so no absolute term arises here).
* ``acc_0 = d_0**2 (1 + b_0)`` and ``acc_f = (d_f**2 + acc_{f-1})(1 + b_f)``,
  |b_f| <= u: one rounding per step instead of a rounded square plus a
  rounded add.
* Term 0 carries (1 + a_0)**2 and b_0..b_{F-1}: F + 2 factors. Term f >= 1
  carries (1 + a_f)**2 and b_f..b_{F-1}: F - f + 2 <= F + 1 factors. All terms
  are non-negative, so the computed value lies in
  [S (1 - u)**(F+2), S (1 + u)**(F+2)], a relative error of at most
  gamma_{F+2} = (F+2)u / (1 - (F+2)u) <= 1.01 (F+2)u for F <= 1.6e5. The
  separate multiply and add of the CPU screen give the same F + 2 for term
  0 and F + 3 - f for term f, so the fused form is never looser.
* Underflow adds an absolute error of at most 2**-150 per rounding, F of
  them, against 2F - 1 before; sqrt(F 2**-150) < 1e-21 is inside the
  ``1e-18`` term of ``e``.
* ``e = 2.02u(wmax + |x|)`` bounds the move from (w, x) to (w32, x32), as
  before: the input rounding is unchanged.
* The bound needs a finite runner-up ``s2``. A float32 value that overflows
  to inf has an exact value of at least 2**128 - 2**103 > FLT_MAX at that
  step, and the steps only add non-negative terms, so a node whose value
  overflowed lies above any finite ``s2`` and the bound covers it. An
  overflowed ``s2`` bounds nothing (``lo2`` would be inf and certify any
  ``s1``), so every screen certifies only when ``s2`` is finite.

Hence the certification test, ``g = 1.01 (F+2) u`` and ``e``, are unchanged,
and a certified row's node is the float64 reference's node. Certified rows
recompute their distance, and every other row rescans the lattice, with the
unchanged float64 code, so labels and distances stay bitwise those of
:func:`aweSOM.inference.best_match_nodes`. Only the set of certified rows
can differ from the CPU screen; :func:`_cuda_screen_reference` emulates the
fused rounding exactly, so its fallback count is the kernel's.
"""
import contextlib
from functools import lru_cache

import numpy as np
from numba import njit, prange

from .inference import _check_inputs, _U32

TILE = 128  # nodes per shared-memory tile


def _screen_parameters(w):
    safe = bool(np.all(np.isfinite(w)) and np.max(np.abs(w)) <= 1e15)
    wmax = float(np.sqrt((w * w).sum(axis=1)).max()) if safe else 0.0
    return safe, wmax


def _layout(features):
    """Rows per thread and the float32 node stride (a multiple of 4, for float4 loads)."""
    rows = 4 if features <= 12 else 2 if features <= 20 else 1
    return rows, -(-features // 4) * 4


@njit(cache=True)
def _fma32(a, b, c):
    """``fmaf(a, b, c)`` of float32 values, rounded once, as CUDA's ``__fmaf_rn``.

    ``a * b`` is exact in float64. The float64 sum is rounded, and rounding
    it again to float32 can differ from one rounding only when it lands
    exactly on a float32 midpoint; its exact error (TwoSum) then decides.
    """
    p = np.float64(a) * np.float64(b)
    cc = np.float64(c)
    s = p + cc
    r = np.float32(s)
    if not np.isfinite(s):
        return r
    v = s - p
    t = (p - (s - v)) + (cc - v)
    rr = np.float64(r) if np.isfinite(r) else np.copysign(2.0**128, s)
    if t == 0.0 or rr == s:
        return r
    o = np.nextafter(r, np.float32(np.inf) if s > rr else np.float32(-np.inf))
    if (rr + np.float64(o)) * 0.5 != s:  # exact: adjacent float32 values
        return r
    return np.fmax(r, o) if t > 0.0 else np.fmin(r, o)


@njit(parallel=True, cache=True)
def _fma_screen_kernel(w, w32, wmax, safe, data, out_idx, out_dist2):
    """Per-row emulation of the CUDA screen: same operations, same rounding, same tests."""
    n = data.shape[0]
    nodes, F = w.shape
    g = (F + 2) * _U32 * 1.01
    fb = np.zeros(n, np.int64)
    inf32 = np.float32(np.inf)
    for q in prange(n):
        x32 = np.empty(F, np.float32)
        for f in range(F):
            x32[f] = np.float32(np.float64(data[q, f]))
        s1 = inf32
        s2 = inf32
        j1 = 0
        if safe:
            for i in range(nodes):
                acc = np.float32(0.0)
                for f in range(F):
                    d = w32[i, f] - x32[f]
                    acc = _fma32(d, d, acc)
                s2 = np.fmin(s2, np.fmax(s1, acc))
                if acc < s1:
                    j1 = i
                s1 = np.fmin(s1, acc)
        certified = False
        if safe:
            xn2 = 0.0
            for f in range(F):
                v = np.float64(data[q, f])
                xn2 += v * v
            e = 2.0 * _U32 * (wmax + np.sqrt(xn2)) * 1.01 + 1e-18
            lo2 = np.sqrt(np.float64(s2) / (1.0 + g)) - e
            hi1 = (np.sqrt(np.float64(s1) / (1.0 - g)) + e) * (1.0 + 1e-12)
            certified = lo2 > hi1 and s2 < inf32
        if certified:
            acc64 = 0.0
            for f in range(F):
                d64 = w[j1, f] - np.float64(data[q, f])
                acc64 += d64 * d64
            out_idx[q] = j1
            out_dist2[q] = acc64
        else:
            best = np.inf
            best_idx = 0
            for i in range(nodes):
                acc64 = 0.0
                for f in range(F):
                    d64 = w[i, f] - np.float64(data[q, f])
                    acc64 += d64 * d64
                if acc64 < best:
                    best = acc64
                    best_idx = i
            out_idx[q] = best_idx
            out_dist2[q] = best
            fb[q] = 1
    return fb.sum()


def _cuda_screen_reference(weights, data, chunk=2**20, return_fallbacks=False):
    """CPU emulation of the CUDA candidate rule, including full-scan fallback.

    Same float32 operations as the kernel, including the fused multiply-add
    (emulated with one rounding, see :func:`_fma32`), the same top-two update
    and the same float64 test, so labels, distances and the number of
    full-lattice candidate sets (``return_fallbacks``) all match the kernel.
    """
    w, x = _check_inputs(weights, data)
    if not 1 <= w.shape[1] <= 32:
        raise ValueError('CUDA search supports 1 through 32 features')
    if chunk < 1:
        raise ValueError('chunk must be positive')
    safe, wmax = _screen_parameters(w)
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x), np.float64)
    w32 = w.astype(np.float32) if safe else np.zeros(w.shape, np.float32)  # unused when unsafe
    fallbacks = 0
    for start in range(0, len(x), chunk):
        stop = min(start + chunk, len(x))
        fallbacks += int(_fma_screen_kernel(w, w32, wmax, safe, np.ascontiguousarray(x[start:stop]),
                                            idx[start:stop], d2[start:stop]))
    return (idx, d2, fallbacks) if return_fallbacks else (idx, d2)


def _source(features):
    rows, pad = _layout(features)
    return ('#define AWESOM_F %d\n#define AWESOM_R %d\n#define AWESOM_P %d\n'
            % (features, rows, pad)) + r'''
    // Thread t of block b handles rows b * blockDim.x * AWESOM_R + t + r * blockDim.x,
    // r < AWESOM_R, so one shared-memory read of a node serves AWESOM_R rows.
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
                          int rows, int nodes, int tile, double wmax, int safe,
                          float4* smem) {
        double* sw = (double*)smem;
        float* sf = (float*)smem;
        const int first = blockIdx.x * blockDim.x * AWESOM_R + threadIdx.x;
        const double inf = __longlong_as_double(0x7ff0000000000000LL);
        float xr[AWESOM_R][AWESOM_F];
        float s1[AWESOM_R], s2[AWESOM_R];
        int j1[AWESOM_R];
        #pragma unroll
        for (int r = 0; r < AWESOM_R; ++r) {
            const int p = first + r * blockDim.x;
            #pragma unroll
            for (int f = 0; f < AWESOM_F; ++f)
                xr[r][f] = p < rows ? __double2float_rn((double)x[p * AWESOM_F + f]) : 0.0f;
            s1[r] = (float)inf;
            s2[r] = (float)inf;
            j1[r] = 0;
        }
        if (safe) {
            for (int base = 0; base < nodes; base += tile) {
                const int count = min(tile, nodes - base);
                // float32 node weights, stride AWESOM_P, zero padding.
                for (int k = threadIdx.x; k < count * AWESOM_P; k += blockDim.x) {
                    const int f = k % AWESOM_P;
                    sf[k] = f < AWESOM_F
                        ? __double2float_rn(w[(base + k / AWESOM_P) * AWESOM_F + f]) : 0.0f;
                }
                __syncthreads();
                for (int j = 0; j < count; ++j) {
                    float wv[AWESOM_P];
                    #pragma unroll
                    for (int q = 0; q < AWESOM_P / 4; ++q) {
                        const float4 v = smem[j * (AWESOM_P / 4) + q];
                        wv[4 * q] = v.x;
                        wv[4 * q + 1] = v.y;
                        wv[4 * q + 2] = v.z;
                        wv[4 * q + 3] = v.w;
                    }
                    const int node = base + j;
                    #pragma unroll
                    for (int r = 0; r < AWESOM_R; ++r) {
                        float acc = 0.0f;
                        #pragma unroll
                        for (int f = 0; f < AWESOM_F; ++f) {
                            const float d = __fsub_rn(wv[f], xr[r][f]);
                            acc = __fmaf_rn(d, d, acc);
                        }
                        // Top two without branches. s2 is the smallest value over every
                        // node but j1; the strict test keeps the lowest index on ties.
                        // acc is NaN only for a NaN row, where every value is NaN and
                        // s1 = s2 = inf stay, as with the branchy update.
                        s2[r] = fminf(s2[r], fmaxf(s1[r], acc));
                        j1[r] = acc < s1[r] ? node : j1[r];
                        s1[r] = fminf(s1[r], acc);
                    }
                }
                __syncthreads();
            }
        }
        #pragma unroll
        for (int r = 0; r < AWESOM_R; ++r) {
            const int p = first + r * blockDim.x;
            double row[AWESOM_F];
            if (p < rows) {
                #pragma unroll
                for (int f = 0; f < AWESOM_F; ++f) row[f] = (double)x[p * AWESOM_F + f];
            }
            bool certified = false;
            if (p < rows && safe) {
                double xn2 = 0.0;
                #pragma unroll
                for (int f = 0; f < AWESOM_F; ++f)
                    xn2 = __dadd_rn(xn2, __dmul_rn(row[f], row[f]));
                // Bound: see the module docstring of aweSOM._gpu_screen.
                const double u = 5.9604644775390625e-8;
                double g = (AWESOM_F + 2) * u * 1.01;
                double e = 2.0 * u * (wmax + __dsqrt_rn(xn2)) * 1.01 + 1e-18;
                double lo2 = __dsqrt_rn(__ddiv_rn((double)s2[r], 1.0 + g)) - e;
                double hi1 = (__dsqrt_rn(__ddiv_rn((double)s1[r], 1.0 - g)) + e) * (1.0 + 1e-12);
                certified = lo2 > hi1 && s2[r] < (float)inf;  // finite s2: see the docstring
            }
            double best = inf;
            int best_i = 0;
            if (p < rows && certified) {
                best = exact(w + j1[r] * AWESOM_F, row);
                best_i = j1[r];
            }
            // All threads must visit barriers, including padding and certified rows.
            // Skip the float64 lattice pass when no row r of this block falls back.
            if (__syncthreads_or(p < rows && !certified)) {
                for (int base = 0; base < nodes; base += tile) {
                    const int count = min(tile, nodes - base);
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
    }
    extern "C" __global__ void bmu_screen_f32(const double* w, const float* x,
        int* idx, double* d2, int rows, int nodes, int tile, double wmax, int safe) {
        extern __shared__ float4 smem[];
        search(w, x, idx, d2, rows, nodes, tile, wmax, safe, smem);
    }
    extern "C" __global__ void bmu_screen_f64(const double* w, const double* x,
        int* idx, double* d2, int rows, int nodes, int tile, double wmax, int safe) {
        extern __shared__ float4 smem[];
        search(w, x, idx, d2, rows, nodes, tile, wmax, safe, smem);
    }
    '''


@lru_cache(maxsize=32)
def _kernels(features):
    import cupy as cp

    # --fmad=false keeps the float64 code unfused; the screen fuses explicitly.
    source = _source(features)
    return tuple(cp.RawKernel(source, name, options=('--fmad=false', '--ftz=false', '--std=c++11'))
                 for name in ('bmu_screen_f32', 'bmu_screen_f64'))


def cuda_available():
    """True if CuPy imports and sees at least one CUDA device; never raises.

    Uses only the CuPy runtime device count (no system-information reads).
    """
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() >= 1
    except Exception:
        return False


def _on_device(cp, device):
    """Context that makes ``device`` current; a no-op for None (current device)."""
    return contextlib.nullcontext() if device is None else cp.cuda.Device(device)


def _pinned(cp, shape, dtype):
    owner = cp.cuda.alloc_pinned_memory(int(np.prod(shape)) * np.dtype(dtype).itemsize)
    return owner, np.ndarray(shape, dtype=dtype, buffer=owner)


class GpuScreenSearcher:
    """Search fixed weights with the screen kernel, bounded device memory and host staging.

    ``__call__`` is synchronous and not thread-safe: no work remains pending
    when it returns, including on an error. ``submit`` queues a block and
    returns while the last subchunks still run, so a caller such as
    :func:`aweSOM.stream.predict_stream` keeps the device busy between its
    chunks; ``wait`` completes the queued work. Subchunks alternate between
    two streams, overlapping one subchunk's staging and copies with the
    previous one's search, also across ``submit`` calls.
    ``overlap=False`` uses one slot for the probe's serial comparison.
    ``tile`` is the number of nodes per shared-memory tile; each thread
    screens up to four rows. ``device`` (CUDA index) holds the weights,
    buffers and streams, and every call runs there; None uses the device
    current at construction.
    """

    def __init__(self, weights, xdim, ydim, *, chunk=2**20, tile=TILE,
                 block=128, overlap=True, device=None):
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
        self.device = None if device is None else int(device)
        self.nodes, self.features = w.shape
        self.chunk, self.tile, self.block = int(chunk), int(tile), int(block)
        self.rows_per_thread, pad = _layout(self.features)  # the grid covers n rows
        self.shared = self.tile * max(8 * self.features, 4 * pad)
        safe, wmax = _screen_parameters(w)
        # Kernel arguments after (w, x, idx, d2, rows, nodes, tile); runtime
        # values, so a new lattice reuses the compiled kernel.
        self.args = (np.float64(wmax), np.int32(safe))
        with _on_device(cp, self.device):
            self.w = cp.asarray(w)
            cp.cuda.get_current_stream().synchronize()
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
            if self.shared > props['sharedMemPerBlock'] or block > props['maxThreadsPerBlock']:
                raise ValueError('tile or block exceeds the device limit')
            self.kernels = _kernels(self.features)
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
        with _on_device(self.cp, self.device):
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
        with _on_device(self.cp, self.device):
            return self._submit(x, out, d2, clusters)

    def _submit(self, x, out, d2, clusters):
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
