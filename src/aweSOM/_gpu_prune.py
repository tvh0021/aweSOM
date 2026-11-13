"""Exact CUDA box pruning, with a CPU emulation of the device algorithm.

Bounds use the same round-to-nearest arithmetic as node distances. See
``docs/gpu_prune.md`` for the monotonicity proof and benchmark instructions.
CuPy is imported only when constructing a device searcher.
"""
from functools import lru_cache

import numpy as np
from numba import njit



def _prepare(weights, lattice_shape):
    nodes, features = weights.shape
    if not 1 <= features <= 32:
        raise ValueError('CUDA prune search supports 1 through 32 features')
    shape = (nodes, 1) if lattice_shape is None else tuple(lattice_shape)
    if (len(shape) != 2 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in shape)
            or shape[0] * shape[1] != nodes):
        raise ValueError('lattice_shape must contain two positive dimensions matching weights')
    nx, ny = shape
    # 8x8 tiles on a 2D lattice; 64-node runs without a lattice shape.
    sx, sy = (64, 1) if ny == 1 else (8, 8)
    groups = [np.array([y * nx + x for y in range(y0, min(y0 + sy, ny))
                       for x in range(x0, min(x0 + sx, nx))], np.int32)
              for y0 in range(0, ny, sy) for x0 in range(0, nx, sx)]
    perm = np.concatenate(groups)
    offsets = np.array([0] + list(np.cumsum([len(g) for g in groups])), np.int32)
    boxes = np.ascontiguousarray([[weights[g].min(axis=0), weights[g].max(axis=0)]
                                  for g in groups])
    return perm, offsets, boxes, bool(np.isfinite(weights).all())


def _hints(hint, rows, nodes):
    if hint is None:
        return np.full(rows, -1, np.int32)
    h = np.asarray(hint)
    if h.shape != (rows,) or h.dtype != np.int32:
        raise ValueError('hint must be an int32 array with one entry per row')
    if np.any(h < 0) or np.any(h >= nodes):
        raise ValueError('hint contains an out-of-range node')
    return np.ascontiguousarray(h)


@njit(cache=True)
def _distance(w, row, j):
    acc = 0.0
    for f in range(w.shape[1]):
        d = w[j, f] - np.float64(row[f])
        acc = acc + d * d
    return acc


@njit(cache=True)
def _bound(boxes, row, b):
    acc = 0.0
    for f in range(boxes.shape[2]):
        x = np.float64(row[f])
        d = 0.0
        if x < boxes[b, 0, f]:
            d = boxes[b, 0, f] - x
        elif x > boxes[b, 1, f]:
            d = x - boxes[b, 1, f]
        acc = acc + d * d
    return acc


@njit(cache=True)
def _reference_kernel(w, x, hints, perm, offsets, boxes, finite):
    idx = np.zeros(len(x), np.int32)
    d2 = np.full(len(x), np.inf)
    counts = np.zeros(len(x), np.int64)
    for p in range(len(x)):
        best, bi, checked = np.inf, 0, 0
        full = not finite
        for f in range(x.shape[1]):
            if not np.isfinite(x[p, f]):
                full = True
        if not full:
            seed = -1
            if hints[p] >= 0:
                a = _distance(w, x[p], hints[p])
                checked += 1
                if a < best:
                    best, bi = a, hints[p]
            else:
                lb = np.inf
                seed = 0
                for b in range(len(boxes)):
                    a = _bound(boxes, x[p], b)
                    if a < lb:
                        lb, seed = a, b
            # Seed block first, followed by ascending block IDs. Hints remove
            # the seed pass entirely. No per-row heap or sorting workspace.
            for visit in range(len(boxes) + 1):
                b = seed if visit == 0 else visit - 1
                if b < 0 or (visit > 0 and b == seed):
                    continue
                if _bound(boxes, x[p], b) > best:
                    continue
                for k in range(offsets[b], offsets[b + 1]):
                    j = perm[k]
                    a = _distance(w, x[p], j)
                    checked += 1
                    if a < best or (a == best and j < bi):
                        best, bi = a, j
                if checked - int(hints[p] >= 0) > len(w) // 2:
                    full = True
                    break
        if full:
            best, bi = np.inf, 0
            for j in range(len(w)):
                a = _distance(w, x[p], j)
                if a < best:
                    best, bi = a, j
            checked += len(w)
        idx[p], d2[p], counts[p] = bi, best, checked
    return idx, d2, counts


def _cuda_prune_reference(weights, data, lattice_shape=None, hint=None,
                          chunk=2**20, return_evaluations=False):
    """CPU emulation; optional third result counts node checks per row.

    Counts include the hint and any repeated work before full-scan fallback.
    """
    from .inference import _check_inputs
    w, x = _check_inputs(weights, data)
    if int(chunk) < 1:
        raise ValueError('chunk must be positive')
    metadata = _prepare(w, lattice_shape)
    h = _hints(hint, len(x), len(w))
    idx, d2, counts = np.empty(len(x), np.int32), np.empty(len(x)), np.empty(len(x), np.int64)
    for start in range(0, len(x), int(chunk)):
        stop = min(start + int(chunk), len(x))
        idx[start:stop], d2[start:stop], counts[start:stop] = _reference_kernel(
            w, x[start:stop], h[start:stop], *metadata)
    return (idx, d2, counts) if return_evaluations else (idx, d2)


@lru_cache(maxsize=32)
def _kernels(features):
    import cupy as cp
    source = '#define AWESOM_PF %d\n' % features + r'''
    __device__ double distance(const double* w, const double* row, int j) {
        double acc = 0.0;
        #pragma unroll
        for (int f = 0; f < AWESOM_PF; ++f) {
            double d = __dsub_rn(w[j * AWESOM_PF + f], row[f]);
            acc = __dadd_rn(acc, __dmul_rn(d, d));
        }
        return acc;
    }
    __device__ double bound(const double* box, const double* row, int b) {
        double acc = 0.0;
        #pragma unroll
        for (int f = 0; f < AWESOM_PF; ++f) {
            double x = row[f], d = 0.0;
            if (x < box[(b * 2) * AWESOM_PF + f])
                d = __dsub_rn(box[(b * 2) * AWESOM_PF + f], x);
            else if (x > box[(b * 2 + 1) * AWESOM_PF + f])
                d = __dsub_rn(x, box[(b * 2 + 1) * AWESOM_PF + f]);
            acc = __dadd_rn(acc, __dmul_rn(d, d));
        }
        return acc;
    }
    template <typename T>
    __device__ void search(const double* w, const T* x, const int* hints,
        const int* perm, const int* offsets, const double* boxes,
        int* idx, double* d2, int rows, int nodes, int blocks, int finite,
        int shared_boxes, double* shared) {
        if (shared_boxes) {
            for (int k = threadIdx.x; k < blocks * 2 * AWESOM_PF; k += blockDim.x)
                shared[k] = boxes[k];
        }
        __syncthreads();
        const double* box = shared_boxes ? shared : boxes;
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= rows) return;
        double row[AWESOM_PF];
        bool full = !finite;
        #pragma unroll
        for (int f = 0; f < AWESOM_PF; ++f) {
            row[f] = (double)x[p * AWESOM_PF + f];
            if (!isfinite(row[f])) full = true;
        }
        double inf = __longlong_as_double(0x7ff0000000000000LL);
        double best = inf;
        int bi = 0, checked = 0;
        if (!full) {
            int seed = -1;
            if (hints[p] >= 0) {
                double a = distance(w, row, hints[p]);
                ++checked;
                if (a < best) { best = a; bi = hints[p]; }
            } else {
                double lb = inf;
                seed = 0;
                for (int b = 0; b < blocks; ++b) {
                    double a = bound(box, row, b);
                    if (a < lb) { lb = a; seed = b; }
                }
            }
            for (int visit = 0; visit <= blocks; ++visit) {
                int b = visit == 0 ? seed : visit - 1;
                if (b < 0 || (visit > 0 && b == seed)) continue;
                if (bound(box, row, b) > best) continue;
                for (int k = offsets[b]; k < offsets[b + 1]; ++k) {
                    int j = perm[k];
                    double a = distance(w, row, j);
                    ++checked;
                    if (a < best || (a == best && j < bi)) { best = a; bi = j; }
                }
                if (checked - (hints[p] >= 0 ? 1 : 0) > nodes / 2) { full = true; break; }
            }
        }
        if (full) {
            best = inf; bi = 0;
            for (int j = 0; j < nodes; ++j) {
                double a = distance(w, row, j);
                if (a < best) { best = a; bi = j; }
            }
        }
        idx[p] = bi; d2[p] = best;
    }
    extern "C" __global__ void prune_f32(const double* w, const float* x,
        const int* h, const int* perm, const int* off, const double* box,
        int* idx, double* d2, int rows, int nodes, int blocks, int finite, int sm) {
        extern __shared__ double s[];
        search(w, x, h, perm, off, box, idx, d2, rows, nodes, blocks, finite, sm, s);
    }
    extern "C" __global__ void prune_f64(const double* w, const double* x,
        const int* h, const int* perm, const int* off, const double* box,
        int* idx, double* d2, int rows, int nodes, int blocks, int finite, int sm) {
        extern __shared__ double s[];
        search(w, x, h, perm, off, box, idx, d2, rows, nodes, blocks, finite, sm, s);
    }
    '''
    return tuple(cp.RawKernel(source, name, options=('--fmad=false', '--std=c++11'))
                 for name in ('prune_f32', 'prune_f64'))


class PruneGpuSearcher:
    """Reusable fixed-lattice search; metadata is built and uploaded once.

    Host input/output, synchronous calls, bounded chunk staging. The kernel
    itself takes device pointers, ready for a future resident AsterX caller.
    ``device`` (CUDA index) holds the metadata and runs every call; None uses
    the device current at construction.
    """

    def __init__(self, weights, lattice_shape=None, *, chunk=2**20, device=None):
        from .inference import _check_inputs
        w, _ = _check_inputs(weights, np.empty((0, np.shape(weights)[-1])))
        perm, off, boxes, self.finite = _prepare(w, lattice_shape)
        if int(chunk) < 1:
            raise ValueError('chunk must be positive')
        self.chunk = min(int(chunk), 2**20)
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError('CUDA prune search needs CuPy; install aweSOM[cuda]') from exc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('CUDA prune search needs a CUDA device')
        from ._gpu import _on_device
        self.cp = cp
        self.device = None if device is None else int(device)
        self.nodes, self.features = w.shape
        self.blocks = len(boxes)
        with _on_device(cp, self.device):
            self.w, self.perm, self.off, self.boxes = map(cp.asarray, (w, perm, off, boxes))
            cp.cuda.get_current_stream().synchronize()
        self.shared = boxes.nbytes if boxes.nbytes <= 32768 else 0
        self.kernels = _kernels(self.features)

    def __call__(self, data, idx, d2, hint=None):
        from ._gpu import _on_device
        with _on_device(self.cp, self.device):
            return self._search(data, idx, d2, hint)

    def _search(self, data, idx, d2, hint=None):
        x = np.asarray(data)
        if x.ndim != 2 or x.shape[1] != self.features or x.dtype not in (np.float32, np.float64):
            raise ValueError('data must be float32 or float64 with matching features')
        if idx.shape != (len(x),) or idx.dtype != np.int32 or d2.shape != (len(x),) or d2.dtype != np.float64:
            raise ValueError('outputs must be int32 idx and float64 d2, one per row')
        h = _hints(hint, len(x), self.nodes)
        cp = self.cp
        for start in range(0, len(x), self.chunk):
            stop = min(start + self.chunk, len(x))
            n = stop - start
            dx = cp.asarray(np.ascontiguousarray(x[start:stop]))
            dh = cp.asarray(h[start:stop])
            di, dd = cp.empty(n, cp.int32), cp.empty(n, cp.float64)
            self.kernels[int(x.dtype == np.float64)](
                ((n + 127) // 128,), (128,),
                (self.w, dx, dh, self.perm, self.off, self.boxes, di, dd,
                 np.int32(n), np.int32(self.nodes), np.int32(self.blocks),
                 np.int32(self.finite), np.int32(bool(self.shared))), shared_mem=self.shared)
            idx[start:stop], d2[start:stop] = cp.asnumpy(di), cp.asnumpy(dd)
        return 0


def best_match_nodes_cuda_prune(weights, data, lattice_shape=None, hint=None, chunk=2**20):
    """Return exact int32 BMUs and float64 squared distances, optionally warm-started.

    ``hint`` must contain a valid int32 node index per row. Wrong but valid
    hints affect work only. Reuse :class:`PruneGpuSearcher` across time steps
    to avoid rebuilding metadata and uploading weights.
    """
    from .inference import _check_inputs
    w, x = _check_inputs(weights, data)
    h = _hints(hint, len(x), len(w))
    search = PruneGpuSearcher(w, lattice_shape, chunk=chunk)
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x), np.float64)
    search(x, idx, d2, h if hint is not None else None)
    return idx, d2
