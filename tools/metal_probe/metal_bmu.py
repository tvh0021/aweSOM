"""Float32 Metal analogue of aweSOM._gpu (development probe, not a backend).

Mirrors the CUDA BMU search in src/aweSOM/_gpu.py so its structure can run on
an Apple GPU through MLX custom Metal kernels. Metal has no float64, so the
arithmetic is float32 throughout and only the float32 contract can be checked:

- weights are rounded to float32 on the host (CUDA keeps them float64);
- float64 input blocks are rounded to float32 on the host while staging
  (CUDA converts float32 input to double inside the kernel);
- the kernel returns float32 distances, upcast exactly to float64 on the host
  so ``d2`` has the same dtype as in ``GpuSearcher``.

Everything else follows the CUDA code: one thread per row, threadgroup tile of
weights loaded cooperatively (partial last tile included), both barriers
reached by inactive threads, feature-order subtract/multiply/add with no FMA
contraction, strict ``<`` so the lowest node index wins a tie, subchunks of at
most ``chunk`` rows alternating between two slots and streams.
"""
from functools import lru_cache

import numpy as np

# Apple GPUs (M1 and later) give 32 KiB of threadgroup memory per threadgroup.
# MLX does not expose the limit, so it is fixed here.
THREADGROUP_BYTES = 32768

# One kernel body for every feature count and tile; F and TILE are template
# parameters because Metal threadgroup arrays need a compile-time size (the
# CUDA kernel uses dynamic shared memory sized at launch instead).
#
# FMA: MLX compiles custom kernels with fp contraction enabled inside a
# statement, so `acc = acc + d * d` becomes fma(d, d, acc) (measured: 23% of
# 2**20 random sums differ from two roundings). The pragma switches
# contraction off for the kernel body, and the product is also a separate
# statement. test_metal_probe.py checks both with a contraction-sensitive tie
# and with a positive control built from the unguarded source.
_GUARD = '#pragma clang fp contract(off)\n'

_BODY = r'''
    threadgroup float sw[TILE * F];
    const int p = (int)(threadgroup_position_in_grid.x * threads_per_threadgroup.x
                        + thread_position_in_threadgroup.x);
    float row[F];
    if (p < rows) {
        for (int f = 0; f < F; ++f) row[f] = x[p * F + f];
    }
    float best = as_type<float>(0x7f800000u);
    int best_i = 0;
    for (int base = 0; base < nodes; base += tile) {
        int count = min(tile, nodes - base);
        for (int k = (int)thread_position_in_threadgroup.x; k < count * F;
             k += (int)threads_per_threadgroup.x)
            sw[k] = w[base * F + k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (p < rows) {
            for (int j = 0; j < count; ++j) {
                float acc = 0.0f;
                for (int f = 0; f < F; ++f) {
                    float d = sw[j * F + f] - row[f];
                    @SQUARE@
                }
                if (acc < best) { best = acc; best_i = base + j; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (p < rows) { idx[p] = best_i; d2[p] = best; }
'''


def kernel_source(contract_guard=True):
    """Metal body as passed to MLX. ``contract_guard=False`` is the test's
    positive control: the natural one-statement form, which MLX contracts."""
    if contract_guard:
        return _GUARD + _BODY.replace('@SQUARE@', 'float sq = d * d;\n                    acc = acc + sq;')
    return _BODY.replace('@SQUARE@', 'acc = acc + d * d;')


@lru_cache(maxsize=4)
def _kernel(contract_guard=True):
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name='awesom_bmu_f32' + ('' if contract_guard else '_contracted'),
        input_names=['w', 'x', 'rows', 'nodes', 'tile'],
        output_names=['idx', 'd2'],
        source=kernel_source(contract_guard),
    )


class MetalSearcher:
    """Float32 Metal counterpart of ``aweSOM._gpu.GpuSearcher``.

    Same constructor and call signature, same validation messages (with
    "Metal" for "CUDA"), same slot/subchunk bookkeeping. MLX arrays live in
    unified memory, so there is no pinned staging: each subchunk is copied from
    the float32 host slot into a new MLX array, the kernel is queued on the
    slot's stream with ``mx.async_eval`` and the slot is drained before reuse.
    """

    def __init__(self, weights, xdim, ydim, *, chunk=2**20, tile=32,
                 block=128, overlap=True, contract_guard=True):
        from aweSOM.inference import _check_inputs

        w, _ = _check_inputs(weights, np.empty((0, np.shape(weights)[-1])))
        if not 1 <= w.shape[1] <= 32:
            raise ValueError('Metal search supports 1 through 32 features')
        if xdim * ydim != len(w):
            raise ValueError('lattice dimensions do not match weights')
        if not 1 <= chunk <= 2**20 or not 1 <= block <= 1024 or tile < 1:
            raise ValueError('need 1 <= chunk <= 2**20, 1 <= block <= 1024 and tile > 0')
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise ImportError('Metal BMU search needs MLX; pip install mlx') from exc
        if not mx.metal.is_available():
            raise RuntimeError('Metal BMU search needs a Metal device')
        self.mx = mx
        self.w64 = w
        self.w = mx.array(w.astype(np.float32))  # deviation: float32 weights
        mx.eval(self.w)
        self.nodes, self.features = w.shape
        self.chunk, self.tile, self.block = int(chunk), int(tile), int(block)
        self.shared = self.tile * self.features * 4
        if self.shared > THREADGROUP_BYTES:
            raise ValueError('tile or block exceeds the device limit')
        self.kernel = _kernel(bool(contract_guard))
        self.streams = [mx.new_stream(mx.gpu) for _ in range(2 if overlap else 1)]
        self.slots = []
        self.capacity = 0
        self.dtype = None

    def _allocate(self, rows, dtype):
        capacity = min(rows, self.chunk)
        if dtype == self.dtype and capacity <= self.capacity:
            return
        # The host slot is float32 for either input dtype; float64 blocks are
        # rounded to nearest when copied in (CUDA keeps a float64 slot).
        self.slots = [dict(host=np.empty((capacity, self.features), np.float32), pending=None)
                      for _ in self.streams]
        self.capacity, self.dtype = capacity, dtype

    def launch(self, x32, stream):
        """Queue the kernel on one contiguous float32 subchunk; returns lazy (idx, d2)."""
        mx = self.mx
        n = len(x32)
        return self.kernel(
            inputs=[self.w, mx.array(x32), n, self.nodes, self.tile],
            template=[('F', self.features), ('TILE', self.tile)],
            grid=(((n + self.block - 1) // self.block) * self.block, 1, 1),
            threadgroup=(self.block, 1, 1),
            output_shapes=[(n,), (n,)],
            output_dtypes=[mx.int32, mx.float32],
            stream=stream,
        )

    def __call__(self, block, idx, d2, d2_32=None):
        """Fill ``idx`` (int32) and ``d2`` (float64); returns 0 fallbacks.

        ``d2_32`` (optional float32 array, one per row) receives the kernel's
        float32 distances before the upcast, for bitwise checks.
        """
        x = np.asarray(block)
        if x.ndim != 2 or x.shape[1] != self.features or x.dtype not in (np.float32, np.float64):
            raise ValueError('block must be float32 or float64 with matching features')
        if idx.shape != (len(x),) or idx.dtype != np.int32 or d2.shape != (len(x),) or d2.dtype != np.float64:
            raise ValueError('outputs must be int32 idx and float64 d2, one per row')
        if not len(x):
            return 0
        self._allocate(len(x), x.dtype)
        mx = self.mx

        def finish(slot):
            if slot['pending'] is not None:
                start, stop, out = slot['pending']
                mx.eval(out)
                idx[start:stop] = np.asarray(out[0])
                dist = np.asarray(out[1])
                d2[start:stop] = dist
                if d2_32 is not None:
                    d2_32[start:stop] = dist
                slot['pending'] = None

        try:
            for k, start in enumerate(range(0, len(x), self.capacity)):
                slot = self.slots[k % len(self.slots)]
                stream = self.streams[k % len(self.slots)]
                finish(slot)
                stop = min(start + self.capacity, len(x))
                n = stop - start
                host = slot['host']
                host[:n] = x[start:stop]
                out = self.launch(host[:n], stream)
                mx.async_eval(out)
                slot['pending'] = start, stop, out
            for slot in self.slots:
                finish(slot)
        finally:
            for stream in self.streams:
                mx.synchronize(stream)
            for slot in self.slots:
                slot['pending'] = None
        return 0


def best_match_nodes_metal(weights, data, chunk=2**20, **kwargs):
    """Float32 BMU indices (int32) and squared distances (float32 upcast to float64)."""
    from aweSOM.inference import _check_inputs

    w, x = _check_inputs(weights, data)
    search = MetalSearcher(w, len(w), 1, chunk=min(int(chunk), 2**20), **kwargs)
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x), np.float64)
    search(x, idx, d2)
    return idx, d2
