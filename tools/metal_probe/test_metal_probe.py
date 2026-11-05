"""Float32 Metal probe of the CUDA BMU search structure.

Skipped unless MLX imports and a Metal device exists. Nothing here checks
float64 arithmetic: Metal has no float64, so the CUDA kernel's bitwise float64
contract still needs a CUDA GPU (tests/gpu_test.py).
"""
import datetime
import functools
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core', reason='MLX is not installed')
if not mx.metal.is_available():
    pytest.skip('No Metal device', allow_module_level=True)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import reference as R  # noqa: E402
from metal_bmu import MetalSearcher, best_match_nodes_metal  # noqa: E402

from aweSOM import _gpu  # noqa: E402
from aweSOM.inference import _bmu_kernel  # noqa: E402
from aweSOM.stream import _BMU, train_batch_stream  # noqa: E402

FIXTURES = HERE.parent.parent / 'tests' / 'fixtures'


def run(search, x):
    idx = np.full(len(x), -1, np.int32)
    d2 = np.full(len(x), np.nan)
    d32 = np.full(len(x), np.nan, np.float32)
    assert search(x, idx, d2, d32) == 0
    np.testing.assert_array_equal(d2, d32.astype(np.float64))  # exact upcast
    return idx, d32


def assert_bitwise(w, x, idx, d32):
    ri, rd = R.bmu_f32(w, x)
    np.testing.assert_array_equal(idx, ri)
    np.testing.assert_array_equal(d32.view(np.uint32), rd.view(np.uint32))


# ----------------------------------------------------------- (a) parity

@pytest.mark.parametrize('features', [1, 2, 3, 7, 10, 31, 32])
@pytest.mark.parametrize('nodes', [1, 37, 100])
def test_reference_agrees_with_numpy(features, nodes):
    """The numba reference matches numpy's per-operation float32 rounding."""
    rng = np.random.default_rng(features * 1000 + nodes)
    w, x = rng.normal(size=(nodes, features)), rng.normal(size=(500, features))
    a, b = R.bmu_f32(w, x), R.bmu_f32_numpy(w, x)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1].view(np.uint32), b[1].view(np.uint32))


@pytest.mark.parametrize('features', [1, 2, 3, 7, 10, 31, 32])
@pytest.mark.parametrize('nodes', [1, 37, 100])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'fortran'])
def test_parity(features, nodes, dtype, layout):
    """Rows not a multiple of the block, nodes not a multiple of the tile,
    four subchunks with an odd last one, both input dtypes and layouts."""
    rng = np.random.default_rng(features * 7919 + nodes)
    w = rng.normal(size=(nodes, features))
    x = rng.normal(scale=1.5, size=(3 * 1001, features)).astype(dtype)
    if layout == 'strided':
        x = x[::3]
    elif layout == 'fortran':
        x = np.asfortranarray(x[:1001])
    else:
        x = np.ascontiguousarray(x[:1001])
    assert len(x) == 1001
    search = MetalSearcher(w, nodes, 1, chunk=300)
    idx, d32 = run(search, x)
    assert_bitwise(w, x, idx, d32)


@pytest.mark.parametrize('tile,block', [(1, 1), (1, 128), (7, 96), (32, 128), (64, 33),
                                        (256, 1024), (1000, 64)])
@pytest.mark.parametrize('overlap', [True, False])
def test_launch_shapes(tile, block, overlap):
    """Tiles smaller and larger than the node count, odd and maximal blocks."""
    features = 7 if tile * 7 * 4 <= 32768 else 1
    rng = np.random.default_rng(tile + block)
    w = rng.normal(size=(83, features))
    x = rng.normal(size=(777, features)).astype(np.float32)
    search = MetalSearcher(w, 83, 1, chunk=250, tile=tile, block=block, overlap=overlap)
    idx, d32 = run(search, x)
    assert_bitwise(w, x, idx, d32)


def test_lattice_size():
    """5824 nodes (91x64) and 7 features, as in production, in one call."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20000, 7)).astype(np.float32)
    w = x[rng.choice(len(x), 91 * 64, replace=False)].astype(np.float64)
    idx, d32 = run(MetalSearcher(w, 91, 64), x)
    assert_bitwise(w, x, idx, d32)
    # Where the margin certificate holds, the exact float64 search agrees.
    ok, arg, _ = R.certified_rows(w, x)
    assert ok.mean() > 0.99 and (idx[ok] == arg[ok]).all()


def test_staging_reuse_and_validation():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(7, 4))
    search = MetalSearcher(w, 7, 1, chunk=3)
    x = rng.normal(size=(22, 4))[::2]
    idx, d32 = run(search, x)
    assert_bitwise(w, x, idx, d32)
    slots = search.slots
    run(search, x[:2])
    assert search.slots is slots and search.capacity == 3  # no reallocation
    run(search, x.astype(np.float32))
    assert search.slots is not slots and search.dtype == np.float32
    assert all(s['pending'] is None for s in search.slots)
    empty = np.empty((0, 4))
    assert search(empty, np.empty(0, np.int32), np.empty(0)) == 0
    with pytest.raises(ValueError, match='matching features'):
        search(x[:, :3], np.empty(11, np.int32), np.empty(11))
    with pytest.raises(ValueError, match='matching features'):
        search(x.astype(np.int64), np.empty(11, np.int32), np.empty(11))
    with pytest.raises(ValueError, match='int32 idx'):
        search(x, np.empty(11, np.int64), np.empty(11))
    with pytest.raises(ValueError, match='int32 idx'):
        search(x, np.empty(11, np.int32), np.empty(11, np.float32))
    with pytest.raises(ValueError, match='1 through 32'):
        MetalSearcher(np.zeros((2, 33)), 2, 1)
    with pytest.raises(ValueError, match='lattice'):
        MetalSearcher(np.zeros((6, 3)), 2, 2)
    with pytest.raises(ValueError, match='chunk'):
        MetalSearcher(np.zeros((6, 3)), 6, 1, chunk=2**20 + 1)
    with pytest.raises(ValueError, match='device limit'):
        MetalSearcher(np.zeros((6, 32)), 6, 1, tile=257)


def test_best_match_nodes_metal_caps_chunk():
    rng = np.random.default_rng(2)
    w, x = rng.normal(size=(9, 3)), rng.normal(size=(50, 3))
    idx, d2 = best_match_nodes_metal(w, x, chunk=2**30)
    ri, rd = R.bmu_f32(w, x)
    np.testing.assert_array_equal(idx, ri)
    np.testing.assert_array_equal(d2, rd.astype(np.float64))


# ----------------------------------------------------- (b) ties and FMA

@pytest.mark.parametrize('features', [2, 3, 7, 32])
def test_first_minimum(features):
    """Each row ties exactly between its pair's two nodes; the lower index
    wins, including when the pair straddles tiles in either order."""
    rng = np.random.default_rng(features)
    w, x, expect = R.equidistant_pairs(features, pairs=40, rows_per_pair=7,
                                       extra_nodes=13, rng=rng)
    assert len(w) % 32 and len(x) % 128
    ref_idx, ref_d2 = R.bmu_f32(w, x)
    np.testing.assert_array_equal(ref_idx, expect)
    for tile in (1, 5, 32):
        idx, d32 = run(MetalSearcher(w, len(w), 1, chunk=100, tile=tile), x)
        np.testing.assert_array_equal(idx, expect)
        np.testing.assert_array_equal(d32, ref_d2)
    # The other node of each pair is at the same distance: a real tie.
    dist = ((w.astype(np.float32)[None] - x.astype(np.float32)[:, None]) ** 2).sum(-1)
    assert ((dist == ref_d2[:, None]).sum(axis=1) >= 2).all()


def test_duplicate_nodes():
    rng = np.random.default_rng(9)
    base = rng.normal(size=(5, 4))
    w = np.vstack([rng.normal(size=(33, 4)) + 50, base, base, base])  # copies at 33, 38, 43
    x = (base[rng.integers(0, 5, 333)] + 1e-3 * rng.normal(size=(333, 4))).astype(np.float32)
    idx, d32 = run(MetalSearcher(w, len(w), 1, chunk=128), x)
    assert_bitwise(w, x, idx, d32)
    assert (idx >= 33).all() and (idx < 38).all()


def test_no_fma_contraction():
    """Node 0 ties node 1 only without FMA. The guarded kernel keeps the tie
    (index 0, distance 5); the unguarded `acc + d * d` form, compiled the same
    way, reproduces the fused answer, which shows the test can see FMA."""
    rng = np.random.default_rng(11)
    cases = R.fma_cases(64, rng, features=3)
    assert any(c[3] == 1 for c in cases) and any(c[3] == 0 for c in cases)
    x = np.zeros((1, 3), np.float32)
    for w, idx0, d0, fused_idx, fused_d2 in cases:
        ri, rd = R.bmu_f32(w, x)
        assert (ri[0], rd[0]) == (idx0, d0)
        idx, d32 = run(MetalSearcher(w, 2, 1), x)
        assert (idx[0], d32[0]) == (idx0, d0)
        idx, d32 = run(MetalSearcher(w, 2, 1, contract_guard=False), x)
        assert (idx[0], d32[0]) == (fused_idx, fused_d2)


def test_subnormals_flush_to_zero():
    """Known limit, pinned so a change is noticed: Metal float32 flushes
    subnormal products to zero, where the CPU reference keeps them. The CUDA
    kernel's float64 subnormal handling cannot be probed here at all."""
    rng = np.random.default_rng(13)
    w = rng.normal(size=(37, 4)) * 1e-21
    x = (rng.normal(size=(500, 4)) * 1e-21).astype(np.float32)
    _, ref_d2 = R.bmu_f32(w, x)
    assert (ref_d2 > 0).all() and (ref_d2 < np.finfo(np.float32).tiny).all()
    _, d32 = run(MetalSearcher(w, 37, 1), x)
    assert (d32 == 0).all()


def test_random_sums_uncontracted():
    """Bitwise distances over many random sums: 23% differ under FMA."""
    rng = np.random.default_rng(12)
    w = rng.normal(size=(1, 7))
    x = rng.normal(size=(100_000, 7)).astype(np.float32)
    idx, d32 = run(MetalSearcher(w, 1, 1), x)
    assert_bitwise(w, x, idx, d32)
    _, fused = run(MetalSearcher(w, 1, 1, contract_guard=False), x)
    assert (fused != d32).mean() > 0.05


# --------------------------------------------------------- (c) training

class Recording(MetalSearcher):
    """MetalSearcher that checks each call against the exact float64 search."""

    calls = []

    def __init__(self, weights, xdim, ydim, chunk=2**20, require_certified=True):
        super().__init__(weights, xdim, ydim, chunk=chunk)
        self.require = require_certified

    def __call__(self, block, idx, d2, d2_32=None):
        result = super().__call__(block, idx, d2)
        ok, arg, _ = R.certified_rows(self.w64, block)
        exact_idx, exact_d2 = np.empty_like(idx), np.empty_like(d2)
        _bmu_kernel(self.w64, block, exact_idx, exact_d2)
        np.testing.assert_array_equal(arg, exact_idx)
        if self.require:
            assert ok.all(), f'{(~ok).sum()} rows are not certified'
        np.testing.assert_array_equal(idx, exact_idx)
        type(self).calls.append((len(block), int((~ok).sum())))
        return result  # idx and the float32-upcast d2 go to the trainer as is


def train_both(monkeypatch, data, searcher, **options):
    Recording.calls = []
    monkeypatch.setattr(_gpu, 'GpuSearcher', searcher)
    gpu, stats = train_batch_stream(data, bmu='cuda', **options)
    assert stats['bmu_kind'][0] == 'cuda'
    assert set(stats['fallbacks']) == {0}
    cpu, _ = train_batch_stream(data, bmu='numba', **options)
    return gpu, cpu


def test_training_fixture_sampled(monkeypatch):
    """scale_out_cpu fixture, 12x8 nodes, sampled init (seed 7). Every row of
    every epoch is certified, so the float32 argmin equals the float64 one and
    the trained weights must equal the stored CPU result bit for bit."""
    ref = np.load(FIXTURES / 'scale_out_cpu.npz')
    meta = json.loads((FIXTURES / 'scale_out_cpu.json').read_text())
    options = dict(meta['options'], seed=meta['sampled_seed'])
    options.pop('bmu')
    gpu, cpu = train_both(monkeypatch, ref['data'], Recording, **options)
    assert len(Recording.calls) == 12 and sum(c[1] for c in Recording.calls) == 0
    np.testing.assert_array_equal(gpu.view(np.uint64), cpu.view(np.uint64))
    np.testing.assert_array_equal(gpu.view(np.uint64), ref['expected_sampled'].view(np.uint64))


def test_training_separated(monkeypatch):
    """200k rows from 8 tight clusters 40+ apart, a 4x2 lattice started on the
    cluster centres, Metal subchunks of 7001 rows: certified by a wide margin."""
    rng = np.random.default_rng(3)
    centres = np.array([[(i >> b) & 1 for b in range(3)] + [0] * 4 for i in range(8)], float) * 40
    centres += rng.uniform(-1, 1, centres.shape)
    x = (centres[rng.integers(0, 8, 200_000)] + rng.normal(size=(200_000, 7))).astype(np.float32)
    init = centres + rng.normal(scale=0.5, size=centres.shape)
    options = dict(xdim=4, ydim=2, epochs=5, chunk=2**16, groups=8, final_pass='full',
                   prefetch=False, n_threads=8, initial_weights=init)
    searcher = functools.partial(Recording, chunk=7001)
    gpu, cpu = train_both(monkeypatch, x, searcher, **options)
    assert sum(c[1] for c in Recording.calls) == 0
    np.testing.assert_array_equal(gpu.view(np.uint64), cpu.view(np.uint64))


def test_training_fixture_observed(monkeypatch):
    """Given-init fixture run: one row in one call is inside the conservative
    bound, so equality is observed rather than guaranteed. The index check in
    Recording still runs on every row."""
    ref = np.load(FIXTURES / 'scale_out_cpu.npz')
    meta = json.loads((FIXTURES / 'scale_out_cpu.json').read_text())
    options = dict(meta['options'], initial_weights=ref['initial'])
    options.pop('bmu')
    searcher = functools.partial(Recording, require_certified=False)
    gpu, cpu = train_both(monkeypatch, ref['data'], searcher, **options)
    print('uncertified rows per call:', [c[1] for c in Recording.calls])
    np.testing.assert_array_equal(gpu.view(np.uint64), ref['expected'].view(np.uint64))


def test_training_unmodified_searcher(monkeypatch):
    """Plain MetalSearcher (float32 d2 upcast, no float64 correction) in the
    trainer: the weights depend only on idx, so they still match; the QE,
    which sums sqrt(d2), is float32-accurate only."""
    ref = np.load(FIXTURES / 'scale_out_cpu.npz')
    meta = json.loads((FIXTURES / 'scale_out_cpu.json').read_text())
    options = dict(meta['options'], seed=meta['sampled_seed'])
    options.pop('bmu')
    monkeypatch.setattr(_gpu, 'GpuSearcher', MetalSearcher)
    gpu, stats = train_batch_stream(ref['data'], bmu='cuda', **options)
    cpu, cstats = train_batch_stream(ref['data'], bmu='numba', **options)
    np.testing.assert_array_equal(gpu.view(np.uint64), cpu.view(np.uint64))
    np.testing.assert_allclose(stats['qe'], cstats['qe'], rtol=1e-5)


# ----------------------------------------------------------- (d) timing

def _rate(fn, rows, repeat=1):
    fn()  # warm-up (compilation, staging allocation)
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return rows / best, best


@pytest.mark.skipif(os.environ.get('METAL_PROBE_TIMING') == '0', reason='METAL_PROBE_TIMING=0')
def test_timing():
    import numba

    rng = np.random.default_rng(0)
    nodes, features = 91 * 64, 7
    sizes = [10**6, 10**7]
    x_all = rng.normal(size=(sizes[-1], features)).astype(np.float32)
    w = x_all[rng.choice(10**6, nodes, replace=False)].astype(np.float64)
    old = numba.get_num_threads()
    numba.set_num_threads(min(8, numba.config.NUMBA_NUM_THREADS))
    rows_out = []
    try:
        for n in sizes:
            x = x_all[:n]
            idx, d2 = np.empty(n, np.int32), np.empty(n)
            metal = MetalSearcher(w, 91, 64)
            serial = MetalSearcher(w, 91, 64, overlap=False)
            screen = _BMU(w, 'screen', 91, 64)
            entry = dict(rows=n)
            entry['metal_rows_per_s'], entry['metal_s'] = _rate(lambda: metal(x, idx, d2), n, 3)
            entry['metal_serial_rows_per_s'], _ = _rate(lambda: serial(x, idx, d2), n, 3)
            m = min(n, 2**20)  # kernel rows never exceed the staged input
            xm = mx.array(x[:m])
            mx.eval(xm)
            grid = -(-m // 128) * 128
            kernel_only = lambda: mx.eval(metal.kernel(  # noqa: E731
                inputs=[metal.w, xm, m, nodes, 32], template=[('F', 7), ('TILE', 32)],
                grid=(grid, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(m,), (m,)],
                output_dtypes=[mx.int32, mx.float32]))
            entry['metal_kernel_only_rows_per_s'], _ = _rate(kernel_only, m, 5)
            entry['numba_bruteforce_f64_rows_per_s'], entry['numba_bruteforce_s'] = \
                _rate(lambda: _bmu_kernel(w, x, idx, d2), n, 1)
            entry['numba_screen_rows_per_s'], _ = _rate(lambda: screen(x, idx, d2), n, 3)
            rows_out.append(entry)
    finally:
        numba.set_num_threads(old)
    meta = dict(date=str(datetime.date.today()), machine=platform.machine(),
                device=mx.device_info().get('device_name'), mlx=mx.__version__,
                numba=numba.__version__, numpy=np.__version__, python=platform.python_version(),
                numba_threads=min(8, numba.config.NUMBA_NUM_THREADS), nodes=nodes, features=features,
                input='float32 normal(0, 1); weights sampled from the rows', results=rows_out)
    path = HERE / f'results-{meta["date"]}.json'
    path.write_text(json.dumps(meta, indent=1) + '\n')
    for e in rows_out:
        print(f"rows={e['rows']:.0e} metal={e['metal_rows_per_s'] / 1e6:.2f} Mrows/s "
              f"(serial {e['metal_serial_rows_per_s'] / 1e6:.2f}, kernel only "
              f"{e['metal_kernel_only_rows_per_s'] / 1e6:.2f}) numba f64 brute force "
              f"{e['numba_bruteforce_f64_rows_per_s'] / 1e6:.3f} Mrows/s, numba screen "
              f"{e['numba_screen_rows_per_s'] / 1e6:.2f} Mrows/s")
