"""Run the bucketed-search kernel source on the CPU, one thread per CUDA thread.

The text from :func:`aweSOM._gpu_bucket._source` is compiled as C++ with the
prelude of ``gpu_screen_host_test`` plus:

* warps: ``__match_any_sync``, ``__shfl_sync`` and ``__shfl_xor_sync`` exchange
  values through per-warp slots between two per-warp barriers (every kernel
  calls them with all 32 lanes, as on the device);
* ``atomicAdd`` on ints, ``__popc``, ``__ffs``, and ``__shared__`` variables
  as statics (one block runs at a time);
* ``__fadd_ru``, ``__fmul_ru`` and ``__double2float_ru`` computed in the
  upward rounding mode (``-frounding-math``, per-thread ``fesetround``).

The driver launches the seven kernels in the device order (seed, histogram,
scan, scatter, bucketed pass, fb_screen, fb_exact) with small blocks and a
small fixed fallback grid, so rows cross blocks, tiles, warps and grid
strides. Labels and distances must equal the float64 reference bitwise; the
seeds, bucket counts and the three counters must equal the CPU models', and
``perm`` must list every row once, grouped by bucket. Skipped without
``clang++`` or ``g++``.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from gpu_bucket_test import _bits, lattice, rows
from gpu_screen_host_test import PRELUDE

from aweSOM import _gpu_bucket as B
from aweSOM.inference import best_match_nodes

CXX = shutil.which('clang++') or shutil.which('g++')
# On GitHub's macOS runners one OS thread per CUDA thread takes 5-9 min per case
# (about 2 s on Linux CI and on a local Mac), so the file runs on Linux CI only.
pytestmark = [
    pytest.mark.skipif(CXX is None, reason='no C++ compiler'),
    pytest.mark.skipif(sys.platform == 'darwin' and bool(os.environ.get('CI')),
                       reason='too slow on macOS CI runners; Linux CI runs it'),
]

EXTRA = r'''
#include <cfenv>
#include <memory>
static Dim gridDim;
static std::barrier<>* g_wbar[32];
static long long g_wslot[32][32];
#define __forceinline__ inline
#define __shared__ static
using std::nextafter;
static inline float __fadd_ru(float a, float b) {
    const int mode = std::fegetround(); std::fesetround(FE_UPWARD);
    volatile float va = a, vb = b; volatile float r = va + vb;
    std::fesetround(mode); return r;
}
static inline float __fmul_ru(float a, float b) {
    const int mode = std::fegetround(); std::fesetround(FE_UPWARD);
    volatile float va = a, vb = b; volatile float r = va * vb;
    std::fesetround(mode); return r;
}
static inline float __double2float_ru(double a) {
    const int mode = std::fegetround(); std::fesetround(FE_UPWARD);
    volatile double va = a; volatile float r = (float)va;
    std::fesetround(mode); return r;
}
static inline int __popc(unsigned v) { return __builtin_popcount(v); }
static inline int __ffs(unsigned v) { return __builtin_ffs((int)v); }
static inline int atomicAdd(int* p, int v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
template <class T> static T warp_xchg(T v, int src) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    long long slot = 0;
    std::memcpy(&slot, &v, sizeof(T));
    g_wslot[w][l] = slot;
    g_wbar[w]->arrive_and_wait();
    long long o = g_wslot[w][src & 31];
    T r;
    std::memcpy(&r, &o, sizeof(T));
    g_wbar[w]->arrive_and_wait();
    return r;
}
template <class T> static T __shfl_sync(unsigned, T v, int src) { return warp_xchg(v, src); }
template <class T> static T __shfl_xor_sync(unsigned, T v, int off) { return warp_xchg(v, (int)(threadIdx.x & 31) ^ off); }
static unsigned __match_any_sync(unsigned, int v) {
    const int w = threadIdx.x >> 5;
    g_wslot[w][threadIdx.x & 31] = v;
    g_wbar[w]->arrive_and_wait();
    unsigned m = 0;
    for (int l = 0; l < 32; ++l)
        if ((int)g_wslot[w][l] == v) m |= 1u << l;
    g_wbar[w]->arrive_and_wait();
    return m;
}
'''

DRIVER = r'''
static std::vector<char> load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) std::abort();
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> data(size + 16);
    if (std::fread(data.data(), 1, size, f) != (size_t)size) std::abort();
    std::fclose(f);
    return data;
}
static const long SMEM = 1 << 17;
static const int GUARD = 64;
static std::vector<float4> g_buf(SMEM / 16 + GUARD);

template <class K> static void launch(int grid, int block, long shared, K kern) {
    if (shared > SMEM || block % 32) std::abort();
    gridDim.x = grid;
    blockDim.x = block;
    g_smem = g_buf.data();
    for (int b = 0; b < grid; ++b) {
        blockIdx.x = b;
        std::memset(g_buf.data(), 0xff, SMEM);  // NaN
        unsigned char* guard = (unsigned char*)g_buf.data() + shared;
        std::memset(guard, 0x5a, GUARD * 16);
        std::barrier<> bar(block);
        g_bar = &bar;
        std::vector<std::unique_ptr<std::barrier<>>> warps;
        for (int w = 0; w < block / 32; ++w) {
            warps.emplace_back(new std::barrier<>(32));
            g_wbar[w] = warps.back().get();
        }
        std::vector<std::thread> threads;
        for (int t = 0; t < block; ++t)
            threads.emplace_back([&, t] { threadIdx.x = t; kern(); });
        for (auto& t : threads) t.join();
        for (int k = 0; k < GUARD * 16; ++k)
            if (guard[k] != 0x5a) { std::fprintf(stderr, "shared memory overrun\n"); std::exit(3); }
    }
}

int main(int argc, char** argv) {
    std::string dir = argv[1];
    int a = 2;
    const int n = std::atoi(argv[a++]), nodes = std::atoi(argv[a++]), P = std::atoi(argv[a++]);
    const int is64 = std::atoi(argv[a++]), block = std::atoi(argv[a++]), ct = std::atoi(argv[a++]);
    const int fbgrid = std::atoi(argv[a++]), fbblock = std::atoi(argv[a++]);
    const double wmax_c = std::strtod(argv[a++], nullptr), cmax = std::strtod(argv[a++], nullptr);
    const float g1 = std::strtof(argv[a++], nullptr);
    const int safe_c = std::atoi(argv[a++]);
    const double eta = std::strtod(argv[a++], nullptr), one_m_eps = std::strtod(argv[a++], nullptr);
    const double fb_wmax = std::strtod(argv[a++], nullptr);
    const int fb_safe = std::atoi(argv[a++]);
    const long sh_seed = std::atol(argv[a++]), sh_pass = std::atol(argv[a++]);
    const long sh_fbs = std::atol(argv[a++]), sh_fbe = std::atol(argv[a++]);
    auto w = load(dir + "/w.bin"), x = load(dir + "/x.bin"), c32 = load(dir + "/c32.bin"),
         cw = load(dir + "/cw.bin"), cid = load(dir + "/cid.bin"), clo = load(dir + "/cand_lo.bin"),
         chi = load(dir + "/cand_hi.bin"), ex32 = load(dir + "/ex32.bin"), elo = load(dir + "/ex_lo.bin"),
         ehi = load(dir + "/ex_hi.bin"), rare = load(dir + "/rare32.bin");
    std::vector<int> seed(n, -7), perm(n, -7), fb_rows(n, -7), list2(n, -7), zbuf(4 + P, 12345),
        boff(P + 1, -7), toff(P + 1, -7), cursor(P, -7), idx(n, -1);
    std::vector<double> d2(n, -2.0);
    const double* W = (const double*)w.data();
    const float* C32 = (const float*)c32.data();
    const float* CW = (const float*)cw.data();
    const int* CID = (const int*)cid.data();
    const int* CLO = (const int*)clo.data();
    const int* CHI = (const int*)chi.data();
    const float* EX = (const float*)ex32.data();
    const int* ELO = (const int*)elo.data();
    const int* EHI = (const int*)ehi.data();
    const int* RARE = (const int*)rare.data();
    const int R = AWESOM_R, rb = block * R;
    const int hist_block = 64, scatter_block = 64;
    auto run = [&](auto* X) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(X)>>;
        launch((n + 64 * R - 1) / (64 * R), 64, sh_seed, [&] {
            if constexpr (sizeof(T) == 8) seed_f64(C32, X, seed.data(), n, P, zbuf.data(), 4 + P);
            else seed_f32(C32, X, seed.data(), n, P, zbuf.data(), 4 + P);
        });
        launch(std::min((n + hist_block - 1) / hist_block, 2), hist_block, P * 4,
               [&] { bucket_hist(seed.data(), n, P, zbuf.data() + 4); });
        launch(1, 1024, 0, [&] { bucket_scan(zbuf.data() + 4, P, rb, boff.data(), cursor.data(), toff.data()); });
        launch((n + scatter_block * SR - 1) / (scatter_block * SR), scatter_block, 2 * P * 4,
               [&] { bucket_scatter(seed.data(), n, P, cursor.data(), perm.data()); });
        launch((n + rb - 1) / rb + P, block, sh_pass, [&] {
            if constexpr (sizeof(T) == 8)
                bpass_f64(W, X, perm.data(), boff.data(), toff.data(), CW, CID, CLO, CHI, EX, ELO, EHI, idx.data(),
                          d2.data(), fb_rows.data(), zbuf.data(), RARE, P, ct, wmax_c, cmax, g1, safe_c, eta, one_m_eps);
            else
                bpass_f32(W, X, perm.data(), boff.data(), toff.data(), CW, CID, CLO, CHI, EX, ELO, EHI, idx.data(),
                          d2.data(), fb_rows.data(), zbuf.data(), RARE, P, ct, wmax_c, cmax, g1, safe_c, eta, one_m_eps);
        });
        launch(fbgrid, fbblock, sh_fbs, [&] {
            if constexpr (sizeof(T) == 8)
                fb_screen_f64(W, X, fb_rows.data(), zbuf.data(), idx.data(), d2.data(), list2.data(), zbuf.data() + 2,
                              nodes, fb_wmax, fb_safe);
            else
                fb_screen_f32(W, X, fb_rows.data(), zbuf.data(), idx.data(), d2.data(), list2.data(), zbuf.data() + 2,
                              nodes, fb_wmax, fb_safe);
        });
        launch(fbgrid, fbblock, sh_fbe, [&] {
            if constexpr (sizeof(T) == 8)
                fb_exact_f64(W, X, list2.data(), zbuf.data() + 2, idx.data(), d2.data(), nodes);
            else
                fb_exact_f32(W, X, list2.data(), zbuf.data() + 2, idx.data(), d2.data(), nodes);
        });
    };
    if (is64) run((const double*)x.data()); else run((const float*)x.data());
    FILE* f = std::fopen((dir + "/out.bin").c_str(), "wb");
    std::fwrite(idx.data(), sizeof(int), n, f);
    std::fwrite(d2.data(), sizeof(double), n, f);
    std::fwrite(seed.data(), sizeof(int), n, f);
    std::fwrite(perm.data(), sizeof(int), n, f);
    std::fwrite(zbuf.data(), sizeof(int), 4 + P, f);
    std::fclose(f);
    return 0;
}
'''


def host_source(F):
    source = B._source(F)
    for old, new in [('extern __shared__ float4 smem[];', 'float4* smem = g_smem;'),
                     ('extern __shared__ int sh[];', 'int* sh = (int*)g_smem;'),
                     ('extern __shared__ double fsd[];', 'double* fsd = (double*)g_smem;'),
                     ('extern __shared__ float fsf[];', 'float* fsf = (float*)g_smem;')]:
        assert old in source, old
        source = source.replace(old, new)
    assert 'extern __shared__' not in source
    return PRELUDE + EXTRA + source + DRIVER


@pytest.fixture(scope='module')
def build(tmp_path_factory):
    root = tmp_path_factory.mktemp('bucket_host')
    built = {}

    def get(F):
        if F not in built:
            src = root / f'bucket{F}.cpp'
            src.write_text(host_source(F))
            exe = root / f'bucket{F}'
            res = subprocess.run([CXX, '-std=c++20', '-O1', '-ffp-contract=off', '-frounding-math', '-pthread',
                                  '-o', str(exe), str(src)], capture_output=True, text=True)
            assert res.returncode == 0, res.stderr[-4000:]
            built[F] = exe
        return root, built[F]
    return get


def run(build, w, x, T, block=64, ct=16, fbgrid=3, fbblock=64):
    F = w.shape[1]
    root, exe = build(F)
    sc = T['sc']
    for name in ('c32', 'cw', 'cid', 'cand_lo', 'cand_hi', 'ex32', 'ex_lo', 'ex_hi', 'rare32'):
        np.ascontiguousarray(T[name]).tofile(root / f'{name}.bin')
    np.ascontiguousarray(w, np.float64).tofile(root / 'w.bin')
    np.ascontiguousarray(x).tofile(root / 'x.bin')
    safe, wmax = B.screen_parameters(w)
    pad, px = B._pads(F)
    P, n = T['P'], len(x)
    args = [n, len(w), P, int(x.dtype == np.float64), block, ct, fbgrid, fbblock, repr(sc['wmax']), repr(sc['cmax']),
            repr(float(sc['G1'])), int(sc['safe']), repr(B.ETA), repr(1.0 - B._feature_eps(F)), repr(wmax), int(safe),
            P * pad * 4, ct * max(pad * 4 + 4, px * 4), B.FT * F * 4, B.FT * F * 8]
    res = subprocess.run([str(exe), str(root)] + [str(v) for v in args], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    raw = np.fromfile(root / 'out.bin', np.uint8)
    o = 0
    out = {}
    for name, dt, m in (('idx', np.int32, n), ('d2', np.float64, n), ('seed', np.int32, n), ('perm', np.int32, n),
                        ('zbuf', np.int32, 4 + P)):
        size = np.dtype(dt).itemsize * m
        out[name] = raw[o:o + size].view(dt)
        o += size
    return out


def check(build, w, x, T, **kw):
    out = run(build, w, x, T, **kw)
    ri, rd = best_match_nodes(w, x)
    np.testing.assert_array_equal(out['idx'], ri)
    np.testing.assert_array_equal(_bits(out['d2']), _bits(rd))
    mi, md, info = B._bucket_reference(w, x, T)
    P = T['P']
    np.testing.assert_array_equal(out['seed'], info['seed'])
    counts = np.bincount(info['seed'], minlength=P)
    np.testing.assert_array_equal(out['zbuf'][4:], counts)
    assert out['zbuf'][0] == (info['code'] != 0).sum()
    assert out['zbuf'][1] == (info['code'] == 1).sum()
    assert out['zbuf'][2] == len(info['screen_left'])
    assert out['zbuf'][3] == ((info['code'] != 0) & T['rare'][info['seed']]).sum()  # fallback in rare buckets
    perm = out['perm']
    np.testing.assert_array_equal(np.sort(perm), np.arange(len(x)))
    np.testing.assert_array_equal(info['seed'][perm], np.repeat(np.arange(P), counts))
    return info


@pytest.mark.parametrize('F', [1, 3, 5, 10])
@pytest.mark.parametrize('kind', ['trained', 'dup', 'tied', 'edge'])
def test_kernel_source_on_host(build, monkeypatch, F, kind):
    monkeypatch.setattr(B, 'LEAF', 16)
    w = lattice(kind, F)
    with np.errstate(over='ignore', invalid='ignore'):
        x64 = rows(w, 160)
    for dtype in (np.float32, np.float64):
        with np.errstate(over='ignore'):
            x = x64.astype(dtype)
        T = B.build_tables(w, x[::2])
        info = check(build, w, x, T)
        if kind == 'trained' and F >= 3:
            assert (info['code'] == 0).sum() > len(x) // 2  # the bucketed pass decides most rows


@pytest.mark.parametrize('lists', ['random', 'rare'])
def test_kernel_source_any_lists(build, monkeypatch, lists):
    monkeypatch.setattr(B, 'LEAF', 16)
    F = 5
    w = lattice('dup', F)
    x = rows(w, 160)
    grp = B.build_groups(w)
    sc = B.screen_scalars(w, grp)
    P = grp['P']
    rng = np.random.default_rng(11)
    if lists == 'random':
        K, rare = rng.integers(1, P + 1, size=P), rng.random(P) < 0.3
    else:
        K, rare = np.full(P, P), np.ones(P, bool)
    T = B.pack(w, grp, sc, K, rare)
    T.update(sc=sc, grp=grp)
    info = check(build, w, x, T, block=32, ct=8, fbgrid=2, fbblock=32)
    assert ((info['code'] != 0) & rare[info['seed']]).any()  # the overlap counter is exercised


def test_lane_ties_in_fallback(build, monkeypatch):
    """Equal float64 distances in different lanes and fallback tiles: the lowest index wins."""
    monkeypatch.setattr(B, 'LEAF', 16)
    F = 3
    rng = np.random.default_rng(2)
    w = rng.normal(size=(600, F))
    for j in (5, 37, 300, 301, 599):  # lanes 5, 5, 12, 13, 23; tiles 0, 0, 1, 1, 2
        w[j] = w[0]
    w[290] = w[33] = w[2] = [1.0, 1.0, 1.0]  # copies across tiles; the rows below sit between pairs
    x = np.concatenate([np.repeat(w[:1], 40, axis=0), np.full((40, F), 1.0),
                        np.repeat([[0.0, 0.0, 0.0]], 5, axis=0)])
    T = B.build_tables(w, x)
    info = check(build, w, x, T, fbgrid=2)
    assert (info['code'] != 0).sum() >= 80  # ties never certify: they reach the fallback kernels
