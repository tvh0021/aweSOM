"""Run the generated CUDA screen kernel source on the CPU, one thread per CUDA thread.

The kernel text from :func:`aweSOM._gpu_screen._source` is compiled as C++
with a small prelude: CUDA intrinsics map to IEEE float and double
operations (``__fmaf_rn`` to ``std::fmaf``, correctly rounded), barriers to
``std::barrier``, and shared memory to one buffer per block, filled with NaN
bytes before each block. Contraction is off, as with ``--fmad=false``. This
checks the row blocking, tiling, float4 staging, barriers and fallback on
real kernel code without a GPU: labels and distances must equal the float64
reference bitwise, and the number of certified rows must equal the CPU
emulation's. Skipped without ``clang++`` or ``g++``.
"""
import re
import shutil
import subprocess

import numpy as np
import pytest

from aweSOM import _gpu_screen
from aweSOM.inference import best_match_nodes

CXX = shutil.which('clang++') or shutil.which('g++')
pytestmark = pytest.mark.skipif(CXX is None, reason='no C++ compiler')

PRELUDE = r'''
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
struct float4 { float x, y, z, w; };
struct Dim { unsigned x; };
static thread_local Dim threadIdx;
static Dim blockIdx, blockDim;
static std::barrier<>* g_bar;
static std::atomic<int> g_or{0};
static std::atomic<long> g_certified{0};
static float4* g_smem;
static int g_safe;
static double g_wmax;
#define __device__
#define __global__
static inline void __syncthreads() { g_bar->arrive_and_wait(); }
static inline int __syncthreads_or(int p) {
    if (p) g_or.fetch_or(1);
    g_bar->arrive_and_wait();
    int r = g_or.load();
    g_bar->arrive_and_wait();
    if (threadIdx.x == 0) g_or.store(0);
    g_bar->arrive_and_wait();
    return r;
}
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmaf_rn(float a, float b, float c) { return std::fmaf(a, b, c); }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __ddiv_rn(double a, double b) { return a / b; }
static inline double __dsqrt_rn(double a) { return std::sqrt(a); }
static inline float __double2float_rn(double a) { return (float)a; }
static inline double __longlong_as_double(long long v) { double d; std::memcpy(&d, &v, 8); return d; }
using std::min;
'''

DRIVER = r'''
static std::vector<char> load(const char* path) {
    FILE* f = std::fopen(path, "rb");
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> data(size);
    if (std::fread(data.data(), 1, size, f) != (size_t)size) std::abort();
    std::fclose(f);
    return data;
}
int main(int argc, char** argv) {
    int n = std::atoi(argv[1]), nodes = std::atoi(argv[2]), is64 = std::atoi(argv[3]);
    int block = std::atoi(argv[4]), tile = std::atoi(argv[5]), rows_per_thread = std::atoi(argv[6]);
    long shared = std::atol(argv[7]);
    g_safe = std::atoi(argv[8]);
    g_wmax = std::strtod(argv[9], nullptr);
    std::vector<char> w = load(argv[10]), x = load(argv[11]);
    std::vector<int> idx(n, -1);
    std::vector<double> d2(n, -1.0);
    std::vector<float4> smem(shared / 16 + 1);
    g_smem = smem.data();
    blockDim.x = block;
    int per = block * rows_per_thread;
    for (int b = 0; b < (n + per - 1) / per; ++b) {
        blockIdx.x = b;
        std::memset(smem.data(), 0xff, smem.size() * sizeof(float4));  // NaN
        std::barrier<> bar(block);
        g_bar = &bar;
        std::vector<std::thread> threads;
        for (int t = 0; t < block; ++t)
            threads.emplace_back([&, t] {
                threadIdx.x = t;
                if (is64)
                    bmu_screen_f64((const double*)w.data(), (const double*)x.data(), idx.data(),
                                   d2.data(), n, nodes, tile);
                else
                    bmu_screen_f32((const double*)w.data(), (const float*)x.data(), idx.data(),
                                   d2.data(), n, nodes, tile);
            });
        for (auto& t : threads) t.join();
    }
    FILE* f = std::fopen(argv[12], "wb");
    std::fwrite(idx.data(), sizeof(int), n, f);
    std::fwrite(d2.data(), sizeof(double), n, f);
    std::fclose(f);
    std::printf("%ld\n", g_certified.load());
    return 0;
}
'''


def host_source(features):
    source = _gpu_screen._source(features, True, 1.0)
    source = re.sub(r'#define AWESOM_SAFE .*', '#define AWESOM_SAFE g_safe', source)
    source = re.sub(r'#define AWESOM_WMAX .*', '#define AWESOM_WMAX g_wmax', source)
    for old, new in [('extern __shared__ float4 smem[];', 'float4* smem = g_smem;'),
                     ('certified = lo2 > hi1;', 'certified = lo2 > hi1; if (certified) ++g_certified;')]:
        assert source.count(old) >= 1, old
        source = source.replace(old, new)
    return PRELUDE + source + DRIVER


@pytest.fixture(scope='module')
def binaries(tmp_path_factory):
    root = tmp_path_factory.mktemp('screen_host')
    built = {}

    def build(features):
        if features not in built:
            src = root / f'screen{features}.cpp'
            src.write_text(host_source(features))
            exe = root / f'screen{features}'
            subprocess.run([CXX, '-std=c++20', '-O1', '-ffp-contract=off', '-pthread',
                            '-o', str(exe), str(src)], check=True, capture_output=True)
            built[features] = exe
        return built[features]
    return root, build


def run(binaries, w, x, block, tile):
    root, build = binaries
    features = w.shape[1]
    rows_per_thread, pad = _gpu_screen._layout(features)
    safe, wmax = _gpu_screen._screen_parameters(w)
    np.ascontiguousarray(w, np.float64).tofile(root / 'w.bin')
    np.ascontiguousarray(x).tofile(root / 'x.bin')
    shared = tile * max(8 * features, 4 * pad)
    out = subprocess.run([str(build(features)), str(len(x)), str(len(w)), str(int(x.dtype == np.float64)),
                          str(block), str(tile), str(rows_per_thread), str(shared), str(int(safe)),
                          repr(wmax), str(root / 'w.bin'), str(root / 'x.bin'), str(root / 'out.bin')],
                         check=True, capture_output=True, text=True)
    raw = np.fromfile(root / 'out.bin', np.uint8)
    idx = raw[:4 * len(x)].view(np.int32)
    d2 = raw[4 * len(x):].view(np.float64)
    return idx, d2, int(out.stdout)


def case(name, features, n, dtype, seed=5):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(73, features))
    x = rng.normal(size=(n, features))
    if name == 'smooth':  # trained-like lattice, rows near nodes: mostly certified
        a, b = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 8))
        w = np.column_stack([np.sin(a.ravel() * (f + 1)) + b.ravel() for f in range(features)])
        x = w[rng.integers(len(w), size=n)] + rng.normal(size=(n, features)) * 0.05
    elif name == 'mixed':  # duplicate nodes, exact ties, near-ties, NaN and huge rows
        w[1] = w[0]
        w[5] = w[4]
        w[5, 0] = np.nextafter(w[4, 0], np.inf)
        x[:7] = w[0]
        x[7:11] = w[4]
        x[11] = np.nan
        x[12] = np.inf
        x[13] = 1e30
        x[14:17] = (w[2] + w[3]) / 2  # equidistant from two nodes
    elif name == 'large':
        w *= 1e12
        x *= 1e12
    elif name == 'unsafe':
        w[3] = np.nan
    return w, x.astype(dtype)


@pytest.mark.parametrize('features', [1, 3, 7, 8, 10, 13, 32])
@pytest.mark.parametrize('name', ['random', 'smooth', 'mixed', 'large', 'unsafe'])
def test_kernel_source_on_host(binaries, features, name):
    for dtype, n, block, tile in [(np.float32, 301, 32, 128), (np.float64, 157, 8, 16)]:
        w, x = case(name, features, n, dtype)
        idx, d2, certified = run(binaries, w, x, block, tile)
        ri, rd = best_match_nodes(w, x)
        np.testing.assert_array_equal(idx, ri)
        np.testing.assert_array_equal(d2.view(np.uint64), rd.view(np.uint64))
        _, _, fallbacks = _gpu_screen._cuda_screen_reference(w, x, return_fallbacks=True)
        assert certified == len(x) - fallbacks
        if name == 'smooth':
            assert certified > len(x) // 2  # the screen path, not only the fallback, is exercised
        if name == 'unsafe':
            assert certified == 0


def test_source_contract():
    for features in [1, 7, 10, 32]:
        source = _gpu_screen._source(features, True, 2.5)
        rows, pad = _gpu_screen._layout(features)
        assert '#include' not in source
        assert '__fmaf_rn(d, d, acc)' in source
        assert f'#define AWESOM_R {rows}\n' in source and f'#define AWESOM_P {pad}\n' in source
        assert pad % 4 == 0 and features <= pad < features + 4
        assert '#define AWESOM_WMAX 2.50000000000000000e+00' in source
        # every row of a block is visited exactly once
        block = 8
        covered = sorted(b * block * rows + t + r * block
                         for b in range(3) for t in range(block) for r in range(rows))
        assert covered == list(range(3 * block * rows))
