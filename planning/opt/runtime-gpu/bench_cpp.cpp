// Benchmark of exact BMU kernels against awesom::best_match (the reference).
// Usage: bench_cpp W.bin X.bin ref_idx.bin ref_d2.bin F N_single reps threads
// W.bin: nodes x F float64, X.bin: N x F float64 (already scaled), ref_*: numba output.
// Prints one JSON line.
#pragma STDC FP_CONTRACT OFF
#include "../../../cpp/awesom_classify.hpp"
#include "fast_classify.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

template <class T>
static std::vector<T> read_bin(const char* path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  std::size_t bytes = in.tellg();
  in.seekg(0);
  std::vector<T> v(bytes / sizeof(T));
  in.read(reinterpret_cast<char*>(v.data()), bytes);
  return v;
}

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// GPU divergence proxy: in SIMT a warp of 32 neighbouring cells executes the union of
// the blocks its threads visit. Returns {mean blocks per point, mean union per warp}.
template <int FF>
static void warp_union(const awesom_fast::BlockTable& t, const double* X, std::size_t n, double& per_pt, double& per_warp) {
  long long vis = 0, uni = 0, warps = 0;
  std::vector<char> mark(t.nb);
  for (std::size_t w0 = 0; w0 + 32 <= n; w0 += 32) {
    std::fill(mark.begin(), mark.end(), 0);
    for (std::size_t p = w0; p < w0 + 32; ++p) {
      const double* x = X + p * FF;
      int bmin = 0;
      double lbmin = std::numeric_limits<double>::infinity();
      for (int b = 0; b < t.nb; ++b) {
        double l = t.block_lb<FF>(b, x);
        if (l < lbmin) { lbmin = l; bmin = b; }
      }
      double best = std::numeric_limits<double>::infinity();
      int bidx = INT_MAX;
      t.eval_block<FF>(bmin, x, best, bidx);
      mark[bmin] = 1;
      ++vis;
      for (int b = 0; b < t.nb; ++b) {
        if (b == bmin || t.block_lb<FF>(b, x) > best) continue;
        t.eval_block<FF>(b, x, best, bidx);
        mark[b] = 1;
        ++vis;
      }
    }
    for (char c : mark) uni += c;
    ++warps;
  }
  per_pt = static_cast<double>(vis) / (32.0 * warps);
  per_warp = static_cast<double>(uni) / warps;
}

int main(int argc, char** argv) {
  if (argc < 10) {
    std::fprintf(stderr, "usage: W X ref_idx ref_d2 F N_single reps threads N_par\n");
    return 1;
  }
  const int F = std::atoi(argv[5]);
  std::vector<double> W = read_bin<double>(argv[1]);
  std::vector<double> X = read_bin<double>(argv[2]);
  std::vector<int> ref_idx = read_bin<int>(argv[3]);
  std::vector<double> ref_d2 = read_bin<double>(argv[4]);
  const std::size_t N = X.size() / F;
  const std::size_t Ns = std::min<std::size_t>(std::strtoull(argv[6], nullptr, 10), N);
  const int reps = std::atoi(argv[7]);
  const int threads = std::atoi(argv[8]);
  const std::size_t Np = std::min<std::size_t>(std::strtoull(argv[9], nullptr, 10), N);
#ifdef _OPENMP
  omp_set_num_threads(threads);
#endif

  awesom::SOMTable m;
  m.F = F;
  m.nodes = static_cast<int>(W.size() / F);
  m.weights = W;

  awesom_fast::SoATable soa(W.data(), m.nodes, F);
  awesom_fast::BlockTable srt(W.data(), m.nodes, F);

  // kernels: 0 ref, 1 soa, 2 block bound, 3 block bound without bound array (GPU form), 4 two-level (previous point's BMU)
  const char* names[] = {"ref_aos_scalar", "soa_simd", "block_bound", "block_bound_recompute", "block_bound_2level"};
  const int K = 5;
  auto run = [&](int k, std::size_t n, int* idx, double* d2, bool parallel) {
    const long long nn = static_cast<long long>(n);
    if (k == 0) {
#pragma omp parallel for schedule(static) if (parallel)
      for (long long p = 0; p < nn; ++p) idx[p] = awesom::best_match(m, &X[p * F], d2[p]);
    } else if (k == 1) {
#pragma omp parallel for schedule(static) if (parallel)
      for (long long p = 0; p < nn; ++p) idx[p] = soa.best_match(&X[p * F], d2[p]);
    } else if (k == 2) {
#pragma omp parallel for schedule(static) if (parallel)
      for (long long p = 0; p < nn; ++p) idx[p] = srt.best_match(&X[p * F], d2[p], -1);
    } else if (k == 3) {
#pragma omp parallel for schedule(static) if (parallel)
      for (long long p = 0; p < nn; ++p) idx[p] = srt.best_match_recompute(&X[p * F], d2[p]);
    } else {
#pragma omp parallel for schedule(static) if (parallel)
      for (long long p = 0; p < nn; ++p) idx[p] = srt.best_match2(&X[p * F], d2[p]);
    }
  };

  std::vector<int> idx(N);
  std::vector<double> d2(N);
  // correctness on the full data set (parallel), against numba
  long long mism_idx[K], mism_d2[K];
  for (int k = 0; k < K; ++k) {
    run(k, N, idx.data(), d2.data(), true);
    mism_idx[k] = mism_d2[k] = 0;
    for (std::size_t p = 0; p < N; ++p) {
      mism_idx[k] += idx[p] != ref_idx[p];
      mism_d2[k] += std::memcmp(&d2[p], &ref_d2[p], sizeof(double)) != 0;
    }
  }
  // timing: single thread on the first Ns rows, then all threads on all rows; interleaved
  std::vector<double> t1[K], tp[K];
  for (int r = -1; r < reps; ++r) {
    for (int k = 0; k < K; ++k) {
      double t0 = now();
      run(k, Ns, idx.data(), d2.data(), false);
      double a = now() - t0;
      t0 = now();
      run(k, Np, idx.data(), d2.data(), true);
      double b = now() - t0;
      if (r >= 0) {
        t1[k].push_back(a);
        tp[k].push_back(b);
      }
    }
  }
  // pruning statistics of the sorted kernel (nodes evaluated per point)
  double visited_cold = srt.mean_visited(X.data(), Ns, false);
  double visited_warm = srt.mean_visited(X.data(), Ns, true);
  double wp = 0, ww = 0;
  if (F == 4) warp_union<4>(srt, X.data(), Ns, wp, ww);
  if (F == 6) warp_union<6>(srt, X.data(), Ns, wp, ww);
  if (F == 10) warp_union<10>(srt, X.data(), Ns, wp, ww);

  std::printf("{\"N\": %zu, \"N_single\": %zu, \"N_parallel\": %zu, \"F\": %d, \"nodes\": %d, \"threads\": %d, \"reps\": %d",
              N, Ns, Np, F, m.nodes, threads, reps);
  for (int k = 0; k < K; ++k) {
    std::printf(", \"%s\": {\"pts_per_s_1thread\": %.4g, \"pts_per_s_parallel\": %.4g, "
                "\"ratio_1thread\": %.3f, \"ratio_parallel\": %.3f, \"idx_mismatch\": %lld, \"d2_bit_mismatch\": %lld}",
                names[k], Ns / median(t1[k]), Np / median(tp[k]), median(t1[0]) / median(t1[k]),
                median(tp[0]) / median(tp[k]), mism_idx[k], mism_d2[k]);
  }
  std::printf(", \"n_blocks\": %d, \"mean_blocks_visited_cold\": %.1f, \"mean_blocks_visited_warm\": %.1f, \"warp32_blocks_per_point\": %.2f, \"warp32_union_blocks\": %.2f}\n",
              srt.nb, visited_cold, visited_warm, wp, ww);
  return 0;
}
