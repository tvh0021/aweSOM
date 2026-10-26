// Exactness stress test of the prototype kernels against awesom::best_match.
// Integer-valued weights and points make exact ties very common; duplicated nodes,
// non-finite points and a table with a NaN weight cover the fallbacks.
// Prints one JSON line; exit code 1 on any mismatch.
#pragma STDC FP_CONTRACT OFF
#include "../../../cpp/awesom_classify.hpp"
#include "fast_classify.hpp"

#include <cstdio>
#include <cstring>
#include <random>

int main() {
  std::mt19937_64 rng(20261026);
  long long checked = 0, ties = 0, bad_soa = 0, bad_block = 0, bad_block_warm = 0;
  const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
  for (int trial = 0; trial < 60; ++trial) {
    int F = 2 + trial % 9;  // 2..10
    int nodes = 1 + static_cast<int>(rng() % 700);
    int grid = 2 + static_cast<int>(rng() % 4);
    bool integer = trial % 3 != 2;
    std::uniform_real_distribution<double> U(-1, 1);
    std::vector<double> w(static_cast<std::size_t>(nodes) * F);
    for (auto& v : w) v = integer ? static_cast<double>(rng() % grid) : U(rng);
    for (int i = 1; i < nodes; i += 7)  // duplicate nodes
      for (int f = 0; f < F; ++f) w[static_cast<std::size_t>(i) * F + f] = w[static_cast<std::size_t>(i - 1) * F + f];
    if (trial % 10 == 9) w[3 % w.size()] = nan;  // non-finite table -> fallback path
    awesom::SOMTable m;
    m.F = F;
    m.nodes = nodes;
    m.weights = w;
    awesom_fast::SoATable soa(w.data(), nodes, F);
    awesom_fast::BlockTable blk(w.data(), nodes, F);
    int guess = -1;
    for (int p = 0; p < 3000; ++p) {
      std::vector<double> x(F);
      for (auto& v : x) v = integer ? static_cast<double>(rng() % (grid + 1)) - 0.5 * (rng() % 2) : 1.5 * U(rng);
      if (p % 500 == 1) x[rng() % F] = nan;
      if (p % 500 == 2) x[rng() % F] = inf;
      if (p % 500 == 3) x[rng() % F] = 1e200;  // squared distances overflow to inf
      double d_ref, d_soa, d_blk, d_wrm;
      int i_ref = awesom::best_match(m, x.data(), d_ref);
      int i_soa = soa.best_match(x.data(), d_soa);
      int i_blk = blk.best_match(x.data(), d_blk);
      int i_wrm = blk.best_match(x.data(), d_wrm, guess);
      guess = i_wrm;
      // count points whose minimum is attained by more than one node
      int nmin = 0;
      for (int i = 0; i < nodes; ++i) {
        double acc = 0;
        for (int f = 0; f < F; ++f) {
          double d = w[static_cast<std::size_t>(i) * F + f] - x[f];
          acc += d * d;
        }
        nmin += acc == d_ref;
      }
      ties += nmin > 1;
      ++checked;
      bad_soa += i_soa != i_ref || std::memcmp(&d_soa, &d_ref, 8) != 0;
      bad_block += i_blk != i_ref || std::memcmp(&d_blk, &d_ref, 8) != 0;
      bad_block_warm += i_wrm != i_ref || std::memcmp(&d_wrm, &d_ref, 8) != 0;
    }
  }
  std::printf("{\"points\": %lld, \"points_with_exact_ties\": %lld, \"mismatch_soa\": %lld, "
              "\"mismatch_block\": %lld, \"mismatch_block_warm\": %lld}\n",
              checked, ties, bad_soa, bad_block, bad_block_warm);
  return (bad_soa || bad_block || bad_block_warm) ? 1 : 0;
}
