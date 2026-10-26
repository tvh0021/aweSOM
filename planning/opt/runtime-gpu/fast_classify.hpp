// fast_classify.hpp -- prototype exact BMU kernels (same result as awesom::best_match).
//
// Both kernels compute each node's squared distance with the reference arithmetic
// (d = w[i,f] - x[f]; acc = 0; acc += d*d over f = 0..F-1, no FMA) and pick the
// lexicographic minimum of (acc, node index), i.e. the lowest index among exact
// ties, which is what the strict-< ascending scan of the reference returns.
// NaN distances are never selected; if no distance is < inf the result is node 0.
#pragma once
#pragma STDC FP_CONTRACT OFF

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace awesom_fast {

// Structure-of-arrays table: wT[f * Np + i], nodes padded to a multiple of L with NaN.
// The node loop is vectorised across L nodes; each node's feature sum keeps its order.
struct SoATable {
  static constexpr int L = 8;
  int nodes, F, Np;
  std::vector<double> wT;
  SoATable(const double* w, int nodes_, int F_) : nodes(nodes_), F(F_) {
    Np = (nodes + L - 1) / L * L;
    wT.assign(static_cast<std::size_t>(F) * Np, std::numeric_limits<double>::quiet_NaN());
    for (int i = 0; i < nodes; ++i)
      for (int f = 0; f < F; ++f) wT[static_cast<std::size_t>(f) * Np + i] = w[static_cast<std::size_t>(i) * F + f];
  }

  template <int FF>
  int scan(const double* x, double& dist2) const {
    double best[L];
    int bidx[L];
    for (int j = 0; j < L; ++j) {
      best[j] = std::numeric_limits<double>::infinity();
      bidx[j] = 0;
    }
    const int FR = FF > 0 ? FF : F;
    const double* w = wT.data();
    for (int i = 0; i < Np; i += L) {
      double acc[L];
      for (int j = 0; j < L; ++j) {
        double d = w[i + j] - x[0];
        acc[j] = d * d;  // == 0.0 + d*d exactly
      }
      for (int f = 1; f < FR; ++f) {
        const double xf = x[f];
        const double* wf = w + static_cast<std::size_t>(f) * Np + i;
        for (int j = 0; j < L; ++j) {
          double d = wf[j] - xf;
          acc[j] = acc[j] + d * d;
        }
      }
      for (int j = 0; j < L; ++j) {
        bool lt = acc[j] < best[j];
        best[j] = lt ? acc[j] : best[j];
        bidx[j] = lt ? i + j : bidx[j];
      }
    }
    double b = best[0];
    int bi = bidx[0];
    for (int j = 1; j < L; ++j)
      if (best[j] < b || (best[j] == b && bidx[j] < bi)) {
        b = best[j];
        bi = bidx[j];
      }
    if (!(b < std::numeric_limits<double>::infinity())) bi = 0;  // reference: no finite distance -> node 0
    dist2 = b;
    return bi;
  }

  int best_match(const double* x, double& dist2) const {
    switch (F) {
      case 2: return scan<2>(x, dist2);
      case 3: return scan<3>(x, dist2);
      case 4: return scan<4>(x, dist2);
      case 5: return scan<5>(x, dist2);
      case 6: return scan<6>(x, dist2);
      case 8: return scan<8>(x, dist2);
      case 10: return scan<10>(x, dist2);
      default: return scan<0>(x, dist2);
    }
  }
};

// Nodes sorted along one feature axis a. For node i, the reference accumulator
// satisfies acc_i >= fl((w[i,a] - x[a])^2) because every term is >= 0 and rounding
// is monotone, and fl(w - x[a]) is monotone in w. So walking outward from x[a] in
// the sorted order, a side can be abandoned once that lower bound exceeds the
// current best (strictly, so equal-distance lower-index nodes are still seen).
// Exact for finite inputs; non-finite points fall back to the full scan.
struct SortedTable {
  int nodes, F, axis = 0;
  std::vector<double> ws;   // nodes x F, sorted order, row-major (one node per row)
  std::vector<double> key;  // ws[k * F + axis]
  std::vector<int> orig;    // original node index of sorted row k
  std::vector<double> w0;   // original weights (fallback)
  bool finite_weights = true;

  SortedTable(const double* w, int nodes_, int F_) : nodes(nodes_), F(F_), w0(w, w + static_cast<std::size_t>(nodes_) * F_) {
    double bestvar = -1;
    for (int f = 0; f < F; ++f) {
      double s = 0, s2 = 0;
      for (int i = 0; i < nodes; ++i) {
        double v = w[static_cast<std::size_t>(i) * F + f];
        finite_weights = finite_weights && std::isfinite(v);
        s += v;
        s2 += v * v;
      }
      double var = s2 / nodes - (s / nodes) * (s / nodes);
      if (var > bestvar) {
        bestvar = var;
        axis = f;
      }
    }
    orig.resize(nodes);
    std::iota(orig.begin(), orig.end(), 0);
    if (finite_weights)
      std::stable_sort(orig.begin(), orig.end(), [&](int a, int b) {
        return w[static_cast<std::size_t>(a) * F + axis] < w[static_cast<std::size_t>(b) * F + axis];
      });
    ws.resize(static_cast<std::size_t>(nodes) * F);
    key.resize(nodes);
    for (int k = 0; k < nodes; ++k) {
      for (int f = 0; f < F; ++f) ws[static_cast<std::size_t>(k) * F + f] = w[static_cast<std::size_t>(orig[k]) * F + f];
      key[k] = ws[static_cast<std::size_t>(k) * F + axis];
    }
  }

  inline double node_dist2(const double* wr, const double* x) const {
    double acc = 0.0;
    for (int f = 0; f < F; ++f) {
      double d = wr[f] - x[f];
      acc += d * d;
    }
    return acc;
  }

  int full_scan(const double* x, double& dist2) const {
    double best = std::numeric_limits<double>::infinity();
    int idx = 0;
    for (int i = 0; i < nodes; ++i) {
      double acc = node_dist2(&w0[static_cast<std::size_t>(i) * F], x);
      if (acc < best) {
        best = acc;
        idx = i;
      }
    }
    dist2 = best;
    return idx;
  }

  // guess: a node index to evaluate first (e.g. the neighbouring cell's BMU), or -1.
  int best_match(const double* x, double& dist2, int guess, long long* visited = nullptr) const {
    bool fin = finite_weights;
    for (int f = 0; f < F; ++f) fin = fin && std::isfinite(x[f]);
    if (!fin) return full_scan(x, dist2);
    const double inf = std::numeric_limits<double>::infinity();
    double best = inf;
    int bidx = INT_MAX;
    long long nv = 0;
    if (guess >= 0) {
      best = node_dist2(&w0[static_cast<std::size_t>(guess) * F], x);
      bidx = guess;
      ++nv;
    }
    const double xa = x[axis];
    int hi = static_cast<int>(std::lower_bound(key.begin(), key.end(), xa) - key.begin());
    int lo = hi - 1;
    for (;;) {
      double dlo = lo >= 0 ? key[lo] - xa : 0, dhi = hi < nodes ? key[hi] - xa : 0;
      double lblo = lo >= 0 ? dlo * dlo : inf, lbhi = hi < nodes ? dhi * dhi : inf;
      bool take_lo = lblo <= lbhi;
      double lb = take_lo ? lblo : lbhi;
      if (lb > best || lb == inf) break;  // both sides exhausted or pruned
      int k = take_lo ? lo-- : hi++;
      double acc = node_dist2(&ws[static_cast<std::size_t>(k) * F], x);
      ++nv;
      int o = orig[k];
      if (acc < best || (acc == best && o < bidx)) {
        best = acc;
        bidx = o;
      }
    }
    if (visited) *visited += nv;
    if (bidx == INT_MAX || !(best < inf)) bidx = 0;
    dist2 = best;
    return bidx;
  }

  double mean_visited(const double* X, std::size_t n, bool warm) const {
    long long v = 0;
    int guess = -1;
    for (std::size_t p = 0; p < n; ++p) {
      double d2;
      int g = best_match(X + p * F, d2, warm ? guess : -1, &v);
      guess = g;
    }
    return static_cast<double>(v) / n;
  }
};

}  // namespace awesom_fast

namespace awesom_fast {

// Nodes grouped into blocks of B = 16 (recursive bisection of weight space, split
// along the widest box side), each block stored SoA with its per-feature bounding
// box [lo_f, hi_f]. Lower bound of a block, accumulated in the reference order:
//   c_f = x<lo ? lo-x : (x>hi ? x-hi : 0);  lb = 0; lb += c_f*c_f
// For any node w in the box, |fl(w_f - x_f)| >= c_f (rounding is monotone), so each
// term is <= the node's term and, again by monotone rounding, lb <= acc. A block
// with lb > best cannot contain the answer (strict, so exact ties are still seen).
// Nodes are compared lexicographically on (acc, original index): the result is the
// reference's lowest-index minimum, bit for bit. Non-finite input -> full scan.
struct BlockTable {
#ifndef AWESOM_BLOCK
#define AWESOM_BLOCK 16
#endif
  static constexpr int B = AWESOM_BLOCK;
  int nodes, F, nb;
  std::vector<double> wb;   // (b * F + f) * B + j
  std::vector<int> ib;      // b * B + j -> original index (INT_MAX for padding)
  std::vector<double> lo, hi;  // b * F + f
  std::vector<double> loT, hiT;  // f * nb + b (for the vectorised bound)
  // two-level variant: super-blocks of S consecutive leaves (the bisection emits
  // leaves depth first, so consecutive leaves are close in weight space)
  static constexpr int S = 8;
  int ns = 0;
  std::vector<double> slo, shi;  // s * F + f
  std::vector<double> w0;
  bool finite_weights = true;

  BlockTable(const double* w, int nodes_, int F_) : nodes(nodes_), F(F_), w0(w, w + static_cast<std::size_t>(nodes_) * F_) {
    for (double v : w0) finite_weights = finite_weights && std::isfinite(v);
    std::vector<int> idx(nodes);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<std::vector<int>> leaves;
    split(idx, 0, nodes, leaves);
    nb = static_cast<int>(leaves.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    wb.assign(static_cast<std::size_t>(nb) * F * B, nan);
    ib.assign(static_cast<std::size_t>(nb) * B, INT_MAX);
    lo.assign(static_cast<std::size_t>(nb) * F, std::numeric_limits<double>::infinity());
    hi.assign(static_cast<std::size_t>(nb) * F, -std::numeric_limits<double>::infinity());
    for (int b = 0; b < nb; ++b) {
      std::vector<int>& lf = leaves[b];
      std::sort(lf.begin(), lf.end());
      for (int j = 0; j < static_cast<int>(lf.size()); ++j) {
        ib[b * B + j] = lf[j];
        for (int f = 0; f < F; ++f) {
          double v = w[static_cast<std::size_t>(lf[j]) * F + f];
          wb[(static_cast<std::size_t>(b) * F + f) * B + j] = v;
          lo[b * F + f] = std::min(lo[b * F + f], v);
          hi[b * F + f] = std::max(hi[b * F + f], v);
        }
      }
    }
    ns = (nb + S - 1) / S;
    slo.assign(static_cast<std::size_t>(ns) * F, std::numeric_limits<double>::infinity());
    shi.assign(static_cast<std::size_t>(ns) * F, -std::numeric_limits<double>::infinity());
    for (int b = 0; b < nb; ++b)
      for (int f = 0; f < F; ++f) {
        slo[(b / S) * F + f] = std::min(slo[(b / S) * F + f], lo[b * F + f]);
        shi[(b / S) * F + f] = std::max(shi[(b / S) * F + f], hi[b * F + f]);
      }
    loT.resize(lo.size());
    hiT.resize(hi.size());
    for (int b = 0; b < nb; ++b)
      for (int f = 0; f < F; ++f) {
        loT[static_cast<std::size_t>(f) * nb + b] = lo[b * F + f];
        hiT[static_cast<std::size_t>(f) * nb + b] = hi[b * F + f];
      }
  }

  void split(std::vector<int>& idx, int a, int e, std::vector<std::vector<int>>& leaves) {
    int n = e - a;
    if (n <= B) {
      leaves.emplace_back(idx.begin() + a, idx.begin() + e);
      return;
    }
    int fbest = 0;
    double wbest = -1;
    for (int f = 0; f < F; ++f) {
      double mn = std::numeric_limits<double>::infinity(), mx = -mn;
      for (int k = a; k < e; ++k) {
        double v = w0[static_cast<std::size_t>(idx[k]) * F + f];
        mn = std::min(mn, v);
        mx = std::max(mx, v);
      }
      if (mx - mn > wbest) {
        wbest = mx - mn;
        fbest = f;
      }
    }
    int m = a + std::max(B, (n / (2 * B)) * B);
    std::nth_element(idx.begin() + a, idx.begin() + m, idx.begin() + e, [&](int p, int q) {
      return w0[static_cast<std::size_t>(p) * F + fbest] < w0[static_cast<std::size_t>(q) * F + fbest];
    });
    split(idx, a, m, leaves);
    split(idx, m, e, leaves);
  }

  int full_scan(const double* x, double& dist2) const {
    double best = std::numeric_limits<double>::infinity();
    int idx = 0;
    for (int i = 0; i < nodes; ++i) {
      double acc = 0.0;
      for (int f = 0; f < F; ++f) {
        double d = w0[static_cast<std::size_t>(i) * F + f] - x[f];
        acc += d * d;
      }
      if (acc < best) {
        best = acc;
        idx = i;
      }
    }
    dist2 = best;
    return idx;
  }

  template <int FF>
  inline void eval_block(int b, const double* x, double& best, int& bidx) const {
    double acc[B];
    const double* wf = &wb[static_cast<std::size_t>(b) * FF * B];
    for (int j = 0; j < B; ++j) {
      double d = wf[j] - x[0];
      acc[j] = d * d;
    }
    for (int f = 1; f < FF; ++f) {
      const double xf = x[f];
      for (int j = 0; j < B; ++j) {
        double d = wf[f * B + j] - xf;
        acc[j] = acc[j] + d * d;
      }
    }
    double m = acc[0];
    for (int j = 1; j < B; ++j) m = acc[j] < m ? acc[j] : m;
    if (!(m <= best)) return;  // also skips all-NaN blocks
    const int* ii = &ib[static_cast<std::size_t>(b) * B];
    for (int j = 0; j < B; ++j)
      if (acc[j] < best || (acc[j] == best && ii[j] < bidx)) {
        best = acc[j];
        bidx = ii[j];
      }
  }

  template <int FF>
  int search(const double* x, double& dist2, int guess, long long* visited) const {
    constexpr int kMaxBlocks = 4096;
    double lb[kMaxBlocks];
    int bmin = 0;
    double lbmin = std::numeric_limits<double>::infinity();
    // branchless, vectorised across blocks: max(l - x, 0) + max(x - h, 0) is exactly
    // fl(l - x), fl(x - h) or 0 because l <= h means at most one term is positive
    for (int b = 0; b < nb; ++b) lb[b] = 0.0;
    for (int f = 0; f < FF; ++f) {
      const double xf = x[f];
      const double* lf = &loT[static_cast<std::size_t>(f) * nb];
      const double* hf = &hiT[static_cast<std::size_t>(f) * nb];
      for (int b = 0; b < nb; ++b) {
        double a = lf[b] - xf, c = xf - hf[b];
        a = a > 0.0 ? a : 0.0;
        c = c > 0.0 ? c : 0.0;
        double t = a + c;
        lb[b] = lb[b] + t * t;
      }
    }
    for (int b = 0; b < nb; ++b)
      if (lb[b] < lbmin) {
        lbmin = lb[b];
        bmin = b;
      }
    double best = std::numeric_limits<double>::infinity();
    int bidx = INT_MAX;
    if (guess >= 0) {
      double acc = 0.0;
      for (int f = 0; f < FF; ++f) {
        double d = w0[static_cast<std::size_t>(guess) * FF + f] - x[f];
        acc += d * d;
      }
      best = acc;
      bidx = guess;
    }
    long long nv = 1;
    eval_block<FF>(bmin, x, best, bidx);
    for (int b = 0; b < nb; ++b) {
      if (b == bmin || lb[b] > best) continue;
      ++nv;
      eval_block<FF>(b, x, best, bidx);
    }
    if (visited) *visited += nv;
    if (bidx == INT_MAX || !(best < std::numeric_limits<double>::infinity())) bidx = 0;
    dist2 = best;
    return bidx;
  }

  // GPU-style variant: no per-point bound array (registers only). Pass 1 finds the
  // block with the smallest bound, pass 2 recomputes each bound and visits the block
  // if bound <= best. Same exactness argument as search().
  template <int FF>
  inline double block_lb(int b, const double* x) const {
    double acc = 0.0;
    for (int f = 0; f < FF; ++f) {
      double a = lo[b * FF + f] - x[f], c = x[f] - hi[b * FF + f];
      a = a > 0.0 ? a : 0.0;
      c = c > 0.0 ? c : 0.0;
      double t = a + c;
      acc += t * t;
    }
    return acc;
  }

  template <int FF>
  int search_recompute(const double* x, double& dist2) const {
    int bmin = 0;
    double lbmin = std::numeric_limits<double>::infinity();
    for (int b = 0; b < nb; ++b) {
      double l = block_lb<FF>(b, x);
      if (l < lbmin) {
        lbmin = l;
        bmin = b;
      }
    }
    double best = std::numeric_limits<double>::infinity();
    int bidx = INT_MAX;
    eval_block<FF>(bmin, x, best, bidx);
    for (int b = 0; b < nb; ++b) {
      if (b == bmin || block_lb<FF>(b, x) > best) continue;
      eval_block<FF>(b, x, best, bidx);
    }
    if (bidx == INT_MAX || !(best < std::numeric_limits<double>::infinity())) bidx = 0;
    dist2 = best;
    return bidx;
  }

  int best_match_recompute(const double* x, double& dist2) const {
    bool fin = finite_weights;
    for (int f = 0; f < F; ++f) fin = fin && std::isfinite(x[f]);
    if (!fin) return full_scan(x, dist2);
    switch (F) {
      case 2: return search_recompute<2>(x, dist2);
      case 3: return search_recompute<3>(x, dist2);
      case 4: return search_recompute<4>(x, dist2);
      case 5: return search_recompute<5>(x, dist2);
      case 6: return search_recompute<6>(x, dist2);
      case 7: return search_recompute<7>(x, dist2);
      case 8: return search_recompute<8>(x, dist2);
      case 9: return search_recompute<9>(x, dist2);
      case 10: return search_recompute<10>(x, dist2);
      default: return full_scan(x, dist2);
    }
  }

  template <int FF>
  static inline double box_lb(const double* l, const double* h, const double* x) {
    double acc = 0.0;
    for (int f = 0; f < FF; ++f) {
      double a = l[f] - x[f], c = x[f] - h[f];
      a = a > 0.0 ? a : 0.0;
      c = c > 0.0 ? c : 0.0;
      double t = a + c;
      acc += t * t;
    }
    return acc;
  }

  // Two-level search: super-block bound, then block bound, then nodes. The super box
  // contains its blocks' boxes, so its bound is <= theirs <= every node's acc.
  template <int FF>
  int search2(const double* x, double& dist2) const {
    constexpr int kMaxSuper = 512;
    double slb[kMaxSuper];
    int smin = 0;
    double smv = std::numeric_limits<double>::infinity();
    for (int q = 0; q < ns; ++q) {
      slb[q] = box_lb<FF>(&slo[q * FF], &shi[q * FF], x);
      if (slb[q] < smv) {
        smv = slb[q];
        smin = q;
      }
    }
    double best = std::numeric_limits<double>::infinity();
    int bidx = INT_MAX;
    {  // seed with the best-bounded block of the best-bounded super-block
      int b0 = smin * S, b1 = std::min(nb, b0 + S), bm = b0;
      double lm = std::numeric_limits<double>::infinity();
      for (int b = b0; b < b1; ++b) {
        double l = box_lb<FF>(&lo[b * FF], &hi[b * FF], x);
        if (l < lm) {
          lm = l;
          bm = b;
        }
      }
      eval_block<FF>(bm, x, best, bidx);
    }
    for (int q = 0; q < ns; ++q) {
      if (slb[q] > best) continue;
      int b0 = q * S, b1 = std::min(nb, b0 + S);
      for (int b = b0; b < b1; ++b)
        if (!(box_lb<FF>(&lo[b * FF], &hi[b * FF], x) > best)) eval_block<FF>(b, x, best, bidx);
    }
    if (bidx == INT_MAX || !(best < std::numeric_limits<double>::infinity())) bidx = 0;
    dist2 = best;
    return bidx;
  }

  int best_match2(const double* x, double& dist2) const {
    bool fin = finite_weights;
    for (int f = 0; f < F; ++f) fin = fin && std::isfinite(x[f]);
    if (!fin || ns > 512) return full_scan(x, dist2);
    switch (F) {
      case 2: return search2<2>(x, dist2);
      case 3: return search2<3>(x, dist2);
      case 4: return search2<4>(x, dist2);
      case 5: return search2<5>(x, dist2);
      case 6: return search2<6>(x, dist2);
      case 7: return search2<7>(x, dist2);
      case 8: return search2<8>(x, dist2);
      case 9: return search2<9>(x, dist2);
      case 10: return search2<10>(x, dist2);
      default: return full_scan(x, dist2);
    }
  }

  int best_match(const double* x, double& dist2, int guess = -1, long long* visited = nullptr) const {
    bool fin = finite_weights;
    for (int f = 0; f < F; ++f) fin = fin && std::isfinite(x[f]);
    if (!fin || nb > 4096) return full_scan(x, dist2);
    switch (F) {
      case 2: return search<2>(x, dist2, guess, visited);
      case 3: return search<3>(x, dist2, guess, visited);
      case 4: return search<4>(x, dist2, guess, visited);
      case 5: return search<5>(x, dist2, guess, visited);
      case 6: return search<6>(x, dist2, guess, visited);
      case 7: return search<7>(x, dist2, guess, visited);
      case 8: return search<8>(x, dist2, guess, visited);
      case 9: return search<9>(x, dist2, guess, visited);
      case 10: return search<10>(x, dist2, guess, visited);
      default: return full_scan(x, dist2);
    }
  }

  double mean_visited(const double* X, std::size_t n, bool warm) const {
    long long v = 0;
    int guess = -1;
    for (std::size_t p = 0; p < n; ++p) {
      double d2;
      guess = best_match(X + p * F, d2, warm ? guess : -1, &v);
    }
    return static_cast<double>(v) / n;
  }
};

}  // namespace awesom_fast
