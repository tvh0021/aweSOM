// awesom_classify.hpp -- header-only reference classifier for aweSOM models.
//
// Reads the plain-text model written by aweSOM.model.SOMModel.save_text and
// reproduces aweSOM.inference.predict bitwise for float64 input:
//   scaled[f] = (x[f] - offset[f]) / scale[f]
//   for each node i in ascending order:
//     acc = 0; for f in 0..F-1: { d = w[i,f] - scaled[f]; acc += d*d; }
//     if (acc < best) { best = acc; idx = i; }      // ties -> lowest index
//   label = node_cluster[idx], dist = sqrt(best)
//
// Build with -ffp-contract=off (GCC/Clang) so that no fused multiply-add
// changes the rounding of acc += d*d; the pragma below asks the same of
// compilers that honour it.
//
// C++17, standard library only.
// TODO: optional load_hdf5(path) under #ifdef AWESOM_HAVE_HDF5.

#ifndef AWESOM_CLASSIFY_HPP
#define AWESOM_CLASSIFY_HPP

#pragma STDC FP_CONTRACT OFF

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace awesom {

struct SOMTable {
  int xdim = 0, ydim = 0, nodes = 0, F = 0;
  std::vector<double> weights;  // nodes * F, row-major, node index = x + y * xdim
  std::vector<double> offset, scale, node_dist_q99;
  std::vector<int> node_cluster;  // -1 = unassigned
  double reject_distance = std::numeric_limits<double>::quiet_NaN();
  std::string scaler_kind;
  std::vector<std::string> feature_names;
  std::map<std::string, std::string> metadata;
};

namespace detail {

// strtod parses "%.17g" output (including nan/inf) to the identical double.
inline double parse_double(const std::string& tok, const std::string& what) {
  const char* begin = tok.c_str();
  char* end = nullptr;
  double v = std::strtod(begin, &end);
  if (end == begin || *end != '\0')
    throw std::runtime_error("awesom: cannot parse '" + tok + "' in " + what);
  return v;
}

inline std::vector<double> parse_doubles(const std::string& s, const std::string& what) {
  std::istringstream in(s);
  std::vector<double> out;
  std::string tok;
  while (in >> tok) out.push_back(parse_double(tok, what));
  return out;
}

inline int parse_int(const std::string& tok, const std::string& what) {
  const char* begin = tok.c_str();
  char* end = nullptr;
  long v = std::strtol(begin, &end, 10);
  if (end == begin || *end != '\0')
    throw std::runtime_error("awesom: cannot parse '" + tok + "' in " + what);
  return static_cast<int>(v);
}

inline std::string trim(const std::string& s) {
  const char* ws = " \t\r";
  std::size_t a = s.find_first_not_of(ws);
  if (a == std::string::npos) return "";
  std::size_t b = s.find_last_not_of(ws);
  return s.substr(a, b - a + 1);
}

}  // namespace detail

// Load a model written by SOMModel.save_text. Throws std::runtime_error on
// malformed input.
inline SOMTable load_text(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("awesom: cannot open " + path);
  std::map<std::string, std::string> header;
  std::vector<std::string> rows;
  std::string line;
  while (std::getline(in, line)) {
    std::string t = detail::trim(line);
    if (t.empty()) continue;
    if (t[0] == '#') {
      t = detail::trim(t.substr(1));
      std::size_t sp = t.find(' ');
      std::string key = t.substr(0, sp);
      std::string value = sp == std::string::npos ? "" : detail::trim(t.substr(sp + 1));
      header[key] = value;
    } else {
      rows.push_back(t);
    }
  }
  if (header["format"] != "awesom-som-text")
    throw std::runtime_error("awesom: " + path + " is not an aweSOM text model");

  SOMTable m;
  m.xdim = detail::parse_int(header.at("xdim"), "xdim");
  m.ydim = detail::parse_int(header.at("ydim"), "ydim");
  m.F = detail::parse_int(header.at("n_features"), "n_features");
  m.nodes = m.xdim * m.ydim;
  m.reject_distance = detail::parse_double(header.at("reject_distance"), "reject_distance");
  m.scaler_kind = header.at("scaler_kind");
  m.offset = detail::parse_doubles(header.at("scaler_offset"), "scaler_offset");
  m.scale = detail::parse_doubles(header.at("scaler_scale"), "scaler_scale");
  m.node_dist_q99 = detail::parse_doubles(header.at("node_dist_q99"), "node_dist_q99");
  {
    std::istringstream names(header.at("feature_names"));
    std::string name;
    while (std::getline(names, name, ',')) m.feature_names.push_back(name);
  }
  if (static_cast<int>(m.offset.size()) != m.F || static_cast<int>(m.scale.size()) != m.F ||
      static_cast<int>(m.feature_names.size()) != m.F)
    throw std::runtime_error("awesom: scaler/feature_names length != n_features");
  if (static_cast<int>(m.node_dist_q99.size()) != m.nodes ||
      static_cast<int>(rows.size()) != m.nodes)
    throw std::runtime_error("awesom: expected one node_dist_q99 value and one row per node");

  static const char* reserved[] = {"format",        "xdim",         "ydim",
                                   "n_features",    "reject_distance", "scaler_kind",
                                   "bulk_range",    "feature_names", "scaler_offset",
                                   "scaler_scale",  "node_dist_q99"};
  for (const auto& kv : header) {
    bool is_reserved = false;
    for (const char* r : reserved) is_reserved = is_reserved || kv.first == r;
    if (!is_reserved) m.metadata[kv.first] = kv.second;
  }

  m.weights.reserve(static_cast<std::size_t>(m.nodes) * m.F);
  m.node_cluster.reserve(m.nodes);
  for (const std::string& row : rows) {
    std::istringstream rs(row);
    std::vector<std::string> toks;
    std::string tok;
    while (rs >> tok) toks.push_back(tok);
    if (static_cast<int>(toks.size()) != m.F + 1)
      throw std::runtime_error("awesom: node row must have n_features weights + cluster id");
    for (int f = 0; f < m.F; ++f) m.weights.push_back(detail::parse_double(toks[f], "weights"));
    m.node_cluster.push_back(detail::parse_int(toks[m.F], "node_cluster"));
  }
  return m;
}

// Best-matching node of an already scaled point; dist2 receives the squared
// Euclidean distance. Ties go to the lowest node index.
inline int best_match(const SOMTable& m, const double* x_scaled, double& dist2) {
  double best = std::numeric_limits<double>::infinity();
  int idx = 0;
  const double* w = m.weights.data();
  for (int i = 0; i < m.nodes; ++i) {
    double acc = 0.0;
    for (int f = 0; f < m.F; ++f) {
      double d = w[static_cast<std::size_t>(i) * m.F + f] - x_scaled[f];
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

// Scale a raw point with the model's scaler, find its best-matching node and
// return that node's cluster id; dist receives the Euclidean BMU distance in
// the scaled space. Supports up to 64 features without heap allocation.
inline int classify(const SOMTable& m, const double* x_raw, double& dist) {
  constexpr int kMaxF = 64;
  double stack_buf[kMaxF];
  std::vector<double> heap_buf;
  double* xs = stack_buf;
  if (m.F > kMaxF) {
    heap_buf.resize(m.F);
    xs = heap_buf.data();
  }
  for (int f = 0; f < m.F; ++f) xs[f] = (x_raw[f] - m.offset[f]) / m.scale[f];
  double dist2;
  int idx = best_match(m, xs, dist2);
  dist = std::sqrt(dist2);
  return m.node_cluster[idx];
}

// Classify n raw points stored row-major in x_raw (n * F). labels and dist
// must each hold n values; dist may be nullptr.
inline void classify_many(const SOMTable& m, const double* x_raw, std::size_t n, int* labels,
                          double* dist) {
  for (std::size_t p = 0; p < n; ++p) {
    double d;
    labels[p] = classify(m, x_raw + p * static_cast<std::size_t>(m.F), d);
    if (dist) dist[p] = d;
  }
}

// True if dist exceeds the global rejection threshold or the 99th percentile
// of the training distances of node idx (unknown when that node had no data).
inline bool is_unfamiliar(const SOMTable& m, int node_idx, double dist) {
  double q = m.node_dist_q99[node_idx];
  return dist > m.reject_distance || (!std::isnan(q) && dist > q);
}

}  // namespace awesom

#endif  // AWESOM_CLASSIFY_HPP
