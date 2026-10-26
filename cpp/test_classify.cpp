// test_classify: classify a CSV of raw points with an aweSOM text model.
//
// Usage: test_classify MODEL.txt POINTS.csv
// POINTS.csv holds one point per line, n_features comma-separated values.
// Prints "label dist" per point, dist with 17 significant digits.

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "awesom_classify.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " MODEL.txt POINTS.csv\n";
    return 2;
  }
  try {
    const awesom::SOMTable model = awesom::load_text(argv[1]);
    std::ifstream in(argv[2]);
    if (!in) throw std::runtime_error(std::string("cannot open ") + argv[2]);
    std::vector<double> x(model.F);
    std::string line;
    long lineno = 0;
    while (std::getline(in, line)) {
      ++lineno;
      if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
      std::istringstream ls(line);
      std::string tok;
      int f = 0;
      while (std::getline(ls, tok, ',')) {
        if (f >= model.F) break;
        x[f++] = awesom::detail::parse_double(awesom::detail::trim(tok), "points");
      }
      if (f != model.F || std::getline(ls, tok, ','))
        throw std::runtime_error("line " + std::to_string(lineno) + ": wrong number of features");
      double dist;
      int label = awesom::classify(model, x.data(), dist);
      std::printf("%d %.17g\n", label, dist);
    }
  } catch (const std::exception& e) {
    std::cerr << "test_classify: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
