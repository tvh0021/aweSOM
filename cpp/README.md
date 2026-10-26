# aweSOM C++ reference classifier

`awesom_classify.hpp` is a header-only (C++17, standard library only)
classifier for models written by `aweSOM.model.SOMModel.save_text`. It
gives the same labels as `SOMModel.predict` / `aweSOM.inference.predict`,
bit for bit, for float64 input. It is the reference that a runtime consumer
(the planned AsterX thorn `AsterCurrentSheets`) has to match.

## API

```cpp
#include "awesom_classify.hpp"

awesom::SOMTable m = awesom::load_text("model.txt");
double dist;
int label = awesom::classify(m, x_raw, dist);        // scales x_raw, then BMU search
int node  = awesom::best_match(m, x_scaled, dist2);  // already-scaled input, squared distance
awesom::classify_many(m, x_raw, n, labels, dists);   // n points, row-major n x F
bool flag = awesom::is_unfamiliar(m, node, dist);    // dist > reject_distance or > node_dist_q99[node]
```

`SOMTable` holds `xdim, ydim, nodes, F`, `weights` (nodes x F, row-major,
node index = x + y * xdim), `offset`, `scale`, `node_dist_q99`,
`node_cluster` (-1 = unassigned), `reject_distance`, `scaler_kind`,
`feature_names` and every other header line as `metadata`.

HDF5 loading (`load_hdf5` under `AWESOM_HAVE_HDF5`) is not implemented yet;
convert with `SOMModel.load("m.h5").save_text("m.txt")`.

## Exactness contract

```
scaled[f] = (x[f] - offset[f]) / scale[f]
for node i = 0 .. nodes-1:
    acc = 0.0
    for f = 0 .. F-1:  d = w[i][f] - scaled[f];  acc += d * d
    if (acc < best) { best = acc; idx = i }      // strict: ties go to the lowest index
label = node_cluster[idx];  dist = sqrt(best)
```

All arithmetic is IEEE double. Compile with `-ffp-contract=off` (the CMake
target sets it), otherwise Clang and GCC may fuse `acc += d * d` into an
FMA and round differently from Python. Floats in the text file are written
with `%.17g` and parsed with `strtod`, so they read back exactly.

Float32 input in Python is promoted to float64 before the arithmetic above.
Labels then match a C++ run on the same promoted values, but can differ from
float64 data at near-ties.

## Build and test

```sh
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
build/test_classify tests/fixtures/iris_like_model.txt tests/fixtures/iris_like_points.csv
```

`test_classify MODEL.txt POINTS.csv` prints `label dist` (dist with 17
significant digits) for each comma-separated raw point. `tests/cpp_test.py`
compiles it with the first of `clang++`, `g++`, `c++` it finds
(`-O2 -std=c++17 -ffp-contract=off`) and requires exact label equality with
`SOMModel.predict` on 2000 random points and on the committed fixture. The
test is skipped when no compiler is found.

## Fixture for downstream codes

`tests/fixtures/` holds an iris-like model (3 blobs, 4 features with raw
scales from 0.2 to 25, manual scaler, 10 x 8 lattice, 3 clusters) as
`iris_like_model.txt` and `.h5`, 1000 held-out raw points
(`iris_like_points.npy` / `.csv`), and the expected output
(`iris_like_expected_labels.npy`, `iris_like_expected_dist.npy`, and
`iris_like_expected.csv` with `label,dist` per line). Regenerate with
`python tests/fixtures/make_fixtures.py`. On this machine the C++ output
matches the expected labels and distances bitwise.

## Throughput

Apple M1 Max (10 cores), 2,000,000 float64 points, F = 6, 2368 nodes
(64 x 37), random weights, `aweSOM.inference.benchmark()`, numba 0.68:

| kernel | points/s |
| --- | --- |
| `aweSOM.inference.best_match_nodes` (numba, 10 threads) | 1.61e6 |
| `Lattice.best_match` (old numba kernel, 10 threads) | 1.00e5 |
| `awesom::classify_many` (C++, 1 thread, `-O2`, 200,000 points) | 1.42e5 |

Labels from the two Python kernels agreed on all 2,000,000 points. Per
thread, the C++ loop and the numba kernel run at about the same rate.
