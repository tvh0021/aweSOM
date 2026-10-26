"""Regenerate the SOM model equivalence fixtures in this directory.

Run from the repository root::

    python tests/fixtures/make_fixtures.py

Writes an "iris-like" model (three Gaussian blobs in four features with very
different raw scales, manual scaler, 10 x 8 lattice) as text and HDF5, 1000
held-out raw float64 points, and the labels and BMU distances that
``SOMModel.predict`` assigns to them. A C++ consumer (the planned
``AsterCurrentSheets`` thorn) can load ``iris_like_model.txt`` and
``iris_like_points.csv`` and must reproduce ``iris_like_expected.csv``
(label, distance) exactly.
"""

import contextlib
import io
from pathlib import Path

import numpy as np

from aweSOM import Lattice, SOMModel, Scaler

HERE = Path(__file__).resolve().parent
FEATURES = ["b_perp", "j_par", "rho", "temp"]
MEANS = np.array([[1.0, 0.0, 5.0, 100.0], [3.0, 2.0, 5.5, 300.0], [2.0, -1.5, 7.0, 200.0]])
SIGMAS = np.array([0.3, 0.4, 0.2, 25.0])


def sample(rng: np.random.Generator, n: int) -> np.ndarray:
    blob = rng.integers(0, len(MEANS), size=n)
    return MEANS[blob] + rng.normal(size=(n, len(SIGMAS))) * SIGMAS


def main() -> None:
    rng = np.random.default_rng(20261026)
    train_raw = sample(rng, 1500)
    test_raw = sample(rng, 1000)

    scaler = Scaler.fit(train_raw, kind="manual")
    train_scaled = scaler.transform(train_raw)
    som = Lattice(10, 8, 0.3, 2000)
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(train_scaled, FEATURES)
    model = SOMModel.from_lattice(
        som, train_scaled, scaler=scaler, merge_cost=0.25,
        metadata={"dataset": "tests/fixtures/make_fixtures.py iris-like blobs, seed 20261026"},
    )
    model.save_text(HERE / "iris_like_model.txt")
    model.save(HERE / "iris_like_model.h5")

    labels, dist = model.predict(test_raw, return_distance=True, backend="numba")
    np.save(HERE / "iris_like_points.npy", test_raw)
    np.save(HERE / "iris_like_expected_labels.npy", labels)
    np.save(HERE / "iris_like_expected_dist.npy", dist)
    np.savetxt(HERE / "iris_like_points.csv", test_raw, fmt="%.17g", delimiter=",")
    with open(HERE / "iris_like_expected.csv", "w") as fh:
        for lab, d in zip(labels, dist):
            fh.write(f"{lab},{d:.17g}\n")
    print(f"clusters: {np.unique(model.node_cluster).tolist()}, "
          f"label counts: {np.bincount(labels).tolist()}")


if __name__ == "__main__":
    main()
