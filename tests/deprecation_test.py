"""The old label route (map_data_to_lattice + assign_cluster_to_data) is
deprecated since 1.5.0; it must warn and still give the labels of predict."""

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

from aweSOM import Lattice

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def iris_lattice():
    data = np.load(FIXTURES / "iris_like_points.npy").astype(np.float64)
    som = Lattice(12, 8, 0.3, 2000)
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(data, [f"f{i}" for i in range(data.shape[1])])
        som.assign_cluster_to_lattice(merge_cost=0.2)
    return som, data


def test_deprecated_route_warns_and_equals_predict(iris_lattice):
    som, data = iris_lattice
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.warns(DeprecationWarning, match="map_data_to_lattice.*Lattice.predict"):
            projection_2d = som.map_data_to_lattice()
        with pytest.warns(DeprecationWarning, match="assign_cluster_to_data.*Lattice.predict"):
            old = som.assign_cluster_to_data(projection_2d, som.lattice_assigned_clusters)
    new = som.predict(data)
    assert old.dtype == np.int32 and new.dtype == np.int32
    assert np.array_equal(old, new)
    assert som.projection_2d.shape == (len(data), 2)
