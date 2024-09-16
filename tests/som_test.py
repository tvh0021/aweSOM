import pytest
import numpy as np
from aweSOM import Lattice

# Set up the parameters for the lattice; can change to stress test
training_steps = 10000
data_dims = (500, 4)
features_names = ["feature1", "feature2", "feature3", "feature4"]
alpha_0 = 0.5
xdim = 20
ydim = 10
data = np.random.rand(data_dims[0], data_dims[1])


@pytest.fixture
def map():
    print("Setting up the test", flush=True)

    # Initialize the lattice
    map = Lattice(
        xdim=xdim,
        ydim=ydim,
        alpha_0=alpha_0,
        train=training_steps,
        alpha_type="static",
        sampling_type="uniform",
    )
    return map


@pytest.fixture
def trained_map(map: Lattice):
    map.train_lattice(data, features_names)
    return map


@pytest.fixture
def map_after_mapping(trained_map: Lattice):
    trained_map.map_data_to_lattice()
    trained_map.assign_cluster_to_lattice()
    # add more here
    return trained_map


def test_lattice_initialization(map: Lattice):
    print("Testing lattice initialization", flush=True)
    assert map.xdim == xdim
    assert map.ydim == ydim
    assert map.alpha_0 == alpha_0
    assert map.alpha == map.alpha_0
    assert map.train == training_steps
    assert map.alpha_type == 0
    assert map.init == "uniform"
    assert map.seed == 42
    assert map.epoch == 0
    assert map.save_frequency == map.train // 200


def test_train_lattice(trained_map: Lattice):
    print("Testing lattice training", flush=True)

    assert trained_map.data_array.shape == data.shape
    assert trained_map.features_names == features_names
    assert trained_map.lattice.shape == (xdim * ydim, data_dims[1])
    assert (
        trained_map.alpha == pytest.approx(alpha_0 * 0.75**25, rel=1e-4)
        if trained_map.alpha_type == 1
        else alpha_0
    )
    assert trained_map.epoch == training_steps


def test_Gamma(trained_map: Lattice):
    print("Testing Neighborhood function (gamma)", flush=True)
    m = np.arange(xdim * ydim).reshape((-1, 1))
    m2Ds = trained_map.coordinate(m, xdim)
    bmu_index = np.random.randint(0, xdim * ydim)
    alpha = alpha_0
    nsize = 8

    # Basic test
    gamma = trained_map.Gamma(bmu_index, m2Ds, alpha, nsize)
    assert gamma.shape == (xdim * ydim,)
    assert gamma[bmu_index] == pytest.approx(alpha, rel=1e-4)

    # Test with a reference node location
    reference_index = np.random.randint(0, xdim * ydim)
    dist_2d = np.abs(m2Ds - m2Ds[bmu_index])
    chebyshev_distance = np.max(dist_2d[reference_index])
    expected_value = alpha * np.exp(-(chebyshev_distance**2) / (2 * (nsize / 3) ** 2))
    assert gamma[reference_index] == pytest.approx(expected_value, rel=1e-4)


def test_map_data_to_lattice(map_after_mapping: Lattice):
    print("Testing mapping data to lattice", flush=True)
    assert map_after_mapping.projection_1d.shape == (data_dims[0], 1)
    assert map_after_mapping.projection_2d.shape == (data_dims[0], 2)


def test_assign_cluster_to_lattice(map_after_mapping: Lattice):
    print("Testing assigning cluster ids to lattice", flush=True)
    number_of_clusters = np.max(map_after_mapping.lattice_assigned_clusters) + 1
    assert map_after_mapping.lattice_assigned_clusters.shape == (xdim, ydim)

    # Test with smoothing; asserting that there are less (or at least equal) clusters with smoothing
    clusters_smooth = map_after_mapping.assign_cluster_to_lattice(smoothing=2)
    number_of_clusters_smooth = np.max(clusters_smooth) + 1
    assert number_of_clusters_smooth <= number_of_clusters

    # Test with merge cost; asserting that there are less (or at least equal) clusters with merge cost
    clusters_merge = map_after_mapping.assign_cluster_to_lattice(merge_cost=0.4)
    number_of_clusters_merge = np.max(clusters_merge) + 1
    assert number_of_clusters_merge <= number_of_clusters


def test_assign_cluster_to_data(map_after_mapping: Lattice):
    print("Testing assigning cluster ids to data", flush=True)
    som_labels = map_after_mapping.assign_cluster_to_data(
        map_after_mapping.projection_2d, map_after_mapping.lattice_assigned_clusters
    )
    assert som_labels.shape == (data_dims[0],)


def test_best_match(map_after_mapping: Lattice):
    print("Testing best match unit", flush=True)

    # Obtain a random node, change the value slightly to mimic an observation, and check if the best_match is the same node
    for _ in range(data_dims[0]):
        node_index = np.random.randint(0, xdim * ydim)
        weight_before = map_after_mapping.lattice[node_index]
        weight_after = weight_before + np.random.rand(data_dims[1]) * 1.0e-5
        weight_after = np.reshape(weight_after, (1, -1))

        best_match_node = map_after_mapping.best_match(
            map_after_mapping.lattice, weight_after
        )
        assert best_match_node == node_index


def test_compute_centroids(map_after_mapping: Lattice):
    print("Testing computing centroids", flush=True)
    centroids = map_after_mapping.compute_centroids()
    assert "centroid_x" in centroids
    assert "centroid_y" in centroids
    assert centroids["centroid_x"].shape == (xdim, ydim)
    assert centroids["centroid_y"].shape == (xdim, ydim)


def test_get_unique_centroids(map_after_mapping: Lattice):
    print("Testing unique centroids", flush=True)
    centroids = map_after_mapping.compute_centroids()
    centroid_map = [
        (centroids["centroid_x"][x, y], centroids["centroid_y"][x, y])
        for x in range(xdim)
        for y in range(ydim)
    ]
    unique_centroids = map_after_mapping.get_unique_centroids(centroids)
    assert len(unique_centroids["position_x"]) == len(set(centroid_map))
    assert len(unique_centroids["position_y"]) == len(set(centroid_map))
    assert sum(list(map_after_mapping.nodes_count.values())) == xdim * ydim


def test_compute_umat(map_after_mapping: Lattice):
    print("Testing U-matrix computation", flush=True)
    umat = (
        map_after_mapping.compute_umat()
    )  # recompute the U-matrix in case the lattice has been modified
    assert umat.shape == (xdim, ydim)

    # Pick a few random nodes (interior, so no edge effect), and check if the U-matrix is computed correctly
    nodes_position = np.zeros((10, 2), dtype=int)
    for i in range(10):
        edge = True
        while edge:
            node_index = np.asanyarray(np.random.randint(0, xdim * ydim)).reshape(
                (-1, 1)
            )
            node_position = map_after_mapping.coordinate(node_index, xdim)
            if (
                node_position[0, 0] != 0
                and node_position[0, 0] != xdim - 1
                and node_position[0, 1] != 0
                and node_position[0, 1] != ydim - 1
            ):
                edge = False
        nodes_position[i] = node_position
    print("test nodes position", nodes_position)

    from sklearn.metrics.pairwise import euclidean_distances

    d = euclidean_distances(map_after_mapping.lattice, map_after_mapping.lattice) / (
        xdim * ydim
    )
    heat_arr = np.zeros((nodes_position.shape[0]))

    # iterate over the inner nodes and compute their umat values
    for i in range(nodes_position.shape[0]):
        ix = nodes_position[i, 0]
        iy = nodes_position[i, 1]
        print(ix, iy)
        sum = (
            d[map_after_mapping.rowix(ix, iy), map_after_mapping.rowix(ix - 1, iy - 1)]
            + d[map_after_mapping.rowix(ix, iy), map_after_mapping.rowix(ix, iy - 1)]
            + d[
                map_after_mapping.rowix(ix, iy), map_after_mapping.rowix(ix + 1, iy - 1)
            ]
            + d[map_after_mapping.rowix(ix, iy), map_after_mapping.rowix(ix + 1, iy)]
            + d[
                map_after_mapping.rowix(ix, iy), map_after_mapping.rowix(ix + 1, iy + 1)
            ]
            + d[map_after_mapping.rowix(ix, iy), map_after_mapping.rowix(ix, iy + 1)]
            + d[
                map_after_mapping.rowix(ix, iy), map_after_mapping.rowix(ix - 1, iy + 1)
            ]
            + d[map_after_mapping.rowix(ix, iy), map_after_mapping.rowix(ix - 1, iy)]
        ) / 8
        heat_arr[i] = sum

    print("heat arr", heat_arr)
    print(
        "umat",
        umat[nodes_position[:, 0], nodes_position[:, 1]],
    )

    diff_arr = np.abs(umat[nodes_position[:, 0], nodes_position[:, 1]] - heat_arr)
    print(diff_arr)

    for k in range(10):
        assert diff_arr[k] < 1e-5


if __name__ == "__main__":
    pytest.main()
