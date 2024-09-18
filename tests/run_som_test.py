import pytest
import numpy as np
import os

from aweSOM.run_som import (
    batch_separator,
    number_of_nodes,
    initialize_lattice,
    manual_scaling,
    save_som_object,
    save_cluster_labels,
)


# Mock class for Lattice to use in tests
class MockLattice:
    def __init__(
        self, xdim, ydim, alpha_0, train, alpha_type="decay", sampling_type="uniform"
    ):
        self.xdim = xdim
        self.ydim = ydim
        self.alpha_0 = alpha_0
        self.train = train
        self.alpha_type = alpha_type
        self.sampling_type = sampling_type
        self.lattice = np.zeros((xdim, ydim))
        self.seed = 42

    def train_lattice(
        self, data, feature_list, number_of_steps=None, restart_lattice=None
    ):
        pass

    def map_data_to_lattice(self):
        return np.zeros((self.xdim, self.ydim))

    def assign_cluster_to_lattice(self, smoothing=None, merge_cost=0.2):
        return np.zeros((self.xdim, self.ydim))

    def assign_cluster_to_data(self, projection_2d, clusters):
        return np.zeros(projection_2d.shape[0])


@pytest.fixture
def sample_data():
    return np.random.rand(100, 10)


def test_batch_separator(sample_data):
    batches = batch_separator(sample_data, 5)
    assert batches.shape == (5, 20, 10)


def test_number_of_nodes():
    assert number_of_nodes(100, 10) == int(5 * np.sqrt(100 * 10) / 6)


def test_initialize_lattice(sample_data):
    xdim, ydim = initialize_lattice(sample_data, 0.5)
    assert xdim * ydim == pytest.approx(number_of_nodes(100, 10), 5)


def test_manual_scaling(sample_data):
    scaled_data = manual_scaling(sample_data)
    assert scaled_data.shape == sample_data.shape


def test_save_som_object():
    som = MockLattice(10, 10, 0.5, 100)
    save_som_object(som, 10, 10, 0.5, 100, name_of_dataset="test")
    file_name = "som_object.test-10-10-0.5-100-1s.pkl"
    assert os.path.isfile(
        file_name
    ), f"{file_name} does not exist in the current working directory."
    os.remove(file_name)


def test_save_cluster_labels():
    labels = np.zeros((10, 10))
    save_cluster_labels(labels, 10, 10, 0.5, 100, name_of_dataset="test")
    file_name = "labels.test-10-10-0.5-100-1s.npy"
    assert os.path.isfile(
        file_name
    ), f"{file_name} does not exist in the current working directory."
    os.remove(file_name)


# Run the tests
if __name__ == "__main__":
    pytest.main()
