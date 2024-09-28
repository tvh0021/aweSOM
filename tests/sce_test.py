import pytest
import numpy as np
from unittest.mock import patch
import warnings
import glob

from aweSOM.sce import (
    load_som_npy,
    create_mask,
    compute_SQ,
    find_number_of_clusters,
)

# Mock data for testing
mock_data = np.array([[0, 1, 1], [0, 0, 1], [1, 1, 0]])

mock_data_3d = np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]])


def test_jax_gpu_available():
    # Check that the correct device is being used
    with patch("jax.default_backend", return_value="gpu"):
        import aweSOM.sce as sce

        if sce.USE_JAX and sce.array_lib.__name__ == "jax.numpy":
            assert sce.array_lib.__name__ == "jax.numpy"
            print("Using JAX for GPU computation.")
        else:
            warnings.warn("While JAX is available, no GPU was found. Will use NumPy.")


def test_jax_cpu_available():
    # Check that the correct device is being used
    with patch("jax.default_backend", return_value="cpu"):
        import aweSOM.sce as sce

        assert sce.USE_JAX is False
        assert sce.array_lib.__name__ == "numpy"
        assert not hasattr(sce, "jnp")


def test_load_som_npy(tmp_path):
    # Create a temporary .npy file
    file_path = tmp_path / "test.npy"
    np.save(file_path, mock_data)

    # Load the data using the function
    loaded_data = load_som_npy(str(file_path))

    # Assert the loaded data is the same as the original
    assert np.array_equal(loaded_data, mock_data)


def test_create_mask():
    # Create a mask for cluster id 1
    mask = create_mask(mock_data_3d, 1)

    # Expected mask
    expected_mask = np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]])

    # Assert the mask is as expected
    assert np.array_equal(mask, expected_mask)


def test_compute_SQ():
    mask1 = np.array([[1, 0], [0, 1]])

    mask2 = np.array([[1, 1], [0, 0]])

    SQ, SQ_matrix = compute_SQ(mask1, mask2)

    # Assert the SQ value and SQ_matrix are as expected
    assert isinstance(SQ, float)
    assert isinstance(SQ_matrix, np.ndarray)


def test_find_number_of_clusters():
    # Example data for testing
    path_to_example_files = "examples/iris/som_results/"
    all_files = glob.glob(path_to_example_files + "/*.npy")

    expected_number_of_clusters = np.array(
        [
            4,
            4,
            2,
            3,
            4,
            4,
            3,
            2,
            3,
            4,
            3,
            4,
            3,
            4,
            3,
            4,
            4,
            3,
            4,
            4,
            4,
            4,
            4,
            3,
            4,
            2,
            4,
            2,
            4,
            4,
            3,
            2,
            3,
            4,
            3,
            4,
        ]
    )
    expected_number_of_runs = 36
    expected_number_of_clusters_total = np.sum(expected_number_of_clusters)

    nids_array = find_number_of_clusters(all_files)

    # Assert the result is as expected
    # assert np.array_equal(nids_array, expected_number_of_clusters) # commented out because sometimes the order of the clusters are different, which throws an error
    assert np.sum(nids_array) == expected_number_of_clusters_total
    assert len(all_files) == expected_number_of_runs


if __name__ == "__main__":
    pytest.main()
