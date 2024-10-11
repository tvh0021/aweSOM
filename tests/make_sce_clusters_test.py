import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

from aweSOM.make_sce_clusters import (
    plot_gsum_values,
    plot_gsum_deriv,
    get_gsum_values,
    get_sce_cluster_separation,
    combine_separated_clusters,
    make_file_name,
)


def test_plot_gsum_values(tmp_path):
    gsum_values = [1, 2, 3, 4, 5]
    minimas = [1, 3]
    file_path = tmp_path / "gsum_values.png"
    plot_gsum_values(gsum_values, minimas, str(tmp_path))
    assert file_path.exists()

    plt.close("all")
    with patch("matplotlib.pyplot.show"):
        plot_gsum_values(gsum_values, minimas)

    fig = plt.gcf()
    assert fig is not None, "Figure is not created"
    assert len(fig.get_axes()) > 0, "No axes are created"
    plt.close(fig)


def test_plot_gsum_deriv(tmp_path):
    gsum_deriv = np.array([1, -1, 2, -2, 3])
    threshold = -0.5
    minimas = [1, 3]
    file_path = tmp_path / "gsum_deriv.png"
    plot_gsum_deriv(gsum_deriv, threshold, minimas, str(tmp_path))
    assert file_path.exists()

    plt.close("all")
    with patch("matplotlib.pyplot.show"):
        plot_gsum_deriv(gsum_deriv, threshold, minimas)

    fig = plt.gcf()
    assert fig is not None, "Figure is not created"
    assert len(fig.get_axes()) > 0, "No axes are created"


def test_get_gsum_values(tmp_path):
    mapping_file = tmp_path / "multimap_mappings.txt"
    with open(mapping_file, "w") as f:
        f.write("cluster-1\n1 0.5\n2 0.3\ncluster-2\n3 0.7\n4 0.2\n")
    gsum_values, map_list = get_gsum_values(str(mapping_file))
    assert gsum_values == [0.7, 0.5, 0.3, 0.2]
    assert map_list == [
        [0.7, 3, "cluster-2"],
        [0.5, 1, "cluster-1"],
        [0.3, 2, "cluster-1"],
        [0.2, 4, "cluster-2"],
    ]


def test_get_sce_cluster_separation():
    gsum_deriv = np.array([1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6])
    threshold = 2.0
    cluster_ranges, minimas = get_sce_cluster_separation(gsum_deriv, threshold)
    assert minimas == [7, 9]
    assert cluster_ranges == [[0, 7], [7, 9], [9, 11]]


def test_combine_separated_clusters():
    from scipy.signal import savgol_filter

    file_path = "examples/iris/som_results/SCE"
    dims = [150]
    gsum_values, map_list = get_gsum_values(file_path + "/multimap_mappings.txt")
    smooth_fraction = 10
    order = 4
    smoothed_map = gsum_values.copy()
    print("Applying Savitzky-Golay filter")
    smoothed_map = savgol_filter(
        smoothed_map, len(gsum_values) // smooth_fraction, order, deriv=0
    )

    # compute the derivative of the gsum values to find the drop
    gsum_deriv = (
        savgol_filter(smoothed_map, len(gsum_values) // smooth_fraction, order, deriv=1)
        / smoothed_map
    )

    # iterate through the derivative and find the local minima
    threshold = -0.05
    cluster_ranges, _ = get_sce_cluster_separation(gsum_deriv, threshold)

    combined_clusters = combine_separated_clusters(
        map_list, cluster_ranges, dims, file_path
    )
    assert combined_clusters.shape == (7, 150)


def test_make_file_name():
    assert make_file_name(1, "png") == "0001.png"
    assert make_file_name(10, "png") == "0010.png"
    assert make_file_name(100, "png") == "0100.png"


if __name__ == "__main__":
    pytest.main()
