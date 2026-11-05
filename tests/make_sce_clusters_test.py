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


def test_get_gsum_values_exponent_form(tmp_path):
    # a Gsum written as 2.5e-05 contains a '-' but is not a run name
    mapping_file = tmp_path / "multimap_mappings.txt"
    with open(mapping_file, "w") as f:
        f.write("/abs/labels.run-1\n0 3.0\n1 2.5e-05\n/abs/labels.run-2\n0 1e-07\n1 2.0\n")
    gsum_values, map_list = get_gsum_values(str(mapping_file))
    assert gsum_values == [3.0, 2.0, 2.5e-05, 1e-07]
    assert map_list[2] == [2.5e-05, 1, "/abs/labels.run-1"]
    assert map_list[3] == [1e-07, 0, "/abs/labels.run-2"]


def test_get_sce_cluster_separation_few_minima():
    # one minimum below the threshold: popped, so a single range covering everything
    deriv = np.array([0.0, 0.0, -1.0, 0.0, 0.0])
    ranges, minimas = get_sce_cluster_separation(deriv, -0.5)
    assert minimas == [] and ranges == [[0, 5]]
    # two minima: the first is popped, the second splits the list in two
    deriv = np.array([0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0])
    ranges, minimas = get_sce_cluster_separation(deriv, -0.5)
    assert minimas == [4] and ranges == [[0, 4], [4, 7]]
    # no minimum at all
    ranges, minimas = get_sce_cluster_separation(np.zeros(4), -0.5)
    assert minimas == [] and ranges == [[0, 4]]


def test_stack_sce_labels(tmp_path):
    from aweSOM.make_sce_clusters import stack_sce_labels, gsum_derivative
    from aweSOM.sce import loop_over_all_clusters, find_number_of_clusters

    rng = np.random.default_rng(1)
    truth = rng.integers(0, 3, 600)
    files = []
    for r in range(4):
        lab = truth.copy()
        flip = rng.random(600) < 0.05
        lab[flip] = rng.integers(0, 3, flip.sum())
        files.append(str(tmp_path / f"labels.run-{r}.npy"))
        np.save(files[-1], lab.astype(np.uint8))
    (tmp_path / "SCE").mkdir()
    loop_over_all_clusters(files, find_number_of_clusters(files), np.array([600]), str(tmp_path / "SCE"), write_masks=False)
    labels, signals, info = stack_sce_labels(str(tmp_path / "SCE" / "multimap_mappings.txt"), str(tmp_path), [600], threshold=-0.015)
    assert labels.shape == (600,) and signals.shape == (len(info["cluster_ranges"]), 600)
    assert np.array_equal(labels, np.argmax(signals, axis=0))
    assert info["cluster_ranges"][0][0] == 0 and info["cluster_ranges"][-1][1] == 12
    # short lists (12 entries) still get a valid Savitzky-Golay window
    assert gsum_derivative(info["gsum_values"]).shape == (12,)


def test_adjusted_rand_from_table():
    from math import comb
    from sklearn.metrics import adjusted_rand_score
    from aweSOM.make_sce_clusters import adjusted_rand_from_table

    assert adjusted_rand_from_table(np.diag([4, 3, 5])) == 1.0
    table = np.array([[5, 1, 0], [1, 4, 1], [0, 2, 6]])
    n = table.sum()
    sum_ij = sum(comb(int(v), 2) for v in table.ravel())
    sa = sum(comb(int(v), 2) for v in table.sum(axis=1))
    sb = sum(comb(int(v), 2) for v in table.sum(axis=0))
    expected = sa * sb / comb(int(n), 2)
    ari = (sum_ij - expected) / ((sa + sb) / 2 - expected)
    assert adjusted_rand_from_table(table) == pytest.approx(ari, rel=1e-12)
    # the same partitions as label vectors
    a = np.repeat(np.repeat(np.arange(3), 3), table.ravel())
    b = np.repeat(np.tile(np.arange(3), 3), table.ravel())
    assert adjusted_rand_from_table(table) == pytest.approx(adjusted_rand_score(a, b), rel=1e-12)


def test_anchored_sce_labels(tmp_path):
    from aweSOM.make_sce_clusters import anchored_sce_labels
    from aweSOM.sce import loop_over_all_clusters, find_number_of_clusters

    rng = np.random.default_rng(3)
    truth = rng.permutation(np.repeat(np.arange(3), [100, 120, 80]))
    runs = [truth, np.array([2, 0, 1])[truth], np.array([1, 2, 0])[truth]]  # same clusters, other ids
    flip = rng.choice(300, 10, replace=False)
    runs[2][flip] = (runs[2][flip] + 1) % 3  # run 2 differs from the others on 10 cells
    files = []
    for r, lab in enumerate(runs):
        files.append(str(tmp_path / f"labels.run-{r}.npy"))
        np.save(files[-1], lab.astype(np.uint8))
    (tmp_path / "SCE").mkdir()
    loop_over_all_clusters(files, find_number_of_clusters(files), np.array([300]), str(tmp_path / "SCE"), write_masks=False)
    labels, signals, info = anchored_sce_labels(str(tmp_path / "SCE" / "multimap_mappings.txt"), str(tmp_path), [300])
    assert labels.shape == (300,) and labels.dtype == np.uint8
    assert signals.shape == (3, 300)
    # the medoid is run 0 (tied with run 1, first in order), so the vote returns its ids
    assert info["medoid"].endswith("labels.run-0")
    assert np.array_equal(labels, truth)
    assert np.array_equal(signals.sum(axis=0), np.full(300, 3.0, dtype=np.float32))
    assert [len(g) for g in info["groups"]] == [3, 3, 3]


def test_make_file_name():
    assert make_file_name(1, "png") == "0001.png"
    assert make_file_name(10, "png") == "0010.png"
    assert make_file_name(100, "png") == "0100.png"


if __name__ == "__main__":
    pytest.main()
