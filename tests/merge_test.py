"""The non-recursive merge must give exactly what the original recursive merge gave.

reference_merge is a frozen copy of Lattice.merge_similar_centroids as it was before
the rewrite (recursive, every pairwise cost recomputed each level).
"""

import contextlib
import inspect
import io
import sys
import time

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from aweSOM import Lattice


def reference_merge(lattice_obj, naive_centroids, threshold):
    heat = lattice_obj.umat
    centroids = naive_centroids.copy()
    unique_centroids = lattice_obj.get_unique_centroids(centroids)

    cost_between_centroids = []

    for i in range(len(unique_centroids["position_x"]) - 1):
        for j in range(i + 1, len(unique_centroids["position_x"])):
            a = [
                unique_centroids["position_x"][i],
                unique_centroids["position_y"][i],
            ]
            b = [
                unique_centroids["position_x"][j],
                unique_centroids["position_y"][j],
            ]
            num_sample = 5 * int(np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2))
            x, y = np.linspace(a[0], b[0], num_sample), np.linspace(
                a[1], b[1], num_sample
            )
            umat_dist = map_coordinates(heat**2, np.vstack((x, y)))
            total_cost = np.sum(umat_dist)
            cost_between_centroids.append([a, b, total_cost])

    if len(cost_between_centroids) == 0:
        return centroids

    sorted_cost = sorted(cost_between_centroids, key=lambda x: x[2])
    sorted_cost = [[a, b, c / sorted_cost[-1][2]] for a, b, c in sorted_cost]

    if sorted_cost[0][2] < threshold:
        centroid_a = tuple(sorted_cost[0][0])
        centroid_b = tuple(sorted_cost[0][1])
        nodes_a = lattice_obj.nodes_count[centroid_a]
        nodes_b = lattice_obj.nodes_count[centroid_b]

        if nodes_a < nodes_b:
            centroids = lattice_obj.replace_value(centroids, centroid_a, centroid_b)
        else:
            centroids = lattice_obj.replace_value(centroids, centroid_b, centroid_a)

        centroids = reference_merge(lattice_obj, centroids, threshold)
    else:
        lattice_obj.get_unique_centroids(centroids)
        return centroids

    return centroids


def _copy_centroids(c):
    return {k: v.copy() for k, v in c.items()}


def _trained_map(seed):
    rng = np.random.default_rng(seed)
    xdim, ydim = rng.integers(8, 25, size=2)
    n_blobs = rng.integers(3, 9)
    centres = rng.normal(scale=4.0, size=(n_blobs, 4))
    data = centres[rng.integers(0, n_blobs, 2000)] + rng.normal(size=(2000, 4))
    som = Lattice(int(xdim), int(ydim), 0.3, 400)
    som.seed = seed
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(data, list("abcd"), history=False)
    return som


@pytest.mark.parametrize("seed", range(20))
def test_merge_matches_recursive_reference(seed):
    som = _trained_map(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        naive = som.compute_centroids(False)
        for threshold in (0.1, 0.2, 0.4):
            want = reference_merge(som, _copy_centroids(naive), threshold)
            want_counts = dict(som.nodes_count)
            got = som.merge_similar_centroids(_copy_centroids(naive), threshold)
            assert np.array_equal(got["centroid_x"], want["centroid_x"])
            assert np.array_equal(got["centroid_y"], want["centroid_y"])
            assert som.nodes_count == want_counts

            clusters_new = som.assign_cluster_to_lattice(merge_cost=threshold)
            original = som.merge_similar_centroids
            som.merge_similar_centroids = lambda c, t: reference_merge(som, c, t)
            try:
                clusters_ref = som.assign_cluster_to_lattice(merge_cost=threshold)
            finally:
                som.merge_similar_centroids = original
            assert np.array_equal(clusters_new, clusters_ref)


def test_large_lattice_merge_has_no_recursion_limit():
    # a rough random U-matrix has ~270 local minima, so ~270 merge rounds; the
    # recursive version needed one stack frame per merge
    som = Lattice(60, 40, 0.3, 100)
    som.umat = np.random.default_rng(0).random((60, 40))
    with contextlib.redirect_stdout(io.StringIO()):
        naive = som.compute_centroids(False)
        n_naive = len(som.get_unique_centroids(naive)["position_x"])
        # leave fewer spare frames than there are merges, so a recursive merge fails
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(len(inspect.stack()) + 150)
        try:
            start = time.perf_counter()
            merged = som.merge_similar_centroids(_copy_centroids(naive), 0.3)
            elapsed = time.perf_counter() - start
        finally:
            sys.setrecursionlimit(limit)
    n_merged = len(som.get_unique_centroids(merged)["position_x"])
    assert n_naive > 100
    assert n_naive - n_merged > 100  # more merges than a shallow recursion would allow
    assert elapsed < 120
