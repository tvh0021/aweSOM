import time
import numpy as np
import pandas as pd

from aweSOM import Lattice
from aweSOM.run_som import initialize_lattice

import argparse


def generate_data(number_of_features, number_of_points):
    data_dims = (number_of_points, number_of_features)
    training_steps = data_dims[0]
    features_names = [f"feature{i}" for i in range(1, number_of_features + 1)]
    alpha_0 = 0.5

    xdim, ydim = initialize_lattice(random_data(data_dims), 0.5)
    data = random_data(data_dims)

    params = (xdim, ydim, alpha_0, training_steps, "uniform")

    return data, features_names, params


def random_data(data_dims: tuple):
    return np.random.rand(data_dims[0], data_dims[1])


def test_popsom(generated_data):
    data, features_names, params = generated_data

    print(
        f"Benchmarking POPSOM with {data.shape[0]} points and {data.shape[1]} features",
        flush=True,
    )
    start = time.time()

    xdim, ydim, alpha_0, training_steps, sampling_type = params
    data_frame = pd.DataFrame(data, columns=features_names)
    labels = np.random.randint(0, xdim * ydim, data_frame.shape[0])

    from popsom import map

    som = map(xdim, ydim, alpha_0, training_steps)

    print("Training POPSOM", flush=True)
    som.fit(data_frame, labels)
    print("Training done", flush=True)
    training_time = time.time()
    print("Time taken: ", training_time - start, flush=True)

    print("Computing U-matrix", flush=True)
    umat = som.compute_umat()
    print("U-matrix computed", flush=True)
    umat_time = time.time()
    print("Time taken: ", umat_time - training_time, flush=True)

    print("Computing SOM projection", flush=True)
    data_matrix = som.projection()
    data_Xneuron = data_matrix["x"]
    data_Yneuron = data_matrix["y"]
    print("SOM projection computed", flush=True)
    som_labels_time = time.time()
    print("Time taken: ", som_labels_time - umat_time, flush=True)

    print("Computing centroids", flush=True)
    centroids = som.compute_combined_clusters(umat, False, 0.15)
    centr_x = centroids["centroid_x"]
    centr_y = centroids["centroid_y"]

    # create list of centroid _locations
    nx, ny = np.shape(centr_x)

    centr_locs = []
    for i in range(nx):
        for j in range(ny):
            cx = centr_x[i, j]
            cy = centr_y[i, j]

            centr_locs.append((cx, cy))

    unique_ids = list(set(centr_locs))
    # print(unique_ids)
    n_clusters = len(unique_ids)
    print("Number of clusters")
    print(n_clusters)

    mapping = {}
    for I, key in enumerate(unique_ids):
        # print(key, I)
        mapping[key] = I

    clusters = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            key = (centr_x[i, j], centr_y[i, j])
            I = mapping[key]

            clusters[i, j] = I

    print("Centroids computed", flush=True)
    centroids_time = time.time()
    print("Time taken: ", centroids_time - som_labels_time, flush=True)

    print("Computing cluster labels", flush=True)
    data_back = np.zeros((data.shape[0]))

    for ix in range(data.shape[0]):
        data_back[ix] = clusters[data_Xneuron[ix], data_Yneuron[ix]]
        if ix % (data.shape[0] // 10) == 0:
            print(ix)
    print("Cluster labels computed", flush=True)
    cluster_labels_time = time.time()
    print("Time taken: ", cluster_labels_time - centroids_time, flush=True)

    end = time.time()
    print(f"Total time taken: {end - start:.3f} s", flush=True)

    return end - start


def test_aweSOM(generated_data):
    data, features_names, params = generated_data

    print(
        f"Benchmarking aweSOM with {data.shape[0]} points and {data.shape[1]} features",
        flush=True,
    )
    start = time.time()

    xdim, ydim, alpha_0, training_steps, sampling_type = params
    labels = np.random.randint(0, xdim * ydim, data.shape[0])

    lattice = Lattice(
        xdim,
        ydim,
        alpha_0,
        training_steps,
        alpha_type="static",
        sampling_type=sampling_type,
    )

    print("Training aweSOM", flush=True)
    lattice.train_lattice(
        data,
        features_names,
        labels,
    )
    print("Training done", flush=True)
    training_time = time.time()
    print("Time taken: ", training_time - start, flush=True)

    print("Computing U-matrix", flush=True)
    lattice.compute_umat()
    print("U-matrix computed", flush=True)
    umat_time = time.time()
    print("Time taken: ", umat_time - training_time, flush=True)

    print("Computing SOM projection and centroids", flush=True)
    projection_2d = lattice.map_data_to_lattice()
    final_clusters = lattice.assign_cluster_to_lattice(smoothing=None, merge_cost=0.0)
    print("SOM projection and centroids computed", flush=True)
    som_labels_time = time.time()
    print("Time taken: ", som_labels_time - umat_time, flush=True)

    print("Computing cluster labels", flush=True)
    som_labels = lattice.assign_cluster_to_data(projection_2d, final_clusters)
    print("Cluster labels computed", flush=True)
    cluster_labels_time = time.time()
    print("Time taken: ", cluster_labels_time - som_labels_time, flush=True)

    end = time.time()
    print(f"Total time taken: {end - start:.3f} s", flush=True)

    return end - start


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Perform benchmarking of aweSOM and POPSOM"
    )
    parser.add_argument("--N", type=int, default=10000, help="Number of data points")
    parser.add_argument("--F", type=int, default=10, help="Number of features")
    args = parser.parse_args()

    # number_of_features = np.arange(2, 11, 4)
    # number_of_points = [1000, 10000, 100000, 1000000]

    n = args.N
    f = args.F

    generated_data = generate_data(f, n)
    data, features_names, params = generated_data
    xdim, ydim, alpha_0, training_steps, sampling_type = params

    print(f"SOM lattice of size {xdim}x{ydim} trained for {training_steps} steps")

    results = []

    time_aweSOM = test_aweSOM(generated_data)
    print("---------------------------------------------------", flush=True)
    # time_popsom = test_popsom(generated_data)

    results.append(
        {
            "N": n,
            "F": f,
            # "POPSOM": time_popsom,
            "aweSOM": time_aweSOM,
            # "ratio": time_popsom / time_aweSOM,
        }
    )

    print(results)
    print("--------------------", flush=True)
