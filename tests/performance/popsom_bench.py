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


def test_popsom(generated_data, procedure):
    """test POPSOM code for scaling

    Args:
        generated_data (tuple): all the initial data
        procedure (str, optional): 'training', 'mapping', or 'both'.

    Returns:
        float: time taken to run the procedure in seconds
    """
    data, features_names, params = generated_data

    print(
        f"Benchmarking POPSOM with {data.shape[0]} points and {data.shape[1]} features \n",
        flush=True,
    )
    start = time.time()

    xdim, ydim, alpha_0, training_steps, sampling_type = params
    data_frame = pd.DataFrame(data, columns=features_names)
    labels = np.random.randint(0, xdim * ydim, data_frame.shape[0])

    from popsom import map

    som = map(xdim, ydim, alpha_0, training_steps)

    if procedure == "both" or procedure == "training":
        print("Training POPSOM", flush=True)
        som.fit(data_frame, labels)
        print("Training done", flush=True)
        training_time = time.time()
        print("Time taken: ", training_time - start, flush=True)
        print("\n", flush=True)
    else:
        print("Skipping training", flush=True)
        som.train = 10  # train for only 10 steps (should take almost no time) to generate some data for the next steps
        som.fit(data_frame, labels)
        print("Lattice values populated", flush=True)
        start = time.time()  # reset start time
        training_time = start
        print("\n", flush=True)

    if procedure == "both" or procedure == "mapping":
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

    if procedure == "mapping":
        return {"map": end - start}
    elif procedure == "training":
        return {"train": end - start}
    else:
        return {"train": training_time - start, "map": end - training_time}


def test_aweSOM(generated_data, procedure):
    data, features_names, params = generated_data

    print(
        f"Benchmarking aweSOM with {data.shape[0]} points and {data.shape[1]} features \n",
        flush=True,
    )
    start = time.time()

    xdim, ydim, alpha_0, training_steps, sampling_type = params
    labels = np.random.randint(0, xdim * ydim, data.shape[0])

    if procedure == "both" or procedure == "training":
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
        print("\n", flush=True)
    else:
        lattice = Lattice(
            xdim,
            ydim,
            alpha_0,
            100,  # train for only 100 steps (should take almost no time) to generate some data for the next steps
            alpha_type="static",
            sampling_type=sampling_type,
        )
        print("Begin dummy training for 100 steps", flush=True)
        lattice.train_lattice(
            data,
            features_names,
            labels,
        )
        print("Lattice values populated", flush=True)
        start = time.time()  # reset start time
        training_time = start
        print("\n", flush=True)

    if procedure == "both" or procedure == "mapping":
        print("Computing U-matrix", flush=True)
        lattice.compute_umat()
        print("U-matrix computed", flush=True)
        umat_time = time.time()
        print("Time taken: ", umat_time - training_time, flush=True)

        print("Computing SOM projection and centroids", flush=True)
        projection_2d = lattice.map_data_to_lattice()
        final_clusters = lattice.assign_cluster_to_lattice(
            smoothing=None, merge_cost=0.0
        )
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

    if procedure == "mapping":
        return {"map": end - start}
    elif procedure == "training":
        return {"train": end - start}
    else:
        return {"train": training_time - start, "map": end - training_time}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Perform benchmarking of aweSOM and POPSOM"
    )
    parser.add_argument("--N", type=int, default=10000, help="Number of data points")
    parser.add_argument("--F", type=int, default=10, help="Number of features")
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Number of nodes in lattice",
        required=False,
    )
    parser.add_argument(
        "--procedure",
        type=str,
        default="both",
        help="Procedure to run: 'training', 'mapping', or 'both'",
    )
    parser.add_argument("--popsom", action="store_true", help="Run POPSOM")
    parser.add_argument("--awesom", action="store_true", help="Run aweSOM")
    args = parser.parse_args()

    n = args.N
    f = args.F
    procedure = args.procedure
    if procedure not in ["training", "mapping", "both"]:
        raise ValueError("Invalid procedure")
    generated_data = generate_data(f, n)
    data, features_names, params = generated_data

    if args.nodes:
        number_of_nodes = args.nodes
        _, _, alpha_0, training_steps, sampling_type = params
        ydim = int(np.sqrt(number_of_nodes / 2))
        xdim = ydim * 2
        # modify params to reflect manually set lattice size
        generated_data = tuple(
            [data, features_names, (xdim, ydim, alpha_0, training_steps, sampling_type)]
        )
    else:
        xdim, ydim, alpha_0, training_steps, sampling_type = params
        number_of_nodes = xdim * ydim

    print("---------------------------------------", flush=True)
    print("| SCALING TEST FOR SOM IMPLEMENTATION |", flush=True)
    print("---------------------------------------", flush=True)
    if procedure == "training":
        print("Training only", flush=True)
    elif procedure == "mapping":
        print("Mapping only", flush=True)
    else:
        print("Training and mapping", flush=True)
    print(f"SOM lattice of size {xdim}x{ydim} training for {training_steps} steps")
    print("\n", flush=True)

    results = []

    if args.awesom:
        # print("Running aweSOM", flush=True)
        time_aweSOM = test_aweSOM(generated_data, procedure=procedure)
        print("---------------------------------------------------", flush=True)
        results.append(
            {
                "N": n,
                "F": f,
                "N_nodes": number_of_nodes,
                "aweSOM": time_aweSOM,
            }
        )
    elif args.popsom:
        # print("Running POPSOM", flush=True)
        time_popsom = test_popsom(generated_data, procedure=procedure)
        results.append(
            {
                "N": n,
                "F": f,
                "N_nodes": number_of_nodes,
                "POPSOM": time_popsom,
            }
        )
    else:
        time_aweSOM = test_aweSOM(generated_data, procedure=procedure)
        print("---------------------------------------------------", flush=True)
        time_popsom = test_popsom(generated_data, procedure=procedure)

        results.append(
            {
                "N": n,
                "F": f,
                "N_nodes": number_of_nodes,
                "POPSOM": time_popsom,
                "aweSOM": time_aweSOM,
            }
        )

    print("Done", flush=True)
    print(results)
    print("--------------------", flush=True)
