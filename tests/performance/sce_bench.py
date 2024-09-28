import time
import numpy as np
import argparse
import os
import shutil
import subprocess

from aweSOM.sce import *
import sce_adapted as sce_adapted
from aweSOM.make_sce_clusters import make_file_name


def generate_mock_clusters(
    dimensions: tuple[int], number_of_clusters: int
) -> np.ndarray:

    return np.random.randint(0, number_of_clusters, dimensions)


def save_mock_clusters(clusters: np.ndarray, filename: str) -> None:

    np.save(filename, clusters)


def test_aweSOM(folder, number_of_points):
    print("Testing aweSOM", flush=True)
    start = time.time()
    print("Starting SCE", flush=True)
    os.chdir(folder)
    cluster_files = glob.glob("*.npy")

    # --------------------------------------------------
    # data
    subfolder = "SCE"
    print(cluster_files)

    # --------------------------------------------------
    # calculate unique number of clusters per run
    nids_array = find_number_of_clusters(cluster_files)
    print("nids_array:", nids_array, flush=True)
    print("There are {} runs".format(len(cluster_files)), flush=True)
    print("There are {} clusters in total".format(np.sum(nids_array)), flush=True)
    count_time = time.time()
    print("Time taken: ", count_time - start, flush=True)

    # --------------------------------------------------
    # generate index for multimap_mapping as the loop runs. Avoid declaring a dict beforehand to avoid memory leaks

    try:  # try to create subfolder, if it exists, pass
        os.mkdir(subfolder)
    except FileExistsError:
        pass

    with open(subfolder + "/multimap_mappings.txt", "w") as f:
        f.write("")

    # --------------------------------------------------
    # make shape of the data
    data_dims = np.array(number_of_points)

    # --------------------------------------------------
    # loop over data files reading image by image and do pairwise comparisons
    # all wrapped inside the loop_over_all_clusters function, which uses JAX for fast computation
    loop_over_all_clusters(cluster_files, nids_array, data_dims)
    end = time.time()
    print("Total time taken: ", end - start, flush=True)

    return end - start


def test_ensemble_learning():
    print("Testing ensemble learning", flush=True)
    start = time.time()

    # --------------------------------------------------
    # run script

    subprocess.run(["python", "sce_adapted.py"])
    end = time.time()
    print("Total time taken: ", end - start, flush=True)

    return end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCE benchmark")
    parser.add_argument("--N", type=int, default=10000, help="Number of data points")
    parser.add_argument("--R", type=int, default=20, help="Number of realizations")

    args = parser.parse_args()

    number_of_points = args.N
    number_of_realizations = args.R
    number_of_clusters = [
        np.random.randint(6, 10) for _ in range(number_of_realizations)
    ]

    cluster_files = [
        f"labels.{make_file_name(i,'npy')}" for i in range(number_of_realizations)
    ]
    folder = "som_out/"

    try:  # try to create subfolder, if it exists, pass
        os.mkdir(folder)
    except FileExistsError:
        pass

    print(f"Generating {number_of_realizations} mock clusters", flush=True)
    for i in range(number_of_realizations):
        clusters = generate_mock_clusters(number_of_points, number_of_clusters[i])
        save_mock_clusters(clusters, folder + cluster_files[i])
    print(f"Mock clusters generated and saved in {folder}", flush=True)

    time_aweSOM = test_aweSOM(folder, args.N)
    print("---------------------------------------------------", flush=True)
    os.chdir("../")
    time_ensemble = test_ensemble_learning()

    print("Done, cleaning up", flush=True)
    shutil.rmtree(folder)
    print(f"Deleted all files inside {folder}", flush=True)

    print("Results:")
    print("Using N =", number_of_points, "and R =", number_of_realizations)
    print(f"aweSOM time: {time_aweSOM}")
    print(f"Ensemble learning time: {time_ensemble}")
    print(f"Ratio: {time_ensemble / time_aweSOM}")
