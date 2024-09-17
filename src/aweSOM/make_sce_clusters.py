import argparse
import os
import matplotlib.pyplot as plt
import shutil
import numpy as np
from scipy.signal import savgol_filter


def plot_gsum_values(
    gsum_values: list[float], minimas: list[int] = None, file_path: str = None
):
    """
    Plot the gsum values with optional minima markers.

    Args:
        gsum_values (list[float]): A list of gsum values to plot.
        minimas (list[int], optional): A list of indices indicating the minima to highlight. Defaults to None.
        file_path (str, optional): The directory path where the plot will be saved. If None, the plot will be displayed. Defaults to None.

    Returns:
        None: This function does not return a value. It either displays the plot or saves it to a file.
    """
    plt.figure(dpi=300)
    plt.plot(
        list(range(len(gsum_values))),
        gsum_values,
        marker="o",
        c="k",
        markersize=2,
        linewidth=1,
    )
    if minimas is not None:
        plt.scatter(
            minimas,
            [gsum_values[i] for i in minimas],
            c="b",
            marker="x",
            label="Minimas",
        )
        plt.legend()
    plt.title(f"Sorted gsum values")
    plt.xlabel("Ranked clusters")
    plt.ylabel("Gsum value")
    plt.grid()
    if file_path is None:
        plt.show()
    else:
        plt.savefig(f"{file_path}/gsum_values.png")
        print("Saved gsum values plot")


def plot_gsum_deriv(
    gsum_deriv: np.ndarray,
    threshold: float,
    minimas: list[int] = None,
    file_path: str = None,
):
    """
    Plots the gsum derivative with optional minima highlighted.

    Args:
        gsum_deriv (np.ndarray): An array of gsum derivative values to be plotted.
        threshold (float): The threshold value to draw a horizontal line on the plot.
        minimas (list[int], optional): A list of indices representing the minima to be highlighted on the plot. Defaults to None.
        file_path (str, optional): The file path where the plot will be saved. If None, the plot will be displayed instead. Defaults to None.

    Returns:
        None: This function does not return a value. It either displays the plot or saves it to a file.
    """

    x_range = list(range(len(gsum_deriv)))
    plt.figure(dpi=300)
    print("minimas", minimas, flush=True)
    plt.plot(x_range, gsum_deriv, marker="o", c="k", markersize=2, linewidth=1)
    if minimas is not None:
        plt.scatter(minimas, [gsum_deriv[i] for i in minimas], c="b", marker="x")
    plt.ylim(threshold * 5, 0.0)
    plt.title(f"Sorted gsum derivatives")
    plt.xlabel("Ranked clusters")
    plt.ylabel("Gsum derivative")
    plt.grid()
    plt.hlines(threshold, 0, x_range[-1], colors="r", linestyles="--")
    if file_path is None:
        plt.show()
    else:
        plt.savefig(f"{file_path}/gsum_deriv.png")
        print("Saved gsum derivative plot")


def get_gsum_values(mapping_file: str):
    """Get the gsum values from the mapping file

    Args:
        mapping_file (str): path to the mapping file

    Returns:
        list: gsum values
        dict: mapping of gsum values to cluster id and cluster name
    """
    mapping = dict()
    with open(mapping_file, "r") as f:
        for line in f:
            line = line.strip("\n")
            if "-" in line:
                key_name = line
                mapping[key_name] = []
            else:
                mapping[key_name].append(line.split(" "))

    map_list = []
    for key in mapping.keys():
        map_list.extend([[float(i[1]), int(i[0]), key] for i in mapping[key]])

    map_list.sort(key=lambda map_list: map_list[0], reverse=True)

    gsum_values = [map_list[i][0] for i in range(len(map_list))]

    return gsum_values, map_list


def get_sce_cluster_separation(gsum_deriv: np.ndarray, threshold: float):
    """
    Identify the separation of clusters in a given derivative array based on a specified threshold.

    Args:
        gsum_deriv (np.ndarray): A 1D array representing the derivative values.
        threshold (float): The threshold value used to determine cluster separation.

    Returns:
        tuple: A tuple containing:
            - list: A list of ranges for the identified clusters, where each range is represented as a list of two integers.
            - list: A list of indices representing the local minima found below the threshold.
    """

    threshold_crossed = False  # True if gsum_deriv[0] < threshold else False

    minimas = []
    for i in range(1, len(gsum_deriv) - 1):
        if (
            (gsum_deriv[i] < threshold)
            & (gsum_deriv[i] < gsum_deriv[i - 1])
            & (gsum_deriv[i] < gsum_deriv[i + 1])
            & (threshold_crossed == True)
        ):
            minimas.append(i)
            threshold_crossed = False

        if (gsum_deriv[i] > threshold) & (threshold_crossed == False):
            threshold_crossed = True

    minimas.pop(
        0
    )  # remove the first minimum because it is usually part of the first cluster
    # from the local minima, find the ranges of the clusters
    cluster_ranges = []
    for i in range(len(minimas) - 1):
        if i == 0:
            cluster_ranges.append([0, minimas[i]])

        cluster_ranges.append([minimas[i], minimas[i + 1]])

        if i == len(minimas) - 2:
            cluster_ranges.append([minimas[i + 1], len(gsum_deriv)])

    return cluster_ranges, minimas


def combine_separated_clusters(
    map_list: list, cluster_ranges: list[list[int]], dims: int, file_path: str
) -> np.ndarray:
    """
    Combine separated clusters by summing their corresponding gsum masks.

    Args:
        map_list (list): A list of instances representing the binary maps.
        cluster_ranges (list[list[int]]): A list of ranges indicating the start and end indices for each cluster.
        dims (int): The dimensions of the binary maps.
        file_path (str): The file path where the binary maps are stored.

    Returns:
        np.ndarray: A numpy array containing the summed binary maps for each cluster.
    """

    remapped_clusters = dict()

    for i in range(len(cluster_ranges)):
        start_pointer, end_pointer = cluster_ranges[i]
        remapped_clusters[i] = []
        for j in range(start_pointer, end_pointer):
            remapped_clusters[i].append(map_list[j])

    print(
        "Length of remapped clusters : ",
        [len(remapped_clusters[k]) for k in remapped_clusters.keys()],
        flush=True,
    )

    # Add values of the binary map of each cluster to obtain a new map
    # read in each binary map within a cluster_range, then sum them up
    all_signals_map = np.empty(([len(remapped_clusters)] + dims), dtype=np.float32)
    for cluster in remapped_clusters.keys():
        print("Currently analyzing cluster : ", cluster, flush=True)
        print(
            "Number of instances in cluster : ",
            len(remapped_clusters[cluster]),
            flush=True,
        )

        # cannot use jax here because it uses too much memory; cannot use numba because it does not support np.load; loading all binary maps in each cluster at once will use more memory, but is also ~30% faster than loading them sequentially and adding to total every step.
        this_cluster_signal_map = np.zeros(
            ([len(remapped_clusters[cluster])] + dims), dtype=np.float32
        )
        for i, instance in enumerate(remapped_clusters[cluster]):
            if i % 10 == 0:
                print("Instance", i, flush=True)
            this_cluster_signal_map[i] = np.reshape(
                np.load(file_path + f"/mask-{instance[2]}-id{instance[1]}.npy"),
                newshape=dims,
            )

        all_signals_map[cluster] = np.sum(this_cluster_signal_map, axis=0)

    return all_signals_map


def makeFilename(n: int) -> str:
    """Make a filename based on the number given.

    Args:
        n (int): number to be converted to a filename

    Returns:
        str: filename
    """
    if n < 10:
        file_n = "000" + str(n)
    elif (n >= 10) & (n < 100):
        file_n = "00" + str(n)
    else:
        file_n = "0" + str(n)

    return f"{file_n}.png"


def parse_args():
    """argument parser for the make_sce_clusters.py script"""
    parser = argparse.ArgumentParser(
        description="Use multimap mapping to analyze and segment groups of features"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        dest="file_path",
        default=os.getcwd(),
        help="Multimap mapping file path",
    )
    parser.add_argument(
        "--copy_clusters",
        dest="copy_clusters",
        action="store_true",
        help="Copy the clusters to a new folder",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        dest="threshold",
        default=-0.015,
        help="Threshold for the derivative of the gsum values",
    )
    parser.add_argument(
        "--return_gsum",
        dest="return_gsum",
        action="store_true",
        help="Return the sorted gsum values plot",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        action="store",
        type=int,
        dest="dims",
        default=[640, 640, 640],
        help="Dimensions of the data",
    )
    parser.add_argument(
        "--save_combined_map",
        dest="save_combined_map",
        action="store_true",
        help="Save the combined map of all clusters",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    gsum_values, map_list = get_gsum_values(args.file_path + "/multimap_mappings.txt")
    print("Length of sorted map", len(gsum_values), flush=True)

    # now iterate through the list and copy the files to the appropriate cluster folder
    if args.copy_clusters:
        ranked_clusters_dir = os.path.join(args.file_path, "ranked-clusters")
        if not os.path.exists(ranked_clusters_dir):
            os.makedirs(ranked_clusters_dir)

        for i in range(len(gsum_values)):
            origin_file_name = "{}/mask-{}_id{}.png".format(
                args.file_path, map_list[i][2], map_list[i][1]
            )
            destination_file_name = "{}/ranked-clusters/{}".format(
                args.file_path, makeFilename(i)
            )
            shutil.copyfile(origin_file_name, destination_file_name)

        print("Done copying files")

    # apply a Savitzky-Golay filter to smooth the gsum values
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
    threshold = args.threshold
    cluster_ranges, minimas = get_sce_cluster_separation(gsum_deriv, threshold)

    print("Minimas", minimas, flush=True)
    print("Cluster ranges", cluster_ranges, flush=True)
    print("Number of clusters", len(cluster_ranges), flush=True)

    # plot the gsum and gsum_deriv values
    if args.return_gsum:
        plot_gsum_values(gsum_values, minimas, args.file_path)
        plot_gsum_deriv(gsum_deriv, threshold, minimas, args.file_path)

    # save the separated SCE clusters
    if args.save_combined_map:
        combined_sce_clusters = combine_separated_clusters(
            map_list, cluster_ranges, args.dims, args.file_path
        )
        # save the new binary map
        np.save(
            args.file_path + f"/sce_clusters_{threshold}.npy", combined_sce_clusters
        )
        print(
            f"Saved new combined clusters as {args.file_path}/sce_clusters_{threshold}.npy"
        )
