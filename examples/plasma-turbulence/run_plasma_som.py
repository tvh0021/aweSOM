## Example script to run the SOM code on plasma turbulence data
## using the multiprocessing module to parallelize the runs

import os
import h5py as h5
import argparse

from aweSOM import Lattice
from aweSOM.run_som import *

import itertools
from multiprocessing import Pool, cpu_count


def save_som_run(
    params, normalized_data, feature_list, merge_threshold, name_of_dataset
):
    H, alpha_0, train = params
    print("H: {}, alpha_0: {}, train: {}".format(H, alpha_0, train))
    xdim, ydim = initialize_lattice(normalized_data, H)
    map = Lattice(xdim, ydim, alpha_0, train, sampling_type="uniform")
    map.train_lattice(normalized_data, feature_list)
    projection_2d = map.map_data_to_lattice()
    final_clusters = map.assign_cluster_to_lattice(
        smoothing=None, merge_cost=merge_threshold
    )
    som_labels = map.assign_cluster_to_data(projection_2d, final_clusters)
    save_cluster_labels(
        som_labels,
        xdim,
        ydim,
        alpha_0,
        train,
        initial="u",
        name_of_dataset=name_of_dataset,
    )
    save_som_object(
        map,
        xdim,
        ydim,
        alpha_0,
        train,
        initial="u",
        name_of_dataset=name_of_dataset,
    )


def parse_args():
    """argument parser for the run_plasma_som.py script"""
    parser = argparse.ArgumentParser(
        description="Run SOM code given a range of parameters"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        dest="ratio",
        nargs="+",
        help="Ratio of the lattice",
        default=[0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--alpha_0",
        type=float,
        dest="alpha_0",
        nargs="+",
        help="Initial learning rate",
        default=[0.05, 0.1, 0.2, 0.4],
    )
    parser.add_argument(
        "--train",
        type=int,
        dest="train",
        nargs="+",
        help="Number of training steps",
        default=[2097152, 4194304],
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    parameters = dict()
    parameters["ratio"] = args.ratio
    parameters["alpha_0"] = args.alpha_0
    parameters["train"] = args.train

    merge_threshold = 0.2

    working_dir = os.getcwd() + "/"
    file_name = "features_2j1b1e0r_5000_jasym.h5"  # change this to the right file name

    name_of_dataset = file_name.split("_")[2].split(".h5")[0]
    sampling_type = "u"  # uniform sampling; can be changed to "s" for random sampling

    # load data
    with h5.File(working_dir + file_name, "r") as f:
        x = f["features"][()]
        feature_list = f["names"][()]
    feature_list = [f.decode("utf-8") for f in feature_list]

    # normalize data
    normalized_data = manual_scaling(x)

    combinations = list(
        itertools.product(
            parameters["ratio"], parameters["alpha_0"], parameters["train"]
        )
    )

    with Pool(cpu_count()) as p:
        items = [
            (c, normalized_data, feature_list, merge_threshold, name_of_dataset)
            for c in combinations
        ]

        p.starmap(save_som_run, items)
