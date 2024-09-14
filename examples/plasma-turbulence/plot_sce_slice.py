import argparse
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--return_fig", dest="return_fig", action="store_true", help="Return the figure"
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        dest="reference_file",
        default="/mnt/home/tha10/ceph/SOM-tests/lr-d3x128/production/features_2j1b1e0r_5000_jasym.h5",
        help="Reference file to compare the clusters to",
        required=False,
    )
    parser.add_argument(
        "--slice",
        type=int,
        dest="slice",
        default=27,
        help="Slice number, make sure this matches the slice number in the sce_slice.py call",
    )
    return parser.parse_args()


if __name__ == "__main__":
    remapped_clusters = dict()

    for i in range(len(cluster_ranges)):
        start_pointer = cluster_ranges[i][0]
        end_pointer = cluster_ranges[i][1]

        key_name = str(i)
        remapped_clusters[key_name] = []

        for j in range(start_pointer, end_pointer):
            remapped_clusters[key_name].append(map_list[j])

    print(
        "Length of remapped clusters : ",
        [len(remapped_clusters[k]) for k in remapped_clusters.keys()],
        flush=True,
    )
    # print ("First cluster : ", remapped_clusters['0'])

    # add values of the binary map of each cluster to obtain a new map
    # read in the binary map
    dims = args.dims
    all_binary_maps = np.empty(([len(remapped_clusters)] + dims), dtype=np.float32)
    for cluster in remapped_clusters.keys():
        print("Currently analyzing cluster : ", cluster, flush=True)
        print(
            "Number of instances in cluster : ",
            len(remapped_clusters[cluster]),
            flush=True,
        )

        # cannot use jax here because it uses too much memory; cannot use numba because it does not support np.load; loading all binary maps in each cluster at once will use more memory, but is also ~30% faster than loading them sequentially and adding to total every step.
        cluster_binary_map = np.zeros(
            ([len(remapped_clusters[cluster])] + dims), dtype=np.float32
        )
        for i, instance in enumerate(remapped_clusters[cluster]):
            print("Instance", i, flush=True)
            cluster_binary_map[i] = np.reshape(
                np.load(
                    args.file_path
                    + "/mask-{}-id{}.npy".format(instance[2], instance[1])
                ),
                newshape=dims,
            )

        all_binary_maps[int(cluster)] = np.sum(cluster_binary_map, axis=0)
