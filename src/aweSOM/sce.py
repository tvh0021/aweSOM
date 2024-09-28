## Statistically Combined Ensemble (SCE) code for N-dimensional data
## See Ha et al. (2024) for implementation details
## For detailed mathematical description, see Bussov & Nattila (2021)
## https://github.com/mkruuse/segmenting-turbulent-simulations-with-ensemble-learning
## If GPU-enabled, capable of handling 1000^3 data with 1 A100-80GB GPU.

import os
import glob
import argparse
import numpy as np

# Use JAX if GPU/the jax package is installed, otherwise use NumPy
import jax

try:
    default_device = jax.default_backend()
    if default_device == "gpu":
        USE_JAX = True
    else:
        USE_JAX = False
except:
    USE_JAX = False

# Define a unified interface for the functions
if USE_JAX:
    from jax import numpy as jnp
    from jax import jit

    print("Using JAX for GPU computation")
    array_lib = jnp  # Use JAX's numpy interface
else:
    print("Using NumPy for CPU computation")
    array_lib = np  # Use NumPy


# Define a conditional jit decorator
def conditional_jit(func):
    if USE_JAX:
        return jit(func)
    else:
        return func


# read array from clusterID.npy
def load_som_npy(path: str) -> array_lib.ndarray:
    return array_lib.load(path, "r")


@conditional_jit
def create_mask(img: array_lib.ndarray, cid: int) -> array_lib.ndarray:
    """Create a mask for a given cluster id

    Args:
        img (jnp.ndarray): 3D array of cluster ids
        cid (int): cluster id to mask

    Returns:
        (j)np.ndarray: masked cluster, 1 where cluster id is cid, 0 elsewhere
    """
    return array_lib.where(img == cid, 1, 0)


def compute_SQ(mask: array_lib.ndarray, maskC: array_lib.ndarray):
    """Compute the quality index between two masks

    Args:
        mask ((j)np.ndarray): mask of cluster C
        maskC ((j)np.ndarray): mask of cluster C'

    Returns:
        SQ (float): quality index, equals to S/Q
        SQ_matrix ((j)np.ndarray): pixelwise quality index, equals to S/Q * mask
    """
    # --------------------------------------------------
    # product of two masked arrays; corresponds to intersection
    I = array_lib.multiply(mask, maskC)

    # --------------------------------------------------
    # sum of two masked arrays; corresponds to union
    U = array_lib.ceil((mask + maskC) * 0.5)
    # U_area = array_lib.sum(U) / (nx * ny * nz)

    # --------------------------------------------------
    # Intersection signal strength of two masked arrays, S
    S = array_lib.sum(I) / array_lib.sum(U)

    # --------------------------------------------------
    # Union quality of two masked arrays, Q
    if array_lib.max(mask) == 0 or array_lib.max(maskC) == 0:
        return 0.0, array_lib.zeros(mask.shape)

    Q = array_lib.sum(U) / (array_lib.sum(mask) + array_lib.sum(maskC)) - array_lib.sum(
        I
    ) / (array_lib.sum(mask) + array_lib.sum(maskC))
    if Q == 0.0:
        return 0.0, array_lib.zeros(
            mask.shape
        )  # break here because this causes NaNs that accumulate.

    # --------------------------------------------------
    # final measure for this comparison is (S/Q) x Union
    SQ = S / Q
    SQ_matrix = SQ * mask

    return SQ, SQ_matrix


def loop_over_all_clusters(
    all_files: list[str], number_of_clusters: array_lib.ndarray, dimensions: np.ndarray
) -> int:
    """
    Loops over all clusters in the given data, compute goodness-of-fit, then save Gsum values to file.

    Args:
        all_files (list[str]): A list of data files saved in '.npy' format.
        number_of_clusters ((j)np.ndarray): An array of the number of cluster ids in each run.
        dimensions (np.ndarray): A 1d array representing the dimensions of the clusters (can be any dimension but nx*ny*nz has to be equal to number of data points).

    Returns:
        Save Gsum value of each cluster C to a file.
    """
    pass

    runs = all_files  # [file.strip('.npy') for file in all_files]

    # loop over data files reading image by image
    for i in range(len(runs)):
        run = runs[i]

        clusters_1d = load_som_npy(run)
        print("-----------------------")
        print("Run : ", run, flush=True)

        with open(subfolder + "/multimap_mappings.txt", "a") as f:
            f.write("{}\n".format(run.strip(".npy")))

        # nx x ny x nz size maps
        # nz,ny,nx = array_lib.cbrt(clusters_1d.shape[0]).astype(int), array_lib.cbrt(clusters_1d.shape[0]).astype(int), array_lib.cbrt(clusters_1d.shape[0]).astype(int)
        # clusters = clusters_1d.reshape(nz,ny,nx)
        clusters = clusters_1d.reshape(dimensions)

        # unique ids
        nids = number_of_clusters[i]  # number of cluster ids in this run
        # ids = np.arange(nids)
        print("nids : ", nids)

        for cid in range(nids):
            # print('  -----------------------')
            # print('  cid : ', cid, flush=True)

            # create masked array where only id == cid are visible
            mask = create_mask(clusters, cid)

            total_mask = array_lib.zeros(dimensions, dtype=float)

            total_SQ_scalar = 0.0

            for j in range(len(runs)):
                runC = runs[j]

                if j == i:  # don't compare to itself
                    continue

                clustersC_1d = load_som_npy(runC)
                clustersC = clustersC_1d.reshape(dimensions)

                # print('    -----------------------')
                # print('   ',runC, flush=True)

                nidsC = number_of_clusters[j]  # number of cluster ids in this run
                # print('    nidsC : ', nidsC)

                for cidC in range(nidsC):
                    maskC = create_mask(clustersC, cidC)

                    SQ, SQ_matrix = compute_SQ(mask, maskC)

                    # pixelwise stacking of 2 masks
                    total_mask += SQ_matrix  # for numpy array

                    total_SQ_scalar += SQ

            # save total mask to file
            # print("Saving total mask to file", flush=True)
            array_lib.save(
                subfolder + "/mask-{}-id{}.npy".format(run.strip(".npy"), cid),
                total_mask,
            )

            # print("Saving total SQ scalar to multimap_mapping", flush=True)
            with open(subfolder + "/multimap_mappings.txt", "a") as f:
                f.write("{} {}\n".format(cid, total_SQ_scalar))

    return 0


def find_number_of_clusters(cluster_files: list[str]) -> array_lib.ndarray:
    """
    Find the number of clusters in each run.

    Args:
        cluster_files (list[str]): A list of data files saved in '.npy' format.

    Returns:
        number_of_clusters ((j)np.ndarray): An array of the number of cluster ids in each run.
    """

    number_of_clusters = np.empty(len(cluster_files), dtype=int)
    for run in range(len(cluster_files)):
        clusters = load_som_npy(cluster_files[run])
        ids = array_lib.unique(clusters)
        number_of_clusters[run] = len(ids)

    return number_of_clusters


def parse_args():
    """argument parser for the sce.py script"""
    parser = argparse.ArgumentParser(description="SCE code")
    parser.add_argument(
        "--folder", type=str, dest="folder", default=os.getcwd(), help="Folder name"
    )
    parser.add_argument(
        "--subfolder", type=str, dest="subfolder", default="SCE", help="Subfolder name"
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

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    print("Starting SCE3d", flush=True)
    folder = args.folder
    os.chdir(folder)
    cluster_files = glob.glob("*.npy")

    # --------------------------------------------------
    # data
    subfolder = args.subfolder
    print(cluster_files)

    # --------------------------------------------------
    # calculate unique number of clusters per run
    nids_array = find_number_of_clusters(cluster_files)
    print("nids_array:", nids_array, flush=True)
    print("There are {} runs".format(len(cluster_files)), flush=True)
    print("There are {} clusters in total".format(np.sum(nids_array)), flush=True)

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
    data_dims = np.array(args.dims)

    # --------------------------------------------------
    # loop over data files reading image by image and do pairwise comparisons
    # all wrapped inside the loop_over_all_clusters function, which uses JAX for fast computation
    loop_over_all_clusters(cluster_files, nids_array, data_dims)
