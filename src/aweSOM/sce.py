## Statistically Combined Ensemble (SCE) code for N-dimensional data
## See Ha et al. (2024) for implementation details
## For detailed mathematical description, see Bussov & Nattila (2021)
## https://github.com/mkruuse/segmenting-turbulent-simulations-with-ensemble-learning
## If GPU-enabled, capable of handling 1000^3 data with 1 A100-80GB GPU.
##
## loop_over_all_clusters works on per-run contingency tables (cluster-pair cell counts
## from np.bincount), so the cost is O(runs^2 * cells) instead of O(clusters^2 * cells)
## and the ensemble loop runs on the CPU in seconds (28 maps x 128^3 cells: 10 s against
## 411 s CPU / 230 s Metal for the original mask-by-mask loop). Outputs are bitwise
## identical to the mask-by-mask loop on the NumPy backend. compute_SQ / create_mask are
## kept (JAX-accelerated when a GPU or Metal backend is present) for direct use.

import os
import glob
import argparse
import numpy as np

# Use JAX if GPU and the jax package is installed, otherwise use NumPy
# added support for experimental JAX backend with METAL on Apple silicon
try:
    import jax

    default_device = jax.default_backend()
    if default_device == "gpu" or default_device == "METAL":
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
    # an empty mask has no overlap with anything; return before dividing by
    # the (possibly zero) union size
    if array_lib.max(mask) == 0 or array_lib.max(maskC) == 0:
        return 0.0, array_lib.zeros(mask.shape)

    # --------------------------------------------------
    # Intersection signal strength of two masked arrays, S
    S = array_lib.sum(I) / array_lib.sum(U)

    # --------------------------------------------------
    # Union quality of two masked arrays, Q
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


def pair_quality_from_counts(n_a: int, n_b: int, n_i: int) -> float:
    """Quality index S/Q of two masks from their cell counts.

    This reproduces ``compute_SQ`` operation for operation: with 0/1 masks the
    intersection ``I = mask * maskC`` has ``n_i`` cells, the union
    ``ceil((mask + maskC) / 2)`` has ``n_a + n_b - n_i`` cells, and every sum
    in ``compute_SQ`` is an exact integer, so evaluating the same divisions on
    the counts gives the same float64 result bit for bit.

    Args:
        n_a (int): number of cells in cluster C.
        n_b (int): number of cells in cluster C'.
        n_i (int): number of cells in both.

    Returns:
        float: S/Q, or 0.0 when either mask is empty or Q is zero.
    """
    if n_a == 0 or n_b == 0:
        return 0.0
    n_u = n_a + n_b - n_i
    S = n_i / n_u
    Q = n_u / (n_a + n_b) - n_i / (n_a + n_b)
    if Q == 0.0:
        return 0.0
    return S / Q


def _load_labels(path: str) -> np.ndarray:
    """Load one label file as a compact integer array (NumPy, never JAX).

    Non-integer values can never equal a cluster id in ``create_mask`` so they
    are mapped to -1, which the contingency table ignores.
    """
    raw = np.load(path, mmap_mode="r")
    labels = np.asarray(raw).reshape(-1)
    as_int = labels.astype(np.int64)
    bad = as_int != labels
    bad |= (as_int < -1) | (as_int >= np.iinfo(np.int32).max)  # would wrap when narrowed
    if bad.any():
        as_int = as_int.copy()
        as_int[bad] = -1
    if as_int.min() >= 0 and as_int.max() < 255:
        return as_int.astype(np.uint8)
    if as_int.min() >= -1 and as_int.max() < 2**15:
        return as_int.astype(np.int16)
    return as_int.astype(np.int32)


_CONTINGENCY_CHUNK = 1 << 24
_TABLE_CACHE_BYTES = 1 << 28  # reuse each run pair's table for both runs below this total

try:
    import numba as _numba

    @_numba.njit(parallel=True, cache=True)
    def _contingency_numba(labels_a, n_a, labels_b, n_b, n_parts):
        n = labels_a.shape[0]
        partial = np.zeros((n_parts, n_a * n_b), dtype=np.int64)
        step = (n + n_parts - 1) // n_parts
        for p in _numba.prange(n_parts):
            start = p * step
            stop = min(n, start + step)
            row = partial[p]
            for k in range(start, stop):
                x = np.int64(labels_a[k])
                y = np.int64(labels_b[k])
                if x >= 0 and x < n_a and y >= 0 and y < n_b:
                    row[x * n_b + y] += 1
        return partial.sum(axis=0)

    @_numba.njit(parallel=True, cache=True)
    def _all_pair_tables(stack, nids, pair_i, pair_j, offsets, out):
        """Contingency tables of every listed run pair, one pair per thread.

        Four interleaved sub-histograms per pair break the dependency chain of
        repeated increments to the same bin (labels come in long runs of one id).
        """
        n = stack.shape[1]
        n4 = n - n % 4
        for q in _numba.prange(pair_i.shape[0]):
            i = pair_i[q]
            j = pair_j[q]
            n_a = nids[i]
            n_b = nids[j]
            m = n_a * n_b
            la = stack[i]
            lb = stack[j]
            loc = np.zeros(4 * m, dtype=np.int64)
            for k in range(0, n4, 4):
                for u in range(4):
                    x = np.int64(la[k + u])
                    y = np.int64(lb[k + u])
                    if x >= 0 and x < n_a and y >= 0 and y < n_b:
                        loc[u * m + x * n_b + y] += 1
            for k in range(n4, n):
                x = np.int64(la[k])
                y = np.int64(lb[k])
                if x >= 0 and x < n_a and y >= 0 and y < n_b:
                    loc[x * n_b + y] += 1
            o = offsets[q]
            for c in range(m):
                out[o + c] = loc[c] + loc[m + c] + loc[2 * m + c] + loc[3 * m + c]

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is a hard dependency
    HAVE_NUMBA = False


def contingency_table(
    labels_a: np.ndarray, n_a: int, labels_b: np.ndarray, n_b: int, use_numba: bool = True
) -> np.ndarray:
    """Count cells with label ``i`` in ``labels_a`` and ``j`` in ``labels_b``.

    Args:
        labels_a (np.ndarray): flat integer labels of run A.
        n_a (int): number of cluster ids considered in run A (ids ``0..n_a-1``).
        labels_b (np.ndarray): flat integer labels of run B, same length.
        n_b (int): number of cluster ids considered in run B.

    Returns:
        np.ndarray: ``[n_a, n_b]`` int64 counts. Labels outside the id range are
            ignored, exactly as ``create_mask`` never selects them.
    """
    if HAVE_NUMBA and use_numba:
        # one private table per part; keep their total under ~64 MB for large id counts
        n_parts = min(_numba.get_num_threads() * 4, labels_a.shape[0] // 65536 + 1)
        n_parts = max(1, min(n_parts, (1 << 26) // (8 * n_a * n_b + 1)))
        return _contingency_numba(
            np.ascontiguousarray(labels_a), int(n_a), np.ascontiguousarray(labels_b), int(n_b), n_parts
        ).reshape(n_a, n_b)
    table = np.zeros(n_a * n_b, dtype=np.int64)
    for start in range(0, labels_a.shape[0], _CONTINGENCY_CHUNK):
        a = labels_a[start : start + _CONTINGENCY_CHUNK].astype(np.int64)
        b = labels_b[start : start + _CONTINGENCY_CHUNK].astype(np.int64)
        valid = (a >= 0) & (a < n_a) & (b >= 0) & (b < n_b)
        if not valid.all():
            a = a[valid]
            b = b[valid]
        table += np.bincount(a * n_b + b, minlength=n_a * n_b)
    return table.reshape(n_a, n_b)


def loop_over_all_clusters(
    all_files: list[str],
    number_of_clusters: array_lib.ndarray,
    dimensions: np.ndarray,
    subfolder: str = "SCE",
    write_masks: bool = True,
) -> int:
    """
    Loops over all clusters in the given data, compute goodness-of-fit, then save Gsum values to file.

    The comparison of every cluster with every cluster of every other run is
    done on contingency tables (one ``np.bincount`` per pair of runs) instead
    of on full-size masks. Each pair's quality index depends only on the two
    cluster sizes and their overlap (``pair_quality_from_counts``), and the
    saved per-cell map is the cluster mask times its total Gsum, so the output
    files are identical bit for bit to the original mask-by-mask loop while
    the work drops from O(clusters^2 x cells) to O(runs^2 x cells). This path
    always runs in NumPy; ``create_mask`` and ``compute_SQ`` remain available
    (and JAX-jitted on a GPU) for callers that want per-pair maps.

    Args:
        all_files (list[str]): A list of data files saved in '.npy' format.
        number_of_clusters ((j)np.ndarray): An array of the number of cluster ids in each run.
        dimensions (np.ndarray): A 1d array representing the dimensions of the clusters (can be any dimension but nx*ny*nz has to be equal to number of data points).
        subfolder (str): The name of the subfolder to save the results to.
        write_masks (bool): write one full-size float64 ``mask-<run>-id<cid>.npy`` per cluster
            (the default). Each is the run's cluster mask times its Gsum, so
            ``make_sce_clusters.combine_separated_clusters_from_labels`` can rebuild the
            combined map from the label files and ``multimap_mappings.txt`` alone; pass
            False to skip these writes (8 bytes per cell per cluster).

    Returns:
        Save Gsum value of each cluster C to a file.
    """
    runs = list(all_files)
    nids = [int(n) for n in np.asarray(number_of_clusters).reshape(-1)]
    dimensions = tuple(int(d) for d in np.asarray(dimensions).reshape(-1))
    labels = [_load_labels(run) for run in runs]
    n_cells = int(np.prod(dimensions))
    for run, lab in zip(runs, labels):
        if lab.shape[0] != n_cells:
            raise ValueError(f"{run} has {lab.shape[0]} cells, expected {n_cells} from dims {dimensions}")

    # sizes[i][cid] = cells of cluster cid in run i. Contingency tables are built on
    # demand (each run pair twice) so at most one nids[i] x nids[j] table is live at a time.
    sizes = [np.bincount(lab[(lab >= 0) & (lab < n)].astype(np.int64), minlength=n) for lab, n in zip(labels, nids)]

    # Each unordered pair's table is built once and its transpose reused for the other run,
    # as long as all tables together fit in _TABLE_CACHE_BYTES; counts are integers, so
    # table(j, i) == table(i, j).T exactly. A table is dropped once both runs used it.
    cache_tables = 8 * sum(nids) ** 2 <= _TABLE_CACHE_BYTES
    cached = {}
    if cache_tables and HAVE_NUMBA and len(runs) > 1:
        # all tables at once, run pairs in parallel (same integer counts as contingency_table)
        stack = np.empty((len(runs), n_cells), dtype=np.result_type(*[lab.dtype for lab in labels]))
        for r in range(len(runs)):
            stack[r] = labels[r]
            labels[r] = stack[r]
        pair_i, pair_j = np.triu_indices(len(runs), 1)
        nid_arr = np.asarray(nids, dtype=np.int64)
        table_sizes = nid_arr[pair_i] * nid_arr[pair_j]
        offsets = np.concatenate([[0], np.cumsum(table_sizes)[:-1]]).astype(np.int64)
        flat = np.zeros(int(table_sizes.sum()), dtype=np.int64)
        _all_pair_tables(stack, nid_arr, pair_i.astype(np.int64), pair_j.astype(np.int64), offsets, flat)
        for i, j, o, m in zip(pair_i, pair_j, offsets, table_sizes):
            cached[(int(i), int(j))] = flat[o : o + m].reshape(nids[i], nids[j])

    def overlap(i, j):
        if not cache_tables:
            return contingency_table(labels[i], nids[i], labels[j], nids[j])
        if j < i:
            return cached.pop((j, i)).T
        if (i, j) not in cached:
            cached[(i, j)] = contingency_table(labels[i], nids[i], labels[j], nids[j])
        return cached[(i, j)]

    for i, run in enumerate(runs):
        print("-----------------------")
        print("Run : ", run, flush=True)
        with open(subfolder + "/multimap_mappings.txt", "a") as f:
            f.write("{}\n".format(run.removesuffix(".npy")))
        print("nids : ", nids[i])

        # totals[cid] accumulates SQ over the other runs in ascending run order and, within a
        # run, ascending cluster order: the same sequence of additions as the mask-by-mask loop.
        totals = [0.0] * nids[i]
        for j in range(len(runs)):
            if j == i:  # don't compare to itself
                continue
            table = overlap(i, j)
            for cid in range(nids[i]):
                n_a = int(sizes[i][cid])
                n_i_row = table[cid]
                for cidC in range(nids[j]):
                    totals[cid] += pair_quality_from_counts(n_a, int(sizes[j][cidC]), int(n_i_row[cidC]))
            table = n_i_row = None  # drop the view and the table before the next pair's is built

        for cid in range(nids[i]):
            total_SQ_scalar = totals[cid]
            # every cell of this cluster accumulated exactly the same SQ values in the same order
            if write_masks:
                total_mask = np.where(labels[i] == cid, total_SQ_scalar, 0.0).reshape(dimensions)
                np.save(subfolder + "/mask-{}-id{}.npy".format(run.removesuffix(".npy"), cid), total_mask)
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
        clusters = np.load(cluster_files[run], mmap_mode="r")
        if np.issubdtype(clusters.dtype, np.integer) and clusters.size:
            lo, hi = int(clusters.min()), int(clusters.max())
            if lo >= 0 and hi < 1 << 24:  # count distinct ids without a sort
                number_of_clusters[run] = np.count_nonzero(np.bincount(clusters.reshape(-1), minlength=1))
                continue
        number_of_clusters[run] = len(np.unique(clusters))

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

    parser.add_argument(
        "--no_masks",
        dest="write_masks",
        action="store_false",
        help="Do not write the full-size mask-*.npy files; make_sce_clusters --labels_path rebuilds the combined map from the label files",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    print("Starting SCE", flush=True)
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
    loop_over_all_clusters(cluster_files, nids_array, data_dims, subfolder, write_masks=args.write_masks)
