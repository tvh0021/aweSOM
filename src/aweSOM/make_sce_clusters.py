import argparse
import os
import matplotlib.pyplot as plt
import shutil
import numpy as np
import numba
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


def _is_gsum_line(line: str) -> bool:
    """True for a ``"<cid> <Gsum>"`` line of multimap_mappings.txt.

    Run names used to be recognised by a '-' in the line, which also matches a
    Gsum written in exponent form (``"3 2.5e-05"``); test the line's shape instead.
    """
    parts = line.split(" ")
    if len(parts) != 2:
        return False
    try:
        int(parts[0])
        float(parts[1])
    except ValueError:
        return False
    return True


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
            if not line:
                continue
            if _is_gsum_line(line):
                mapping[key_name].append(line.split(" "))
            else:  # a run name (a label file path without .npy)
                key_name = line
                mapping[key_name] = []

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

    if minimas:
        # remove the first minimum because it is usually part of the first cluster
        minimas.pop(0)
    # from the local minima, find the ranges of the clusters: [0, m0], [m0, m1], ..., [m_last, n].
    # With fewer than two minima left this still covers every entry (one or two ranges).
    bounds = [0] + minimas + [len(gsum_deriv)]
    cluster_ranges = [[bounds[i], bounds[i + 1]] for i in range(len(bounds) - 1)]

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

        this_cluster_signal_map = np.zeros(
            ([len(remapped_clusters[cluster])] + dims), dtype=np.float32
        )
        for i, instance in enumerate(remapped_clusters[cluster]):
            if i % 10 == 0:
                print("Instance", i, flush=True)
            this_cluster_signal_map[i] = np.reshape(
                np.load(file_path + f"/mask-{instance[2]}-id{instance[1]}.npy"),
                dims,
            )

        all_signals_map[cluster] = np.sum(this_cluster_signal_map, axis=0)

    return all_signals_map


def combine_separated_clusters_from_labels(
    map_list: list, cluster_ranges: list[list[int]], dims: list[int], labels_path: str, mask_dtype=np.float32
) -> np.ndarray:
    """Same result as ``combine_separated_clusters``, built from the label files.

    Each SCE mask file is ``np.where(labels == cid, Gsum, 0.0)``, so the combined
    map of a cluster range is the float32 sum, in ``map_list`` order, of
    ``float32(Gsum)`` over the instances whose label matches the cell. Adding the
    zeros of non-matching masks is exact, and ``np.sum(axis=0)`` adds the stack
    in order whenever there is more than one cell (a one-cell map repeats
    NumPy's pairwise reduction), so the result is bitwise identical to
    summing the mask files, without writing or reading them (mask files take 4
    bytes per cell per SCE cluster in float32, 8 in float64).

    Args:
        map_list (list): ``[Gsum, cid, run_name]`` entries from ``get_gsum_values``.
        cluster_ranges (list[list[int]]): start and end indices of each combined cluster.
        dims (list[int]): dimensions of the label maps.
        labels_path (str): folder holding ``<run_name>.npy`` label files.
        mask_dtype (np.dtype or str): the ``mask_dtype`` the mask files were written with; each Gsum is
            rounded to it (float32 rounds at about 6e-8 relative, ``np.float64`` as in v1.1.3) before the sum.

    Returns:
        np.ndarray: float32 array of shape ``[len(cluster_ranges)] + dims``.
    """
    dims = [int(d) for d in dims]
    n_cells = int(np.prod(dims))
    run_index = {}
    for _, _, run_name in map_list:
        run_index.setdefault(run_name, len(run_index))
    raw = [np.load(os.path.join(labels_path, name + ".npy"), mmap_mode="r").reshape(-1) for name in run_index]
    lo = min(int(r.min()) for r in raw) if raw else 0
    hi = max(int(r.max()) for r in raw) if raw else 0
    # compact copy of all runs; every value equals the file's (non-integer files keep their dtype)
    dtype = np.result_type(*[r.dtype for r in raw]) if raw else np.int64
    if np.issubdtype(dtype, np.integer):
        dtype = np.uint8 if 0 <= lo and hi < 256 else np.int16 if -(2**15) <= lo and hi < 2**15 else np.int64
    labels = np.empty((len(run_index), n_cells), dtype=dtype)
    for r, lab in enumerate(raw):
        labels[r] = lab
    all_signals_map = np.empty([len(cluster_ranges)] + dims, dtype=np.float32)
    for cluster, (start_pointer, end_pointer) in enumerate(cluster_ranges):
        inst = map_list[start_pointer:end_pointer]
        runs = np.array([run_index[name] for _, _, name in inst], dtype=np.int64)
        cids = np.array([int(cid) for _, cid, _ in inst], dtype=np.int64)
        # the mask files hold Gsum in mask_dtype and combine_separated_clusters copies them into float32
        gsums = np.array([np.float64(g) for g, _, _ in inst], dtype=np.float64).astype(mask_dtype).astype(np.float32)
        if n_cells == 1:
            # NumPy sums a one-cell stack pairwise rather than in order; repeat its reduction
            vals = np.where(labels[runs, 0] == cids, gsums, np.float32(0.0))
            all_signals_map[cluster] = np.sum(vals.reshape([len(inst)] + dims), axis=0)
        else:
            all_signals_map[cluster] = _sum_matching(labels, runs, cids, gsums).reshape(dims)
    return all_signals_map


@numba.njit(parallel=True, cache=True)
def _sum_matching(labels, runs, cids, gsums):
    """out[c] = float32 sum, in instance order, of gsums[k] over k with labels[runs[k], c] == cids[k]."""
    n = labels.shape[1]
    out = np.zeros(n, dtype=np.float32)
    for c in numba.prange(n):
        acc = np.float32(0.0)
        for k in range(runs.shape[0]):
            if labels[runs[k], c] == cids[k]:
                acc += gsums[k]
        out[c] = acc
    return out


def gsum_derivative(gsum_values: list[float], smooth_fraction: int = 10, order: int = 4) -> np.ndarray:
    """Relative derivative of the ranked Gsum curve, as in the make_sce_clusters CLI.

    The curve is smoothed with a Savitzky-Golay filter of window
    ``len(gsum_values) // smooth_fraction`` and polynomial ``order``; the result is
    ``d(smoothed)/di / smoothed``. The window is raised to ``order + 1`` when the
    list is too short for the CLI's rule (small ensembles), and lowered to the
    list length when that is shorter still.

    Args:
        gsum_values (list[float]): Gsum values sorted in descending order.
        smooth_fraction (int): window = len // smooth_fraction. Defaults to 10.
        order (int): Savitzky-Golay polynomial order. Defaults to 4.

    Returns:
        np.ndarray: relative derivative, one value per ranked cluster.
    """
    n = len(gsum_values)
    window = min(n, max(n // smooth_fraction, order + 1))
    order = min(order, window - 1)
    smoothed = savgol_filter(np.asarray(gsum_values, dtype=np.float64), window, order, deriv=0)
    return savgol_filter(smoothed, window, order, deriv=1) / smoothed


def stack_sce_labels(
    mapping_file: str,
    labels_path: str,
    dims: list[int],
    threshold: float = -0.015,
    smooth_fraction: int = 10,
    order: int = 4,
):
    """SCE Gsum file -> combined signal maps -> one stacked label per cell.

    Runs the make_sce_clusters steps in order: rank the Gsum values
    (``get_gsum_values``), find the breaks in the ranked curve
    (``gsum_derivative`` and ``get_sce_cluster_separation`` with ``threshold``),
    sum each range's Gsum-weighted masks into a signal map
    (``combine_separated_clusters_from_labels``, so no mask files are needed),
    and give each cell the index of its strongest signal map.

    Args:
        mapping_file (str): multimap_mappings.txt written by ``sce.loop_over_all_clusters``.
        labels_path (str): folder of the ``<run_name>.npy`` label files (ignored for absolute run names).
        dims (list[int]): dimensions of the label maps.
        threshold (float): derivative threshold of ``get_sce_cluster_separation``. Defaults to -0.015.
        smooth_fraction (int): see ``gsum_derivative``. Defaults to 10.
        order (int): see ``gsum_derivative``. Defaults to 4.

    Returns:
        tuple: ``labels`` (int array of ``dims``: index of the largest signal map),
            ``signals`` (float32 ``[n_ranges] + dims``), and a dict with
            ``gsum_values``, ``gsum_deriv``, ``minimas``, ``cluster_ranges`` and
            ``map_list``.
    """
    gsum_values, map_list = get_gsum_values(mapping_file)
    gsum_deriv = gsum_derivative(gsum_values, smooth_fraction, order)
    cluster_ranges, minimas = get_sce_cluster_separation(gsum_deriv, threshold)
    signals = combine_separated_clusters_from_labels(map_list, cluster_ranges, list(dims), labels_path)
    n_ranges = len(cluster_ranges)
    labels = np.argmax(signals, axis=0).astype(np.uint8 if n_ranges < 256 else np.int32)
    info = {
        "gsum_values": gsum_values,
        "gsum_deriv": gsum_deriv,
        "minimas": minimas,
        "cluster_ranges": cluster_ranges,
        "map_list": map_list,
    }
    return labels, signals, info


def adjusted_rand_from_table(table: np.ndarray) -> float:
    """Adjusted Rand index of two labelings from their contingency table."""
    t = np.asarray(table, dtype=np.float64)
    comb = lambda v: v * (v - 1.0) / 2.0
    n = t.sum()
    sum_ij = comb(t).sum()
    sa = comb(t.sum(axis=1)).sum()
    sb = comb(t.sum(axis=0)).sum()
    expected = sa * sb / comb(n) if n > 1 else 0.0
    top = 0.5 * (sa + sb)
    return 1.0 if top == expected else float((sum_ij - expected) / (top - expected))


def anchored_sce_labels(
    mapping_file: str,
    labels_path: str,
    dims: list[int],
    weighting: str = "vote",
    n_pass: int = 2,
):
    """Stack realizations by cluster identity (SCE clusters grouped by overlap).

    ``stack_sce_labels`` groups clusters by their rank in the Gsum list, so one
    range can hold clusters of different physical regimes that happen to be
    equally reproducible. This variant groups them by overlap instead and keeps
    the rest of the SCE machinery:

    1. The anchor is the medoid realization (largest mean adjusted Rand index
       with the others, from ``sce.contingency_table``).
    2. Every cluster of every realization joins the anchor cluster with which it
       has the largest Jaccard index. (Raw overlap would send a small cluster
       that is shifted relative to the anchor's version, e.g. a 3 % current
       sheet sharing 1 % of cells with the anchor's sheet, to the large
       background cluster that contains most of its cells.)
    3. Each group's signal map is built with ``combine_separated_clusters_from_labels``
       with weight 1 (``weighting="vote"``: the signals of a cell add up to the
       number of realizations and the argmax is a plurality vote) or
       ``Gsum / mean Gsum of the group`` (``"gsum"``). Gsum weights are heavy
       tailed (S/Q grows like size / mismatch) and are 0 for a cluster that is
       reproduced exactly (the Q = 0 guard of ``pair_quality_from_counts``), so
       one realization can dominate a group; ``"vote"`` is the robust default.
    4. The argmax becomes the anchor of the next pass (``n_pass`` passes), which
       removes most of the dependence on the medoid's own boundaries.

    Args:
        mapping_file (str): multimap_mappings.txt written by ``sce.loop_over_all_clusters``.
        labels_path (str): folder of the ``<run_name>.npy`` label files (ignored for absolute run names).
        dims (list[int]): dimensions of the label maps.
        weighting (str): ``"vote"`` or ``"gsum"``. Defaults to ``"vote"``.
        n_pass (int): matching passes. Defaults to 2.

    Returns:
        tuple: ``labels`` (int array of ``dims``, ids 0..k-1), ``signals``
            (float32 ``[k] + dims``, row i is the signal of label i), and a dict
            with ``medoid``, ``mean_ari`` per run, the final ``groups`` (list of
            ``[weight, cid, run_name]`` per label) and ``map_list``.
    """
    from .sce import contingency_table

    if weighting not in ("gsum", "vote"):
        raise ValueError("weighting must be 'gsum' or 'vote'")
    dims = [int(d) for d in dims]
    _, map_list = get_gsum_values(mapping_file)
    gsum = {(name, int(cid)): float(g) for g, cid, name in map_list}
    runs = sorted({name for _, _, name in map_list})
    labs = [np.load(os.path.join(labels_path, r + ".npy"), mmap_mode="r").reshape(-1) for r in runs]
    labs = [np.ascontiguousarray(l) for l in labs]
    nids = [int(l.max()) + 1 for l in labs]  # ids below 0 are ignored by contingency_table

    R = len(runs)
    ari = np.eye(R)
    for i in range(R):
        for j in range(i + 1, R):
            ari[i, j] = ari[j, i] = adjusted_rand_from_table(contingency_table(labs[i], nids[i], labs[j], nids[j]))
    mean_ari = (ari.sum(axis=1) - 1.0) / max(R - 1, 1)
    medoid = int(np.argmax(mean_ari))
    anchor = labs[medoid]

    for _ in range(n_pass):
        n_anchor = int(anchor.max()) + 1
        groups = [[] for _ in range(n_anchor)]
        for r, (name, lab) in enumerate(zip(runs, labs)):
            table = contingency_table(lab, nids[r], anchor, n_anchor).astype(np.float64)
            union = table.sum(axis=1, keepdims=True) + table.sum(axis=0, keepdims=True) - table
            jac = table / np.maximum(union, 1.0)
            for cid in range(nids[r]):
                if table[cid].sum() > 0 and (name, cid) in gsum:
                    groups[int(np.argmax(jac[cid]))].append([gsum[(name, cid)], cid, name])
        groups = [g for g in groups if g]
        new_map, ranges = [], []
        for g in groups:
            mean_g = np.mean([e[0] for e in g])
            w = [(e[0] / mean_g if mean_g > 0 else 1.0) if weighting == "gsum" else 1.0 for e in g]
            ranges.append([len(new_map), len(new_map) + len(g)])
            new_map.extend([[wi, e[1], e[2]] for wi, e in zip(w, g)])
        signals = combine_separated_clusters_from_labels(new_map, ranges, dims, labels_path)
        winner = np.argmax(signals.reshape(len(groups), -1), axis=0)
        used, anchor = np.unique(winner, return_inverse=True)
        anchor = anchor.astype(np.int64)
        signals = signals[used]
        groups = [new_map[a:b] for a, b in (ranges[u] for u in used)]

    labels = anchor.astype(np.uint8 if len(groups) < 256 else np.int32).reshape(dims)
    info = {"medoid": runs[medoid], "mean_ari": dict(zip(runs, mean_ari.tolist())), "groups": groups, "map_list": map_list}
    return labels, signals, info


def make_file_name(n: int, ext: str) -> str:
    """Make a filename based on the number and the extension given.

    Args:
        n (int): number to be converted to a filename
        ext (str): file extension

    Returns:
        str: filename
    """
    if n < 10:
        file_n = "000" + str(n)
    elif (n >= 10) & (n < 100):
        file_n = "00" + str(n)
    else:
        file_n = "0" + str(n)

    return f"{file_n}.{ext}"


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
    parser.add_argument(
        "--labels_path",
        type=str,
        dest="labels_path",
        default=None,
        help="Folder with the SCE input label files; build the combined map from them instead of the mask files",
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
                args.file_path, make_file_name(i)
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
        if args.labels_path is not None:
            combined_sce_clusters = combine_separated_clusters_from_labels(
                map_list, cluster_ranges, args.dims, args.labels_path
            )
        else:
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
