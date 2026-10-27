## Given segmentation labels, combine clumps of the same label that are connected

import numba
import numpy as np
from tqdm import tqdm


def generate_neighbors(connectivity):
    """Generate relative neighbor positions based on connectivity. If data is 3D, assume z-axis is the first axis."""
    if connectivity == 4:  # 2D direct neighbors only
        deltas = [(0, -1), (-1, 0), (1, 0), (0, 1)]
    elif connectivity == 8:  # 2D
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    elif connectivity == 6:  # 3D direct neighbors only
        deltas = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if abs(dx) + abs(dy) + abs(dz) == 1
        ]
    elif connectivity == 10:  # 3D, 8-connectivity in 2D plus +z and -z
        deltas = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if abs(dx) + abs(dy) + abs(dz) == 1
        ] + [(0, -1, -1), (0, 1, -1), (0, -1, 1), (0, 1, 1)]
    elif connectivity == 26:  # 3D
        deltas = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
    else:
        raise ValueError(
            "Unsupported connectivity. For 2D, use 4 or 8; for 3D, use 6, 10, or 26."
        )
    return deltas


def get_neighbors(
    element: int, data_array: np.ndarray, connectivity: np.ndarray
) -> list:
    """Return all neighbors of an element with the same value in coordinate positions"""
    neighbors = []
    if data_array.ndim == 2:
        element_value = data_array[element[0], element[1]]
        # define 8-connectivity
        for dx, dy in connectivity:
            x, y = element[0] + dx, element[1] + dy
            if 0 <= x < data_array.shape[0] and 0 <= y < data_array.shape[1]:
                if data_array[x, y] == element_value:
                    neighbors.append((x, y))
    elif data_array.ndim == 3:
        element_value = data_array[element[0], element[1], element[2]]
        # define 26-connectivity
        for dx, dy, dz in connectivity:
            x, y, z = element[0] + dx, element[1] + dy, element[2] + dz
            if (
                0 <= x < data_array.shape[0]
                and 0 <= y < data_array.shape[1]
                and 0 <= z < data_array.shape[2]
            ):
                if data_array[x, y, z] == element_value:
                    neighbors.append((x, y, z))
    return neighbors


def get_clumps(data_array: np.ndarray, connectivity: np.ndarray) -> list:
    """Return all clumps of elements with the same value in the array"""
    visited = np.zeros(data_array.shape, dtype=bool)
    clumps = list()

    def dfs(element, clump):
        stack = [element]
        while stack:
            current = stack.pop()
            if not visited[current]:
                visited[current] = True
                clump.append(current)
                neighbors = get_neighbors(current, data_array, connectivity)
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        stack.append(neighbor)

    if data_array.ndim == 2:
        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                if not visited[i, j]:
                    clump = []
                    dfs((i, j), clump)
                    if clump:
                        clumps.append(clump)
    elif data_array.ndim == 3:
        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                for k in range(data_array.shape[2]):
                    if not visited[i, j, k]:
                        clump = []
                        dfs((i, j, k), clump)
                        if clump:
                            clumps.append(clump)
    else:
        raise ValueError("Input array must be 2D or 3D.")

    return clumps


def _structure_from_deltas(connectivity, ndim):
    """3^ndim boolean structuring element holding the centre and the given neighbor offsets."""
    structure = np.zeros((3,) * ndim, dtype=bool)
    structure[(1,) * ndim] = True
    for delta in connectivity:
        structure[tuple(1 + d for d in delta)] = True
        structure[tuple(1 - d for d in delta)] = True
    return structure


def label_clumps(data_array: np.ndarray, connectivity, periodic: bool = False, use_cc3d: bool = True, renumber: bool = True):
    """Vectorized ``get_clumps``: label connected regions of equal value.

    Args:
        data_array (np.ndarray): 2D or 3D array of (integer) labels.
        connectivity: neighbor offsets as returned by ``generate_neighbors``.
        periodic (bool): also connect cells across opposite faces.
        use_cc3d (bool): use the ``cc3d`` package (multi-label, one pass) when it is
            installed and supports the connectivity (3D 6/18/26, periodic only for 6).
        renumber (bool): renumber clumps by first cell in C order (costs one sort);
            without it ids are still a valid partition but in backend order.

    Returns:
        (np.ndarray, int): int32 clump id per cell and the number of clumps. Ids are
            numbered in order of each clump's first cell in C order, which is the
            order of the list returned by ``get_clumps`` (for ``periodic=False``).
    """
    data_array = np.asarray(data_array)
    structure = _structure_from_deltas(connectivity, data_array.ndim)
    out = _label_cc3d(data_array, structure, periodic) if use_cc3d else None
    if out is not None:
        offset = int(out.max()) + 1
    else:
        from scipy import ndimage

        out = np.zeros(data_array.shape, dtype=np.int64)
        offset = 0
        for value in np.unique(data_array):
            lab, n = ndimage.label(data_array == value, structure=structure)
            out[lab > 0] = lab[lab > 0] + offset
            offset += n
        out -= 1
        if periodic:
            out, offset = _merge_periodic(data_array, out, offset, connectivity)
    if not renumber:
        return out, offset
    # renumber by first occurrence in C order
    ids, n = _first_seen_renumber(out.reshape(-1), offset)
    return ids.reshape(data_array.shape), n


@numba.njit(cache=True)
def _first_seen_renumber(flat, n_labels):
    """Relabel 0..n_labels-1 in order of first appearance (one sequential pass)."""
    new_id = np.full(n_labels, -1, dtype=np.int32)
    out = np.empty(flat.size, dtype=np.int32)
    n = 0
    for k in range(flat.size):
        v = flat[k]
        if new_id[v] < 0:
            new_id[v] = n
            n += 1
        out[k] = new_id[v]
    return out, n


def _label_cc3d(data_array, structure, periodic):
    """cc3d multi-label connected components, or None if cc3d cannot do this case."""
    try:
        import cc3d
    except ImportError:
        return None
    if data_array.ndim != 3 or not np.issubdtype(data_array.dtype, np.integer):
        return None
    n_nb = int(structure.sum()) - 1
    if n_nb not in (6, 18, 26) or (periodic and n_nb != 6):
        return None
    full = np.zeros((3, 3, 3), dtype=bool)
    full[1, 1, 1] = True
    reference = {6: (np.abs(np.indices((3, 3, 3)) - 1).sum(axis=0) <= 1),
                 18: (np.abs(np.indices((3, 3, 3)) - 1).sum(axis=0) <= 2),
                 26: np.ones((3, 3, 3), dtype=bool)}[n_nb]
    if not np.array_equal(structure, reference):  # e.g. the custom 10-neighbour set
        return None
    lo = int(data_array.min())
    if int(data_array.max()) - lo + 1 > np.iinfo(np.uint32).max:  # would wrap and merge distinct values
        return None
    shifted = (data_array.astype(np.int64) - lo + 1).astype(np.uint32)  # cc3d treats 0 as background
    lab = cc3d.connected_components(shifted, connectivity=n_nb, periodic_boundary=periodic, out_dtype=np.uint32)
    return lab.astype(np.int64) - 1


def _merge_periodic(data_array, labels, n_labels, connectivity):
    """Join clumps that touch across the periodic boundary (union over wrapped neighbor pairs)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    src, dst = [], []
    shape = data_array.shape
    for delta in connectivity:
        # only cells whose neighbor wraps around: a slab of width 1 on each axis with delta != 0
        for axis, d in enumerate(delta):
            if d == 0:
                continue
            sl = [slice(None)] * data_array.ndim
            sl[axis] = slice(shape[axis] - 1, shape[axis]) if d > 0 else slice(0, 1)
            sl = tuple(sl)
            v = data_array[sl]
            nb_index = []
            for ax, dd in enumerate(delta):
                idx = np.arange(shape[ax])[sl[ax]]
                nb_index.append((idx + dd) % shape[ax])
            grid = np.ix_(*nb_index)
            v_nb = data_array[grid]
            same = v == v_nb
            src.append(labels[sl][same])
            dst.append(labels[grid][same])
    src = np.concatenate(src) if src else np.empty(0, np.int64)
    dst = np.concatenate(dst) if dst else np.empty(0, np.int64)
    graph = coo_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n_labels, n_labels))
    n, comp = connected_components(graph, directed=False)
    return comp[labels], n


def merge_double_sheets(
    double_sheet_dict: dict, positive_sheet_dict: dict, negative_sheet_dict: dict
):
    merged_double_sheets = dict()
    neighbor_deltas = generate_neighbors(10)

    for double_key, double_positions in double_sheet_dict.items():
        merged_clump = set()
        print("Double sheet:", double_key)

        has_positive_neighbor = False
        has_negative_neighbor = False

        visited_positive_keys = set()
        visited_negative_keys = set()

        for double_position in tqdm(double_positions):
            # print('Double position:', double_position)
            neighbors = [
                tuple(np.add(double_position, delta)) for delta in neighbor_deltas
            ]

            # check positive neighbors
            for positive_key, positive_positions in positive_sheet_dict.items():
                if any([neighbor in positive_positions for neighbor in neighbors]):
                    has_positive_neighbor = True
                    merged_clump.update(positive_positions)
                    visited_positive_keys.add(positive_key)
                    break

            if has_positive_neighbor:
                # check negative neighbors
                for negative_key, negative_positions in negative_sheet_dict.items():
                    if any([neighbor in negative_positions for neighbor in neighbors]):
                        has_negative_neighbor = True
                        merged_clump.update(negative_positions)
                        visited_negative_keys.add(negative_key)
                        break

            # break condition stops the loop as soon as one positive and one negative neighbor is found
            # but we want to merge all the clumps that are connected to the double sheet
            # if has_positive_neighbor and has_negative_neighbor:
            #     break

        if has_positive_neighbor and has_negative_neighbor:
            merged_clump.update(double_positions)
            merged_double_sheets[double_key] = merged_clump

            # remove the visited keys to avoid overlap between clusters
            for positive_key in visited_positive_keys:
                del positive_sheet_dict[positive_key]
            for negative_key in visited_negative_keys:
                del negative_sheet_dict[negative_key]
        else:
            merged_double_sheets[double_key] = double_positions

    return merged_double_sheets


# Example usage
# if __name__ == "__main__":
#     # Example 2D array
#     array_2d = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 2, 2], [2, 2, 2, 0]])

#     clumped_array, clump_labels = find_clumps_and_mask(array_2d)
#     print("Original Array:")
#     print(array_2d)
#     print("\nClumped Array:")
#     print(clumped_array)
#     print("\nClump Labels:")
#     print(clump_labels)

#     # Example 3D array
#     array_3d = np.array(
#         [
#             [[0, 1, 0], [1, 1, 0], [0, 0, 0]],
#             [[1, 1, 0], [0, 0, 2], [2, 2, 2]],
#             [[0, 0, 0], [2, 2, 0], [0, 0, 0]],
#         ]
#     )

#     clumped_array_3d, clump_labels_3d = find_clumps_and_mask(array_3d)
#     print("\nOriginal 3D Array:")
#     print(array_3d)
#     print("\nClumped 3D Array:")
#     print(clumped_array_3d)
#     print("\nClump Labels 3D:")
#     print(clump_labels_3d)


def merge_double_sheets_fast(double_sheet_dict: dict, positive_sheet_dict: dict, negative_sheet_dict: dict):
    """Array version of ``merge_double_sheets`` with the same result and the same in-place deletions.

    The original walks each double sheet's cells in order and, for every cell, takes the first
    positive clump (dict order) holding one of its 10 neighbors, then (once any positive neighbor
    was seen for this sheet) the first such negative clump. Clumps of one dict are disjoint, so
    "first in dict order" is the smallest dict ordinal among the cell's neighbors; that is a
    gather on an ordinal volume instead of a ``tuple in list`` search per neighbor and clump.
    Merged entries are sets, unmerged ones the original position objects, as in the original.
    Requires every clump of a dict to be disjoint from the other clumps of that dict.
    """
    deltas = np.array(generate_neighbors(10), dtype=np.int64)
    all_pos = [np.asarray(v, dtype=np.int64).reshape(-1, 3) for d in (double_sheet_dict, positive_sheet_dict, negative_sheet_dict) for v in d.values()]
    if not any(p.size for p in all_pos):  # nothing can merge; the original returns every sheet unchanged
        return dict(double_sheet_dict)
    shape = tuple(int(x) + 3 for x in np.max([p.max(axis=0) for p in all_pos if p.size], axis=0))  # pad by one: no wrap, as in the original

    def ordinal_volume(d):
        keys = list(d.keys())
        vol = np.full(shape, -1, dtype=np.int64)
        for k, key in enumerate(keys):
            p = np.asarray(d[key], dtype=np.int64).reshape(-1, 3) + 1
            if (vol[tuple(p.T)] >= 0).any():
                raise ValueError("clumps of one dict overlap")
            vol[tuple(p.T)] = k
        return keys, vol

    pos_keys, pos_vol = ordinal_volume(positive_sheet_dict)
    neg_keys, neg_vol = ordinal_volume(negative_sheet_dict)
    big = np.iinfo(np.int64).max
    merged_double_sheets = dict()
    for double_key, double_positions in double_sheet_dict.items():
        p = np.asarray(double_positions, dtype=np.int64).reshape(-1, 3) + 1
        nb = p[:, None, :] + deltas[None, :, :]  # cells x 10 x 3
        idx = tuple(nb[..., a] for a in range(3))
        pos_ord = pos_vol[idx]
        first_pos = np.where(pos_ord >= 0, pos_ord, big).min(axis=1)
        has_pos_cell = first_pos < big
        merged = False
        if has_pos_cell.any():
            start = int(np.argmax(has_pos_cell))  # sticky flag: negatives are checked from here on
            neg_ord = neg_vol[tuple(a[start:] for a in idx)]
            first_neg = np.where(neg_ord >= 0, neg_ord, big).min(axis=1)
            if (first_neg < big).any():
                merged = True
                visited_pos = np.unique(first_pos[has_pos_cell])
                visited_neg = np.unique(first_neg[first_neg < big])
                merged_clump = set()
                for k in visited_pos:
                    merged_clump.update(positive_sheet_dict[pos_keys[k]])
                for k in visited_neg:
                    merged_clump.update(negative_sheet_dict[neg_keys[k]])
                merged_clump.update(double_positions)
                merged_double_sheets[double_key] = merged_clump
                for k in visited_pos:
                    pp = np.asarray(positive_sheet_dict.pop(pos_keys[k]), dtype=np.int64).reshape(-1, 3) + 1
                    pos_vol[tuple(pp.T)] = -1
                for k in visited_neg:
                    pp = np.asarray(negative_sheet_dict.pop(neg_keys[k]), dtype=np.int64).reshape(-1, 3) + 1
                    neg_vol[tuple(pp.T)] = -1
        if not merged:
            merged_double_sheets[double_key] = double_positions
    return merged_double_sheets
