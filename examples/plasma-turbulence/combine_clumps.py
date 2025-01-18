## Given segmentation labels, combine clumps of the same label that are connected

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
