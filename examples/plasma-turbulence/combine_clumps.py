## Given segmentation labels, combine clumps of the same label that are connected

import numpy as np
from scipy.ndimage import label


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
        return None

    return clumps


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
