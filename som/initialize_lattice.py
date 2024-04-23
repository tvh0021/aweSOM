## The goal of this script is to initialize the lattice of the SOM based on Kohonen's advice
## The rectangular lattice should have a total of 5 sqrt(N*f) neurons, where N is the number 
## of data points and f is the number of features. The lattice should have a height to width
## ratio of ~0.6 (one preferred direction)

import numpy as np

def number_of_nodes(N : int, f : int) -> int:
    return int(5 * np.sqrt(N * f) / 4)

def initialize_lattice(data : np.ndarray, ratio : float) -> list[int]:
    """ initialize_lattice - given a N x f dataset and a ratio, return the dimensions of the SOM lattice

    Args:
        data (np.ndarray): N x f dataset, N is the number of data points and f is the number of features
        ratio (float): height to width ratio of the lattice, between 0 and 1.

    Returns:
        list[int]: [xdim, ydim] dimensions of the lattice
    """
    N = data.shape[0]
    f = data.shape[1]
    nodes = number_of_nodes(N, f)
    xdim = int(np.ceil(np.sqrt(nodes / ratio)))
    ydim = int(nodes / xdim)

    return [xdim, ydim]