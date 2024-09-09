## Script to initialize and train SOM network with generic (1D) data

import h5py as h5
import sys
import argparse

from sklearn.preprocessing import MinMaxScaler
import numpy as np

import pandas as pd
import pickle

def batch_separator(data : np.ndarray, number_of_batches : int) -> np.ndarray:
    """ batch_separator - given a dataset and a number of batches, return a list of datasets
    each containing the same number of data points

    Args:
        data (np.ndarray): N x f dataset, N is the number of data points and f is the number of features
        number_of_batches (int): number of batches to create (b)

    Returns:
        np.ndarray: b x N//b x f list of datasets
    """
    N = data.shape[0]
    f = data.shape[1]
    batch_size = N // number_of_batches

    batches = np.zeros((number_of_batches, batch_size, f))
    for i in range(number_of_batches):
        batches[i] = data[i*batch_size:(i+1)*batch_size]

    return batches

def number_of_nodes(N : int, f : int) -> int:
    return int(5 * np.sqrt(N * f) / 6)

def initialize_lattice(data : np.ndarray, ratio : float) -> list[int]:
    """ initialize_lattice - given a N x f dataset and a ratio, return the dimensions of the SOM lattice
        based on Kohonen's advice

    Args:
        data (np.ndarray): N x f dataset, N is the number of data points and f is the number of features
        ratio (float): height to width ratio of the lattice, between 0 and 1.

    Returns:
        list[int]: [xdim, ydim] dimensions of the lattice
    """
    N = data.shape[0]
    f = data.shape[1]
    nodes = number_of_nodes(N, f)
    xdim = int(np.sqrt(nodes / ratio))
    ydim = int(nodes / xdim)

    return [xdim, ydim]

def manual_scaling(data : np.ndarray, bulk_range : float = 1.) -> np.ndarray:
    """manual_scaling - scale data to a range that centers on 0. and contains 95% of the data within the range

    Args:
        data (np.ndarray): 2d array of data (N x f)
        bulk_range (float, optional): The extent to which 95% of the data resides in. Defaults to 1..

    Returns:
        np.ndarray: scaled data
    """
    two_sigma = 2. * np.std(data, axis=0)
    return (data - np.mean(data, axis=0)) / two_sigma * bulk_range


if __name__ == "__main__":
    import parallel_som as psom

    parser = argparse.ArgumentParser(description='SOM code')
    parser.add_argument("--features_path", type=str, dest='features_path', default='/mnt/ceph/users/tha10/SOM-tests/hr-d3x640/')
    parser.add_argument("--file", type=str, dest='file', default='features_4j1b1e_2800.h5')
    parser.add_argument("--init_lattice", type=str, dest='init_lattice', default='uniform', help='Initial values of lattice. uniform or sampling')
    parser.add_argument('--xdim', type=int, dest='xdim', default=None, help='X dimension of the lattice', required=False)
    parser.add_argument('--ydim', type=int, dest='ydim', default=None, help='Y dimension of the lattice', required=False)
    parser.add_argument('--ratio', type=float, dest='ratio', default=0.7, help='Height to width ratio of the lattice', required=False)
    parser.add_argument('--alpha', type=float, dest='alpha', default=0.5, help='Initial learning parameter')
    parser.add_argument('--train', type=int, dest='train', default=None, help='Number of training steps')
    parser.add_argument('--batch', type=int, dest='batch', default=1, help='Number of batches', required=False)
    parser.add_argument('--pretrained', action="store_true", dest='pretrained', help='Pass this argument if supplying a pre-trained model', required=False)
    parser.add_argument('--neurons_path', type=str, dest='neurons_path', default=None, help='Path to file containing neuron values', required=False)
    parser.add_argument('--threshold', type=float, dest='threshold', default=0.2, help='Threshold for merging clusters', required=False)

    args = parser.parse_args()

    #--------------------------------------------------
    if (args.pretrained == True) & (args.neurons_path is None):
        sys.exit("Cannot run, no neuron values provided.")

    # CLI arguments
    features_path = args.features_path
    file_name = args.file
    init_lattice = args.init_lattice
    xdim = args.xdim
    ydim = args.ydim
    ratio = args.ratio
    alpha = args.alpha
    train = args.train
    batch = args.batch
    pretrained = args.pretrained
    neurons_path = args.neurons_path
    threshold = args.threshold
    
    name_of_dataset = file_name.split("_")[2].split(".h5")[0] # all the data laps to process

    f5 = h5.File(features_path+file_name, 'r')
    x = f5['features'][()]
    feature_list = f5['names'][()]

    feature_list = [n.decode('utf-8') for n in feature_list]
    f5.close()

    if train is None:
        train = len(x)
        print(f"Training steps not provided, defaulting to # steps = # data points", flush=True)
        # print(f"Training steps: {train}", flush=True)

    # initialize lattice
    if xdim is None or ydim is None:
        # NOTE: try PCA here to figure out the ratio of the map
        print("No lattice dimensions provided, initializing lattice based on Kohonen's advice", flush=True)
        [xdim, ydim] = initialize_lattice(x, ratio)
    
    print(f"Initialized lattice dimensions: {xdim}x{ydim}", flush=True)
    print(f"File loaded, parameters: {name_of_dataset}-{xdim}-{ydim}-{alpha}-{train}-{batch}", flush=True)

    # normalize data
    scale_method = "manual"
    if scale_method == "manual":
        data_transformed = manual_scaling(x)
    else:
        scaler = MinMaxScaler()
        data_transformed = scaler.fit_transform(x)

    # initialize SOM lattice
    som = psom.Lattice(xdim, ydim, alpha, train, alpha_type="decay", sampling_type=init_lattice)
    
    # train SOM
    if batch == 1:
        som.train_lattice(data_transformed, feature_list,)
    else:
        print(f"Training batch 1/{batch}", flush=True)
        data_by_batch = batch_separator(data_transformed, batch)
        som.train_lattice(data_by_batch[0], feature_list, number_of_steps=data_by_batch[0].shape[0])
        lattice_weights = som.lattice

        for i in range(1, batch):
            print(f"Training batch {i+1}/{batch}", flush=True)
            som.train_lattice(data_by_batch[i], feature_list, number_of_steps=data_by_batch[i].shape[0], restart_lattice=lattice_weights)
            lattice_weights = som.lattice

    print(f"Random seed: {som.seed}", flush=True)

    # map data to lattice
    som.data_array = data_transformed # recover the full dataset instead of the batch
    projection_2d = som.map_data_to_lattice()
    
    # assign cluster ids to the lattice
    clusters = som.assign_cluster_to_lattice(smoothing=None,merge_cost=threshold)

    # assign cluster ids to the data
    som_labels = som.assign_cluster_to_data(projection_2d, clusters)

    if init_lattice == "sampling":
        initial = "s"
    else:
        initial = "u"

    # save cluster ids
    np.save(f"{name_of_dataset}-{xdim}-{ydim}-{alpha}-{train}-{batch}{initial}"+"_labels.npy", som_labels)
    print(f"Cluster labels saved to {name_of_dataset}-{xdim}-{ydim}-{alpha}-{train}-{batch}{initial}_labels.npy")

    # save som object
    with open(f'som_object_{xdim}_{ydim}_{alpha}_{train}{initial}.pkl', 'wb') as file:
        pickle.dump(som, file)
    print(f"SOM object saved to som_object_{xdim}_{ydim}_{alpha}_{train}{initial}.pkl")