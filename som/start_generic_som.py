## Script to initialize and train SOM network with generic (1D) data

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import parallel_som as psom
import pandas as pd
import pickle

import h5py as h5
import sys
import argparse

import initialize_lattice as il

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SOM code')
    parser.add_argument("--features_path", type=str, dest='features_path', default='/mnt/ceph/users/tha10/SOM-tests/hr-d3x640/')
    parser.add_argument("--file", type=str, dest='file', default='features_4j1b1e_2800.h5')
    parser.add_argument("--init_lattice", type=str, dest='init_lattice', default='uniform', help='Initial values of lattice. uniform or sampling')
    parser.add_argument('--alpha', type=float, dest='alpha', default=0.5, help='Initial learning parameter')
    parser.add_argument('--train', type=int, dest='train', default=10000, help='Number of training steps')
    parser.add_argument('--batch', type=int, dest='batch', default=None, help='Number of batches', required=False)
    parser.add_argument('--pretrained', action="store_true", dest='pretrained', help='Pass this argument if supplying a pre-trained model', required=False)
    parser.add_argument('--neurons_path', type=str, dest='neurons_path', default=None, help='Path to file containing neuron values', required=False)
    parser.add_argument('--save_neuron_values', dest='save_neuron_values', help='Pass this argument if you want to save neuron values', action="store_true")

    args = parser.parse_args()

    #--------------------------------------------------
    if (args.pretrained == True) & (args.neurons_path is None):
        sys.exit("Cannot run, no neuron values provided.")

    # CLI arguments
    features_path = args.features_path
    file_name = args.file
    init_lattice = args.init_lattice
    alpha = args.alpha
    train = args.train
    batch = args.batch
    pretrained = args.pretrained
    neurons_path = args.neurons_path
    save_neuron_values = args.save_neuron_values
    
    if save_neuron_values is None:
            save_neuron_values = False
    
    name_of_dataset = file_name.split("_")[2].split(".h5")[0] # all the data laps to process

    f5 = h5.File(features_path+file_name, 'r')
    x = f5['features'][()]
    feature_list = f5['names'][()]

    feature_list = [n.decode('utf-8') for n in feature_list]
    f5.close()

    # initialize lattice
    [xdim, ydim] = il.initialize_lattice(x, 0.6)

    print(f"File loaded, parameters: {name_of_dataset}-{xdim}-{ydim}-{alpha}-{train}-{batch}", flush=True)

    # normalize data
    scaler = MinMaxScaler()
    data_transformed = scaler.fit_transform(x)

    # initialize SOM
    if batch is None:
        som = psom.map(xdim, ydim, alpha, train, epoch=0, alpha_type="decay", sampling_type=init_lattice)
    else:
        data_by_batch = batch_separator(data_transformed, batch)
        som = psom.map(xdim, ydim, alpha, train, epoch=0, alpha_type="decay", sampling_type=init_lattice, number_of_batches=batch)
    
    # train SOM
    if batch is None:
        attr = pd.DataFrame(data_transformed)
        attr.columns = feature_list
        som.fit(attr)
    else:
        attr = pd.DataFrame(data_by_batch[0])
        attr.columns = feature_list
        som.fit(attr)
        neurons = som.all_neurons()
        
        for i in range(1, batch):
            attr = pd.DataFrame(data_by_batch[i])
            attr.columns = feature_list
            som.fit(data_by_batch[i], restart=True, neurons=neurons)
            neurons = som.all_neurons()
    
    # assign cluster ids to the lattice
    clusters = som.assign_cluster_to_lattice(smoothing=None,merge_cost=0.3)

    # assign cluster ids to the data
    data_matrix=som.projection()
    data_Xneuron=data_matrix[:,0]
    data_Yneuron=data_matrix[:,1]

    som_labels = som.assign_cluster_to_data(data_Xneuron, data_Yneuron, clusters)

    # save cluster ids
    np.save(f"{name_of_dataset}-{xdim}-{ydim}-{alpha}-{train}-{batch}"+"_labels.npy", som_labels)
    print(f"Cluster labels saved to {name_of_dataset}-{xdim}-{ydim}-{alpha}-{train}-{batch}_labels.npy")

    # save som object
    with open('som_object.pkl', 'wb') as file:
        pickle.dump(som, file)
    print(f"SOM object saved to som_object.pkl")
