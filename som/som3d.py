## Self-organizing map (SOM) with 3d simulation data, parallelized with numba. Certain basic features are based on 
## Bussov & Nattila (2021): https://github.com/mkruuse/segmenting-turbulent-simulations-with-ensemble-learning.git

# -*- coding: utf-8 -*-

import numpy as np
import parallel_som as som
import pandas as pd
from numba import njit, prange

import h5py as h5
import sys
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def get_smaller_domain(data_array : np.ndarray, new_width : int, start_index_x : int, start_index_y : int, start_index_z : int):
    """Get a smaller domain from a full box simulation to save computational time

    Args:
        data_array (numpy 3d array): cubic box of some data [4d array should also work, provided that the first 3 dimensions are spatial]
        fraction (float): fraction of the original domain to keep, between 0 and 1
        start_index (int): starting index of the bigger domain

    Returns:
        numpy 3d(4d) array: cropped cubic box
    """
    if (start_index_z + new_width > data_array.shape[0]) | (start_index_y + new_width > data_array.shape[1]) | (start_index_x + new_width > data_array.shape[2]):
        print("Cannot crop, smaller domain is outside of current domain", flush=True)
        return 
    else:
        print(f"Cropped domain starts at: [{start_index_z},{start_index_y},{start_index_x}], width = {new_width}", flush=True)
        return data_array[start_index_z:start_index_z+new_width, start_index_y:start_index_y+new_width, start_index_x:start_index_x+new_width]
    
def convert_to_4d(data_array_2d : np.ndarray) -> np.ndarray:
        """Convert a n x f array to a a x b x c x f array, where f is values of certain features and a, b, c are grid points

        Args:
            data_array_2d (numpy 2d array)

        Returns:
            numpy 4d array
        """
        nd = int(np.cbrt(data_array_2d.shape[0]))
        data_array_4d = np.zeros((nd,nd,nd,data_array_2d.shape[-1]))

        for f in range(data_array_2d.shape[-1]):
                feature_3d = np.reshape(data_array_2d[:,f], newshape=[nd,nd,nd])
                data_array_4d[:,:,:,f] = feature_3d[:,:,:]
        return data_array_4d

def flatten_to_2d(data_array_4d : np.ndarray) -> np.ndarray:
        """Convert a a x b x c x f array to a n x f, where f is values of certain features and a, b, c are grid points

        Args:
            data_array_4d (numpy 4d array)

        Returns:
            numpy 2d array
        """
        nd = data_array_4d.shape[0]
        data_array_2d = np.zeros((int(nd**3), data_array_4d.shape[-1]))
        for f in range(data_array_4d.shape[-1]):
                data_array_2d[:,f] = data_array_4d[:,:,:,f].flatten()
        return data_array_2d

@njit(parallel=True)
def assign_cluster_id(nx : int, ny : int, nz : int, data_Xneuron : np.ndarray, data_Yneuron : np.ndarray, clusters : np.ndarray) -> np.ndarray:
        """From neuron data and cluster assignments, return the cluster id of the cell

        Args:
            nx (int): length of x-dimension
            ny (int): length of y-dimension
            nz (int): length of z-dimension
            data_Xneuron (np.ndarray): 1d array with x coordinate of the neuron associated with a cell
            data_Yneuron (np.ndarray): 1d array with y coordinate of the neuron associated with a cell
            clusters (np.ndarray): n x n matrix of cluster on neuron map

        Returns:
            np.ndarray: cluster_id
        """
        cluster_id = np.zeros((nz,ny,nx))
        for iz in prange(nz):
                for iy in prange(ny):
                        for ix in prange(nx):
                                j = iz * ny * nx + iy * nx + ix # convert from 3d coordinates to 1d row indices
                                cluster_id[iz,iy,ix] = clusters[int(data_Xneuron[j]), int(data_Yneuron[j])]
        return cluster_id

def batch_training(full_data : np.ndarray, xdim : int, ydim : int, alpha : float, train : int, batch : int, feature_list : list[str], save_neuron_values=False) -> som.map:
        """Function to perform batch training on a full domain

        Args:
            full_data (numpy ndarray): N x F array, where N is the number of data points and F is the number of features
            batch (int): width of the domain to be trained on
            feature_list (list[str]): list of feature names

        Returns:
            class: SOM map
        """
        width_of_new_window = batch
        x_4d = convert_to_4d(full_data)
        history = []
        feature_history = []
        nz, ny, nx = x_4d.shape[0], x_4d.shape[1], x_4d.shape[2]
        epoch = 0
        number_of_batches = (nz // width_of_new_window) * (ny // width_of_new_window) * (nx // width_of_new_window)

        for split_index1 in range(nz // width_of_new_window):
                start_index_crop_z = split_index1 * width_of_new_window
                for split_index2 in range(ny // width_of_new_window):
                        start_index_crop_y = split_index2 * width_of_new_window
                        for split_index3 in range(nx // width_of_new_window):
                                start_index_crop_x = split_index3 * width_of_new_window
                                
                                x_split_4d = get_smaller_domain(x_4d, width_of_new_window, start_index_crop_x, start_index_crop_y, start_index_crop_z)

                                x_split = flatten_to_2d(x_split_4d)
                                attr=pd.DataFrame(x_split)
                                attr.columns=feature_list

                                print(f'constructing batch SOM for xdim={xdim}, ydim={ydim}, alpha={alpha}, train={train}, index=[{start_index_crop_z},{start_index_crop_y},{start_index_crop_x}]...', flush=True)
                                m=som.map(xdim, ydim, alpha, train, epoch, number_of_batches, alpha_type='decay')
                                
                                print("Training step: ", epoch)

                                labels = np.array(list(range(len(x_split))))
                                if (split_index1 == 0) & (split_index2 == 0) & (split_index3 == 0):
                                        m.fit(attr,labels,restart=False)
                                else: # if first window, then initiate random neuron values, else use neurons from last batch
                                        m.fit(attr,labels,restart=True, neurons=neurons)

                                neurons = m.all_neurons()
                                epoch = m.epoch
                                
                                # print("neurons: ", neurons)
                                if save_neuron_values == True:
                                        np.save(f'neurons_{lap}_{xdim}{ydim}_{alpha}_{train}_{split_index1}-{split_index2}-{split_index3}.npy', neurons, allow_pickle=True)
                                        print("Data being saved")
                                
                                # print changes in neuron weights
                                # loss_history = m.loss_history
                                average_loss = m.average_loss
                                np.save(f'evolution_{lap}_{xdim}{ydim}_{alpha}_{epoch}_{split_index1}-{split_index2}-{split_index3}.npy', average_loss, allow_pickle=True)

        # at the end, load the entire domain back to m to assign cluster id
        attr=pd.DataFrame(full_data)
        attr.columns=feature_list
        labels = np.array(list(range(len(x))))
        m.fit_notraining(attr, labels, neurons)

        np.save(f'evolution_{lap}_{xdim}{ydim}_{alpha}_{train}_{batch}_combined.npy', np.array(history), allow_pickle=True)
        np.save(f'feature_evolution_{lap}_{xdim}{ydim}_{alpha}_{train}_{batch}_combined.npy', np.array(feature_history), allow_pickle=True)

        return m


if __name__ == "__main__":

        parser = argparse.ArgumentParser(description='SOM code')
        parser.add_argument("--features_path", type=str, dest='features_path', default='/mnt/ceph/users/tha10/SOM-tests/hr-d3x640/')
        parser.add_argument("--file", type=str, dest='file', default='features_4j1b1e_2800.h5')
        parser.add_argument('--xdim', type=int, dest='xdim', default=20, help='Map x size')
        parser.add_argument('--ydim', type=int, dest='ydim', default=20, help='Map y size')
        parser.add_argument('--alpha', type=float, dest='alpha', default=0.5, help='Learning parameter')
        parser.add_argument('--train', type=int, dest='train', default=10000, help='Number of training steps')
        parser.add_argument('--batch', type=int, dest='batch', default=None, help='Width of domain in a batch', required=False)
        parser.add_argument('--pretrained', action="store_true", dest='pretrained', help='Pass this argument if supplying a pre-trained model', required=False)
        parser.add_argument('--neurons_path', type=str, dest='neurons_path', default=None, help='Path to file containing neuron values', required=False)
        parser.add_argument('--save_neuron_values', dest='save_neuron_values', help='Pass this argument if you want to save neuron values', action="store_true")

        args = parser.parse_args()

        #--------------------------------------------------
        if (args.pretrained == True) & (args.neurons_path is None):
               sys.exit("Cannot run, no neuron values provided.")

        #--------------------------------------------------
        xmin = 0.0
        ymin = 0.0
        xmax = 1.0
        ymax = 1.0

        # CLI arguments
        features_path = args.features_path
        file_name = args.file
        xdim = args.xdim
        ydim = args.ydim
        alpha = args.alpha
        train = args.train
        batch = args.batch
        pretrained = args.pretrained
        neurons_path = args.neurons_path
        save_neuron_values = args.save_neuron_values
        
        if save_neuron_values is None:
                save_neuron_values = False
        
        lap = file_name.split("_")[2].split(".h5")[0] # all the data laps to process

        f5 = h5.File(features_path+file_name, 'r')
        x = f5['features'][()]
        feature_list = f5['names'][()]

        feature_list = [n.decode('utf-8') for n in feature_list]
        f5.close()
        print(f"File loaded, parameters: {lap}-{xdim}-{ydim}-{alpha}-{train}-{batch}", flush=True)
        
        nd = int(np.cbrt(x.shape[0]))
        nz,ny,nx = nd,nd,nd
        print("Shape of data: ", (nx,ny,nz), flush=True)

        # print(feature_list)
        # print("shape after x:", np.shape(x))

        #--------------------------------------------------
        # analyze
        #1. standardize:
        scaler = StandardScaler()
        # scaler = MinMaxScaler()

        scaler.fit(x)
        x = scaler.transform(x)

        # if the SOM is to be divided into smaller batches, separate those batches window by window
        if (batch is None) & (pretrained == False):
                attr=pd.DataFrame(x)
                attr.columns=feature_list

                print(f'constructing full SOM for xdim={xdim}, ydim={ydim}, alpha={alpha}, train={train}...', flush=True)
                m=som.map(xdim, ydim, alpha, train, alpha_type='decay')

                labels = np.array(list(range(len(x))))
                m.fit(attr,labels)
                neurons = m.all_neurons()
                # print("neurons: ", neurons)
                if save_neuron_values == True:
                        np.save(f'neurons_{lap}_{xdim}{ydim}_{alpha}_{train}.npy', neurons, allow_pickle=True)
                        print("Data being saved")
                # print changes in neuron weights
                # loss_history = m.loss_history
                average_loss = m.average_loss
                epoch = m.epoch
                np.save(f'evolution_{lap}_{xdim}{ydim}_{alpha}_{epoch}.npy', average_loss, allow_pickle=True)
        elif (batch is not None) & (pretrained == False):
                print(f"Constructing SOM batch training for xdim={xdim}, ydim={ydim}, alpha={alpha}, train={train}, batch={batch}", flush=True)
                m_batch = batch_training(x, xdim, ydim, alpha, train, batch, feature_list, save_neuron_values)
                # print("Saved neuron values : ", save_neuron_values)
                m = m_batch
        else: # if the run is initialized as a no training run, load these values
                print(f'constructing pre-trained SOM for xdim={xdim}, ydim={ydim}, alpha={alpha}, train={train}...', flush=True)
                m=som.map(xdim, ydim, alpha, train, alpha_type='decay')
                attr=pd.DataFrame(x)
                attr.columns=feature_list
                labels = np.array(list(range(len(x))))
                neurons = np.load(neurons_path)
                m.fit_notraining(attr,labels,neurons)

        #Data matrix with neuron positions:
        print("Calculating projection")
        data_matrix=m.projection()
        data_Xneuron=data_matrix[:,0]
        data_Yneuron=data_matrix[:,1]
        print("data matrix: ", flush=True)
        print(data_matrix[:10,:], flush=True)
        print("Printing Xneuron info", flush=True)
        print("Shape of Xneuron: ", data_Xneuron.shape, flush=True)
        print("Printing Yneuron info", flush=True)
        print("Shape of Yneuron: ", data_Yneuron.shape, flush=True)

        #Neuron matrix with centroids:
        umat = m.compute_umat(smoothing=2)
        centrs = m.compute_combined_clusters(umat, False, 0.15) #0.15
        centr_x = centrs['centroid_x']
        centr_y = centrs['centroid_y']

        #create list of centroid _locations
        neuron_x, neuron_y = np.shape(centr_x)

        centr_locs = []
        for i in range(neuron_x):
                for j in range(neuron_y):
                        cx = centr_x[i,j]
                        cy = centr_y[i,j]

                        centr_locs.append((cx,cy))

        unique_ids = list(set(centr_locs))
        n_clusters = len(unique_ids)
        print("Number of clusters", flush=True)
        print(n_clusters)

        mapping = {}
        for I, key in enumerate(unique_ids):
                # print(key, I)
                mapping[key] = I

        clusters = np.zeros((neuron_x,neuron_y))
        for i in range(neuron_x):
                for j in range(neuron_y):
                        key = (centr_x[i,j], centr_y[i,j])
                        I = mapping[key]

                        clusters[i,j] = I

        print("clusters", flush=True)
        print(clusters, flush=True)
        

        #TRANSFER RESULT BACK INTO ORIGINAL DATA PLOT
        # xinds = np.zeros(len(data_Xneuron))
        # print("shape of xinds:", np.shape(xinds))
        print("Assigning clusters", flush=True)
        
        cluster_id = assign_cluster_id(nx, ny, nz, data_Xneuron, data_Yneuron, clusters)
        np.save(f'clusters_{lap}_{xdim}{ydim}_{alpha}_{train}_{batch}.npy', cluster_id, allow_pickle=True)
        print("Done writing the cluster ID file")    
