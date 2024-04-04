## Self-organizing map base code including training and fitting data, along with limited plotting capabilities
## ported from the POPSOM library by Li Yuan (2018)
## https://github.com/njali2001/popsom.git 


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns					
from random import randint
from sklearn.metrics.pairwise import euclidean_distances
import statsmodels.stats.api as sms     # t-test
import statistics as stat               # F-test
from scipy import stats                 # KS Test
from scipy.stats import f               # F-test
from itertools import combinations

from numba import njit, prange

np.random.seed(42)


# NOTE: numba does not work in a class with pandas DataFrames. Can circumvent with @staticmethod
class map:
	def __init__(self, xdim : int = 10, ydim : int = 10, alpha : float = 0.3, train : int = 1000, epoch : int = 0, number_of_batches : int = 1, alpha_type : str = "decay", norm : bool = False, save_neurons : bool = False):
		""" __init__ -- Initialize the Model 

			parameters:
			- xdim,ydim - the dimensions of the map. Default is 10
			- alpha - the learning rate, should be a positive non-zero real number. Default is 0.3
			- train - number of training iterations. Default is 1000
			- step_counter - current step in the training process. Default is 0
			- number_of_batches - number of batches to train on. Default is 1
			- alpha_type - a string that determines whether the learning rate is static or decaying. Default is "decay"
			- norm - normalize the input data space. Default is False
			- save_neurons - save the neuron values at the end of training. Default is False
    	"""
		self.xdim = xdim
		self.ydim = ydim
		self.alpha = alpha
		self.train = train
		self.epoch = epoch
		self.number_of_batches = number_of_batches
		self.norm = norm
		if alpha_type == "static":
			self.alpha_type = 0
		elif alpha_type == "decay":
			self.alpha_type = 1
		else:
			sys.exit("alpha_type must be either 'static' or 'decay'")
		self.save_neurons = save_neurons

	def fit(self, data : pd.DataFrame, labels : np.ndarray, restart : bool = False, neurons : np.ndarray = None):
		""" fit -- Train the Model with numba JIT acceleration

			parameters:
			- data - a dataframe where each row contains an unlabeled training instance
			- labels - a vector or dataframe with one label for each observation in data
			- restart - a flag that determines whether fit starts with non-randomized values of neurons
			- neurons - vectors for the weights of the neurons from past realizations
    	"""

		if self.norm:
			data = data.div(data.sum(axis=1), axis=0)
			
		self.restart = restart
		
		if self.restart:
			self.restart_neurons  = neurons

		self.data = data
		self.data_array = data.to_numpy()	
		self.labels = labels
		# self.momentum_decay_rate = momentum_decay_rate

		# check if the dims are reasonable
		if (self.xdim < 3 or self.ydim < 3):
			sys.exit("build: map is too small.")

		# generate / train neuron map
		self.fast_som()

		print("Begin matching points with neuron", flush=True)
		visual = self.best_match(self.neurons, self.data_array)

		self.visual = visual

	def fit_notraining(self, data : pd.DataFrame, labels : np.ndarray, neurons : np.ndarray, parallel : bool = False):
		"""fit_notraining -- Provided an array of neurons, load best-fit data into the class

		Args:
			data (pandas DataFrame): data with features as columns
			labels (np.ndarray): range from 0 to number of points in data, acts as label
			neurons (numpy 3d array): xdim x ydim x feature array of neurons
			parallel (bool): indicate whether to use multiple threads to find best fit neurons
		"""

		self.data = data
		self.data_array = data.to_numpy()
		self.labels = labels
		self.neurons = neurons
		
		print("Begin matching points with neuron", flush=True)
		visual = self.best_match(self.neurons, self.data_array)

		self.visual = visual
	
	# neighborhood function
	@staticmethod
	@njit()
	def Gamma(c : int, m2Ds : np.ndarray, alpha : float, nsize : int, gaussian : bool = True):
		""" Gamma -- neighborhood function"""

		# 2d distance between neuron c and the rest of the map
		dist_2d = np.abs(m2Ds[c,:] - m2Ds)
		rectangular_dist = np.zeros(dist_2d.shape[0])
		for i in prange(dist_2d.shape[0]): # numba max does not have axis argument, otherwise this would be rectangular_dist = np.max(dist_2d, axis=1)
			rectangular_dist[i] = np.max(dist_2d[i,:])
  
		#Define the Gaussian function to calculate the neighborhood function
		def gauss(dist, A, sigma):
			""" gauss -- Gaussian function"""
			# sigma = 1.
			return A * np.exp(-(dist) ** 2 / (2 * sigma ** 2))
		
		# if neuron on the grid is in within nsize neighborhood, then h = Gaussian(alpha), else h = 0.0
		if gaussian:
			h = gauss(rectangular_dist, alpha, nsize/3)
		else:
			h = np.where(rectangular_dist <= nsize, alpha, 0.)
		
		h[rectangular_dist > nsize] = 0.

		return h
	

	def fast_som(self):
		""" fast_som -- fast implementation of the self-organizing map algorithm using numba JIT acceleration"""
    	# some constants
		dr = self.data_array.shape[0]
		dc = self.data_array.shape[1]
		nr = self.xdim * self.ydim
		nc = dc  # dim of data and neurons is the same

		if self.restart:
			neurons = self.restart_neurons
		else:
			# vector with small init values for all neurons
			# NOTE: each row represents a neuron, each column represents a dimension.
			neurons = np.random.uniform(-1, 1, (nr,nc))

		alpha = self.alpha # starting learning rate
		if self.alpha_type == 1:
			alpha_freq = self.train // 25 # how often to decay the learning rate

	    # compute the initial neighborhood size and step
		nsize_max = max(self.xdim, self.ydim) + 1
		nsize_min = 8
		nsize_step = self.train // 20 # for the first 5% of the training steps, shrink the neighborhood

		if self.epoch > nsize_step:
			nsize = nsize_min
		else:
			nsize_freq = nsize_step // (nsize_max - nsize_min) # how often to shrink the neighborhood
			nsize = nsize_max - self.epoch // nsize_freq
   
		epoch = self.epoch  # counts the number of epochs per nsize_freq
		# print("starting epoch for this batch is ", epoch, flush=True)

	    # constants for the Gamma function
		m = np.reshape(list(range(nr)), (nr,1))  # a vector with all neuron 1D addresses

	    # x-y coordinate of ith neuron: m2Ds[i,] = c(xi, yi)
		m2Ds = self.coordinate(m, self.xdim)

		# Initialize [train] number of random observations for training
		ix = np.random.randint(0, dr-1, self.train)
		xk = self.data_array[ix,:]

		# Save the stepping of the neurons for termination condition
		neurons_old = neurons.copy()
		frequency = 1000
		# self.weight_history = np.zeros((self.train//frequency, 2))
		# self.feature_weight_history = np.zeros((self.train//frequency, dc))
		i = 0
  
		# implement momentum-based gradient descent
		# momentum_decay_rate = self.momentum_decay_rate
		diff = 0.

		# history of the loss function
		self.loss_history = np.zeros((self.train, dc))
		self.average_loss = np.zeros((self.train//frequency, dc))

		# for epoch in range(self.step_counter, self.train):
		while True:
	        # hood size decreases in disrete nsize steps
			if (epoch % frequency == 0) & (epoch != 0):
				
				this_average_loss = np.mean(self.loss_history[epoch-frequency:epoch], axis=0) # average loss over the last frequency epochs
				self.average_loss[epoch//frequency-1,:] = this_average_loss
    
				# Terminate if the network has not changed much in the last train//100 epochs
				if np.sum(this_average_loss**2)  < 1e-4:
					print("Terminating from small changes at epoch ", epoch, flush=True)
					self.epoch = epoch
					print("Saving final neurons weights", epoch, flush=True)
					np.save(f"neuronsf_{epoch}_{self.xdim}{self.ydim}_{self.alpha}_{self.train}.npy", neurons)
					break

			# if this batch has gone over the step limit for the batch (which is total training steps divided by number of batches), terminate
			if (epoch - self.epoch) >= (self.train // self.number_of_batches):
				print("Terminating from step limit reached at epoch ", epoch, flush=True)
				self.epoch = epoch
				if self.save_neurons:
					print("Saving final neurons weights", epoch, flush=True)
					np.save(f"neuronsf_{epoch}_{self.xdim}{self.ydim}_{self.alpha}_{self.train}.npy", neurons)
				break

	        # competitive step
			xk_m = xk[epoch,:]
			
			# calculate the relative distance between features and neurons, take the closest neuron
			# momentum = diff * momentum_decay_rate # momentum-based gradient descent
			diff = neurons - xk_m
			squ = diff**2
			s = np.sum(squ, axis=1)
			c = np.argmin(s)

			self.loss_history[epoch,:] = s[c]

	        # update step
			gamma_m = np.outer(self.Gamma(c, m2Ds, alpha, nsize), np.ones(nc))
			# neurons -= (diff + momentum) * gamma_m
			neurons -= diff * gamma_m

			# shrink the neighborhood size every frequ epochs
			if epoch % nsize_freq == 0 and nsize > 8:
				nsize = nsize_max - epoch // nsize_freq
				print(f"Shrinking neighborhood size to {nsize} at epoch {epoch}", flush=True)

			# decay the learning rate every frequ epochs
			if epoch % alpha_freq == 0 and self.alpha_type == 1 and epoch != 0:
				alpha *= 0.75
				print(f"Decaying learning rate to {alpha} at epoch {epoch}", flush=True)
    
			# save neuron maps sparingly
			if epoch % 1000000 == 0 and epoch != 0:
				print("Saving neurons at epoch ", epoch, flush=True)
				np.save(f"neurons_{epoch}_{self.xdim}{self.ydim}_{self.alpha}_{self.train}.npy", neurons)	
    
			epoch += 1

		self.neurons = neurons

	@staticmethod
	@njit(parallel=True)
	def best_match(neurons : np.ndarray, obs : np.ndarray, full=False) -> np.ndarray:
		"""best_match -- given observation obs[n,f] (where n is number of different observations, f is number of features per observation), return the best matching neuron

		Args:
			neurons (np.ndarray): values of all the neurons from the map
			obs (np.ndarray): observations
			full (bool, optional): indicate whether to return first and second best match. Defaults to False.

		Returns:
			np.ndarray: return the 1d(2d) array of the best-matched neuron for each observation
		"""

		if full:
			best_match_neuron = np.zeros((obs.shape[0], 2))
		else:
			best_match_neuron = np.zeros((obs.shape[0],1))

		for i in prange(obs.shape[0]):
			diff = neurons - obs[i,:]
			squ = diff ** 2
			s = np.sum(squ, axis=1)

			if full:
				# numba does not support argsort, so we record the top two best matches this way
				o = np.argmin(s)
				best_match_neuron[i,0] = o
				s[o] = np.max(s)
				o = np.argmin(s)
				best_match_neuron[i,1] = o
			else:
				best_match_neuron[i] = np.argmin(s)

			if i % int(1e7) == 0:
				print("i = ", i)

		return best_match_neuron
	
	def projection(self) -> np.ndarray:
		"""projection -- print the association of observations with map elements

		Returns:
			np.ndarray: for each observations, return the x and y value on the neuron map
		"""

		visual_reshaped = np.reshape(self.visual,(len(self.visual),1))

		data_matrix = self.coordinate(visual_reshaped, self.xdim)

		return data_matrix

	def neuron(self, x, y):
		""" neuron -- returns the contents of a neuron at (x,y) on the map as a vector
		
			parameters:
			 - x - map x-coordinate of neuron
			 - y - map y-coordinate of neuron
		
			return value:
			 - a vector representing the neuron
		"""

		ix = self.rowix(x, y)
		return self.neurons[ix]
	
	def all_neurons(self):
		""" all_neurons -- returns the entire neuron maps to load into next step
		"""
		return self.neurons

	@staticmethod
	@njit(parallel=True)
	def coordinate(rowix : np.ndarray, xdim : int) -> np.ndarray:
		"""coordinate -- convert from a list of row index to an array of xy-coordinate

		Args:
			rowix (np.ndarray): 2d array with the 1d indices of the points of interest (n x 1 matrix)
			xdim (int): x dimension of the neuron map

		Returns:
			np.ndarray: array with x and y coordinates of each point in rowix
		"""

		len_rowix = len(rowix)
		coords = np.zeros((len_rowix, 2))

		for k in prange(len_rowix):
			coords[k,:] = np.array([rowix[k,0] % xdim, rowix[k,0] // xdim])

		return coords

	## Any function below this line was not significantly modified from the original POPSOM library
	## Refer to: https://github.com/njali2001/popsom.git for documentation

	def rowix(self, x, y):
		""" rowix -- convert from a map xy-coordinate to a row index  """

		rix = x + y*self.xdim
		return rix
		
	def convergence(self, conf_int=.95, k=50, verb=False, ks=False):
		""" convergence -- the convergence index of a map
		
			Parameters:
			- conf_int - the confidence interval of the quality assessment (default 95%)
			- k - the number of samples used for the estimated topographic accuracy computation
			- verb - if true reports the two convergence components separately, otherwise it will
			         report the linear combination of the two
			- ks - a switch, true for ks-test, false for standard var and means test
			
			Return
			- return value is the convergence index
		"""

		if ks:
			embed = self.embed_ks(conf_int, verb=False)
		else:
			embed = self.embed_vm(conf_int, verb=False)

		topo_ = self.topo(k, conf_int, verb=False, interval=False)

		if verb:
			return {"embed": embed, "topo": topo_}
		else:
			return (0.5*embed + 0.5*topo_)		

	def compute_centroids(self, heat, explicit=False):
		""" compute_centroids -- compute the centroid for each point on the map
		
			parameters:
			- heat - is a matrix representing the heat map representation
			- explicit - controls the shape of the connected component
			
			return value:
			- a list containing the matrices with the same x-y dims as the original map containing the centroid x-y coordinates
		"""

		xdim = self.xdim
		ydim = self.ydim
		centroid_x = np.array([[-1] * ydim for _ in range(xdim)])
		centroid_y = np.array([[-1] * ydim for _ in range(xdim)])

		heat = np.array(heat)

		def compute_centroid(ix, iy):
			# recursive function to find the centroid of a point on the map

			if (centroid_x[ix, iy] > -1) and (centroid_y[ix, iy] > -1):
				return {"bestx": centroid_x[ix, iy], "besty": centroid_y[ix, iy]}

			min_val = heat[ix, iy]
			min_x = ix
			min_y = iy

			# (ix, iy) is an inner map element
			if ix > 0 and ix < xdim-1 and iy > 0 and iy < ydim-1:

				if heat[ix-1, iy-1] < min_val:
					min_val = heat[ix-1, iy-1]
					min_x = ix-1
					min_y = iy-1

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix+1, iy-1] < min_val:
					min_val = heat[ix+1, iy-1]
					min_x = ix+1
					min_y = iy-1

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

				if heat[ix+1, iy+1] < min_val:
					min_val = heat[ix+1, iy+1]
					min_x = ix+1
					min_y = iy+1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

				if heat[ix-1, iy+1] < min_val:
					min_val = heat[ix-1, iy+1]
					min_x = ix-1
					min_y = iy+1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is bottom left corner
			elif ix == 0 and iy == 0:

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

				if heat[ix+1, iy+1] < min_val:
					min_val = heat[ix+1, iy+1]
					min_x = ix+1
					min_y = iy+1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

			# (ix, iy) is bottom right corner
			elif ix == xdim-1 and iy == 0:

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

				if heat[ix-1, iy+1] < min_val:
					min_val = heat[ix-1, iy+1]
					min_x = ix-1
					min_y = iy+1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is top right corner
			elif ix == xdim-1 and iy == ydim-1:

				if heat[ix-1, iy-1] < min_val:
					min_val = heat[ix-1, iy-1]
					min_x = ix-1
					min_y = iy-1

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is top left corner
			elif ix == 0 and iy == ydim-1:

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix+1, iy-1] < min_val:
					min_val = heat[ix+1, iy-1]
					min_x = ix+1
					min_y = iy-1

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

			# (ix, iy) is a left side element
			elif ix == 0 and iy > 0 and iy < ydim-1:

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix+1, iy-1] < min_val:
					min_val = heat[ix+1, iy-1]
					min_x = ix+1
					min_y = iy-1

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

				if heat[ix+1, iy+1] < min_val:
					min_val = heat[ix+1, iy+1]
					min_x = ix+1
					min_y = iy+1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

			# (ix, iy) is a bottom side element
			elif ix > 0 and ix < xdim-1 and iy == 0:

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy
	
				if heat[ix+1, iy+1] < min_val:
					min_val = heat[ix+1, iy+1]
					min_x = ix+1
					min_y = iy+1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

				if heat[ix-1, iy+1] < min_val:
					min_val = heat[ix-1, iy+1]
					min_x = ix-1
					min_y = iy+1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is a right side element
			elif ix == xdim-1 and iy > 0 and iy < ydim-1:

				if heat[ix-1, iy-1] < min_val:
					min_val = heat[ix-1, iy-1]
					min_x = ix-1
					min_y = iy-1

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

				if heat[ix-1, iy+1] < min_val:
					min_val = heat[ix-1, iy+1]
					min_x = ix-1
					min_y = iy+1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is a top side element
			elif ix > 0 and ix < xdim-1 and iy == ydim-1:

				if heat[ix-1, iy-1] < min_val:
					min_val = heat[ix-1, iy-1]
					min_x = ix-1
					min_y = iy-1

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix+1, iy-1] < min_val:
					min_val = heat[ix+1, iy-1]
					min_x = ix+1
					min_y = iy-1

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

	        # if successful
	        # move to the square with the smaller value, i_e_, call
	        # compute_centroid on this new square
	        # note the RETURNED x-y coords in the centroid_x and
	        # centroid_y matrix at the current location
	        # return the RETURNED x-y coordinates

			if min_x != ix or min_y != iy:
				r_val = compute_centroid(min_x, min_y)

	            # if explicit is set show the exact connected component
	            # otherwise construct a connected componenent where all
	            # nodes are connected to a centrol node
				if explicit:

					centroid_x[ix, iy] = min_x
					centroid_y[ix, iy] = min_y
					return {"bestx": min_x, "besty": min_y}

				else:
					centroid_x[ix, iy] = r_val['bestx']
					centroid_y[ix, iy] = r_val['besty']
					return r_val

			else:
				centroid_x[ix, iy] = ix
				centroid_y[ix, iy] = iy
				return {"bestx": ix, "besty": iy}

		for i in range(xdim):
			for j in range(ydim):
				compute_centroid(i, j)

		return {"centroid_x": centroid_x, "centroid_y": centroid_y}

	def compute_combined_clusters(self, heat, explicit, rang):

		# compute the connected components
		centroids = self.compute_centroids(heat, explicit)
		# Get unique centroids
		unique_centroids = self.get_unique_centroids(centroids)
		# Get distance from centroid to cluster elements for all centroids
		within_cluster_dist = self.distance_from_centroids(centroids,
														   unique_centroids,
														   heat)
		# Get average pairwise distance between clusters
		between_cluster_dist = self.distance_between_clusters(centroids,
															  unique_centroids,	
															  heat)
		# Get a boolean matrix of whether two components should be combined
		combine_cluster_bools = self.combine_decision(within_cluster_dist,
													  between_cluster_dist,
													  rang)
		# Create the modified connected components grid
		ne_centroid = self.new_centroid(combine_cluster_bools,
										centroids,
										unique_centroids)

		return ne_centroid

	def get_unique_centroids(self, centroids):
		""" get_unique_centroids -- a function that computes a list of unique centroids from
		                            a matrix of centroid locations.
		
			parameters:
			- centroids - a matrix of the centroid locations in the map
		"""

		# get the dimensions of the map
		xdim = self.xdim
		ydim = self.ydim
		xlist = []
		ylist = []
		x_centroid = centroids['centroid_x']
		y_centroid = centroids['centroid_y']

		for ix in range(xdim):
			for iy in range(ydim):
				cx = x_centroid[ix, iy]
				cy = y_centroid[ix, iy]

		# Check if the x or y of the current centroid is not in the list
		# and if not
		# append both the x and y coordinates to the respective lists
				if not(cx in xlist) or not(cy in ylist):
					xlist.append(cx)
					ylist.append(cy)

		# return a list of unique centroid positions
		return {"position_x": xlist, "position_y": ylist}

	def distance_from_centroids(self, centroids, unique_centroids, heat):
		""" distance_from_centroids -- A function to get the average distance from
		                               centroid by cluster.
			parameters:
			- centroids - a matrix of the centroid locations in the map
			- unique_centroids - a list of unique centroid locations
			- heat - a unified distance matrix
		"""

		centroids_x_positions = unique_centroids['position_x']
		centroids_y_positions = unique_centroids['position_y']
		within = np.zeros(len(centroids_x_positions))

		for i in range(len(centroids_x_positions)):
			cx = centroids_x_positions[i]
			cy = centroids_y_positions[i]

			# compute the average distance
			distance = self.cluster_spread(cx, cy, np.array(heat), centroids)

			# append the computed distance to the list of distances
			within[i] = distance

		return within

	def cluster_spread(self, x, y, umat, centroids):
		""" cluster_spread -- Function to calculate the average distance in
		                      one cluster given one centroid.
		
			parameters:
			- x - x position of a unique centroid
			- y - y position of a unique centroid
			- umat - a unified distance matrix
			- centroids - a matrix of the centroid locations in the map
		"""

		centroid_x = x
		centroid_y = y
		sum = 0
		elements = 0
		xdim = self.xdim
		ydim = self.ydim
		centroid_weight = umat[centroid_x, centroid_y]

		for xi in range(xdim):
			for yi in range(ydim):
				cx = centroids['centroid_x'][xi, yi]
				cy = centroids['centroid_y'][xi, yi]

				if(cx == centroid_x and cy == centroid_y):
					cweight = umat[xi, yi]
					sum = sum + abs(cweight - centroid_weight)
					elements = elements + 1

		average = sum / elements

		return average

	def distance_between_clusters(self, centroids, unique_centroids, umat):
		""" distance_between_clusters -- A function to compute the average pairwise
		                                 distance between clusters.
		
			parameters:
			- centroids - a matrix of the centroid locations in the map
			- unique_centroids - a list of unique centroid locations
			- umat - a unified distance matrix
		"""

		cluster_elements = self.list_clusters(centroids, unique_centroids, umat)

		tmp_1 = np.zeros(shape=(max([len(cluster_elements[i]) for i in range(
				len(cluster_elements))]), len(cluster_elements)))

		for i in range(len(cluster_elements)):
			for j in range(len(cluster_elements[i])):
				tmp_1[j, i] = cluster_elements[i][j]

		columns = tmp_1.shape[1]

		tmp = np.transpose(np.array(list(combinations([i for i in range(columns)], 2))))
		print("shape of tmp: ", tmp.shape)
		print("shape of tmp_1: ", tmp_1.shape)

		tmp_3 = np.zeros(shape=(tmp_1.shape[0], tmp.shape[1]))

		for i in range(tmp.shape[1]):
			tmp_3[:, i] = np.where(tmp_1[:, tmp[1, i]]*tmp_1[:, tmp[0, i]] != 0,
									abs(tmp_1[:, tmp[0, i]]-tmp_1[:, tmp[1, i]]), 0)
	        # both are not equals 0

		mean = np.true_divide(tmp_3.sum(0), (tmp_3 != 0).sum(0))
		index = 0
		mat = np.zeros((columns, columns))

		for xi in range(columns-1):
			for yi in range(xi, columns-1):
				mat[xi, yi + 1] = mean[index]
				mat[yi + 1, xi] = mean[index]
				index = index + 1

		return mat

	def list_clusters(self, centroids, unique_centroids, umat):
		""" list_clusters -- A function to get the clusters as a list of lists.
		
			parameters:
			- centroids - a matrix of the centroid locations in the map
			- unique_centroids - a list of unique centroid locations
			- umat - a unified distance matrix
		"""

		centroids_x_positions = unique_centroids['position_x']
		centroids_y_positions = unique_centroids['position_y']
		cluster_list = []

		for i in range(len(centroids_x_positions)):
			cx = centroids_x_positions[i]
			cy = centroids_y_positions[i]

	    # get the clusters associated with a unique centroid and store it in a list
			cluster_list.append(self.list_from_centroid(cx, cy, centroids, umat))

		return cluster_list

	def list_from_centroid(self, x, y, centroids, umat):
		""" list_from_centroid -- A funtion to get all cluster elements
		                          associated to one centroid.
		
			parameters:
			- x - the x position of a centroid
			- y - the y position of a centroid
			- centroids - a matrix of the centroid locations in the map
			- umat - a unified distance matrix
		"""

		centroid_x = x
		centroid_y = y
		xdim = self.xdim
		ydim = self.ydim

		cluster_list = []
		for xi in range(xdim):
			for yi in range(ydim):
				cx = centroids['centroid_x'][xi, yi]
				cy = centroids['centroid_y'][xi, yi]

				if(cx == centroid_x and cy == centroid_y):
					cweight = np.array(umat)[xi, yi]
					cluster_list.append(cweight)

		return cluster_list

	def combine_decision(self, within_cluster_dist, distance_between_clusters, rang):
		""" combine_decision -- A function that produces a boolean array
		                        representing which clusters should be combined.
		
			parameters:
			- within_cluster_dist - A list of the distances from centroid to cluster elements for all centroids
			- distance_between_clusters - A list of the average pairwise distance between clusters
			- range - The distance where the clusters are merged together.
		"""

		inter_cluster = distance_between_clusters
		centroid_dist = within_cluster_dist
		dim = inter_cluster.shape[1]
		to_combine = np.array([[False]*dim]*dim)

		for xi in range(dim):
			for yi in range(dim):
				cdist = inter_cluster[xi, yi]
				if cdist != 0:
					rx = centroid_dist[xi] * rang
					ry = centroid_dist[yi] * rang
					if (cdist < centroid_dist[xi] + rx or
						cdist < centroid_dist[yi] + ry):
						to_combine[xi, yi] = True

		return to_combine

	def new_centroid(self, bmat, centroids, unique_centroids):
		""" new_centroid -- A function to combine centroids based on matrix of booleans.
		
			parameters:
			- bmat - a boolean matrix containing the centroids to merge
			- centroids - a matrix of the centroid locations in the map
			- unique_centroids - a list of unique centroid locations
		"""

		bmat_rows = bmat.shape[0]
		bmat_columns = bmat.shape[1]
		centroids_x = unique_centroids['position_x']
		centroids_y = unique_centroids['position_y']
		components = centroids

		for xi in range(bmat_rows):
			for yi in range(bmat_columns):
				if bmat[xi, yi]:
					x1 = centroids_x[xi]
					y1 = centroids_y[xi]
					x2 = centroids_x[yi]
					y2 = centroids_y[yi]
					components = self.swap_centroids(x1, y1, x2, y2, components)

		return components

	def swap_centroids(self, x1, y1, x2, y2, centroids):
		""" swap_centroids -- A function that changes every instance of a centroid to
		                      one that it should be combined with.
			parameters:
			- centroids - a matrix of the centroid locations in the map
		"""

		xdim = self.xdim
		ydim = self.ydim
		compn_x = centroids['centroid_x']
		compn_y = centroids['centroid_y']
		for xi in range(xdim):
			for yi in range(ydim):
				if compn_x[xi, 0] == x1 and compn_y[yi, 0] == y1:
					compn_x[xi, 0] = x2
					compn_y[yi, 0] = y2

		return {"centroid_x": compn_x, "centroid_y": compn_y}

	def embed(self, conf_int=.95, verb=False, ks=False):
		""" embed -- evaluate the embedding of a map using the F-test and
		             a Bayesian estimate of the variance in the training data.
		
			parameters:
			- conf_int - the confidence interval of the convergence test (default 95%)
			- verb - switch that governs the return value false: single convergence value
			  		 is returned, true: a vector of individual feature congences is returned.
			
			- return value:
			- return is the cembedding of the map (variance captured by the map so far)

			Hint: 
				  the embedding index is the variance of the training data captured by the map;
			      maps with convergence of less than 90% are typically not trustworthy.  Of course,
			      the precise cut-off depends on the noise level in your training data.
		"""

		if ks:
			return self.embed_ks(conf_int, verb)
		else:
			return self.embed_vm(conf_int, verb)

	def embed_ks(self, conf_int=0.95, verb=False):
		""" embed_ks -- using the kolgomorov-smirnov test """

		# map_df is a dataframe that contains the neurons
		map_df = self.neurons

		# data_df is a dataframe that contain the training data
		data_df = self.data_array

		nfeatures = map_df.shape[1]

		# use the Kolmogorov-Smirnov Test to test whether the neurons and training
		# data appear
		# to come from the same distribution
		ks_vector = []
		for i in range(nfeatures):
			ks_vector.append(stats.mstats.ks_2samp(map_df[:, i], data_df[:, i]))

		prob_v = self.significance(graphics=False)
		var_sum = 0

		# compute the variance captured by the map
		for i in range(nfeatures):

			# the second entry contains the p-value
			if ks_vector[i][1] > (1 - conf_int):
				var_sum = var_sum + prob_v[i]
			else:
				# not converged - zero out the probability
				prob_v[i] = 0

		# return the variance captured by converged features
		if verb:
			return prob_v
		else:
			return var_sum

	def embed_vm(self, conf_int=.95, verb=False):
		""" embed_vm -- using variance and mean tests  """

		# map_df is a dataframe that contains the neurons
		map_df = self.neurons

		# data_df is a dataframe that contain the training data
		data_df = self.data_array

		def df_var_test(df1, df2, conf=.95):

			if df1.shape[1] != df2.shape[1]:
				sys.exit("df_var_test: cannot compare variances of data frames")

			# init our working arrays
			var_ratio_v = [randint(1, 1) for _ in range(df1.shape[1])]
			var_confintlo_v = [randint(1, 1) for _ in range(df1.shape[1])]
			var_confinthi_v = [randint(1, 1) for _ in range(df1.shape[1])]

			def var_test(x, y, ratio=1, conf_level=0.95):

				DF_x = len(x) - 1
				DF_y = len(y) - 1
				V_x = stat.variance(x.tolist())
				V_y = stat.variance(y.tolist())

				ESTIMATE = V_x / V_y

				BETA = (1 - conf_level) / 2
				CINT = [ESTIMATE / f.ppf(1 - BETA, DF_x, DF_y),
						ESTIMATE / f.ppf(BETA, DF_x, DF_y)]

				return {"estimate": ESTIMATE, "conf_int": CINT}

		    # compute the F-test on each feature in our populations
			for i in range(df1.shape[1]):

				t = var_test(df1[:, i], df2[:, i], conf_level=conf)
				var_ratio_v[i] = t['estimate']
				var_confintlo_v[i] = t['conf_int'][0]
				var_confinthi_v[i] = t['conf_int'][1]

			# return a list with the ratios and conf intervals for each feature
			return {"ratio": var_ratio_v,
					"conf_int_lo": var_confintlo_v,
					"conf_int_hi": var_confinthi_v}

		def df_mean_test(df1, df2, conf=0.95):

			if df1.shape[1] != df2.shape[1]:
				sys.exit("df_mean_test: cannot compare means of data frames")

			# init our working arrays
			mean_diff_v = [randint(1, 1) for _ in range(df1.shape[1])]
			mean_confintlo_v = [randint(1, 1) for _ in range(df1.shape[1])]
			mean_confinthi_v = [randint(1, 1) for _ in range(df1.shape[1])]

			def t_test(x, y, conf_level=0.95):
				estimate_x = np.mean(x)
				estimate_y = np.mean(y)
				cm = sms.CompareMeans(sms.DescrStatsW(x), sms.DescrStatsW(y))
				conf_int_lo = cm.tconfint_diff(alpha=1-conf_level, usevar='unequal')[0]
				conf_int_hi = cm.tconfint_diff(alpha=1-conf_level, usevar='unequal')[1]

				return {"estimate": [estimate_x, estimate_y],
						"conf_int": [conf_int_lo, conf_int_hi]}

			# compute the F-test on each feature in our populations
			for i in range(df1.shape[1]):
				t = t_test(x=df1[:, i], y=df2[:, i], conf_level=conf)
				mean_diff_v[i] = t['estimate'][0] - t['estimate'][1]
				mean_confintlo_v[i] = t['conf_int'][0]
				mean_confinthi_v[i] = t['conf_int'][1]

			# return a list with the ratios and conf intervals for each feature
			return {"diff": mean_diff_v,
					"conf_int_lo": mean_confintlo_v,
					"conf_int_hi": mean_confinthi_v}
		# do the F-test on a pair of datasets
		vl = df_var_test(map_df, data_df, conf_int)

		# do the t-test on a pair of datasets
		ml = df_mean_test(map_df, data_df, conf=conf_int)

		# compute the variance captured by the map --
		# but only if the means have converged as well.
		nfeatures = map_df.shape[1]
		prob_v = self.significance(graphics=False)
		var_sum = 0

		for i in range(nfeatures):

			if (vl['conf_int_lo'][i] <= 1.0 and vl['conf_int_hi'][i] >= 1.0 and
				ml['conf_int_lo'][i] <= 0.0 and ml['conf_int_hi'][i] >= 0.0):

				var_sum = var_sum + prob_v[i]
			else:
				# not converged - zero out the probability
				prob_v[i] = 0

		# return the variance captured by converged features
		if verb:
			return prob_v
		else:
			return var_sum

	def topo(self, k=50, conf_int=.95, verb=False, interval=True):
		""" topo -- measure the topographic accuracy of the map using sampling
		
			parameters:
			- k - the number of samples used for the accuracy computation
			- conf_int - the confidence interval of the accuracy test (default 95%)
			- verb - switch that governs the return value, false: single accuracy value
			  		 is returned, true: a vector of individual feature accuracies is returned.
			- interval - a switch that controls whether the confidence interval is computed.
			
			- return value is the estimated topographic accuracy
		"""
		

		# data.df is a matrix that contains the training data
		data_df = self.data_array

		if (k > data_df.shape[0]):
			sys.exit("topo: sample larger than training data.")

		data_sample_ix = [randint(1, data_df.shape[0]) for _ in range(k)]

		# compute the sum topographic accuracy - the accuracy of a single sample
		# is 1 if the best matching unit is a neighbor otherwise it is 0
		acc_v = np.zeros(k)
		for i in range(k):
			acc_v[i] = self.accuracy(data_df.iloc[data_sample_ix[i]-1], data_sample_ix[i])

		# compute the confidence interval values using the bootstrap
		if interval:
			bval = self.bootstrap(conf_int, data_df, k, acc_v)

		# the sum topographic accuracy is scaled by the number of samples -
		# estimated
		# topographic accuracy
		if verb:
			return acc_v
		else:
			val = np.sum(acc_v)/k
			if interval:
				return {'val': val, 'lo': bval['lo'], 'hi': bval['hi']}
			else:
				return val

	def bootstrap(self, conf_int, data_df, k, sample_acc_v):
		""" bootstrap -- compute the topographic accuracies for the given confidence interval """

		ix = int(100 - conf_int*100)
		bn = 200

		bootstrap_acc_v = [np.sum(sample_acc_v)/k]

		for i in range(2, bn+1):

			bs_v = np.array([randint(1, k) for _ in range(k)])-1
			a = np.sum(list(np.array(sample_acc_v)[list(bs_v)]))/k
			bootstrap_acc_v.append(a)

		bootstrap_acc_sort_v = np.sort(bootstrap_acc_v)

		lo_val = bootstrap_acc_sort_v[ix-1]
		hi_val = bootstrap_acc_sort_v[bn-ix-1]

		return {'lo': lo_val, 'hi': hi_val}	

	def accuracy(self, sample, data_ix):
		""" accuracy -- the topographic accuracy of a single sample is 1 is the best matching unit
		             	and the second best matching unit are are neighbors otherwise it is 0
		"""

		o = self.best_match(self.neurons, sample, full=True)
		best_ix = o[0]
		second_best_ix = o[1]

		# sanity check
		coord = self.coordinate(np.reshape(best_ix,(1,1)), self.xdim)
		coord_x = coord[0,0]
		coord_y = coord[0,1]

		map_ix = self.visual[data_ix-1]
		coord = self.coordinate(np.reshape(map_ix,(1,1)), self.xdim)
		map_x = coord[0,0]
		map_y = coord[0,1]

		if (coord_x != map_x or coord_y != map_y or best_ix != map_ix):
			print("Error: best_ix: ", best_ix, " map_ix: ", map_ix, "\n")

		# determine if the best and second best are neighbors on the map
		best_xy = self.coordinate(np.reshape(best_ix,(1,1)), self.xdim)
		second_best_xy = self.coordinate(np.reshape(second_best_ix,(1,1)), self.xdim)
		diff_map = best_xy[0,:] - second_best_xy[0,:]
		diff_map_sq = diff_map * diff_map
		sum_map = np.sum(diff_map_sq)
		dist_map = np.sqrt(sum_map)

		# it is a neighbor if the distance on the map
		# between the bmu and 2bmu is less than 2,   should be 1 or 1.414
		if dist_map < 2:
			return 1
		else:
			return 0

	def significance(self, graphics=False, feature_labels=False):
		""" significance -- compute the relative significance of each feature and plot it
		
			parameters:
			- graphics - a switch that controls whether a plot is generated or not
			- feature_labels - a switch to allow the plotting of feature names vs feature indices
			
			return value:
			- a vector containing the significance for each feature  
		"""

		data_df = self.data
		nfeatures = data_df.shape[1]

	    # Compute the variance of each feature on the map
		var_v = [randint(1, 1) for _ in range(nfeatures)]

		for i in range(nfeatures):
			var_v[i] = np.var(np.array(data_df)[:, i])

	    # we use the variance of a feature as likelihood of
	    # being an important feature, compute the Bayesian
	    # probability of significance using uniform priors

		var_sum = np.sum(var_v)
		prob_v = var_v/var_sum

	    # plot the significance
		if graphics:
			y = max(prob_v)

			plt.axis([0, nfeatures+1, 0, y])

			x = np.arange(1, nfeatures+1)
			tag = list(data_df)

			plt.xticks(x, tag)
			plt.yticks = np.linspace(0, y, 5)

			i = 1
			for xc in prob_v:
				plt.axvline(x=i, ymin=0, ymax=xc)
				i += 1

			plt.xlabel('Features')
			plt.ylabel('Significance')
			plt.show()
		else:
			return prob_v

	def smooth_2d(self, Y, ind=None, weight_obj=None, grid=None, nrow=64, ncol=64, surface=True, theta=None):
		""" smooth_2d -- Kernel Smoother For Irregular 2-D Data """

		def exp_cov(x1, x2, theta=2, p=2, distMat=0):
			x1 = x1*(1/theta)
			x2 = x2*(1/theta)
			distMat = euclidean_distances(x1, x2)
			distMat = distMat**p
			return np.exp(-distMat)

		NN = [[1]*ncol] * nrow
		grid = {'x': [i for i in range(nrow)], "y": [i for i in range(ncol)]}

		if weight_obj is None:
			dx = grid['x'][1] - grid['x'][0]
			dy = grid['y'][1] - grid['y'][0]
			m = len(grid['x'])
			n = len(grid['y'])
			M = 2 * m
			N = 2 * n
			xg = []

			for i in range(N):
				for j in range(M):
					xg.extend([[j, i]])

			xg = np.array(xg)

			center = []
			center.append([int(dx * M/2-1), int((dy * N)/2-1)])

			out = exp_cov(xg, np.array(center),theta=theta)
			out = np.transpose(np.reshape(out, (N, M)))
			temp = np.zeros((M, N))
			temp[int(M/2-1)][int(N/2-1)] = 1

			wght = np.fft.fft2(out)/(np.fft.fft2(temp) * M * N)
			weight_obj = {"m": m, "n": n, "N": N, "M": M, "wght": wght}

		temp = np.zeros((weight_obj['M'], weight_obj['N']))
		temp[0:m, 0:n] = Y
		temp2 = np.fft.ifft2(np.fft.fft2(temp) *
							 weight_obj['wght']).real[0:weight_obj['m'],
													  0:weight_obj['n']]

		temp = np.zeros((weight_obj['M'], weight_obj['N']))
		temp[0:m, 0:n] = NN
		temp3 = np.fft.ifft2(np.fft.fft2(temp) *
							 weight_obj['wght']).real[0:weight_obj['m'],
													  0:weight_obj['n']]

		return temp2/temp3
	
	## PORTED FROM POPSOM; NOT TESTED FOR COMPATIBILITY ---------------------------------------------
	
	def marginal(self, marginal):
		""" marginal -- plot that shows the marginal probability distribution of the neurons and data

		 	parameters:
		 	- marginal is the name of a training data frame dimension or index
		"""
		
		# check if the second argument is of type character
		if type(marginal) == str and marginal in list(self.data):

			f_ind = list(self.data).index(marginal)
			f_name = marginal
			train = np.array(self.data)[:, f_ind]
			neurons = self.neurons[:, f_ind]
			plt.ylabel('Density')
			plt.xlabel(f_name)
			sns.kdeplot(np.ravel(train),
				        label="training data",
						shade=True,
						color="b")
			sns.kdeplot(neurons, label="neurons", shade=True, color="r")
			plt.legend(fontsize=15)
			plt.show()

		elif (type(marginal) == int and marginal < len(list(self.data)) and marginal >= 0):

			f_ind = marginal
			f_name = list(self.data)[marginal]
			train = np.array(self.data)[:, f_ind]
			neurons = self.neurons[:, f_ind]
			plt.ylabel('Density')
			plt.xlabel(f_name)
			sns.kdeplot(np.ravel(train),
						label="training data",
						shade=True,
						color="b")
			sns.kdeplot(neurons, label="neurons", shade=True, color="r")
			plt.legend(fontsize=15)
			plt.show()

		else:
			sys.exit("marginal: second argument is not the name of a training \
						data frame dimension or index")

	def starburst(self, explicit=False, smoothing=2, merge_clusters=True,  merge_range=.25):
		""" starburst -- compute and display the starburst representation of clusters
			
			parameters:
			- explicit - controls the shape of the connected components
			- smoothing - controls the smoothing level of the umat (NULL,0,>0)
			- merge_clusters - a switch that controls if the starburst clusters are merged together
			- merge_range - a range that is used as a percentage of a certain distance in the code
			                to determine whether components are closer to their centroids or
			                centroids closer to each other.
		"""

		umat = self.compute_umat(smoothing=smoothing)
		self.plot_heat(umat,
						explicit=explicit,
						comp=True,
						merge=merge_clusters,
						merge_range=merge_range)

	def compute_umat(self, smoothing=None):
		""" compute_umat -- compute the unified distance matrix
		
			parameters:
			- smoothing - is either NULL, 0, or a positive floating point value controlling the
			              smoothing of the umat representation
			return:
			- a matrix with the same x-y dims as the original map containing the umat values
		"""

		d = euclidean_distances(self.neurons, self.neurons)
		umat = self.compute_heat(d, smoothing)

		return umat

	def compute_heat(self, d, smoothing=None):
		""" compute_heat -- compute a heat value map representation of the given distance matrix
			
			parameters:
			- d - a distance matrix computed via the 'dist' function
			- smoothing - is either NULL, 0, or a positive floating point value controlling the
			        	  smoothing of the umat representation
			
			return:
			- a matrix with the same x-y dims as the original map containing the heat
		"""

		x = self.xdim
		y = self.ydim
		heat = np.array([[0.0] * y for _ in range(x)])

		if x == 1 or y == 1:
			sys.exit("compute_heat: heat map can not be computed for a map \
	                 with a dimension of 1")

		# this function translates our 2-dim map coordinates
		# into the 1-dim coordinates of the neurons
		def xl(ix, iy):

			return ix + iy * x

		# check if the map is larger than 2 x 2 (otherwise it is only corners)
		if x > 2 and y > 2:
			# iterate over the inner nodes and compute their umat values
			for ix in range(1, x-1):
				for iy in range(1, y-1):
					sum = (d[xl(ix, iy), xl(ix-1, iy-1)] +
						   d[xl(ix, iy), xl(ix, iy-1)] +
	                       d[xl(ix, iy), xl(ix+1, iy-1)] +
	                       d[xl(ix, iy), xl(ix+1, iy)] +
	                       d[xl(ix, iy), xl(ix+1, iy+1)] +
	                       d[xl(ix, iy), xl(ix, iy+1)] +
	                       d[xl(ix, iy), xl(ix-1, iy+1)] +
	                       d[xl(ix, iy), xl(ix-1, iy)])

					heat[ix, iy] = sum/8

			# iterate over bottom x axis
			for ix in range(1, x-1):
				iy = 0
				sum = (d[xl(ix, iy), xl(ix+1, iy)] +
	                   d[xl(ix, iy), xl(ix+1, iy+1)] +
	                   d[xl(ix, iy), xl(ix, iy+1)] +
	                   d[xl(ix, iy), xl(ix-1, iy+1)] +
	                   d[xl(ix, iy), xl(ix-1, iy)])

				heat[ix, iy] = sum/5

			# iterate over top x axis
			for ix in range(1, x-1):
				iy = y-1
				sum = (d[xl(ix, iy), xl(ix-1, iy-1)] +
	                   d[xl(ix, iy), xl(ix, iy-1)] +
	                   d[xl(ix, iy), xl(ix+1, iy-1)] +
	                   d[xl(ix, iy), xl(ix+1, iy)] +
	                   d[xl(ix, iy), xl(ix-1, iy)])

				heat[ix, iy] = sum/5

			# iterate over the left y-axis
			for iy in range(1, y-1):
				ix = 0
				sum = (d[xl(ix, iy), xl(ix, iy-1)] +
	                   d[xl(ix, iy), xl(ix+1, iy-1)] +
	                   d[xl(ix, iy), xl(ix+1, iy)] +
	                   d[xl(ix, iy), xl(ix+1, iy+1)] +
	                   d[xl(ix, iy), xl(ix, iy+1)])

				heat[ix, iy] = sum/5

			# iterate over the right y-axis
			for iy in range(1, y-1):
				ix = x-1
				sum = (d[xl(ix, iy), xl(ix-1, iy-1)] +
	                   d[xl(ix, iy), xl(ix, iy-1)] +
	                   d[xl(ix, iy), xl(ix, iy+1)] +
	                   d[xl(ix, iy), xl(ix-1, iy+1)] +
	                   d[xl(ix, iy), xl(ix-1, iy)])

				heat[ix, iy] = sum/5

		# compute umat values for corners
		if x >= 2 and y >= 2:
			# bottom left corner
			ix = 0
			iy = 0
			sum = (d[xl(ix, iy), xl(ix+1, iy)] +
	               d[xl(ix, iy), xl(ix+1, iy+1)] +
	               d[xl(ix, iy), xl(ix, iy+1)])

			heat[ix, iy] = sum/3

			# bottom right corner
			ix = x-1
			iy = 0
			sum = (d[xl(ix, iy), xl(ix, iy+1)] +
	               d[xl(ix, iy), xl(ix-1, iy+1)] +
	               d[xl(ix, iy), xl(ix-1, iy)])
			heat[ix, iy] = sum/3

			# top left corner
			ix = 0
			iy = y-1
			sum = (d[xl(ix, iy), xl(ix, iy-1)] +
	               d[xl(ix, iy), xl(ix+1, iy-1)] +
	               d[xl(ix, iy), xl(ix+1, iy)])
			heat[ix, iy] = sum/3

			# top right corner
			ix = x-1
			iy = y-1
			sum = (d[xl(ix, iy), xl(ix-1, iy-1)] +
	               d[xl(ix, iy), xl(ix, iy-1)] +
	               d[xl(ix, iy), xl(ix-1, iy)])
			heat[ix, iy] = sum/3

		if smoothing is not None:
			if smoothing == 0:
				heat = self.smooth_2d(heat,
									  nrow=x,
									  ncol=y,
									  surface=False)
			elif smoothing > 0:
				heat = self.smooth_2d(heat,
									  nrow=x,
									  ncol=y,
									  surface=False,
									  theta=smoothing)
			else:
				sys.exit("compute_heat: bad value for smoothing parameter")

		return heat

	def plot_heat(self, heat, explicit=False, comp=True, merge=False, merge_range=0.25):
		""" plot_heat -- plot a heat map based on a 'map', this plot also contains the connected
		                 components of the map based on the landscape of the heat map

			parameters:
			- heat - is a 2D heat map of the map returned by 'map'
			- explicit - controls the shape of the connected components
			- comp - controls whether we plot the connected components on the heat map
			- merge - controls whether we merge the starbursts together.
			- merge_range - a range that is used as a percentage of a certain distance in the code
			                to determine whether components are closer to their centroids or
			                centroids closer to each other.
		"""

		umat = heat

		x = self.xdim
		y = self.ydim
		nobs = self.data_array.shape[0]
		count = np.array([[0]*y]*x)

		# need to make sure the map doesn't have a dimension of 1
		if (x <= 1 or y <= 1):
			sys.exit("plot_heat: map dimensions too small")

		heat_tmp = np.squeeze(np.asarray(heat)).flatten()   	# Convert 2D Array to 1D
		tmp = pd.cut(heat_tmp, bins=100, labels=False)
		tmp = np.reshape(tmp, (-1, y))				# Convert 1D Array to 2D
		
		tmp_1 = np.array(np.transpose(tmp))
		
		fig, ax = plt.subplots()
		ax.pcolor(tmp_1, cmap=plt.cm.YlOrRd)
		
		ax.set_xticks(np.arange(x)+0.5, minor=False)
		ax.set_yticks(np.arange(y)+0.5, minor=False)
		plt.xlabel("x")
		plt.ylabel("y")
		ax.set_xticklabels(np.arange(x), minor=False)
		ax.set_yticklabels(np.arange(y), minor=False)
		ax.xaxis.set_tick_params(labeltop='on')
		ax.yaxis.set_tick_params(labelright='on')

		# put the connected component lines on the map
		if comp:
			if not merge:
				# find the centroid for each neuron on the map
				centroids = self.compute_centroids(heat, explicit)
			else:
				# find the unique centroids for the neurons on the map
				centroids = self.compute_combined_clusters(umat, explicit, merge_range)

			# connect each neuron to its centroid
			for ix in range(x):
				for iy in range(y):
					cx = centroids['centroid_x'][ix, iy]
					cy = centroids['centroid_y'][ix, iy]
					plt.plot([ix+0.5, cx+0.5],
	                         [iy+0.5, cy+0.5],
	                         color='grey',
	                         linestyle='-',
	                         linewidth=1.0)

		# put the labels on the map if available
		if not (self.labels is None) and (len(self.labels) != 0):

			# count the labels in each map cell
			for i in range(nobs):

				nix = self.visual[i]
				c = self.coordinate(np.reshape(nix,(1,1)), self.xdim) # NOTE: slow code
				ix = c[0,0]
				iy = c[0,1]

				count[ix-1, iy-1] = count[ix-1, iy-1]+1

			for i in range(nobs):

				c = self.coordinate(np.reshape(self.visual[i],(1,1)), self.xdim) # NOTE: slow code
				ix = c[0,0]
				iy = c[0,1]

				# we only print one label per cell
				if count[ix-1, iy-1] > 0:

					count[ix-1, iy-1] = 0
					ix = ix - .5
					iy = iy - .5
					l = self.labels[i]
					plt.text(ix+1, iy+1, l)

		plt.show()