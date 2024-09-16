## Self-Organizing Map base code including training and fitting data,
## along with limited cluster plotting capabilities, ported from
## the POPSOM library by Li Yuan (2018) https://github.com/njali2001/popsom.git
## with modifications by Trung Ha (2024) for aweSOM

import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})
from sklearn.metrics.pairwise import euclidean_distances

from numba import njit, prange
from scipy.ndimage import map_coordinates

seed = 42
np.random.seed(seed)


class Lattice:
    def __init__(
        self,
        xdim: int = 10,
        ydim: int = 10,
        alpha_0: float = 0.3,
        train: int = 1000,
        alpha_type: str = "decay",
        sampling_type: str = "sampling",
    ):
        """Initialize the SOM lattice.

        Args:
                xdim (int): The x dimension of the map. Default is 10.
                ydim (int): The y dimension of the map. Default is 10.
                alpha_0 (float): The initial learning rate, should be a positive non-zero real number. Default is 0.3.
                train (int): Number of total training iterations; include for all batches. Default is 1000.
                alpha_type (str): A string that determines whether the learning rate is static or decaying. Default is "decay".
                sampling_type (str): A string that determines whether the initial lattice is uniform or randomly sampled from the data. Default is "sampling".

        """
        self.xdim = xdim
        self.ydim = ydim
        self.alpha = alpha_0
        self.train = train
        self.init = sampling_type  # uniform or random sampling initial lattice
        self.seed = seed
        self.epoch = 0

        if alpha_type == "static":
            self.alpha_type = 0
            self.alpha_0 = alpha_0
        elif alpha_type == "decay":
            self.alpha_type = 1
            self.alpha_0 = alpha_0
        else:
            sys.exit("alpha_type must be either 'static' or 'decay'")

        self.save_frequency = self.train // 200  # how often to save the node weights
        self.lattice_history = []
        self.umat_history = []

    def train_lattice(
        self,
        data: np.ndarray,
        features_names: list[str],
        labels: np.ndarray = None,
        number_of_steps: int = -1,
        save_lattice: bool = False,
        restart_lattice: np.ndarray = None,
    ):
        """Train the Model with numba JIT acceleration.

        Args:
                data (np.ndarray): A numpy 2D array where each row contains an unlabeled training instance.
                features_names (list[str]): A list of feature names.
                labels (np.ndarray, optional): A vector with one label (ground truth) for each observation in data. Defaults to None.
                number_of_steps (int): Number of steps taken this batch, used for keeping track of training restarts. Default is self.train.
                save_lattice (bool, optional): A flag that determines whether the node weights are saved to a file at the end of training. Defaults to False.
                restart_lattice (np.ndarray, optional): Vectors for the weights of the nodes from past realizations. Defaults to None.

        """

        self.restart_lattice = restart_lattice
        self.save_lattice = save_lattice

        self.data_array = data
        self.features_names = features_names
        self.labels = labels
        if number_of_steps == -1:
            self.this_batch_train = self.train
        else:
            self.this_batch_train = number_of_steps
        # self.momentum_decay_rate = momentum_decay_rate

        # check if the dims are reasonable
        if self.xdim < 4 or self.ydim < 4:
            sys.exit("build: map is too small.")

        # train SOM
        self.fast_som()

    def fast_som(self):
        """Performs the self-organizing map (SOM) training.

        This method initializes the SOM with random values or a subset of the data, and then trains the SOM by updating the
        node weights based on the input vectors. The training process includes adjusting the learning rate, shrinking the
        neighborhood size, and saving the node weights and U-matrix periodically.

        """
        # some constants
        number_input_vectors = self.data_array.shape[0]
        number_features = self.data_array.shape[1]
        number_nodes = self.xdim * self.ydim
        this_batch_train = (
            self.this_batch_train
        )  # only train for this number of steps; useful for restarting training

        if self.restart_lattice is not None:
            lattice = self.restart_lattice
        else:
            if self.init == "uniform":
                # vector with small init values for all nodes
                # NOTE: each row represents a node, each column represents a feature.
                lattice = np.random.uniform(0.0, 1.0, (number_nodes, number_features))
            else:
                # sample a random subset of the data to initialize the lattice
                ix = np.random.randint(0, number_input_vectors - 1, number_nodes)
                lattice = self.data_array[ix, :]

            self.lattice = lattice.copy()
            self.lattice_history.append(lattice)  # save the initial lattice
            self.umat_history.append(self.compute_umat())  # save the initial U-matrix

        alpha = self.alpha  # starting learning rate
        if self.alpha_type == 1:
            alpha_freq = (
                self.train // 25
            )  # how often to decay the learning rate; at 24 steps, alpha_f ~ 1e-3 alpha_0
        else:
            alpha_freq = 1

            # compute the initial neighborhood size and step
        nsize_max = max(self.xdim, self.ydim) + 1
        nsize_min = 8
        nsize_step = (
            this_batch_train // 4
        )  # for the third quarter of the training steps, shrink the neighborhood
        nsize_freq = nsize_step // (
            nsize_max - nsize_min
        )  # how often to shrink the neighborhood

        # if self.epoch < 2 * nsize_step:
        # 	nsize = nsize_max
        # elif self.epoch >= 3 * nsize_step:
        # 	nsize = nsize_min
        # else:
        # 	nsize = nsize_max - self.epoch // nsize_freq
        nsize = (
            nsize_max  # start with the largest neighborhood size at each training batch
        )

        epoch = self.epoch  # counts the number of epochs per nsize_freq
        stop_epoch = epoch + this_batch_train
        print("starting epoch is: ", epoch, flush=True)
        print("stopping epoch is: ", stop_epoch, flush=True)

        print("Saving lattice every ", self.save_frequency, " epochs", flush=True)

        # constants for the Gamma function
        m = np.reshape(
            list(range(number_nodes)), (number_nodes, 1)
        )  # a vector with all node 1D addresses

        # x-y coordinate of ith node: m2Ds[i,] = c(xi, yi)
        m2Ds = self.coordinate(m, self.xdim)

        # this ensures that the same random order is not repeated if the training is restarted
        if self.epoch > 1:
            self.seed += 1
            np.random.seed(self.seed)

        # Added 06/17/2024: use random number generator, shuffle the data and take the first train samples
        rng = np.random.default_rng()
        indices = np.arange(number_input_vectors)
        rng.shuffle(indices)

        # Added 06/25/2024: if the number of training steps is larger than the number of data points, repeat the shuffled indices
        if this_batch_train > number_input_vectors:
            indices = np.tile(indices, this_batch_train // number_input_vectors + 1)
        xk = self.data_array[indices[:this_batch_train], :]

        # implement momentum-based gradient descent
        # momentum_decay_rate = self.momentum_decay_rate

        # history of the loss function
        # loss_freq = 1000
        # self.loss_history = np.zeros((self.train, number_features))
        # self.average_loss = np.zeros((self.train//loss_freq, number_features))

        print("Begin training", flush=True)
        while True:
            if epoch % int(self.train // 10) == 0:
                print("Evaluating epoch = ", epoch, flush=True)

            # if (epoch % loss_freq == 0) & (epoch != 0):
            # 	this_average_loss = np.mean(self.loss_history[epoch-loss_freq:epoch], axis=0) # average loss over the last [loss_freq] epochs
            # 	self.average_loss[epoch//loss_freq-1,:] = this_average_loss

            # if training step has gone over the step limit, terminate
            if epoch >= stop_epoch:
                print(
                    "Terminating from step limit reached at epoch ", epoch, flush=True
                )
                self.epoch = epoch
                if self.save_lattice:
                    print("Saving final lattice", epoch, flush=True)
                    np.save(
                        f"lattice_{epoch}_{self.xdim}{self.ydim}_{self.alpha}_{self.train}.npy",
                        lattice,
                    )
                break

                # get one random input vector
            xk_m = xk[epoch - self.epoch, :]

            # calculate the relative distance between input vector and nodes, take the closest node as the BMU
            # momentum = diff * momentum_decay_rate # momentum-based gradient descent
            diff = lattice - xk_m
            squ = diff**2
            s = np.sum(squ, axis=1)
            c = np.argmin(s)

            # self.loss_history[epoch,:] = np.sqrt(s[c])

            # update step
            gamma_m = np.outer(
                self.Gamma(c, m2Ds, alpha, nsize), np.ones(number_features)
            )
            # lattice -= (diff + momentum) * gamma_m
            lattice -= diff * gamma_m

            # shrink the neighborhood size every [nsize_freq] epochs
            if (
                (epoch - self.epoch) > 2 * nsize_step
                and (epoch - self.epoch) % nsize_freq == 0
                and nsize > nsize_min
            ):
                nsize = (
                    nsize_max - ((epoch - self.epoch) - 2 * nsize_step) // nsize_freq
                )
                print(
                    f"Shrinking neighborhood size to {nsize} at epoch {epoch}",
                    flush=True,
                )

            # decay the learning rate every [alpha_freq] epochs
            if epoch % alpha_freq == 0 and self.alpha_type == 1 and epoch != 0:
                alpha *= 0.75
                print(f"Decaying learning rate to {alpha} at epoch {epoch}", flush=True)

            # save lattice sparingly
            if epoch % self.save_frequency == 0:
                self.lattice = lattice.copy()
                # print("Saving lattice at epoch ", epoch, flush=True)
                self.lattice_history.append(self.lattice)

                # compute the umatrix and save it
                umat = self.compute_umat()
                # np.save(f"umat_{epoch}_{self.xdim}{self.ydim}_{self.alpha}_{self.train}.npy", umat)
                self.umat_history.append(umat)

            epoch += 1
        print("Training complete", flush=True)

        # update the learning rate, lattice, and Umatrix after training
        self.alpha = alpha
        self.umat = self.compute_umat()
        self.lattice = lattice  # lattice takes the shape of [X*Y, F]

    @staticmethod
    @njit()
    def Gamma(
        index_bmu: int,
        m2Ds: np.ndarray,
        alpha: float,
        nsize: int,
        gaussian: bool = True,
    ):
        """Calculate the neighborhood function for a given BMU on a lattice.

        Args:
                index_bmu (int): The index of the BMU node on the lattice.
                m2Ds (np.ndarray): Lattice coordinate of each node.
                alpha (float): The amplitude parameter for the Gaussian function, AKA the learning rate.
                nsize (int): The size of the neighborhood.
                gaussian (bool, optional): Whether to use Gaussian function or not. Defaults to True.

        Returns:
                np.ndarray: The neighborhood function values for each node on the grid.
        """

        dist_2d = np.abs(
            m2Ds[index_bmu, :] - m2Ds
        )  # 2d distance between the BMU and the rest of the lattice
        chebyshev_dist = np.zeros(
            dist_2d.shape[0]
        )  # initialize the Chebyshev distance array
        for i in prange(
            dist_2d.shape[0]
        ):  # numba max does not have axis argument, otherwise this would be chebyshev_dist = np.max(dist_2d, axis=1)
            chebyshev_dist[i] = np.max(dist_2d[i, :])

        # Define the Gaussian function to calculate the neighborhood function
        def gauss(dist, A, sigma):
            """gauss -- Gaussian function"""
            return A * np.exp(-((dist) ** 2) / (2 * sigma**2))

        # if a node on the lattice is in within nsize neighborhood, then h = Gaussian(alpha), else h = 0.0
        if gaussian:  # use Gaussian function
            h = gauss(chebyshev_dist, alpha, nsize / 3)
        else:  # otherwise everything within nsize is multiplied by alpha, and everything outside is unchanged
            h = np.where(chebyshev_dist <= nsize, alpha, 0.0)

        h[chebyshev_dist > nsize] = (
            0.0  # manually set the values outside the neighborhood to 0
        )

        return h

    def map_data_to_lattice(self):
        """
        After training, map each data point to the nearest node in the lattice.

        Returns:
                np.ndarray[int]: A 2D array with the x and y coordinates of the best matching nodes for each data point.
        """

        print("Begin matching points with nodes", flush=True)
        data_to_lattice_1d = self.best_match(self.lattice, self.data_array)

        self.projection_1d = data_to_lattice_1d

        projection_1d_to_2d = np.reshape(
            self.projection_1d, (len(self.projection_1d), 1)
        )  # make it a 2D array

        projection_2d = self.coordinate(projection_1d_to_2d, self.xdim)
        self.projection_2d = projection_2d

        return projection_2d

    def assign_cluster_to_lattice(self, smoothing=None, merge_cost=0.005):
        """
        Assigns clusters to the lattice based on the computed centroids.

        Args:
                smoothing (float, optional): Smoothing parameter for computing Umatrix. Defaults to None.
                merge_cost (float, optional): Cost threshold for merging similar centroids. Defaults to 0.005.

        Returns:
                numpy.ndarray: Array representing the assigned clusters for each lattice point.
        """

        if smoothing is not None:  # smooth the Umatrix before computing the centroids
            self.umat = self.compute_umat(smoothing)
        naive_centroids = self.compute_centroids(
            False
        )  # all local minima are centroids
        centroids = self.merge_similar_centroids(
            naive_centroids, merge_cost
        )  # merge similar centroids

        x = self.xdim
        y = self.ydim
        centr_locs = []

        # create list of centroid locations
        for ix in range(x):
            for iy in range(y):
                cx = centroids["centroid_x"][ix, iy]
                cy = centroids["centroid_y"][ix, iy]

                centr_locs.append((cx, cy))

        unique_ids = list(set(centr_locs))
        n_clusters = len(unique_ids)
        print(f"Number of clusters : {n_clusters}", flush=True)
        print("Centroids: ", unique_ids, flush=True)

        # mapping = {}
        clusters = 1000 * np.ones((x, y), dtype=np.int32)
        for i in range(n_clusters):
            # mapping[i] = unique_ids[i]
            for ix in range(x):
                for iy in range(y):
                    if (
                        centroids["centroid_x"][ix, iy],
                        centroids["centroid_y"][ix, iy],
                    ) == unique_ids[i]:
                        clusters[ix, iy] = i

        self.lattice_assigned_clusters = clusters
        return clusters

    @staticmethod
    @njit(parallel=True)
    def assign_cluster_to_data(
        projection_2d: np.ndarray, clusters_on_lattice: np.ndarray
    ) -> np.ndarray:
        """
        Given a lattice and cluster assignments on that lattice, return the cluster ids of the data (in a 1d array)

        Args:
                projection_2d (np.ndarray): 2d array with x-y coordinates of the node associated with each data point
                clusters_on_lattice (np.ndarray): X x Y matrix of cluster labels on lattice

        Returns:
                np.ndarray: cluster_id of each data point
        """
        cluster_id = np.zeros(projection_2d.shape[0], dtype=np.int32)
        for i in prange(projection_2d.shape[0]):
            cluster_id[i] = clusters_on_lattice[
                int(projection_2d[i, 0]), int(projection_2d[i, 1])
            ]
        return cluster_id

    @staticmethod
    @njit(parallel=True)
    def best_match(lattice: np.ndarray, obs: np.ndarray, full=False) -> np.ndarray:
        """
        Given input vector inp[n,f] (where n is number of different observations, f is number of features per observation), return the best matching node.

        Args:
                lattice (np.ndarray): weight values of the lattice
                obs (np.ndarray): observations (input vectors)
                full (bool, optional): indicate whether to return first and second best match. Defaults to False.

        Returns:
                np.ndarray: return the 1d index of the best-matched node (within the lattice) for each observation
        """

        if full:
            best_match_node = np.zeros((obs.shape[0], 2))
        else:
            best_match_node = np.zeros((obs.shape[0], 1))

        for i in prange(obs.shape[0]):
            diff = lattice - obs[i]
            squ = diff**2
            s = np.sum(squ, axis=1)

            if full:
                # numba does not support argsort, so we record the top two best matches this way
                o = np.argmin(s)
                best_match_node[i, 0] = o
                s[o] = np.max(s)
                o = np.argmin(s)
                best_match_node[i, 1] = o
            else:
                best_match_node[i] = np.argmin(s)

            if i % int(obs.shape[0] // 10) == 0:
                print("i = ", i)

        return best_match_node

    @staticmethod
    @njit(parallel=True)
    def coordinate(rowix: np.ndarray, xdim: int) -> np.ndarray:
        """
        Convert from a list of row index to an array of xy-coordinates.

        Args:
                rowix (np.ndarray): 1d array with the 1d indices of the points of interest (n x 1 matrix)
                xdim (int): x dimension of the lattice

        Returns:
                np.ndarray: array with x and y coordinates of each point in rowix
        """

        len_rowix = len(rowix)
        coords = np.zeros((len_rowix, 2), dtype=np.int32)

        for k in prange(len_rowix):
            coords[k, :] = np.array([rowix[k, 0] % xdim, rowix[k, 0] // xdim])

        return coords

    def rowix(self, x, y):
        """
        Convert from a xy-coordinate to a row index.

        Args:
                x (int): The x-coordinate of the map.
                y (int): The y-coordinate of the map.

        Returns:
                int: The row index corresponding to the given xy-coordinate.

        """
        rix = x + y * self.xdim
        return rix

    def node_weight(self, x, y):
        """
        Returns the weight values of a node at (x,y) on the lattice.

        Args:
                x (int): x-coordinate of the node.
                y (int): y-coordinate of the node.

        Returns:
                np.ndarray: 1d array of weight values of said node.
        """

        ix = self.rowix(x, y)
        return self.lattice[ix]

    def compute_centroids(self, explicit=False):
        """
        Compute the centroid for each node in the lattice given a precomputed Umatrix.

        Args:
                explicit (bool): Controls the shape of the connected component.

        Returns:
                dict: A dictionary containing the matrices with the same x-y dimensions as the original map,
                containing the centroid x-y coordinates.

        """

        xdim = self.xdim
        ydim = self.ydim
        heat = self.umat
        centroid_x = np.array([[-1] * ydim for _ in range(xdim)])
        centroid_y = np.array([[-1] * ydim for _ in range(xdim)])

        def find_this_centroid(ix, iy):
            # recursive function to find the centroid of a point on the map

            if (centroid_x[ix, iy] > -1) and (centroid_y[ix, iy] > -1):
                return {"bestx": centroid_x[ix, iy], "besty": centroid_y[ix, iy]}

            min_val = heat[ix, iy]
            min_x = ix
            min_y = iy

            # (ix, iy) is an inner map element
            if ix > 0 and ix < xdim - 1 and iy > 0 and iy < ydim - 1:

                if heat[ix - 1, iy - 1] < min_val:
                    min_val = heat[ix - 1, iy - 1]
                    min_x = ix - 1
                    min_y = iy - 1

                if heat[ix, iy - 1] < min_val:
                    min_val = heat[ix, iy - 1]
                    min_x = ix
                    min_y = iy - 1

                if heat[ix + 1, iy - 1] < min_val:
                    min_val = heat[ix + 1, iy - 1]
                    min_x = ix + 1
                    min_y = iy - 1

                if heat[ix + 1, iy] < min_val:
                    min_val = heat[ix + 1, iy]
                    min_x = ix + 1
                    min_y = iy

                if heat[ix + 1, iy + 1] < min_val:
                    min_val = heat[ix + 1, iy + 1]
                    min_x = ix + 1
                    min_y = iy + 1

                if heat[ix, iy + 1] < min_val:
                    min_val = heat[ix, iy + 1]
                    min_x = ix
                    min_y = iy + 1

                if heat[ix - 1, iy + 1] < min_val:
                    min_val = heat[ix - 1, iy + 1]
                    min_x = ix - 1
                    min_y = iy + 1

                if heat[ix - 1, iy] < min_val:
                    min_val = heat[ix - 1, iy]
                    min_x = ix - 1
                    min_y = iy

            # (ix, iy) is bottom left corner
            elif ix == 0 and iy == 0:

                if heat[ix + 1, iy] < min_val:
                    min_val = heat[ix + 1, iy]
                    min_x = ix + 1
                    min_y = iy

                if heat[ix + 1, iy + 1] < min_val:
                    min_val = heat[ix + 1, iy + 1]
                    min_x = ix + 1
                    min_y = iy + 1

                if heat[ix, iy + 1] < min_val:
                    min_val = heat[ix, iy + 1]
                    min_x = ix
                    min_y = iy + 1

            # (ix, iy) is bottom right corner
            elif ix == xdim - 1 and iy == 0:

                if heat[ix, iy + 1] < min_val:
                    min_val = heat[ix, iy + 1]
                    min_x = ix
                    min_y = iy + 1

                if heat[ix - 1, iy + 1] < min_val:
                    min_val = heat[ix - 1, iy + 1]
                    min_x = ix - 1
                    min_y = iy + 1

                if heat[ix - 1, iy] < min_val:
                    min_val = heat[ix - 1, iy]
                    min_x = ix - 1
                    min_y = iy

            # (ix, iy) is top right corner
            elif ix == xdim - 1 and iy == ydim - 1:

                if heat[ix - 1, iy - 1] < min_val:
                    min_val = heat[ix - 1, iy - 1]
                    min_x = ix - 1
                    min_y = iy - 1

                if heat[ix, iy - 1] < min_val:
                    min_val = heat[ix, iy - 1]
                    min_x = ix
                    min_y = iy - 1

                if heat[ix - 1, iy] < min_val:
                    min_val = heat[ix - 1, iy]
                    min_x = ix - 1
                    min_y = iy

            # (ix, iy) is top left corner
            elif ix == 0 and iy == ydim - 1:

                if heat[ix, iy - 1] < min_val:
                    min_val = heat[ix, iy - 1]
                    min_x = ix
                    min_y = iy - 1

                if heat[ix + 1, iy - 1] < min_val:
                    min_val = heat[ix + 1, iy - 1]
                    min_x = ix + 1
                    min_y = iy - 1

                if heat[ix + 1, iy] < min_val:
                    min_val = heat[ix + 1, iy]
                    min_x = ix + 1
                    min_y = iy

            # (ix, iy) is a left side element
            elif ix == 0 and iy > 0 and iy < ydim - 1:

                if heat[ix, iy - 1] < min_val:
                    min_val = heat[ix, iy - 1]
                    min_x = ix
                    min_y = iy - 1

                if heat[ix + 1, iy - 1] < min_val:
                    min_val = heat[ix + 1, iy - 1]
                    min_x = ix + 1
                    min_y = iy - 1

                if heat[ix + 1, iy] < min_val:
                    min_val = heat[ix + 1, iy]
                    min_x = ix + 1
                    min_y = iy

                if heat[ix + 1, iy + 1] < min_val:
                    min_val = heat[ix + 1, iy + 1]
                    min_x = ix + 1
                    min_y = iy + 1

                if heat[ix, iy + 1] < min_val:
                    min_val = heat[ix, iy + 1]
                    min_x = ix
                    min_y = iy + 1

            # (ix, iy) is a bottom side element
            elif ix > 0 and ix < xdim - 1 and iy == 0:

                if heat[ix + 1, iy] < min_val:
                    min_val = heat[ix + 1, iy]
                    min_x = ix + 1
                    min_y = iy

                if heat[ix + 1, iy + 1] < min_val:
                    min_val = heat[ix + 1, iy + 1]
                    min_x = ix + 1
                    min_y = iy + 1

                if heat[ix, iy + 1] < min_val:
                    min_val = heat[ix, iy + 1]
                    min_x = ix
                    min_y = iy + 1

                if heat[ix - 1, iy + 1] < min_val:
                    min_val = heat[ix - 1, iy + 1]
                    min_x = ix - 1
                    min_y = iy + 1

                if heat[ix - 1, iy] < min_val:
                    min_val = heat[ix - 1, iy]
                    min_x = ix - 1
                    min_y = iy

            # (ix, iy) is a right side element
            elif ix == xdim - 1 and iy > 0 and iy < ydim - 1:

                if heat[ix - 1, iy - 1] < min_val:
                    min_val = heat[ix - 1, iy - 1]
                    min_x = ix - 1
                    min_y = iy - 1

                if heat[ix, iy - 1] < min_val:
                    min_val = heat[ix, iy - 1]
                    min_x = ix
                    min_y = iy - 1

                if heat[ix, iy + 1] < min_val:
                    min_val = heat[ix, iy + 1]
                    min_x = ix
                    min_y = iy + 1

                if heat[ix - 1, iy + 1] < min_val:
                    min_val = heat[ix - 1, iy + 1]
                    min_x = ix - 1
                    min_y = iy + 1

                if heat[ix - 1, iy] < min_val:
                    min_val = heat[ix - 1, iy]
                    min_x = ix - 1
                    min_y = iy

            # (ix, iy) is a top side element
            elif ix > 0 and ix < xdim - 1 and iy == ydim - 1:

                if heat[ix - 1, iy - 1] < min_val:
                    min_val = heat[ix - 1, iy - 1]
                    min_x = ix - 1
                    min_y = iy - 1

                if heat[ix, iy - 1] < min_val:
                    min_val = heat[ix, iy - 1]
                    min_x = ix
                    min_y = iy - 1

                if heat[ix + 1, iy - 1] < min_val:
                    min_val = heat[ix + 1, iy - 1]
                    min_x = ix + 1
                    min_y = iy - 1

                if heat[ix + 1, iy] < min_val:
                    min_val = heat[ix + 1, iy]
                    min_x = ix + 1
                    min_y = iy

                if heat[ix - 1, iy] < min_val:
                    min_val = heat[ix - 1, iy]
                    min_x = ix - 1
                    min_y = iy

                    # if successful
                    # move to the square with the smaller value, i_e_, call
                    # find_this_centroid on this new square
                    # note the RETURNED x-y coords in the centroid_x and
                    # centroid_y matrix at the current location
                    # return the RETURNED x-y coordinates

            if min_x != ix or min_y != iy:
                r_val = find_this_centroid(min_x, min_y)

                # if explicit is set show the exact connected component
                # otherwise construct a connected componenent where all
                # nodes are connected to a centrol node
                if explicit:

                    centroid_x[ix, iy] = min_x
                    centroid_y[ix, iy] = min_y
                    return {"bestx": min_x, "besty": min_y}

                else:
                    centroid_x[ix, iy] = r_val["bestx"]
                    centroid_y[ix, iy] = r_val["besty"]
                    return r_val

            else:
                centroid_x[ix, iy] = ix
                centroid_y[ix, iy] = iy
                return {"bestx": ix, "besty": iy}

        for i in range(xdim):
            for j in range(ydim):
                find_this_centroid(i, j)

        return {"centroid_x": centroid_x, "centroid_y": centroid_y}

    @staticmethod
    # @njit(parallel=True) # numba does not support dictionary; so cannot parallelize this function
    def replace_value(
        centroids: dict[str, np.ndarray], centroid_a: tuple, centroid_b: tuple
    ) -> dict[str, np.ndarray]:
        """
        Replaces the values of centroid_a with the values of centroid_b in the given centroids dictionary.

        Args:
                centroids (dict[str, np.ndarray]): A dictionary containing the centroids.
                centroid_a (tuple): The coordinates of the centroid to be replaced.
                centroid_b (tuple): The coordinates of the centroid to replace with.

        Returns:
                dict[str, np.ndarray]: The updated centroids dictionary.
        """
        (xdim, ydim) = centroids["centroid_x"].shape
        for ix in range(xdim):
            for iy in range(ydim):
                if (
                    centroids["centroid_x"][ix, iy] == centroid_a[0]
                    and centroids["centroid_y"][ix, iy] == centroid_a[1]
                ):
                    centroids["centroid_x"][ix, iy] = centroid_b[0]
                    centroids["centroid_y"][ix, iy] = centroid_b[1]

        return centroids

    def merge_similar_centroids(self, naive_centroids: np.ndarray, threshold=0.3):
        """
        Merge centroids that are close enough together.

        Args:
                naive_centroids (np.ndarray): original centroids before merging
                threshold (float, optional): Any centroids with pairwise cost less than this threshold is merged. Defaults to 0.3.

        Returns:
                np.ndarray: new node map with combined centroids
        """

        heat = self.umat
        centroids = naive_centroids.copy()
        unique_centroids = self.get_unique_centroids(
            centroids
        )  # the nodes_count dictionary is also created here, so don't remove this line

        # for each pair of centroids, compute the weighted distance between them via interpolating the umat
        # if the distance is less than the threshold, combine the centroids

        cost_between_centroids = []

        for i in range(len(unique_centroids["position_x"]) - 1):
            for j in range(i + 1, len(unique_centroids["position_x"])):
                a = [
                    unique_centroids["position_x"][i],
                    unique_centroids["position_y"][i],
                ]
                b = [
                    unique_centroids["position_x"][j],
                    unique_centroids["position_y"][j],
                ]
                num_sample = 5 * int(np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2))
                x, y = np.linspace(a[0], b[0], num_sample), np.linspace(
                    a[1], b[1], num_sample
                )
                umat_dist = map_coordinates(heat**2, np.vstack((x, y)))
                total_cost = np.sum(umat_dist)
                # total_cost /= num_sample
                cost_between_centroids.append([a, b, total_cost])

        # cost_between_centroids.sort(key=lambda x: x[2])
        sorted_cost = sorted(
            cost_between_centroids, key=lambda x: x[2]
        )  # this ranks all pairwise cost in ascending order
        # normalize the cost such that the largest cost at each step is always one
        sorted_cost = [[a, b, c / sorted_cost[-1][2]] for a, b, c in sorted_cost]

        # combine the centroids recursively until the threshold is reached
        if sorted_cost[0][2] < threshold:
            centroid_a = tuple(sorted_cost[0][0])
            centroid_b = tuple(sorted_cost[0][1])
            # heat_a = heat[centroid_a[0], centroid_a[1]]
            # heat_b = heat[centroid_b[0], centroid_b[1]]
            nodes_a = self.nodes_count[centroid_a]
            nodes_b = self.nodes_count[centroid_b]

            print(f"Centroid A: {centroid_a}, count: {nodes_a}", flush=True)
            print(f"Centroid B: {centroid_b}, count: {nodes_b}", flush=True)
            # print(f"Centroid A: {centroid_a}, Umatrix value: {heat_a}", flush=True)
            # print(f"Centroid B: {centroid_b}, Umatrix value: {heat_b}", flush=True)
            print("Merging...", flush=True)

            replace_a_with_b = False

            # this method takes the centroid with the larger number of nodes
            if nodes_a < nodes_b:
                replace_a_with_b = True

            # this method takes the centroid with the smaller U-matrix value
            # if heat_a > heat_b:
            #     replace_a_with_b = True

            if replace_a_with_b:
                centroids = self.replace_value(centroids, centroid_a, centroid_b)
            else:
                centroids = self.replace_value(centroids, centroid_b, centroid_a)

            # print("New centroids: \n", centroids, flush=True)
            centroids = self.merge_similar_centroids(centroids, threshold)
        else:
            unique_centroids = self.get_unique_centroids(centroids)
            # print("Centroids: \n", unique_centroids, flush=True)
            print(
                "Number of unique centroids: ",
                len(unique_centroids["position_x"]),
                flush=True,
            )
            print("Minimum cost between centroids: ", sorted_cost[0][2], flush=True)
            return centroids

        return centroids

    def get_unique_centroids(self, centroids):
        """
        Print out a list of unique centroids given a matrix of centroid locations.

        Args:
                centroids: A matrix of the centroid locations in the map.

        Returns:
                A dictionary containing the unique x and y positions of the centroids.
                The dictionary has the following keys:
                position_x: A list of unique x positions.
                position_y: A list of unique y positions.
        """

        # get the dimensions of the map
        xdim = self.xdim
        ydim = self.ydim
        xlist = []
        ylist = []
        centr_locs = []

        # create a list of unique centroid positions
        for ix in range(xdim):
            for iy in range(ydim):
                cx = centroids["centroid_x"][ix, iy]
                cy = centroids["centroid_y"][ix, iy]

                centr_locs.append((cx, cy))

        self.nodes_count = {i: centr_locs.count(i) for i in centr_locs}

        unique_ids = list(set(centr_locs))
        xlist = [x for x, y in unique_ids]
        ylist = [y for x, y in unique_ids]

        return {"position_x": xlist, "position_y": ylist}

    def compute_umat(self, smoothing=None):
        """
        Compute the unified distance matrix.

        Args:
                smoothing (float, optional): A positive floating point value controlling the smoothing of the umat representation. Defaults to None.

        Returns:
                numpy.ndarray: A matrix with the same x-y dimensions as the original map containing the umat values.
        """

        d = euclidean_distances(self.lattice, self.lattice) / (self.xdim * self.ydim)
        umat = self.compute_heat(d, smoothing)

        return umat

    def compute_heat(self, d, smoothing=None):
        """
        Compute a heat value map representation of the given distance matrix.

        Args:
                d (numpy.ndarray): A distance matrix computed via the 'dist' function.
                smoothing (float, optional): A positive floating point value controlling the smoothing of the umat representation. Defaults to None.

        Returns:
                numpy.ndarray: A matrix with the same x-y dimensions as the original map containing the heat.
        """

        x = self.xdim
        y = self.ydim
        heat = np.array([[0.0] * y for _ in range(x)])

        if x == 1 or y == 1:
            sys.exit(
                "compute_heat: heat map can not be computed for a map \
	                 with a dimension of 1"
            )

        # this function translates our 2-dim lattice coordinates
        # into the 1-dim coordinates of the lattice
        def xl(ix, iy):

            return ix + iy * x

        # check if the map is larger than 2 x 2 (otherwise it is only corners)
        if x > 2 and y > 2:
            # iterate over the inner nodes and compute their umat values
            for ix in range(1, x - 1):
                for iy in range(1, y - 1):
                    sum = (
                        d[xl(ix, iy), xl(ix - 1, iy - 1)]
                        + d[xl(ix, iy), xl(ix, iy - 1)]
                        + d[xl(ix, iy), xl(ix + 1, iy - 1)]
                        + d[xl(ix, iy), xl(ix + 1, iy)]
                        + d[xl(ix, iy), xl(ix + 1, iy + 1)]
                        + d[xl(ix, iy), xl(ix, iy + 1)]
                        + d[xl(ix, iy), xl(ix - 1, iy + 1)]
                        + d[xl(ix, iy), xl(ix - 1, iy)]
                    )

                    heat[ix, iy] = sum / 8

            # iterate over bottom x axis
            for ix in range(1, x - 1):
                iy = 0
                sum = (
                    d[xl(ix, iy), xl(ix + 1, iy)]
                    + d[xl(ix, iy), xl(ix + 1, iy + 1)]
                    + d[xl(ix, iy), xl(ix, iy + 1)]
                    + d[xl(ix, iy), xl(ix - 1, iy + 1)]
                    + d[xl(ix, iy), xl(ix - 1, iy)]
                )

                heat[ix, iy] = sum / 5

            # iterate over top x axis
            for ix in range(1, x - 1):
                iy = y - 1
                sum = (
                    d[xl(ix, iy), xl(ix - 1, iy - 1)]
                    + d[xl(ix, iy), xl(ix, iy - 1)]
                    + d[xl(ix, iy), xl(ix + 1, iy - 1)]
                    + d[xl(ix, iy), xl(ix + 1, iy)]
                    + d[xl(ix, iy), xl(ix - 1, iy)]
                )

                heat[ix, iy] = sum / 5

            # iterate over the left y-axis
            for iy in range(1, y - 1):
                ix = 0
                sum = (
                    d[xl(ix, iy), xl(ix, iy - 1)]
                    + d[xl(ix, iy), xl(ix + 1, iy - 1)]
                    + d[xl(ix, iy), xl(ix + 1, iy)]
                    + d[xl(ix, iy), xl(ix + 1, iy + 1)]
                    + d[xl(ix, iy), xl(ix, iy + 1)]
                )

                heat[ix, iy] = sum / 5

            # iterate over the right y-axis
            for iy in range(1, y - 1):
                ix = x - 1
                sum = (
                    d[xl(ix, iy), xl(ix - 1, iy - 1)]
                    + d[xl(ix, iy), xl(ix, iy - 1)]
                    + d[xl(ix, iy), xl(ix, iy + 1)]
                    + d[xl(ix, iy), xl(ix - 1, iy + 1)]
                    + d[xl(ix, iy), xl(ix - 1, iy)]
                )

                heat[ix, iy] = sum / 5

        # compute umat values for corners
        if x >= 2 and y >= 2:
            # bottom left corner
            ix = 0
            iy = 0
            sum = (
                d[xl(ix, iy), xl(ix + 1, iy)]
                + d[xl(ix, iy), xl(ix + 1, iy + 1)]
                + d[xl(ix, iy), xl(ix, iy + 1)]
            )

            heat[ix, iy] = sum / 3

            # bottom right corner
            ix = x - 1
            iy = 0
            sum = (
                d[xl(ix, iy), xl(ix, iy + 1)]
                + d[xl(ix, iy), xl(ix - 1, iy + 1)]
                + d[xl(ix, iy), xl(ix - 1, iy)]
            )
            heat[ix, iy] = sum / 3

            # top left corner
            ix = 0
            iy = y - 1
            sum = (
                d[xl(ix, iy), xl(ix, iy - 1)]
                + d[xl(ix, iy), xl(ix + 1, iy - 1)]
                + d[xl(ix, iy), xl(ix + 1, iy)]
            )
            heat[ix, iy] = sum / 3

            # top right corner
            ix = x - 1
            iy = y - 1
            sum = (
                d[xl(ix, iy), xl(ix - 1, iy - 1)]
                + d[xl(ix, iy), xl(ix, iy - 1)]
                + d[xl(ix, iy), xl(ix - 1, iy)]
            )
            heat[ix, iy] = sum / 3

        if smoothing is not None:
            if smoothing == 0:
                heat = self.smooth_2d(heat, nrow=x, ncol=y, surface=False)
            elif smoothing > 0:
                heat = self.smooth_2d(
                    heat, nrow=x, ncol=y, surface=False, theta=smoothing
                )
            else:
                sys.exit("compute_heat: bad value for smoothing parameter")

        return heat

    def list_clusters(self, centroids, unique_centroids):
        """Get the clusters as a list of lists., not very useful

        Args:
                centroids (matrix): A matrix of the centroid locations in the map.
                unique_centroids (list): A list of unique centroid locations.

        Returns:
                list: A list of clusters associated with each unique centroid.
        """

        centroids_x_positions = unique_centroids["position_x"]
        centroids_y_positions = unique_centroids["position_y"]
        cluster_list = []

        for i in range(len(centroids_x_positions)):
            cx = centroids_x_positions[i]
            cy = centroids_y_positions[i]

            # get the clusters associated with a unique centroid and store it in a list
            cluster_list.append(self.list_from_centroid(cx, cy, centroids))

        return cluster_list

    def list_from_centroid(self, x, y, centroids):
        """Get all cluster elements associated with one centroid.

        Args:
                x (int): The x position of a centroid.
                y (int): The y position of a centroid.
                centroids (numpy.ndarray): A matrix of the centroid locations in the map.

        Returns:
                list: A list of cluster elements associated with the given centroid.
        """

        centroid_x = x
        centroid_y = y
        xdim = self.xdim
        ydim = self.ydim

        cluster_list = []
        for xi in range(xdim):
            for yi in range(ydim):
                cx = centroids["centroid_x"][xi, yi]
                cy = centroids["centroid_y"][xi, yi]

                if (cx == centroid_x) and (cy == centroid_y):
                    cweight = self.umat[xi, yi]
                    cluster_list.append(cweight)

        return cluster_list

    def smooth_2d(
        self,
        Y,
        ind=None,
        weight_obj=None,
        grid=None,
        nrow=64,
        ncol=64,
        surface=True,
        theta=None,
    ):
        """
        Smooths 2D data using a kernel smoother., internal function, no user-facing aspect

        Args:
                Y (array-like): The input data to be smoothed.
                ind (array-like, optional): The indices of the data to be smoothed. Defaults to None.
                weight_obj (dict, optional): The weight object used for smoothing. Defaults to None.
                grid (dict, optional): The grid object used for smoothing. Defaults to None.
                nrow (int, optional): The number of rows in the grid. Defaults to 64.
                ncol (int, optional): The number of columns in the grid. Defaults to 64.
                surface (bool, optional): Flag indicating whether the data represents a surface. Defaults to True.
                theta (float, optional): The theta value used in the exponential covariance function. Defaults to None.

        Returns:
                array-like: The smoothed data.

        Raises:
                None

        """

        def exp_cov(x1, x2, theta=2, p=2, distMat=0):
            x1 = x1 * (1 / theta)
            x2 = x2 * (1 / theta)
            distMat = euclidean_distances(x1, x2)
            distMat = distMat**p
            return np.exp(-distMat)

        NN = [[1] * ncol] * nrow
        grid = {"x": [i for i in range(nrow)], "y": [i for i in range(ncol)]}

        if weight_obj is None:
            dx = grid["x"][1] - grid["x"][0]
            dy = grid["y"][1] - grid["y"][0]
            m = len(grid["x"])
            n = len(grid["y"])
            M = 2 * m
            N = 2 * n
            xg = []

            for i in range(N):
                for j in range(M):
                    xg.extend([[j, i]])

            xg = np.array(xg)

            center = []
            center.append([int(dx * M / 2 - 1), int((dy * N) / 2 - 1)])

            out = exp_cov(xg, np.array(center), theta=theta)
            out = np.transpose(np.reshape(out, (N, M)))
            temp = np.zeros((M, N))
            temp[int(M / 2 - 1)][int(N / 2 - 1)] = 1

            wght = np.fft.fft2(out) / (np.fft.fft2(temp) * M * N)
            weight_obj = {"m": m, "n": n, "N": N, "M": M, "wght": wght}

        temp = np.zeros((weight_obj["M"], weight_obj["N"]))
        temp[0:m, 0:n] = Y
        temp2 = np.fft.ifft2(np.fft.fft2(temp) * weight_obj["wght"]).real[
            0 : weight_obj["m"], 0 : weight_obj["n"]
        ]

        temp = np.zeros((weight_obj["M"], weight_obj["N"]))
        temp[0:m, 0:n] = NN
        temp3 = np.fft.ifft2(np.fft.fft2(temp) * weight_obj["wght"]).real[
            0 : weight_obj["m"], 0 : weight_obj["n"]
        ]

        return temp2 / temp3

    def plot_heat(self, heat, explicit=False, comp=True, merge=False, merge_cost=0.001):
        """
        Plot the heat map of the given data.

        Args:
                heat (array-like): The data to be plotted.
                explicit (bool, optional): A flag indicating whether the connected components are explicit. Defaults to False.
                comp (bool, optional): A flag indicating whether to plot the connected components. Defaults to True.
                merge (bool, optional): A flag indicating whether to merge the connected components. Defaults to False.
                merge_cost (float, optional): The threshold for merging the connected components. Defaults to 0.001.
        """

        x = self.xdim
        y = self.ydim
        nobs = self.data_array.shape[0]
        count = np.array([[0] * y] * x)

        # need to make sure the map doesn't have a dimension of 1
        if x <= 1 or y <= 1:
            sys.exit("plot_heat: map dimensions too small")

        heat_tmp = np.squeeze(heat).flatten()  # Convert 2D Array to 1D
        tmp = np.digitize(
            heat_tmp, np.linspace(heat_tmp.min(), heat_tmp.max(), num=100)
        )
        tmp = np.reshape(tmp, (-1, y))  # Convert 1D Array to 2D

        tmp_1 = np.array(np.transpose(tmp))

        fig, ax = plt.subplots(dpi=200)
        plt.rcParams["font.size"] = 8
        ax.pcolor(tmp_1, cmap=plt.cm.YlOrRd)

        ax.set_xticks(np.arange(0, x, 5) + 0.5, minor=False)
        ax.set_yticks(np.arange(0, y, 5) + 0.5, minor=False)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_xticklabels(np.arange(0, x, 5), minor=False)
        ax.set_yticklabels(np.arange(0, y, 5), minor=False)
        ax.xaxis.set_tick_params(labeltop="on")
        ax.yaxis.set_tick_params(labelright="on")
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        ax.set_aspect("equal")
        ax.grid(True)

        # put the connected component lines on the map
        if comp:

            # find the centroid for each node on the map
            centroids = self.compute_centroids(explicit)
            if merge:
                # find the unique centroids for the nodes on the map
                centroids = self.merge_similar_centroids(centroids, merge_cost)

            unique_centroids = self.get_unique_centroids(centroids)
            print("Unique centroids : ", unique_centroids)

            unique_centroids["position_x"] = [
                x + 0.5 for x in unique_centroids["position_x"]
            ]
            unique_centroids["position_y"] = [
                y + 0.5 for y in unique_centroids["position_y"]
            ]

            plt.scatter(
                unique_centroids["position_x"],
                unique_centroids["position_y"],
                color="red",
                s=10,
            )

            # connect each node to its centroid
            for ix in range(x):
                for iy in range(y):
                    cx = centroids["centroid_x"][ix, iy]
                    cy = centroids["centroid_y"][ix, iy]
                    plt.plot(
                        [ix + 0.5, cx + 0.5],
                        [iy + 0.5, cy + 0.5],
                        color="grey",
                        linestyle="-",
                        linewidth=1.0,
                    )

        # put the labels on the map if available
        if not (self.labels is None) and (len(self.labels) != 0):
            self.map_data_to_lattice()  # obtain the projection_1d array

            # count the labels in each map cell
            for i in range(nobs):

                nix = self.projection_1d[i]
                c = self.coordinate(
                    np.reshape(nix, (1, 1)), self.xdim
                )  # NOTE: slow code
                # print(c)
                ix = int(c[0, 0])
                iy = int(c[0, 1])

                count[ix - 1, iy - 1] = count[ix - 1, iy - 1] + 1

            for i in range(nobs):

                c = self.coordinate(
                    np.reshape(self.projection_1d[i], (1, 1)), self.xdim
                )  # NOTE: slow code
                ix = int(c[0, 0])
                iy = int(c[0, 1])

                # we only print one label per cell
                if count[ix - 1, iy - 1] > 0:

                    count[ix - 1, iy - 1] = 0
                    ix = ix - 0.5
                    iy = iy - 0.5
                    l = self.labels[i]
                    plt.text(ix + 1, iy + 1, l)

        plt.show()
