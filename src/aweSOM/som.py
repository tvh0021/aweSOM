## Self-Organizing Map base code including training and fitting data,
## along with limited cluster plotting capabilities, ported from
## the POPSOM library by Li Yuan (2018) https://github.com/njali2001/popsom.git
## with modifications by Trung Ha (2024) for aweSOM

import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})
from sklearn.metrics.pairwise import euclidean_distances

from numba import njit, prange
from scipy.ndimage import map_coordinates

from .inference import DEFAULT_CHUNK, best_match_nodes
from .inference import predict as _predict_nodes

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

        if self.train > 200:
            self.save_frequency = (
                self.train // 200
            )  # how often to save the node weights
        else:
            self.save_frequency = 5
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
        history: bool = True,
    ):
        """Train the Model with numba JIT acceleration.

        Args:
                data (np.ndarray): A numpy 2D array where each row contains an unlabeled training instance.
                features_names (list[str]): A list of feature names.
                labels (np.ndarray, optional): A vector with one label (ground truth) for each observation in data. Defaults to None.
                number_of_steps (int): Number of steps taken this batch, used for keeping track of training restarts. Default is self.train.
                save_lattice (bool, optional): A flag that determines whether the node weights are saved to a file at the end of training. Defaults to False.
                restart_lattice (np.ndarray, optional): Vectors for the weights of the nodes from past realizations. Defaults to None.
                history (bool, optional): Keep lattice and U-matrix snapshots every save_frequency steps in lattice_history and umat_history. Set to False for large runs; the final lattice and U-matrix are still computed. Defaults to True.

        """

        self.restart_lattice = restart_lattice
        self.save_lattice = save_lattice
        self.history = history

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

        # this ensures that the same random order is not repeated if the training is restarted
        if self.epoch > 1:
            self.seed += 1
        # one seeded generator drives the initial lattice and the sample order,
        # so a run is reproducible for a given (seed, data) pair
        rng = np.random.default_rng(self.seed)

        if self.restart_lattice is not None:
            lattice = np.array(self.restart_lattice, dtype=np.float64, copy=True)
        else:
            if self.init == "uniform":
                # vector with small init values for all nodes
                # NOTE: each row represents a node, each column represents a feature.
                lattice = rng.uniform(0.0, 1.0, (number_nodes, number_features))
            else:
                # sample a random subset of the data to initialize the lattice
                ix = rng.integers(0, number_input_vectors, number_nodes)
                lattice = self.data_array[ix, :].astype(np.float64, copy=True)

            self.lattice = lattice.copy()
            if self.history:
                self.lattice_history.append(lattice)  # save the initial lattice
                self.umat_history.append(
                    self.compute_umat()
                )  # save the initial U-matrix

        alpha = self.alpha  # starting learning rate
        if self.alpha_type == 1:
            alpha_freq = max(
                1, self.train // 25
            )  # how often to decay the learning rate; at 24 steps, alpha_f ~ 1e-3 alpha_0
        else:
            alpha_freq = 1

            # compute the initial neighborhood size and step
        nsize_max = max(self.xdim, self.ydim) + 1
        nsize_min = (
            8
            if this_batch_train // 4 > nsize_max - 8
            else nsize_max - this_batch_train // 4
        )  # minimum neighborhood size; avoid the case where there are less training steps than available to shrink the neighborhood to 8
        nsize_step = (
            this_batch_train // 4
        )  # for the third quarter of the training steps, shrink the neighborhood
        nsize_freq = max(
            1, nsize_step // max(1, nsize_max - nsize_min)
        )  # how often to shrink the neighborhood

        # if self.epoch < 2 * nsize_step:
        # 	nsize = nsize_max
        # elif self.epoch >= 3 * nsize_step:
        # 	nsize = nsize_min
        # else:
        # 	nsize = nsize_max - self.epoch // nsize_freq
        epoch = self.epoch  # counts the number of epochs per nsize_freq
        stop_epoch = epoch + this_batch_train
        print("starting epoch is: ", epoch, flush=True)
        print("stopping epoch is: ", stop_epoch, flush=True)

        print("Saving lattice every ", self.save_frequency, " epochs", flush=True)

        # Added 06/17/2024: use random number generator, shuffle the data and take the first train samples
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

        # the per-step learning rate and neighborhood size; index k is the value used at
        # step k, index this_batch_train is the value after the last step
        alphas, nsizes, shrink_events, decay_events = _som_schedule(
            this_batch_train,
            self.epoch,
            float(alpha),
            self.alpha_type,
            alpha_freq,
            nsize_max,
            nsize_min,
            nsize_step,
            nsize_freq,
        )
        xk = np.ascontiguousarray(xk, dtype=np.float64)
        progress_freq = max(1, self.train // 10)

        print("Begin training", flush=True)
        # the kernel runs the steps between two points where Python has work to do:
        # a progress message or (with history) a lattice snapshot
        k = 0
        while True:
            epoch = self.epoch + k
            if epoch % progress_freq == 0:
                print("Evaluating epoch = ", epoch, flush=True)

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

            k_stop = min(
                this_batch_train, k + progress_freq - epoch % progress_freq
            )  # stop before the next progress message
            if self.history:  # stop after the next snapshot step
                k_snapshot = k + (-epoch) % self.save_frequency
                k_stop = min(k_stop, k_snapshot + 1)

            _som_train_steps(lattice, xk, self.xdim, alphas, nsizes, k, k_stop)

            for j in range(k, k_stop):
                if shrink_events[j]:
                    print(
                        f"Shrinking neighborhood size to {nsizes[j + 1]} at epoch {self.epoch + j}",
                        flush=True,
                    )
                if decay_events[j]:
                    print(
                        f"Decaying learning rate to {float(alphas[j + 1])} at epoch {self.epoch + j}",
                        flush=True,
                    )

            # save lattice sparingly
            if self.history and (self.epoch + k_stop - 1) % self.save_frequency == 0:
                self.lattice = lattice.copy()
                self.lattice_history.append(self.lattice)

                # compute the umatrix and save it
                umat = self.compute_umat()
                self.umat_history.append(umat)

            k = k_stop
        print("Training complete", flush=True)

        # update the learning rate, lattice, and Umatrix after training
        if decay_events.any():
            self.alpha = float(alphas[this_batch_train])
        if self.history:
            # as before: the U-matrix comes from self.lattice, i.e. the last snapshot
            self.umat = self.compute_umat()
            self.lattice = lattice  # lattice takes the shape of [X*Y, F]
        else:
            self.lattice = lattice  # lattice takes the shape of [X*Y, F]
            self.umat = self.compute_umat()

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

    def assign_cluster_to_lattice(self, smoothing=None, merge_cost=0.0):
        """
        Assigns clusters to the lattice based on the computed centroids.

        Args:
                smoothing (float, optional): Smoothing parameter for computing Umatrix. Defaults to None.
                merge_cost (float, optional): Cost threshold for merging similar centroids. Defaults to 0.0.

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
    def best_match(lattice: np.ndarray, obs: np.ndarray, full=False) -> np.ndarray:
        """
        Given input vector inp[n,f] (where n is number of different observations, f is number of features per observation), return the best matching node.

        The single-best search delegates to ``aweSOM.inference.best_match_nodes``
        (chunked, allocation-free, float64 reference arithmetic). It returns the
        same node indices as the original kernel, which is kept as
        ``_best_match_legacy`` for ``full=True`` and for tests.

        Args:
                lattice (np.ndarray): weight values of the lattice
                obs (np.ndarray): observations (input vectors)
                full (bool, optional): indicate whether to return first and second best match. Defaults to False.

        Returns:
                np.ndarray: float array of shape (n, 1) with the 1d index of the best-matched node for each observation, or (n, 2) with the two best if ``full``.
        """

        if full:
            return Lattice._best_match_legacy(lattice, obs, True)
        idx, _ = best_match_nodes(lattice, obs)
        return idx.astype(np.float64).reshape(-1, 1)

    @staticmethod
    @njit(parallel=True)
    def _best_match_legacy(lattice: np.ndarray, obs: np.ndarray, full=False) -> np.ndarray:
        """
        Original per-point kernel (allocates per observation). Kept for ``full=True`` and as the reference in tests.

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

    def predict(
        self,
        data: np.ndarray,
        merge_cost: float = None,
        chunk: int = DEFAULT_CHUNK,
        return_distance: bool = False,
        backend: str = "kdtree",
    ):
        """Assign cluster ids to new data already in the lattice's scaled space.

        Uses the trained ``lattice`` and ``lattice_assigned_clusters``; if the
        latter does not exist yet, ``assign_cluster_to_lattice`` is called with
        ``merge_cost`` (0.0 if None). Equivalent to ``map_data_to_lattice`` +
        ``assign_cluster_to_data`` without the two full-size intermediates
        (exactly with ``backend="numba"``; the default KD-tree can resolve
        exact ties differently).

        Args:
                data (np.ndarray): [N, F] float64 or float32 points in the same space as the training data.
                merge_cost (float, optional): Merge threshold used only if the lattice has no cluster assignment yet.
                chunk (int, optional): Points per kernel call. Defaults to 2**20.
                return_distance (bool, optional): Also return the Euclidean BMU distance. Defaults to False.
                backend (str, optional): ``"kdtree"``, ``"numba"`` or ``"jax"``; see :func:`aweSOM.inference.predict`. Defaults to ``"kdtree"``.

        Returns:
                np.ndarray: int32 [N] cluster labels, plus float64 [N] distances if ``return_distance``.
        """
        clusters = getattr(self, "lattice_assigned_clusters", None)
        if clusters is None:
            clusters = self.assign_cluster_to_lattice(
                merge_cost=0.0 if merge_cost is None else merge_cost
            )
        # clusters is indexed [x, y]; node index = x + y * xdim (Fortran order)
        node_cluster = np.asarray(clusters).ravel(order="F").astype(np.int32)
        return _predict_nodes(
            self.lattice,
            node_cluster,
            data,
            chunk=None if backend == "jax" else chunk,
            return_distance=return_distance,
            backend=backend,
        )

    def train_lattice_batch(
        self,
        data: np.ndarray,
        features_names: list[str],
        labels: np.ndarray = None,
        epochs: int = 40,
        **kwargs,
    ):
        """Train the lattice with the JAX batch map (``aweSOM.batch.train_batch``).

        Data-parallel over every visible JAX device (GPUs, or virtual CPU
        devices). The result is statistically, not bitwise, equivalent to
        ``train_lattice``. Fills ``lattice``, ``umat``, ``epoch`` (= epochs) and
        ``batch_info``. Requires JAX; raises ImportError otherwise.

        Args:
                data (np.ndarray): [N, F] training data (process-local rows in a multi-process run).
                features_names (list[str]): A list of feature names.
                labels (np.ndarray, optional): Ground-truth labels. Defaults to None.
                epochs (int, optional): Number of batch epochs. Defaults to 40.
                **kwargs: Passed to ``train_batch`` (r_max, r_min, chunk, seed, init, mesh, devices, precision, dtype, verbose, callback).

        Returns:
                dict: the ``info`` dictionary returned by ``train_batch``.
        """
        from .batch import train_batch

        if self.xdim < 4 or self.ydim < 4:
            sys.exit("build: map is too small.")

        self.data_array = data
        self.features_names = features_names
        self.labels = labels
        kwargs.setdefault("seed", self.seed)
        kwargs.setdefault("init", self.init)
        weights, info = train_batch(data, self.xdim, self.ydim, epochs=epochs, **kwargs)
        self.lattice = np.asarray(weights, dtype=np.float64)
        self.epoch = epochs
        self.batch_info = info
        self.umat = self.compute_umat()
        return info

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
        heat_squared = heat**2
        centroids = naive_centroids.copy()

        # The pairwise cost depends only on the U-matrix and the two (ordered) endpoints,
        # so it is computed once per ordered pair and reused in later rounds. Each round
        # merges the cheapest pair, then re-ranks the surviving centroids against the new
        # largest cost; this loop is the non-recursive form of the original recursion.
        cost_cache = {}

        while True:
            # same centroid list, unique order and counts as get_unique_centroids, in
            # linear rather than quadratic time; that method still sets self.nodes_count
            # once the merging stops
            centr_locs = [
                (centroids["centroid_x"][ix, iy], centroids["centroid_y"][ix, iy])
                for ix in range(self.xdim)
                for iy in range(self.ydim)
            ]
            nodes_count = Counter(centr_locs)
            unique_ids = list(set(centr_locs))
            position_x = [x for x, y in unique_ids]
            position_y = [y for x, y in unique_ids]

            # for each pair of centroids, compute the weighted distance between them via interpolating the umat
            # if the distance is less than the threshold, combine the centroids
            cost_between_centroids = []

            for i in range(len(position_x) - 1):
                for j in range(i + 1, len(position_x)):
                    a = [position_x[i], position_y[i]]
                    b = [position_x[j], position_y[j]]
                    key = (a[0], a[1], b[0], b[1])
                    total_cost = cost_cache.get(key)
                    if total_cost is None:
                        num_sample = 5 * int(
                            np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
                        )
                        x, y = np.linspace(a[0], b[0], num_sample), np.linspace(
                            a[1], b[1], num_sample
                        )
                        umat_dist = map_coordinates(heat_squared, np.vstack((x, y)))
                        total_cost = np.sum(umat_dist)
                        cost_cache[key] = total_cost
                    cost_between_centroids.append([a, b, total_cost])

            if len(cost_between_centroids) == 0:
                self.get_unique_centroids(centroids)  # sets self.nodes_count
                print("Number of unique centroids:  1", flush=True)
                return centroids

            sorted_cost = sorted(
                cost_between_centroids, key=lambda x: x[2]
            )  # this ranks all pairwise cost in ascending order
            # normalize the cost such that the largest cost at each step is always one
            sorted_cost = [[a, b, c / sorted_cost[-1][2]] for a, b, c in sorted_cost]

            # stop once the cheapest remaining pair is above the threshold
            if not sorted_cost[0][2] < threshold:
                self.get_unique_centroids(centroids)  # sets self.nodes_count
                print("Number of unique centroids: ", len(position_x), flush=True)
                print("Minimum cost between centroids: ", sorted_cost[0][2], flush=True)
                return centroids

            centroid_a = tuple(sorted_cost[0][0])
            centroid_b = tuple(sorted_cost[0][1])
            nodes_a = nodes_count[centroid_a]
            nodes_b = nodes_count[centroid_b]

            print(f"Centroid A: {centroid_a}, count: {nodes_a}", flush=True)
            print(f"Centroid B: {centroid_b}, count: {nodes_b}", flush=True)
            print("Merging...", flush=True)

            # this method takes the centroid with the larger number of nodes
            if nodes_a < nodes_b:
                centroids = self.replace_value(centroids, centroid_a, centroid_b)
            else:
                centroids = self.replace_value(centroids, centroid_b, centroid_a)

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


@njit()
def _som_schedule(
    n_steps: int,
    start_epoch: int,
    alpha: float,
    alpha_type: int,
    alpha_freq: int,
    nsize_max: int,
    nsize_min: int,
    nsize_step: int,
    nsize_freq: int,
):
    """Learning rate and neighborhood size for each step of one online training batch.

    Returns:
            tuple: alphas and nsizes (length n_steps + 1; entry k is used at step k, the
            last entry is the state after training), and boolean arrays marking the steps
            after which the neighborhood shrinks and the learning rate decays.
    """
    alphas = np.empty(n_steps + 1)
    nsizes = np.empty(n_steps + 1, dtype=np.int64)
    shrink_events = np.zeros(n_steps, dtype=np.bool_)
    decay_events = np.zeros(n_steps, dtype=np.bool_)
    nsize = nsize_max
    for k in range(n_steps):
        alphas[k] = alpha
        nsizes[k] = nsize
        epoch = start_epoch + k
        # shrink the neighborhood size every [nsize_freq] epochs
        if k > 2 * nsize_step and k % nsize_freq == 0 and nsize > nsize_min:
            nsize = nsize_max - (k - 2 * nsize_step) // nsize_freq
            shrink_events[k] = True
        # decay the learning rate every [alpha_freq] epochs
        if epoch % alpha_freq == 0 and alpha_type == 1 and epoch != 0:
            alpha *= 0.75
            decay_events[k] = True
    alphas[n_steps] = alpha
    nsizes[n_steps] = nsize
    return alphas, nsizes, shrink_events, decay_events


@njit()
def _pairwise_sum(a: np.ndarray, start: int, n: int) -> float:
    """Sum a[start:start + n] in the same order as NumPy's pairwise summation, so the
    result is bitwise equal to np.sum over that slice."""
    if n < 8:
        res = -0.0
        for i in range(start, start + n):
            res += a[i]
        return res
    elif n <= 128:
        r0 = a[start]
        r1 = a[start + 1]
        r2 = a[start + 2]
        r3 = a[start + 3]
        r4 = a[start + 4]
        r5 = a[start + 5]
        r6 = a[start + 6]
        r7 = a[start + 7]
        i = 8
        while i < n - (n % 8):
            r0 += a[start + i]
            r1 += a[start + i + 1]
            r2 += a[start + i + 2]
            r3 += a[start + i + 3]
            r4 += a[start + i + 4]
            r5 += a[start + i + 5]
            r6 += a[start + i + 6]
            r7 += a[start + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[start + i]
            i += 1
        return res
    else:
        n2 = n // 2
        n2 -= n2 % 8
        return _pairwise_sum(a, start, n2) + _pairwise_sum(a, start + n2, n - n2)


@njit()
def _neighborhood_weight(dist: float, alpha: float, nsize: int) -> float:
    """Lattice.Gamma for one node at Chebyshev distance dist <= nsize from the BMU:
    the same expression in the same order, so the value is bitwise equal."""
    sigma = nsize / 3
    return alpha * np.exp(-((dist) ** 2) / (2 * sigma**2))


@njit()
def _som_train_steps(
    lattice: np.ndarray,
    xk: np.ndarray,
    xdim: int,
    alphas: np.ndarray,
    nsizes: np.ndarray,
    k_start: int,
    k_stop: int,
):
    """Online SOM steps k_start..k_stop-1, updating lattice in place.

    Bitwise equal to _som_train_steps_aos (and so to the NumPy form
    s = np.sum((lattice - x)**2, axis=1), c = np.argmin(s),
    lattice -= (lattice - x) * Gamma(c, ...)), but faster:

    - The lattice is held transposed, [F, nodes], so the distance pass runs
      over contiguous nodes and vectorizes. Each node's sum is still formed in
      NumPy's order (sequential from -0.0 for F < 8; NumPy's 8-accumulator
      pairwise form for 8 <= F < 16).
    - The Gaussian weight depends only on the integer Chebyshev distance
      0..nsize, so it is evaluated nsize + 1 times per step (same expression,
      same inputs) instead of once per node in the window.
    """
    n_nodes, n_features = lattice.shape
    if n_features >= 16 or k_stop <= k_start:
        _som_train_steps_aos(lattice, xk, xdim, alphas, nsizes, k_start, k_stop)
        return
    ydim = n_nodes // xdim
    lt = np.ascontiguousarray(lattice.T)
    s = np.empty(n_nodes)
    q = np.empty((n_features, n_nodes))
    nmax = 0
    for k in range(k_start, k_stop):
        nmax = max(nmax, nsizes[k])
    table = np.empty(nmax + 1)
    wrow = np.empty(xdim)
    for k in range(k_start, k_stop):
        # squared distance between the input vector and every node
        if n_features < 8:
            for i in range(n_nodes):
                s[i] = -0.0
            for f in range(n_features):
                xf = xk[k, f]
                row = lt[f]
                for i in range(n_nodes):
                    diff = row[i] - xf
                    s[i] += diff * diff
        else:
            for f in range(n_features):
                xf = xk[k, f]
                row = lt[f]
                qf = q[f]
                for i in range(n_nodes):
                    diff = row[i] - xf
                    qf[i] = diff * diff
            for i in range(n_nodes):
                s[i] = ((q[0, i] + q[1, i]) + (q[2, i] + q[3, i])) + (
                    (q[4, i] + q[5, i]) + (q[6, i] + q[7, i])
                )
            for f in range(8, n_features):
                qf = q[f]
                for i in range(n_nodes):
                    s[i] += qf[i]

        c = _first_argmin(s)

        # update step over the (2 nsize + 1)^2 window around the BMU, clipped to the map
        alpha = alphas[k]
        nsize = nsizes[k]
        any_zero = False
        for d in range(nsize + 1):
            table[d] = _neighborhood_weight(float(d), alpha, nsize)
            if table[d] == 0.0:
                any_zero = True
        cx = c % xdim
        cy = c // xdim
        x0 = max(0, cx - nsize)
        x1 = min(xdim, cx + nsize + 1)
        for y in range(max(0, cy - nsize), min(ydim, cy + nsize + 1)):
            dy = abs(y - cy)
            base = y * xdim
            for x in range(x0, x1):
                wrow[x] = table[max(abs(x - cx), dy)]
            for f in range(n_features):
                xf = xk[k, f]
                row = lt[f]
                if any_zero:  # nodes with zero weight are skipped, as in the reference
                    for x in range(x0, x1):
                        g = wrow[x]
                        if g != 0.0:
                            diff = row[x + base] - xf
                            row[x + base] -= diff * g
                else:
                    for x in range(x0, x1):
                        diff = row[x + base] - xf
                        row[x + base] -= diff * wrow[x]
    lattice[:, :] = lt.T


@njit()
def _first_argmin(s: np.ndarray) -> int:
    """np.argmin(s) for a 1-D float64 array: the first NaN if there is one, else the
    first index of the minimum. Eight independent running minima, then a scan for the
    first index holding the minimum, so the result equals the sequential loop."""
    n = s.size
    m0 = m1 = m2 = m3 = m4 = m5 = m6 = m7 = np.inf
    nan = False
    i = 0
    while i + 8 <= n:
        a0 = s[i]; a1 = s[i + 1]; a2 = s[i + 2]; a3 = s[i + 3]
        a4 = s[i + 4]; a5 = s[i + 5]; a6 = s[i + 6]; a7 = s[i + 7]
        m0 = a0 if a0 < m0 else m0
        m1 = a1 if a1 < m1 else m1
        m2 = a2 if a2 < m2 else m2
        m3 = a3 if a3 < m3 else m3
        m4 = a4 if a4 < m4 else m4
        m5 = a5 if a5 < m5 else m5
        m6 = a6 if a6 < m6 else m6
        m7 = a7 if a7 < m7 else m7
        nan |= (a0 != a0) | (a1 != a1) | (a2 != a2) | (a3 != a3) | (a4 != a4) | (a5 != a5) | (a6 != a6) | (a7 != a7)
        i += 8
    best = min(min(min(m0, m1), min(m2, m3)), min(min(m4, m5), min(m6, m7)))
    while i < n:
        a = s[i]
        if a < best:
            best = a
        nan |= a != a
        i += 1
    if nan:
        for j in range(n):
            if np.isnan(s[j]):
                return j
    for j in range(n):
        if s[j] == best:
            return j
    return 0


@njit()
def _som_train_steps_aos(
    lattice: np.ndarray,
    xk: np.ndarray,
    xdim: int,
    alphas: np.ndarray,
    nsizes: np.ndarray,
    k_start: int,
    k_stop: int,
):
    """Online SOM steps k_start..k_stop-1 on the [nodes, F] layout, updating lattice in place.

    Reference form of _som_train_steps, used for F >= 16.

    Each step uses the same floating-point operations, in the same order, as the
    NumPy form: s = np.sum((lattice - x)**2, axis=1), c = np.argmin(s),
    lattice -= (lattice - x) * Gamma(c, ...). Only nodes within the neighborhood
    (Chebyshev distance <= nsize, where Gamma is non-zero) are visited; the others
    would have zero subtracted, which leaves them unchanged.
    """
    n_nodes, n_features = lattice.shape
    ydim = n_nodes // xdim
    s = np.empty(n_nodes)
    squ = np.empty(n_features)
    for k in range(k_start, k_stop):
        # squared distance between the input vector and every node
        for i in range(n_nodes):
            if n_features < 8:
                acc = -0.0
                for f in range(n_features):
                    diff = lattice[i, f] - xk[k, f]
                    acc += diff * diff
                s[i] = acc
            else:
                for f in range(n_features):
                    diff = lattice[i, f] - xk[k, f]
                    squ[f] = diff * diff
                s[i] = _pairwise_sum(squ, 0, n_features)

        # best matching unit: first minimum, or first NaN, like np.argmin
        c = 0
        best = s[0]
        if not np.isnan(best):
            for i in range(1, n_nodes):
                if s[i] < best:
                    best = s[i]
                    c = i
                elif np.isnan(s[i]):
                    c = i
                    break

        # update step over the (2 nsize + 1)^2 window around the BMU, clipped to the map
        alpha = alphas[k]
        nsize = nsizes[k]
        cx = c % xdim
        cy = c // xdim
        for y in range(max(0, cy - nsize), min(ydim, cy + nsize + 1)):
            for x in range(max(0, cx - nsize), min(xdim, cx + nsize + 1)):
                g = _neighborhood_weight(
                    float(max(abs(x - cx), abs(y - cy))), alpha, nsize
                )
                if g != 0.0:
                    i = x + y * xdim
                    for f in range(n_features):
                        diff = lattice[i, f] - xk[k, f]
                        lattice[i, f] -= diff * g
