Frequently Asked Questions
==========================

.. *Under Construction*

.. contents:: Table of Contents
    :depth: 2
    :local:

General Questions
-----------------

What is the purpose of this project?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
aweSOM was developed for machine-learning clustering and classification tasks, with a focus on identifying intermittent structures in 3D plasma simulations.
Past implementations of Self-organizing Maps (SOM) and Statistically Combined Ensemble (SCE) were not optimized for large datasets, so aweSOM was developed to address this issue.
Using a combination of JIT-accelerated and parallelized SOM and GPU-accelerated SCE, aweSOM can handle datasets with up to $\sim 10^7$ points running on a single GPU/CPU node.

Additionally, aweSOM is designed to be general-purpose, so it can be used for a variety of clustering and classification tasks beyond its original purpose.
See the :doc:`notebook <notebooks/iris>` for an example application on the classic Iris dataset.

How can I contribute?
~~~~~~~~~~~~~~~~~~~~~
You can contribute by making a fork of the aweSOM repository, making changes, and submitting a pull request.

What if I have a question that is not answered here?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you have a question that is not answered here, you can contact the authors directly.
Please reach out to Trung Ha at `tvha@umass.edu <mailto:tvha@umass.edu>`_.

Technical Questions
-------------------

How can I make aweSOM return more/less clusters?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are three main ways to control the number of clusters returned by aweSOM.

1. Increase the number of nodes in the SOM lattice. The more nodes in the lattice, the more local minimas tend to appear. The default lattice size is 10x10.

2. Modify the smoothing factor of the U-matrix. The smoother the U-matrix, the less local minimas will be found, and the fewer clusters will be returned. The default smoothing factor is None.

3. Modify the merge threshold between cluster centroids. The higher the merge threshold, the more clusters will be merged together, and the fewer clusters will be returned. The default merge threshold is 0.

Given an aweSOM object `map`, you can set the smoothing factor and merge threshold by calling `map.assign_cluster_to_lattice(smoothing=[some_value], merge_cost=[some_value])`.
This will identify/merge cluster centroids, then assign each node to the nearest cluster centroid.

A more subtle way to control the number of clusters is to modify the SOM hyperparameters, such as the aspect ratio of the lattice, the learning rate, and the number of epochs. These parameters are less trivial
in how they modify the number of clusters, so it is recommended to start with the three main ways mentioned above.