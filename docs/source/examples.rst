Examples
========

.. contents:: Table of Contents
    :depth: 3
    :local:

Plasma Simulation
-----------------

The dataset used in this example can be found at [Zenodo link]. The example hdf5 file,
`features_2j1b1e0r_5000_jasym.h5`, contains the simulation snapshot discussed in the paper [link to paper].
This dataset contains 4 features, :math:`j_{\parallel}`, :math:`b_{\perp}`, :math:`e_{\parallel}`,and 
:math:`j_{\rm asym}`. Place the hdf5 file in the `examples/plasma-turbulence/` directory and run the script.

To generate a different set of features, refer to the :ref:`optional-plasma` subsection.

Initialize and train a single SOM realization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run a single SOM realization, use the `examples/plasma-turbulence/run_plasma_som.py` script with only one value
for each argument.

.. code-block:: bash
    
    cd examples/plasma-turbulence
    python run_plasma_som.py --ratio 0.6 --alpha_0 0.1 --train 2097152
    
This will train a SOM realization with an initial learning rate :math:`\alpha_0 = 0.1`, a lattice ratio
:math:`H = 0.6`, and :math:`N = 2,097,152` training steps (the entire simulation domain, :math:`L^3 = 128^3`).
Optional toggles inside the script can be modified as needed. `file_name` points to the hdf5 file containing the
simulation snapshot, `sampling_type` can be either "uniform" (random initial weights between -1 and 1) or "sampling"
(random initial weights sampled from the data), and `merge_threshold` sets the threshold of cost for merging two
cluster centroids.

Two files will be generated: `som_object.*.pkl` and `labels.*.npy`. The former contains the trained SOM object
and the latter contains the cluster IDs for each data point. You can visualize the results following these example 
Jupyter notebook

.. nbsphinx:: fiducial_plasma.ipynb

Train multiple SOM realizations and combine them with SCE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. nbsphinx:: sce_plasma.ipynb


.. _optional-plasma:

Optional - Generate different sets of features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




Iris Dataset
------------

Initialize and train a single SOM realization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Train multiple SOM realizations and combine them with SCE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. .. code-block:: python

..     # Example 1: Simple addition
..     def add(a, b):
..         return a + b

..     print(add(2, 3))  # Output: 5

.. .. code-block:: python

..     # Example 2: Class definition
..     class Dog:
..         def __init__(self, name):
..             self.name = name

..         def bark(self):
..             return f"{self.name} says woof!"

..     my_dog = Dog("Buddy")
..     print(my_dog.bark())  # Output: Buddy says woof!