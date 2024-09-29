Testing
=======

We use `pytest`_ for the test module. The dependency has already been included in the `requirements.txt` file and should be installed automatically with aweSOM.

Functionality tests
-------------------

To run tests for all modules:

.. code-block:: bash

    pytest

You can also run specific test modules by specifying the path to the test file:

.. code-block:: bash

    pytest tests/[module]_test.py

Or run a specific test function within a module:

.. code-block:: bash

    pytest tests/[module]_test.py::test_[function]

If there is no GPU, or if the GPU is not CUDA-compatible, the `sce_test.py` module will fail partially.
This is expected behavior, and SCE analysis should still fall back on the GPU.

Performance tests
-----------------

aweSOM includes many additional features compared to the original implementation of `POPSOM <https://github.com/njali2001/popsom>`_
and `ensemble learning <https://github.com/mkruuse/segmenting-turbulent-simulations-with-ensemble-learning>`_. Therefore, it is not
possible to directly compare the performance of aweSOM with these legacy packages. However, we can still make a rough comparison by 
mimicking the original implementation of ensemble learning and POPSOM in aweSOM, and then benchmarking their performance.

First install the dependencies for these legacy packages:

.. code-block:: bash

    pip install -r tests/performance/requirements.txt

Then run the performance tests inside the ``aweSOM/tests/performance/`` directory.

Benchmarking aweSOM against POPSOM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python popsom_bench.py --N 10000 --F 4

where `N` is the number of points and `F` is the number of features. The script will train a POPSOM map and an aweSOM map
given the same mock dataset, and compare the training time of the two algorithms.

For a personal computer, we recommend using a smaller number of points (:math:`N \sim 10^4`) and features (:math:`F < 5`)
for the test to complete in a reasonable amount of time. More extensive tests can be run on a high-performance computing
cluster. For example, one modern compute node with 40+ cores can perform this benchmark up to :math:`10^6` points and
:math:`\sim 20` features. POPSOM generally cannot handle more than :math:`10^6` points, since training time can exceeds 2
hours at these parameters and/or an out-of-memory error will be raised (even with 1 TB of memory per node).

The expected performance of aweSOM is a speedup of :math:`\sim 8-20` times compared to POPSOM, depending on the number of
points and features.

Benchmarking aweSOM against ensemble learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python sce_bench.py --N 100000 --R 20

where `N` is the number of points and `R` is the number of independent realizations. The script will generate mock cluster
IDs for the dataset and save them as `npy` files. Then it will perform the SCE analysis on the dataset using both the aweSOM
and legacy implementations, and compare the training time of the two algorithms.

In general, the Numpy version of aweSOM is around 2x faster than the legacy implementation. However, the GPU version of
aweSOM is slower than the legacy implementation due to the overhead for small datasets (:math:`N < 5\times10^4`). The GPU 
version of aweSOM is only faster for large datasets (:math:`N > 10^5`), and is exponentially faster as you scale up beyond
:math:`N \sim 10^6`.

We tested the performance of the SCE implementation on a single NVIDIA V-100 GPU with 32 GB of memory. At :math:`N = 10^6`
and :math:`R = 10`, aweSOM is faster than the legacy implementation by a factor of :math:`\sim 15`. At :math:`N = 10^7`,
aweSOM is faster by a factor of :math:`\sim 60`. In high-resolution simulations, :math:`L^3 \gtrsim 500; N = 10^8`, aweSOM
is the only feasible option for performing the SCE analysis.


.. _pytest: https://docs.pytest.org/en/stable/

