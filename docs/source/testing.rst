Testing
=======

We use `pytest`_ for the test module. The dependency has already been included in the `requirements.txt` file and should be installed automatically with aweSOM.

Functionality tests
-------------------

Run tests for all modules in the root directory of the repository:

.. code-block:: bash

    python -m pytest

You can also run specific test modules by specifying the path to the test file:

.. code-block:: bash

    python -m pytest tests/[module]_test.py

Or run a specific test function within a module:

.. code-block:: bash

    python -m pytest tests/[module]_test.py::test_[function]

Two tests in ``sce_test.py`` import JAX with ``pytest.importorskip("jax")``.
Without JAX, pytest skips ``test_jax_gpu_available`` and ``test_jax_cpu_available`` and runs the other four tests.
With JAX installed and no GPU, all six tests pass, and ``test_jax_gpu_available`` issues a warning.
SCE itself runs on the CPU with NumPy when JAX or a GPU is missing.
The ensemble loop itself works on integer contingency tables and does not need a GPU; ``sce_loop_test.py`` checks it
against the original mask-by-mask loop (bitwise on NumPy, to float32 precision on a JAX device).
``combine_clumps_test.py`` tests the clump helpers of the plasma example and is skipped when ``tqdm`` is not installed
(``pip install -r examples/plasma-turbulence/requirements.txt`` installs it).
``model_test.py`` checks ``SOMModel`` and ``Scaler`` across HDF5 and plain-text round-trips. Labels from ``SOMModel``
match ``Lattice.predict`` when the input is scaled at the same precision as the training data, whereas a lattice trained
on float32 features scaled in float32 differs from the float64 ``Scaler`` on 11,544 of :math:`10^7` binary black hole
points, or 0.12%.
``merge_target_test.py`` checks target-count cluster merging, threshold replay with ``merge_cost_used``, the ``float64`` U-matrix requirement, and frozen model metadata round-trips.
``inference_patch_test.py`` checks ``best_match_nodes_patch`` and ``aweSOM.predict(..., backend="patch")`` for bitwise agreement with ``best_match_nodes`` across smooth and random lattices, patch sizes, exact ties, and non-finite inputs, while ``model_test.py`` and ``inference_test.py`` also check ``SOMModel.predict(..., backend="patch")`` and the shared tie and non-finite contract.
``inference_screen_test.py`` checks ``best_match_nodes_screen`` against ``best_match_nodes`` across exact ties, non-finite inputs, single-node lattices, and ``float32`` runner-up overflow fallback.

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

where ``N`` is the number of points and ``F`` is the number of features. The script will train a POPSOM map and an aweSOM map
given the same mock dataset, and compare the training time of the two algorithms.

Additionally, high-level controls include: ``--nodes`` to specify the number of nodes in the lattice, which might be useful 
for isolated scaling tests; ``--procedure [training, mapping, both]`` to specify which part of the algorithm to benchmark; 
and ``--popsom`` or ``--awesom`` to specify one of the two algorithms to benchmark separately.

If you are running a long-duration test that requires dedicated node(s), you can refer to ``examples/slurm_scripts/submit_popsom_bench.cpu``
for an example SLURM script to run this benchmark.

For a personal computer, we recommend using a smaller number of points (:math:`N \sim 10^4`) and features (:math:`F < 5`)
for the test to complete in a reasonable amount of time. More extensive tests can be run on a high-performance computing
cluster. For example, one modern compute node with 40+ cores can perform this benchmark up to :math:`10^6` points and
:math:`\sim 20` features. POPSOM generally cannot handle more than :math:`10^6` points, since training time can exceeds 2
hours at these parameters and/or an out-of-memory error will be raised (even with 1 TB of memory per node).

The expected performance of aweSOM is a speedup of :math:`\sim 8-20 \times` compared to POPSOM, depending on the number of
points and features.

Benchmarking aweSOM against ensemble learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python sce_bench.py --N 100000 --R 20

where ``N`` is the number of points and ``R`` is the number of independent realizations. The script will generate mock cluster
IDs for the dataset and save them as ``npy`` files. Then it will perform the SCE analysis on the dataset using both the aweSOM
and legacy implementations, and compare the training time of the two algorithms.

Additionally, high-level controls include: ``--C`` to specify the number of clusters per realization; ``--legacy`` or 
``--awesom`` to specify one of the two algorithms to benchmark separately.

NOTE: If the test did not complete successfully, there will be a directory named ``som_out`` in the current working directory.
This should be cleaned up manually.

If you are running a long-duration test that requires dedicated node(s), you can refer to ``examples/slurm_scripts/submit_sce_bench.cpu``
and ``examples/slurm_scripts/submit_sce_bench.gpu`` for example SLURM scripts to run this benchmark.

NOTE: the numbers below were measured with the original mask-by-mask ensemble loop. Since v1.2.0 the loop builds
integer contingency tables with a Numba kernel. On 28 plasma label maps of :math:`128^3` cells on an M-series
laptop, the v1.1.3 loop took 338 to 345 s on the CPU; the v1.2.0 loop takes 5.0 s with float64 masks, 2.6 s with
float32 masks (the default) and 0.47 s without masks. The GPU is no longer required for the ensemble step.

In general, the Numpy version of aweSOM is around :math:`2 \times` faster than the legacy implementation. However, the GPU version of
aweSOM is slower than the legacy implementation due to the overhead for small datasets (:math:`N < 5\times10^4`). The GPU 
version of aweSOM is only faster for large datasets (:math:`N > 10^5`), and is exponentially faster as you scale up beyond
:math:`N \sim 10^6`.

We tested the performance of the SCE implementation on a single NVIDIA V-100 GPU with 32 GB of memory. At :math:`N = 10^6`
and :math:`R = 10`, aweSOM is faster than the legacy implementation by a factor of :math:`\sim 15`. At :math:`N = 10^7` and
:math:`R = 10`, aweSOM is faster by a factor of :math:`\sim 60`. In high-resolution simulations where 
:math:`L^3 \gtrsim 500, N = 10^8`, aweSOM is the only feasible option for performing the SCE analysis.


Measured on real data
~~~~~~~~~~~~~~~~~~~~~

Measured on 2026-10-28 on an Apple M1 Max (10 cores), comparing the current code with the original implementation
(commit ``1069cff``). The inputs are the trained BBH lattices (F = 5) and the plasma-turbulence run (F = 4,
:math:`128^3` cells, 28 SCE maps). Each time is the minimum of 3 runs. Other jobs shared the machine (1-minute load
5 to 64), so read the speedups as rough. Every output listed below is bitwise identical to the original: BMU indices,
cluster labels, BMU distances, trained lattices, U-matrices, SCE output files and clump lists. The scripts are
``planning/benchmarks/real_*.py``.

.. list-table:: Inference, all CPU threads (points/s; baseline on the first 1,048,576 points)
   :header-rows: 1

   * - Case
     - Baseline
     - ``numba``
     - ``screen`` (default)
   * - plasma, 64 x 37 lattice, :math:`2.1\times10^6` points
     - 1.5e5
     - 1.0e6
     - 2.1e6
   * - BBH 29 x 20, :math:`10^5` points
     - 3.7e5
     - 3.1e6
     - 5.2e6
   * - BBH 51 x 36, :math:`10^6` points
     - 9.0e4
     - 1.1e6
     - 2.4e6
   * - BBH 91 x 64, :math:`10^7` points
     - 2.9e4
     - 3.7e5
     - 6.9e5

.. list-table:: Training and post-processing
   :header-rows: 1

   * - Step
     - Baseline
     - Current
     - Speedup
   * - Online training, BBH 29 x 20, :math:`10^5` steps
     - 7.40 s
     - 1.03 s (0.48 s with ``history=False``)
     - 7x (15x)
   * - Online training, BBH 51 x 36, :math:`10^6` steps
     - 187 s
     - 19.9 s
     - 9x
   * - SCE, 28 plasma maps
     - 550 s
     - 4.9 s (0.55 s with ``write_masks=False``)
     - 112x (995x)
   * - ``label_clumps`` vs ``get_clumps``, :math:`128^3` sheet masks
     - 7.8 to 34 s
     - 0.06 to 0.13 s
     - 71x to 296x

``screen`` is faster than ``numba`` in every case. Labels from
``SOMModel`` agree with the baseline only when the input is scaled at the same precision as the training data: the
BBH lattices were trained on float32 features scaled in float32, and ``Scaler`` works in float64, so on
:math:`10^7` points 11,544 labels (0.12%) differ.


.. _pytest: https://docs.pytest.org/en/stable/

