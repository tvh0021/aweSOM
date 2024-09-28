Testing
=======

We use `pytest`_ for the test module. The dependency has already been included in the `requirements.txt` file and should be installed automatically with aweSOM.

Functionality tests
-------------------

To run tests for all modules:

.. code-block:: bash

    pytest --ignore=tests/performance

You can also run specific test modules by specifying the path to the test file:

.. code-block:: bash

    pytest tests/[module]_test.py

Or run a specific test function within a module:

.. code-block:: bash

    pytest tests/[module]_test.py::test_[function]

Performance tests
-----------------

To benchmark the performance of aweSOM compared to `POPSOM <https://github.com/njali2001/popsom>`_ and the original 
implementation of `ensemble learning <https://github.com/mkruuse/segmenting-turbulent-simulations-with-ensemble-learning>`_,
first install the dependencies:

.. code-block:: bash

    pip install -r tests/performance/requirements.txt

Then run the performance tests inside the ``aweSOM/tests/performance/`` directory.

Benchmarking aweSOM against POPSOM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python popsom_test.py --n 10000 --f 4

For a personal computer, we recommend using a smaller number of points (:math:`\leq 10000`) and features (:math:`< 10`)
for the test to complete in a reasonable amount of time. More extensive tests can be run on a high-performance computing
cluster. For example, one modern compute node with 40+ cores can perform this benchmark up to :math:`10^6` points and
:math:`\sim 20` features. POPSOM generally cannot handle more than :math:`10^6` points.

The expected performance of aweSOM is a speedup of :math:`\sim 8-20` times compared to POPSOM, depending on the number of
points and features.

Benchmarking aweSOM against ensemble learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. _pytest: https://docs.pytest.org/en/stable/

