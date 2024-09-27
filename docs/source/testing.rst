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

To benchmark the performance of aweSOM compared to `POPSOM <https://github.com/njali2001/popsom>`_ and the original implementation of `ensemble learning <https://github.com/mkruuse/segmenting-turbulent-simulations-with-ensemble-learning>`_, first install the two packages:

.. code-block:: bash
    
    # Install POPSOM
    cd [aweSOM_parent_directory]
    git clone https://github.com/njali2001/popsom.git
    cd popsom
    cp popsom.py ../aweSOM/tests/performance/

    # Install SCE
    cd [aweSOM_parent_directory]
    git clone https://github.com/mkruuse/segmenting-turbulent-simulations-with-ensemble-learning.git
    cd segmenting-turbulent-simulations-with-ensemble-learning
    cp sce.py ../aweSOM/tests/performance/

    # Install the dependencies
    pip install -r tests/performance/requirements.txt

Then run the performance tests inside the ``aweSOM/`` directory:

.. code-block:: bash

    pytest tests/performance


.. _pytest: https://docs.pytest.org/en/stable/

