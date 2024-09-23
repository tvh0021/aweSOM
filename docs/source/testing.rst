Testing
=======

We use `pytest`_ for the test module. The dependency has already been included in the `requirements.txt` file and should be installed automatically with aweSOM.

To run tests for all modules:

.. code-block:: bash

    pytest

You can also run specific test modules by specifying the path to the test file:

.. code-block:: bash

    pytest tests/[module]_test.py

Or run a specific test function within a module:

.. code-block:: bash

    pytest tests/[module]_test.py::test_[function]

.. _pytest: https://docs.pytest.org/en/stable/