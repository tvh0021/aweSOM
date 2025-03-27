.. aweSOM documentation master file, created by
   sphinx-quickstart on Wed Sep 18 16:26:21 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. aweSOM documentation
.. ====================

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.

aweSOM - Accelerated Self-organizing Map and Statistically Combined Ensemble
============================================================================

The `aweSOM <https://github.com/tvh0021/aweSOM>`_ package combines a JIT-accelerated and parallelized implementation of self-organizing map (SOM),
integrating parts of `POPSOM <https://github.com/njali2001/popsom>`_, and a GPU-accelerated implementation of
statistically combined ensemble (SCE) using `ensemble learning <https://github.com/mkruuse/segmenting-turbulent-simulations-with-ensemble-learning>`_.  

.. aweSOM is developed specifically to identify intermittent structures (current sheets) in 3D plasma simulations 
.. (`link to arXiv pre-print <https://doi.org/10.48550/arXiv.2410.01878>`_). However, it can also be used for a variety of clustering and classification tasks.

aweSOM was developed for machine-learning clustering and classification tasks, with a focus on identifying intermittent structures in 3D plasma simulations (`link to arXiv pre-print <https://doi.org/10.48550/arXiv.2410.01878>`_).
Past implementations of SOM and SCE were single-threaded and not optimized for large datasets, so aweSOM was developed to address this issue.
Using a combination of JIT-accelerated and parallelized SOM and GPU-accelerated SCE, aweSOM can handle datasets with up to :math:`\sim 10^8` points running on a single GPU/CPU node.

Additionally, aweSOM has been designed to be general-purpose, so it can be used for a variety of clustering and classification tasks beyond its original purpose.
See the :doc:`notebook <notebooks/iris>` for an example application on the classic Iris dataset.

**Authors:**

`Trung Ha <https://tvh0021.github.io>`_ - University of Massachusetts-Amherst,
`Joonas Nättilä <https://natj.github.io>`_ - University of Helsinki,
and `Jordy Davelaar <https://jordydavelaar.com>`_ - Princeton University.

Current version: 1.1.0

.. [![status](https://joss.theoj.org/papers/b6995d7573124b56e82ec6b875f3b5d1/status.svg)](https://joss.theoj.org/papers/b6995d7573124b56e82ec6b875f3b5d1)


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   testing
   notebooks/plasma
   notebooks/iris
   faqs
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
.. * :ref:`search`

