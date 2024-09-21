.. aweSOM documentation master file, created by
   sphinx-quickstart on Wed Sep 18 16:26:21 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. aweSOM documentation
.. ====================

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.

aweSOM - Accelerated Self-organizing Map (SOM) and Statistically Combined Ensemble (SCE)
========================================================================================

The aweSOM package combines a JIT-accelerated and parallelized implementation of SOM,
integrating parts of `POPSOM <https://github.com/njali2001/popsom>`_ and a GPU-accelerated implementation of
SCE using `ensemble learning <https://github.com/mkruuse/segmenting-turbulent-simulations-with-ensemble-learning>`_. 

.. aweSOM is optimized for large datasets, up to $\sim 10^7$ points. 

aweSOM is developed specifically to identify intermittent structures (current sheets) in 3D plasma simulations 
[link to paper available soon]. However, it can also be used for a variety of clustering and classification tasks.

**Authors:**

`Trung Ha <https://tvh0021.github.io>`_ - University of Massachusetts-Amherst,
`Joonas Nättilä <https://natj.github.io>`_ - University of Helsinki,
and `Jordy Davelaar <https://jordydavelaar.com>`_ - Princeton University.

Current version: 1.4.1


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   examples
   faqs
   modules   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

