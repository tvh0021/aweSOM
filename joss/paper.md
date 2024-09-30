---
title: '`aweSOM`: a CPU/GPU-accelerated Self-organizing Map and Statistically Combined Ensemble framework for Machine-learning Clustering Analysis'
tags:
  - Python
  - astronomy
  - plasma
authors:
  - name: Trung Ha
    orcid: 0000-0001-6600-2517
    corresponding: true
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
  - name: Joonas Nättilä
    orcid: 0000-0002-3226-4575
    affiliation: "4, 5, 1"
  - name: Jordy Davelaar
    orcid: 0000-0002-2685-2434
    affiliation: "6, 7, 1, 5"
affiliations:
 - name: Center for Computational Astrophysics, Flatiron Institute, 162 Fifth Avenue, New York, NY 10010, USA
   index: 1
 - name: Department of Astronomy, University of Massachusetts-Amherst, Amherst, MA 01003, USA
   index: 2
 - name: Department of Physics, University of North Texas, Denton, TX 76203, USA
   index: 3
 - name: Department of Physics, University of Helsinki, P.O. Box 64, University of Helsinki, FI-00014, Finland
   index: 4
 - name: Physics Department and Columbia Astrophysics Laboratory, Columbia University, 538 West 120th Street, New York, NY 10027, USA
   index: 5
 - name: Department of Astrophysical Sciences, Peyton Hall, Princeton University, Princeton, NJ 08544, USA
   index: 6
 - name: NASA Hubble Fellowship Program, Einstein Fellow
   index: 7
date: 01 October 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Magnetized plasma turbulence is a ubiquitous and complex mechanism in high-energy astrophysics and space plasma physics. In a classical Kolmogorov turbulence flow [@kolmogorov1941], kinetic energy is continuously transferred from the driving scale down to smaller scales, where the resulting cascade is self-similar in nature. In a magnetized plasma, however, non-linear interactions between the charged particles and the external magnetic field give rise to *intermittency*--spatio-temporal fluctuations in the turbulent flow. So far, numerical simulations and experiments have demonstrated the manifestation of such intermittency in the form of *current sheets*. 
<!-- These current sheets arise via interactions between magnetic coils within the flow, and are important in regulating the dissipation of magnetic energy via reconnection [e.g., @priest2000] and (non-thermal) particle acceleration [e.g., @lemoine2023].  -->

Until recently, detection and statistically analysis of these intermittent structures have been done mostly by hand, obtained via simple, and often arbitrary, criteria.
<!-- via a combination of simple thresholds in the current density [@zhdankin2013] and manual verification of magnetic-field reversal across a boundary [@kadowaki2018].  -->
In modern, state-of-the-art plasma simulations, there are hundreds of possible locations where these structures are found; manually tracking these location simultaneously is a time-consuming task, and the result can depend strongly on individual observers. Recent advancements in computational techniques, particularly the rapid development of machine learning (ML) methodologies and their application to astrophysics research, have highlighted the need for a more robust and efficient approach to understanding the nature of the intermittency in plasma turbulence. To this end, we introduce `aweSOM`, a Python package based on Self-organizing Maps [SOM, @kohonen1990] that provides a fast and statistically significant framework to perform clustering analysis.

# Statement of need

## The SOM algorithm

SOM is an unsupervised ML algorithm that excels at dimensionality reduction, clustering, and classification tasks.
It consists of a 2-dimension (2D) lattice of nodes. Each node has a weight vector that is of the same dimension as the input data. A SOM performs the clustering by changing the weight vectors such that the lattice's topology conforms to the intrinsic clustering of the input data. In this manner, a SOM lattice can capture multidimensional correlations in the input data.

`aweSOM` is a CPU/GPU-accelerated Python package that greatly extends the capabilities of a generic SOM model. We combine the basic functionalities of `POPSOM` [@hamel2019; @yuan2018], a R-based SOM model, and a Statistically Combined Ensemble (SCE) method [@bussov2021], which allows users to choose between training one or multiple SOM lattices for the purpose of clustering data. In each adaptation, we implement aggressive optimization/parallelization strategies to allow the training and classification of large datasets (up to $N \sim 10^7$ data points in 20 minutes).

## The need for an accelerated SOM implementation

`POPSOM` was developed as a single-threaded, stochastic training algorithm with advanced visualization capabilites built-in. However, due to this single-threaded nature, the algorithm does not scale well with large datasets. For astrophysics applications such as high-resolution simulations, where $N \gtrsim 10^6$, `POPSOM` is unable to complete the training process as the dimensionality of the input data increases due to its much higher memory usage. As a demonstration, we generated a mock dataset with $N = 10^6$ and $F = 6$, then trained it on a SOM lattice of $X = 63$, and $Y = 32$, where $X, Y$ are the dimension of the lattice using both `aweSOM` and `POPSOM`. We used one Intel Icelake node with 64 cores and 1 TB memory for this test. `aweSOM` took $\approx 200$ s and consumed $\approx 450$ MB of memory to complete the training and clustering, while `POPSOM` took $\approx 2200$ s and consumed $\approx 600$ GB of system memory.

To combat these challenges, we rewrite `POPSOM` using more modern `NumPy` functions whenever the lattice (which is a 3D array) is updated. This legacy implementation was originally written in R, then translated to Python, so there are numerous instances where the algorithm could be optimized. Furthermore, for the steps where parallelization could be leveraged, we integrate `Numba` to take advantage of its Just-In-Time (JIT) compiler and simple parallelization of loops. These optimizations translate to up to $20\times$ faster mapping of cluster labels to the input data, and up to $10\times$ faster training time (at $N = 10^6$, $F = 4$, $X = 58$, and $Y = 29$, where $X, Y$ are the dimensions of the lattice).

(Something about the SCE, drawbacks, and our optimizations here)

## SOM implementation

Fundamentally, a SOM is a 2D lattice of nodes that, through training, adapts to the intrinsic orientation of high-dimensional input data. The following steps are followed in constructing and training an `aweSOM` lattice:

1. Initialize a lattice of size $X\times Y \times F$, where $X$ and $Y$ are the dimensions of the lattice, and $F$ denotes the number of features supplied to the model. We follow a tailored formula for the number of nodes: $N_{\rm node} = \frac{5}{6} \sqrt{N \cdot F}$ such that the map both scales with the number of features in addition to the size of the data.
2. The initial weight value of each node, $\omega_0$, can be drawn from a uniform random distribution or based on random sampling of the input data.
3. Multiple considerations are made during training:
- At each epoch, $t$, one input vector is randomly drawn. Then, the Euclidean distances, $D_{\rm E}$, between this vector and all nodes in the lattice are calculated. The node with the smallest distance is chosen as the best-matching unit (BMU). The weight value of each node is updated as follows: $$w_{i,j}(t) = w_{i,j}(t-1) - D_{\mathrm{E}|i,j} \cdot \gamma(t), $$ where $i,j$ represent the node's location in the lattice, and $\gamma(t)$ is the neighborhood function: $$\gamma(t)= \begin{cases} \alpha(t) e^{\frac{-d_{\rm C}^2}{2(s(t)/3)^2}}, & \text{if $d_{\rm C} \leq s(t)$},\\ 0, & \text{if $d_{\rm C} > s(t)$},\end{cases}$$ where $\alpha(t)$ is the learning rate at epoch $t$, $d_{\rm C}$ is the Chebyshev distance between the BMU and the node at $(i,j)$, and $s(t)$ is the neighborhood width at epoch $t$.
- Initially, $s_0 = \mathrm{max}(X,Y)$ such that earlier training steps adjust the weight values across the entire lattice.As training progresses, $s$ gradually decreases until only a small number of nodes (or just the BMU) are updated each epoch. In `aweSOM`, the final neighborhood size is set to $s_{\rm f} = 8$. This ensures that learning localizes to a specific region of the lattice without being overly restrictive, thereby preserving generalization.
- Simultaneously, $\alpha$ also decays exponentially by a factor of 0.75 at regular interval such that $\alpha_{\rm f} = \alpha_0 \times 0.75^{24} \approx 10^{-3}\,\alpha_0$.
4. After training, clustering is performed on the lattice based on the geometry of the unified distance matrix (U-matrix). Cluster centroids are identified by finding local minima in the U-matrix. A ``merging cost" is then calculated by line integration between all pairs of centroids. If the cost is below a normalized threshold (often set to 0.2-0.3), the clusters are merged.
5. Lastly, the cells in the simulation are mapped to the nearest node in the lattice, each of which has been assigned a cluster label. This label is then transferred to the corresponding input vector, resulting in visualization of the clustering in the input space. 

A list of plasma simulations that we applied the `aweSOM` framework on, as well as convergence metrics, are discussed in @ha2024.

## SCE implementation



<!-- Most notably, we use `Numba` for its , using `JAX` to significantly improve the performance of the mask-to-mask stacking procedure.  -->


# Acknowledgements
<!-- 
`squishyplanet` relies on `quadax` [@quadax], an open-source library for numerical quadrature and integration in `JAX`. `squishyplanet` also uses the Kepler's equation solver from `jaxoplanet` [@jaxoplanet] and the finite exposure time correction from `starry` [@starry]. `squishyplanet` is built with the `JAX` library [@jax]. We thank the developers of these packages for their work and for making their code available to the community. -->

# References
