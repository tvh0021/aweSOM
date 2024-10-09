---
title: '`aweSOM`: a CPU/GPU-accelerated Self-organizing Map and Statistically Combined Ensemble Framework for Machine-learning Clustering Analysis'
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
 - name: Department of Astronomy, University of Massachusetts-Amherst, Amherst, MA 01003, USA
   index: 1
 - name: Center for Computational Astrophysics, Flatiron Institute, 162 Fifth Avenue, New York, NY 10010, USA
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
date: 08 October 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

We introduce `aweSOM`, an open-source Python package for machine learning (ML) clustering and classification, based on a Self-organizing Maps [SOM, @kohonen1990] algorithm that incorporates CPU/GPU acceleration to handle large ($N > 10^6$), multidimensional datasets. `aweSOM` consists of two main modules, one that handles the intialization and training of the SOM, and one that takes multiple realizations of the SOMs and stacks their clustering results to obtain a set of statistically significant clusters.

Many Python-based SOM implementations already exist in the literature (e.g., `POPSOM`, @yuan2018; `MiniSom`, @minisom; `sklearn-som`). However, they primarily serve as proof-of-concept demonstrations that are optimized for smaller datasets, while lacking the ability to scale up to large, multidimensional datsets. `aweSOM` provides a solution for this gap in capability, and is especially appropriate for use in high-resolution simulations and large sky surveys, where data comes in $10^6 - 10^9$ individual points, together with multiple features per point. Specifically, we demonstrate that, compared to the legacy implementations `aweSOM` is based on, our package provides between $10-100 \times$ speedup, as well as vastly better memory performance thanks to the many built-in optimizations.

As a companion to this paper, we demonstrate the capabilities of `aweSOM` in detecting the structures of intermittency in plasma turbulence in @ha2024.

# Statement of need

## The self-organizing map method

SOM is an unsupervised ML algorithm that excels at dimensionality reduction, clustering, and classification tasks.
It consists of a 2-dimension (2D) lattice of nodes. Each node has a weight vector that is of the same dimension as the input data. A SOM performs the clustering by changing the weight vectors of a group of nodes such that the lattice's topology eventually conforms to the intrinsic clustering of the input data. In this manner, a SOM lattice can capture multidimensional correlations in the input data.

### `POPSOM`

We base the SOM module in `aweSOM` on `POPSOM` [@hamel2019; @yuan2018], a R-based SOM model. `POPSOM` was developed as a single-threaded, stochastic training algorithm with advanced visualization capabilites built-in. However, due to this single-threaded nature, the algorithm does not scale well with large datasets. For astrophysics problems such as in high-resolution simulations, where $N \gtrsim 10^6$, `POPSOM` is unable to complete the training process as the dimensionality of the input data increases due to its much higher memory usage. As an example, we generate a mock dataset with $N = 10^6$ and $F = 6$ dimensions, then train it on a lattice of $X = 63$, and $Y = 32$, where $X, Y$ are the dimensions of the lattice, using one Intel Icelake node with 64 cores and 1 TB memory. `POPSOM` completed the training in $\approx 2200$ s and consumed $\approx 600$ GB of system memory at its peak.

### Rewriting `POPSOM` into `aweSOM`

To combat the long training time and extremely large memory usage, we rewrite `POPSOM` with multiple optimizations/parallelizations. We use more modern `NumPy` functions whenever the lattice (which is a 3D array) is updated. This legacy implementation was originally written in R, then translated to Python, so there are numerous instances where the algorithm was inefficient. Furthermore, for the steps where parallelization could be leveraged (such as when the cluster labels are mapped to the lattice, then to the input data), we integrate `Numba` [@numba] to take advantage of its Just-In-Time (JIT) compiler and simple parallelization of loops. These optimizations translate to up to $20\times$ faster mapping of cluster labels to the input data, and up to $10\times$ faster training time. In the same example as above, `aweSOM` takes $\approx 200$ s and consumes $\approx 450$ MB of memory to complete the training and clustering. In addition to the $\sim 10 \times$ speedup, `aweSOM` is also $\sim 10^3 \times$ more memory-efficient.

The left hand side of \autoref{fig:sce_scaling} shows a graph of the performance between `aweSOM` and `POPSOM` given a range of $N$ and $F$. `POPSOM` is slightly faster than `aweSOM` for $N \lesssim 10^4$, although both complete their training very quickly. At $N \gtrsim 5 \times 10^5$, `aweSOM` is consistently faster than `POPSOM` by roughly a factor of $10$. Most importantly, `POPSOM` fails to complete its clusters mapping for $N \gtrsim 10^6, F > 4$ because the memory buffer (1 TB) of the test node was exceeded.

## The statistically combined ensemble method

Statistically combined ensembled (SCE) was developed by @bussov2021 to stack multiple independent clustering results into a statistically significant set of clusters. This is a type of ensemble learning. Additionally, SCE can also be used independently from the base SOM algorithm, and is compatible with any general unsupervised classification algorithms. 

### The legacy SCE implementation

In its original version, the SCE was saved as a nested dictionary of boolean arrays, each of which contains the spatial similarity index $g$ between cluster $C$ and cluster $C'$. The total number of operations scales as $N_{C}^R$, where $N_C$ is the number of clusters in each realization, and $R$ is the number of realization. In practice, each SOM realization trained on the plasma simulation (as described in @ha2024) contains on average 7 clusters. When we generate 36 realizations for the SCE analysis, there are a total of $T \approx 7^{36} \sim 10^{30}$ array-to-array comparisons.

### Integrating SCE into `aweSOM` with `JAX`

We use `JAX` [@jax] to significantly improve the performance of the array-to-array comparison procedure by leveraging the GPU's advantage over CPU in parallel computing.  
We implement this optimization by eliminating the need for nested dictionaries, instead replacing them with data arrays. Beyond that, every instance of matrix operation using `NumPy` is converted to `jax.numpy`. Additionally, we implement internal checks such that the SCE code automatically reverts to `NumPy` if GPU-accelerated `JAX` is not available.

Similar to the SOM implementation, the SCE implementation in `aweSOM` scales extremely well with increasing number of data points. The right hand side of \autoref{fig:sce_scaling} shows a graph of the performance between the two implementations given $R = 20$. At $N < 5 \times 10^4$, the legacy implementation is faster due to the overhead from loading `JAX` and the JIT compiler. However, `aweSOM` quickly exceeds the performance of the legacy code, and begins to approach its maximum speed-up of $\sim 100$ at $N \gtrsim 10^7$. On the other hand, simply using `aweSOM` with `NumPy` only yeilds a consistent $2\times$ speedup compared to the legacy implementation. Altogether, it is best to use `aweSOM` with `Numpy` when $N \lesssim 10^5$, and with `JAX` when $N \gtrsim 10^5$.

![Performance scaling for `aweSOM` vs. the legacy SOM (left) and SCE (right) implementation. The top panels show the time for each implementation to complete analysis of $N$ number of data points. The dotted lines shows linear extrapolations from the data in order to estimate the speedup. The bottom panels show the ratio between the time taken by the legacy code divided by the time taken by `aweSOM`. In the SOM analysis, we consider a dataset with $F = 6$ and $F = 10$ dimensions. In the SCE analysis, we test the scaling of both a GPU-accelerated implementation (with `JAX`) and a CPU-only implementation (with `NumPy`). \label{fig:sce_scaling}](joss_scaling.pdf)

<!-- # Mathematical descriptions of `aweSOM`

## SOM implementation

Fundamentally, a SOM is a 2D lattice of nodes that, through training, adapts to the intrinsic orientation of high-dimensional input data. The following steps are followed in constructing and training an `aweSOM` lattice:

1. Initialize a lattice of size $X\times Y \times F$, where $X$ and $Y$ are the dimensions of the lattice, and $F$ denotes the number of features supplied to the model. We follow a tailored formula for the number of nodes: $N_{\rm node} = \frac{5}{6} \sqrt{N \cdot F}$ such that the map both scales with the number of features in addition to the size of the data.
2. The initial weight value of each node, $\omega_0$, can be drawn from a uniform random distribution or based on random sampling of the input data.
3. Multiple considerations are made during training:
- At each epoch, $t$, one input vector is randomly drawn. Then, the Euclidean distances, $D_{\rm E}$, between this vector and all nodes in the lattice are calculated. The node with the smallest distance is chosen as the best-matching unit (BMU). The weight value of each node is updated as follows: $$w_{i,j}(t) = w_{i,j}(t-1) - D_{\mathrm{E}|i,j} \cdot \gamma(t), $$ where $i,j$ represent the node's location in the lattice, and $\gamma(t)$ is the neighborhood function: $$\gamma(t)= \begin{cases} \alpha(t) e^{\frac{-d_{\rm C}^2}{2(s(t)/3)^2}}, & \text{if $d_{\rm C} \leq s(t)$},\\ 0, & \text{if $d_{\rm C} > s(t)$},\end{cases}$$ where $\alpha(t)$ is the learning rate at epoch $t$, $d_{\rm C}$ is the Chebyshev distance between the BMU and the node at $(i,j)$, and $s(t)$ is the neighborhood width at epoch $t$.
- Initially, $s_0 = \mathrm{max}(X,Y)$ such that earlier training steps adjust the weight values across the entire lattice. $s$ gradually decreases as $t$ increases until only a small number of nodes (or just the BMU) are updated each epoch. In `aweSOM`, the final neighborhood size is set to $s_{\rm f} = 8$. This ensures that learning localizes to a specific region of the lattice without being overly restrictive, thereby preserving generalization.
- $\alpha$ also decays exponentially by a factor of 0.75 at regular interval such that $\alpha_{\rm final} = \alpha_0 \times 0.75^{24} \approx 10^{-3}\,\alpha_0$.
4. After training, clustering is performed on the lattice based on the clustering of the unified distance matrix (U-matrix). Initial cluster centroids are identified by finding local minima in the U-matrix. A ``merging cost" is then calculated by line integration between all pairs of centroids. If the cost is below a normalized threshold (often set to 0.2-0.3), the clusters are merged.
5. The input data is mapped to the nearest node in the lattice, each of which has been assigned a cluster label. This label is then transferred to the corresponding input vector, resulting in visualization of the clustering in the input space. 

A list of plasma simulations that we applied the `aweSOM` framework on, as well as convergence metrics, are discussed in @ha2024.

## SCE implementation

The mathematical details of the SCE framework are discussed in @bussov2021. Below, we briefly summarize the key concepts of SCE.

SCE involves a series of steps that stacks $n$~number of SOM realizations. For each cluster $C$ in a SOM realization $R$, its spatial distribution is compared with all other clusters $C'$ in $R' \neq R$ to obtain a goodness-of-fit index $g$. Then, each cluster $C$ is associated with a sum of goodness-of-fit (i.e. ``quality index"): $$G_{\rm sum} = \sum_{C_i' \in R'} g_i.$$

Once all $G_{\rm sum}$ values are obtained, they are ranked in descending order, and groups of similar $G_{\rm sum}$ values are combined to form SCE clusters. 
This approach works because clusters with similar spatial distributions tend to have similar $G_{\rm sum}$ values [see Fig. 6 of @bussov2021]. In practice, we do not rank the $G_{\rm sum}$ values, but instead sum this index point-by-point to obtain a general ``signal strength" of each input vector. Then, we make cuts from this signal strength to obtain the final clustering result. -->

# Acknowledgements

The authors would like to thank Kaze Wong for the valuable input in setting up `JAX` for the SCE analysis. The authors would also like to thank Shirley Ho and Lorenzo Sironi for the useful discussions.
TH acknowledges support from a pre-doctoral program at the Center for Computational Astrophysics, which is part of the Flatiron Institute. JN is supported by an ERC grant (ILLUMINATOR, 101114623). JD is supported by NASA through the NASA Hubble Fellowship grant HST-HF2-51552.001-A, awarded by the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy, Incorporated, under NASA contract NAS5-26555.
`aweSOM` was developed and primarily run at facilities supported by the Scientific Computing Core at the Flatiron Institute. Research at the Flatiron Institute is supported by the Simons Foundation.

# References
