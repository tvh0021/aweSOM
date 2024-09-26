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

Magnetized plasma turbulence is a ubiquitous and complex mechanism in high-energy astrophysics and space plasma physics. In a classical Kolmogorov turbulence flow [@kolmogorov1941], kinetic energy is continuously transferred from a large scale $l_0$ (i.e. the driving scale) down to smaller scales, $l_0 > l_1 > l_2$ ..., where the resulting cascade is self-similar in nature. In a magnetized plasma, however, non-linear interactions between the charged particles and the external magnetic field give rise to *intermittency*--spatio-temporal fluctuations in the turbulent flow. So far, numerical simulations and experiments have demonstrated the manifestation of such intermittency in the form of *current sheets*. These current sheets arise via interactions between magnetic coils within the flow, and are important in regulating the dissipation of magnetic energy via reconnection [e.g., @priest2000] and (non-thermal) particle acceleration [e.g., @lemoine2023]. 

Until recently, detection and statistically analysis of these intermittent structures have been done mostly by hand, via a combination of simple thresholds in the current density [@zhdankin2013] and manual verification of magnetic-field reversal across a boundary [@kadowaki2018]. In modern, state-of-the-art plasma simulations, there are hundreds of possible locations where current sheets are found; tracking these locations by hand is a time-consuming task, and the results can depend strongly on individual observers. Recent advancements in computational techniques, particularly the rapid development of machine learning (ML) methodologies and their application to astrophysics research, have highlighted the need for a more robust and efficient approach to understanding the nature of the intermittency in plasma turbulence. To this end, we introduce `aweSOM`, a Python package based on Self-organizing Maps [SOM, @kohonen1990] that provides a fast and statistically significant framework to segment current sheets in 3-dimensional (3D) plasma simulations.

# Statement of need

SOM is an unsupervised ML algorithm that excels at dimensionality reduction, clustering, and classification tasks. In particular, SOM is appropriate for use in detecting intermittency in plasma turbulence because it does not require prior knowledge about the number of clusters (i.e., *physical structures*). Furthermore, a SOM lattice can capture multidimensional correlations in the input data by exposing the topology of the (possibly nonlinear) manifolds being analyzed. In practice, the complex interactions between the charged particles and the magnetic field in plasma turbulence would require such a model to adequately capture their interplay. Lastly, the resulting clusters obtained from a SOM are easily interpretable.

`aweSOM` is a CPU/GPU-accelerated Python package that greatly extends the capabilities of a generic SOM model. We combine the basic functionalities of `POPSOM` [@hamel2019; @yuan2018], a R-based SOM model, and a Statistically Combined Ensemble (SCE) method [@bussov2021], which allows users to choose between training one or multiple SOM lattices for the purpose of clustering data. In each adaptation, we implement aggressive optimization/parallelization strategies to allow the training and classification of large datasets (up to $\sim 10^7$ data points in 20 minutes).

## SOM implementation

Fundamentally, a SOM is a 2D lattice of nodes that, through training, adapts to the intrinsic orientation of high-dimensional input data. The following steps are followed in constructing and training an `aweSOM` lattice:

1. Initialize a lattice of size $X\times Y \times F$, where $X$ and $Y$ are the number of nodes along each map direction, and $F$ denotes the number of features supplied to the model. We follow a tailored formula for the number of nodes: $N_{\rm node} = \frac{5}{6} \sqrt{N \cdot F}$ such that the map both scales with the number of features in addition to the size of the data, but with a fraction of $\frac{1}{6}$ compensating for the map size quickly becoming too big to train when $N$ and $F$ are both large.
2. The initial weight value of each node, $\omega_0$, is randomly assigned, and can be drawn from a uniform distribution or based on random sampling of the input data.
3. Multiple considerations are made during training:
- At each epoch, $t$, one input vector (a cell within the simulation domain) is randomly drawn. Then, the Euclidean distances, $D_{\rm E}$, between this vector and all nodes in the lattice are calculated. The node with the smallest distance is chosen as the best-matching unit (BMU). The weight value of each node is updated as follows: $$w_{i,j}(t) = w_{i,j}(t-1) - D_{\mathrm{E}|i,j} \cdot \gamma(t), $$ where $i,j$ represent the node's location in the lattice, and $\gamma(t)$ is the neighborhood function: $$\gamma(t)= \begin{cases} \alpha(t) e^{\frac{-d_{\rm C}^2}{2(s(t)/3)^2}}, & \text{if $d_{\rm C} \leq s(t)$},\\ 0, & \text{if $d_{\rm C} > s(t)$},\end{cases}$$ where $\alpha(t)$ is the learning rate at epoch $t$, $d_{\rm C}$ is the Chebyshev distance between the BMU and the node at $(i,j)$, and $s(t)$ is the neighborhood width at epoch $t$.
- Initially, $s_0 = \mathrm{max}(X,Y)$ such that earlier training steps adjust the weight values across the entire lattice.As training progresses, $s$ gradually decreases until only a small number of nodes (or just the BMU) are updated each epoch. In `aweSOM`, the final neighborhood size is set to $s_{\rm f} = 8$. This ensures that learning localizes to a specific region of the lattice without being overly restrictive, thereby preserving generalization.
- Simultaneously, $\alpha$ also decays exponentially by a factor of 0.75 at regular interval such that $\alpha_{\rm f} = \alpha_0 \times 0.75^{24} \approx 10^{-3}\,\alpha_0$.
4. After training, clustering is performed on the lattice based on the geometry of the unified distance matrix (U-matrix). Cluster centroids are identified by finding local minima in the U-matrix. A ``merging cost" is then calculated by line integration between all pairs of centroids. If the cost is below a normalized threshold (often set to 0.2-0.3), the clusters are merged.
5. Lastly, the cells in the simulation are mapped to the nearest node in the lattice, each of which has been assigned a cluster label. This label is then transferred to the corresponding input vector, resulting in visualization of the clustering in the input space. 

A list of plasma simulations that we applied the `aweSOM` framework on, as well as convergence metrics, are discussed in @ha2024.

## SCE implementation



<!-- Most notably, we use `Numba` for its , using `JAX` to significantly improve the performance of the mask-to-mask stacking procedure.  -->


<!-- The study of exoplanets, or planets that orbit stars beyond the sun, is a major focus of the astronomy community. Many of these studies center on the analysis of time series photometric (or spectroscopic) observations collected when a planet happens to pass through the line of sight between an observer and its host star. By modeling the fraction of starlight intercepted by the coincident planet, astronomers can deduce basic properties of the system such as the planet's relative size, its orbital period, and its orbital inclination.

The past 20 years have seen extensive work both on theoretical model development and computationally efficient implementations of these models. Notable examples include @mandel_agol, @batman, and @exoplanet, though many other examples can be found. Though each of these packages make different choices, the majority of them (with notable exceptions, including @ellc[^1]) do share one common assumption: the planet under examination is a perfect sphere.

This is both a reasonable and immensely practical assumption. It is reasonable because firstly, a substantial fraction of planets, especially rocky planets, are likely quite close to perfect spheres (Earth's equatorial radius is only 43 km greater than its polar radius, a difference of 0.3%). Secondly, at the precision of most survey datasets (e.g. *Kepler* and *TESS*), even substantially flattened planets would be nearly indistinguishable from a spherical planet with the same on-sky projected area [@zhu2014]. It is practical since, somewhat miraculously, this assumption enables an analytic solution for the amount of flux blocked by the planet at each timestep. This is true even if the intensity of the stellar surface varies radially according to a nearly arbitrarily complex polynomial [@alfm].

However, for a small but growing number of datasets and targets, the reasonableness of this assumption will break down and lead to biased results. Many gas giant planets, in particular, are expected to be distinctly oblate or triaxial, either due to the effects of tidal deformation or rapid rotation [@barnes2003]. Looking within our own solar system, Jupiter and Saturn have oblateness values of roughly 0.06 and 0.1, respectively, due to their fast spins.

To illustrate the effects of shape deformation on a lightcurve, consider \autoref{fig:example}, which shows a selection of differences between time series generated under the assumption of a spherical planet and those generated assuming a planet with Saturn-like flattening. Depending on the obliquity, precession, impact parameter, and whether the planet is tidally locked, we can generate a wide variety of residual lightcurves. In some cases the deviations from a spherical planet occur almost exclusively in the ingress and egress phases of the transit, while others evolve throughout the transit. Some residual curves are mirrored about the transit midpoint, though in general, they will not always be symmetric [@carter_winn_empirical].

![A sampling of differences between transits of spherical and non-spherical planets. A more complete description of how each of these curves were generated can be found in the [online documentation](https://github.com/ben-cassese/squishyplanet/blob/main/joss/figure.py).\label{fig:example}](deviations.png)

The amplitudes of these effects are quite small compared to the full depth of the transit, but could be detectable with a facility such as JWST, which is capable of a white-light precision of a few 10s of ppm [@ERS_prism].

We leave a detailed description of the mathematics and a corresponding series of visualizations for the online documentation. There we also include confirmation that our implementation, when modeling the limiting case of a spherical planet, agrees with previous well-tested models even for high-order polynomial limb darkening laws. More specifically, we show that that lightcurves of spherical planets generated with `squishyplanet` deviate by no more than 100 ppb from those generated with  `jaxoplanet` [@jaxoplanet], the `JAX`-based rewrite of the popular transit modeling package `exoplanet` [@exoplanet] that also implements the arbitrary-order polynomial limb darkening algorithm presented in @alfm. Finally, we demonstrate `squishyplanet`'s limited support for phase curve modeling. 

We hope that a publicly-available, well-documented, and highly accurate model for non-spherical transiting exoplanets will enable thorough studies of planets' shapes and lead to more data-informed constraints on their interior structures.

[^1]: Though `ellc`, and `squishyplanet` share the same goal of modeling transits of non-spherical planets, they differ in a few key ways. First, `ellc` requires users to select from a set of predefined limb darkening laws, while `squishyplanet` allows for any law that can be cast as a polynomial (e.g. high-order approximations to grid-based models). Second, `ellc` allows for gravity-deformed stars, while `squishyplanet` always models the central star as a sphere and restricts triaxial deformations to the planet only. Third, `ellc` allows users to model radial velocity curves, including the Rossiter-McLaughlin effect, while `squishyplanet` is focused on lightcurve modeling only. In terms of implementation, `ellc` is written in Fortran and wrapped in Python, while `squishyplanet` is written in Python/`JAX`. Also, `ellc` integrates the flux blocked by the planet via 2D numerical integration, while `squishyplanet` uses a 1D numerical integration scheme. We believe that these tools will be complementary and that users will benefit from having both available. -->


# Acknowledgements
<!-- 
`squishyplanet` relies on `quadax` [@quadax], an open-source library for numerical quadrature and integration in `JAX`. `squishyplanet` also uses the Kepler's equation solver from `jaxoplanet` [@jaxoplanet] and the finite exposure time correction from `starry` [@starry]. `squishyplanet` is built with the `JAX` library [@jax]. We thank the developers of these packages for their work and for making their code available to the community. -->

# References
