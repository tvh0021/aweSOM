# aweSOM - Accelerated Self-Organizing Map (SOM) and Statistically Combined Ensemble (SCE)

This package combines a JIT-accelerated and parallelized implementation of SOM, integrating parts of [POPSOM](https://github.com/njali2001/popsom) and a GPU-accelerated implementation of SCE using [ensemble learning](https://github.com/mkruuse/segmenting-turbulent-simulations-with-ensemble-learning). 
It is optimized for large datasets, up to $\sim 10^7$ points. 

aweSOM is developed specifically to identify intermittent structures (current sheets) in 3D plasma simulations (link to paper).
However, it can be used for a variety of clustering and classification tasks.
For example, see the [Iris dataset](examples/iris.ipynb).

## 1. Installation

To install aweSOM:

```bash
git clone https://github.com/tvh0021/aweSOM.git
cd aweSOM
pip install .
```

## 2. Basic Usage

Here are the basic steps to initialize a lattice and train the SOM

```python
import numpy as np
import matplotlib.pyplot as plt
from aweSOM import Lattice

# Create an initial SOM instance
map = Lattice(xdim=40, ydim=15, alpha_0=0.5, train=100000)

# Train the SOM with some data in the shape of $N \times F$.
map.train_lattice(normalized_data, feature_names)
```

The trained SOM is saved at  `map.lattice`

To visualize the SOM with U-matrix, which is saved at the end of training at `map.umat`

```python
# Compute the unique centroids
naive_centroids_matrix = map.compute_centroids() # return the centroid associated with each node
unique_centroids = map.get_unique_centroids(map.compute_centroids()) # return the indivual centroids

plot_centroids['position_x'] = [x+0.5 for x in unique_centroids['position_x']]
plot_centroids['position_y'] = [y+0.5 for y in unique_centroids['position_y']]

X,Y = np.meshgrid(np.arange(xdim)+0.5, np.arange(ydim)+0.5)

plt.figure(dpi=250)
plt.pcolormesh(map.umat.T, cmap='viridis')
plt.scatter(plot_centroids['position_x'],plot_centroids['position_y'], color='red', s=10)
plt.colorbar(fraction=0.02)
plt.contour(X, Y, map.umat.T, levels=np.linspace(np.min(map.umat),np.max(map.umat), 20), colors='black', alpha=0.5)
plt.gca().set_aspect("equal")
plt.title(rf'UMatrix for {xdim}x{ydim} SOM')
```
![U-matrix of a 40x15 SOM trained on Iris dataset](examples/iris/umat.png)

There are 15 centroids in this U-matrix -> there are 15 clusters. 
Now from the geometry of the U-matrix, we can see there are clearly at least two cluster (separated by the large band of high value nodes), and at most four clusters. 

Merge clusters using cost function

```python
merge_threshold = 0.2 # empirical tests reveal a threshold between 0.2 and 0.4 usually works best

# plot U-matrix with the connected components and ground truth labels (if the labels were supplied durint map.train_lattice)
map.plot_heat(map.umat, merge=True, merge_cost=merge_threshold)
```

![U-matrix with labels](examples/iris/heat_labels.png)
