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

Here are the basic steps to initialize a lattice and train the SOM to classify the Iris dataset

```python
import numpy as np
import matplotlib.pyplot as plt
from aweSOM import Lattice
```

First, load the dataset and normalize

```python
from sklearn.datasets import load_iris
iris = load_iris()

print("Shape of the data :", iris.data.shape)
print("Labeled classes :", iris.target_names)
print("Features in the set :", iris.feature_names)
```

Shape of the data : (150, 4)

Labeled classes : ['setosa' 'versicolor' 'virginica']

Features in the set : ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


Normalize the data with a custom scaler

```python
import aweSOM.run_som as rs
iris_data_transformed = rs.manual_scaling(iris.data)
```
Initilize the lattice and train

```python
# Create an initial SOM instance
map = Lattice(xdim=40, ydim=15, alpha_0=0.5, train=100000)

# Train the SOM with some data in the shape of $N \times F$.
true_labels = iris.target
feature_names = iris.feature_names
map.train_lattice(iris_data_transformed, feature_names, true_labels)
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

# plot U-matrix with the connected components and ground truth labels (if the labels were supplied during map.train_lattice)
map.plot_heat(map.umat, merge=True, merge_cost=merge_threshold)
```

![U-matrix with labels](examples/iris/heat_labels.png)

Now, we project each data point onto the lattice and get back cluster-id

```python
projection_2d = map.map_data_to_lattice()
final_clusters = map.assign_cluster_to_lattice(smoothing=None,merge_cost=merge_threshold)
som_labels = map.assign_cluster_to_data(projection_2d, final_clusters)
```

Finally, we compare the aweSOM result to the ground truth

```python
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
scatter_ground = axs[0].scatter(iris.data[:,1], iris.data[:,2], c=iris.target, cmap='viridis')
axs[0].set_xlabel('Sepal Width')
axs[0].set_ylabel('Petal Length')
axs[0].legend(scatter_ground.legend_elements()[0], iris.target_names, loc="upper right", title="Classes")
scatter_som = axs[1].scatter(iris.data[:,1], iris.data[:,2], c=som_labels, cmap='viridis')
axs[1].set_xlabel('Sepal Width')
axs[1].set_ylabel('Petal Length')
axs[1].legend(scatter.legend_elements()[0], np.unique(final_clusters), loc="upper right", title="aweSOM")
plt.show()
```

![Scatter plot comparing ground truth with aweSOM clusters](examples/iris/scatter_res.png)

Clearly, the mapping is: {'setosa' : 2, 'versicolor' : 1, 'virginica' : 0} 

```python
# Assign cluster number to class label; change manually
label_map = {'setosa' : 2, 'versicolor' : 1, 'virginica' : 0}
correct_label = 0

for i in range(len(som_labels)):
    if int(som_labels[i]) == label_map[iris.target_names[iris.target[i]]]:
        correct_label += 1

print("Number of correct predictions: ", correct_label)
print("Accuracy = ", correct_label/len(som_labels) * 100, "%")

# Precision and Recall by class
precision = np.zeros(3)
recall = np.zeros(3)

for i in range(3):
    tp = 0
    fp = 0
    fn = 0
    for j in range(len(som_labels)):
        if int(som_labels[j]) == label_map[iris.target_names[i]]:
            if iris.target[j] == i:
                tp += 1
            else:
                fp += 1
        else:
            if iris.target[j] == i:
                fn += 1
    precision[i] = tp/(tp+fp)
    recall[i] = tp/(tp+fn)

print("Precision: ", [float(np.round(precision[i],4))*100 for i in range(3)], "%")
print("Recall: ", [float(np.round(recall[i],4))*100 for i in range(3)], "%")
```

Number of correct predictions:  141

Accuracy =  94.0 %

Precision:  [100.0, 90.2, 91.84] %

Recall:  [100.0, 92.0, 90.0] %

Is the performance of the aweSOM model.
