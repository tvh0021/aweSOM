"""Retrain the fiducial plasma SOM (2,097,152 points, 4 features) with the online and batch trainers."""
import io, contextlib, json, pickle, sys, time
import numpy as np
sys.path.insert(0, "/Users/tvh0021/git_repos/aweSOM-research/src")
from aweSOM import Lattice
from aweSOM.batch import train_batch, quantization_error, topographic_error

OUT = "/Users/tvh0021/git_repos/aweSOM-research/planning/runs/plasma-2026-10-26"
PKL = "/Users/tvh0021/git_repos/aweSOM/examples/plasma-turbulence/som_object.fiducial-5000-64-37-0.1-2097152-1u.pkl"
fid = pickle.load(open(PKL, "rb"))
data = np.ascontiguousarray(fid.data_array, dtype=np.float64)
names = fid.features_names
xdim, ydim, alpha0, train = fid.xdim, fid.ydim, fid.alpha_0, fid.train
print("data", data.shape, names, xdim, ydim, alpha0, train, fid.init, flush=True)
np.save(f"{OUT}/fiducial_lattice.npy", fid.lattice); np.save(f"{OUT}/fiducial_umat.npy", fid.umat)
np.save(f"{OUT}/fiducial_clusters.npy", fid.lattice_assigned_clusters)
results = {}

def finish(tag, som, t_train):
    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        clusters = som.assign_cluster_to_lattice(smoothing=None, merge_cost=0.2)
        labels, dist = som.predict(data, return_distance=True)
    t_pred = time.perf_counter() - t
    t = time.perf_counter()
    qe = quantization_error(som.lattice, data); te = topographic_error(som.lattice, data, xdim, ydim)
    t_q = time.perf_counter() - t
    np.save(f"{OUT}/{tag}_lattice.npy", som.lattice); np.save(f"{OUT}/{tag}_umat.npy", som.umat)
    np.save(f"{OUT}/{tag}_clusters.npy", clusters); np.save(f"{OUT}/{tag}_labels.npy", labels)
    np.save(f"{OUT}/{tag}_dist.npy", dist.astype(np.float32))
    results[tag] = dict(train_s=t_train, cluster_and_predict_s=t_pred, qe_te_s=t_q, qe=float(qe), te=float(te),
                        n_clusters=int(np.unique(clusters).size), n_centroids=len(som.nodes_count),
                        label_fractions=np.bincount(labels, minlength=int(clusters.max())+1).tolist())
    print(tag, results[tag], flush=True)
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)

# 0. fiducial (pickled) lattice: quality only
t = time.perf_counter()
results["fiducial"] = dict(qe=float(quantization_error(fid.lattice, data)), te=float(topographic_error(fid.lattice, data, xdim, ydim)),
                           n_clusters=int(np.unique(fid.lattice_assigned_clusters).size), n_centroids=len(fid.nodes_count),
                           qe_te_s=time.perf_counter() - t)
print("fiducial", results["fiducial"], flush=True)

# 1. online trainer, same configuration as the fiducial run, history off
som = Lattice(xdim, ydim, alpha0, train, sampling_type=fid.init)
t = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    som.train_lattice(data, names, history=False)
finish("online", som, time.perf_counter() - t)

# 2. batch trainer, 40 epochs, all CPU devices
som_b = Lattice(xdim, ydim, alpha0, train, sampling_type=fid.init)
t = time.perf_counter()
info = som_b.train_lattice_batch(data, names, epochs=40, verbose=True)
t_b = time.perf_counter() - t
json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in info.items()}, open(f"{OUT}/batch_info.json", "w"))
finish("batch", som_b, t_b)
print("DONE", flush=True)
