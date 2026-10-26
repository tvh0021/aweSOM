import io, contextlib, json, pickle, sys, time
import numpy as np
sys.path.insert(0, "/Users/tvh0021/git_repos/aweSOM-research/src")
from aweSOM import Lattice
from aweSOM.batch import quantization_error, topographic_error
fid = pickle.load(open("/Users/tvh0021/git_repos/aweSOM/examples/plasma-turbulence/som_object.fiducial-5000-64-37-0.1-2097152-1u.pkl", "rb"))
data = np.ascontiguousarray(fid.data_array); names = fid.features_names; xdim, ydim = fid.xdim, fid.ydim
out = {}
def record(tag, som, t):
    with contextlib.redirect_stdout(io.StringIO()):
        c = som.assign_cluster_to_lattice(smoothing=None, merge_cost=0.25)
        lab = som.predict(data)
    fr = np.bincount(lab, minlength=c.max() + 1) / len(lab)
    out[tag] = dict(train_s=t, n_clusters=int(np.unique(c).size), background=float(fr.max()), sheetlike=float(1 - fr.max()),
                    qe=float(quantization_error(som.lattice, data[::8])), te=float(topographic_error(som.lattice, data[::8], xdim, ydim)))
    print(tag, out[tag], flush=True); json.dump(out, open("seed_sweep.json", "w"), indent=1)
for seed in (43, 44, 45, 46):
    som = Lattice(xdim, ydim, 0.1, fid.train, sampling_type="uniform"); som.seed = seed
    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(data, names, history=False)
    record(f"online_seed{seed}", som, time.perf_counter() - t)
for seed, epochs in ((43, 40), (44, 40), (42, 80)):
    som = Lattice(xdim, ydim, 0.1, fid.train, sampling_type="uniform")
    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice_batch(data, names, epochs=epochs, seed=seed)
    record(f"batch_seed{seed}_ep{epochs}", som, time.perf_counter() - t)
print("DONE")
