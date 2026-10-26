"""Compare fiducial (paper), online-retrained and batch-trained lattices on the plasma data."""
import io, contextlib, json, pickle, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/tvh0021/git_repos/aweSOM-research/src")
from aweSOM import Lattice

OUT = "."
PKL = "/Users/tvh0021/git_repos/aweSOM/examples/plasma-turbulence/som_object.fiducial-5000-64-37-0.1-2097152-1u.pkl"
fid = pickle.load(open(PKL, "rb"))
data = np.ascontiguousarray(fid.data_array); names = fid.features_names
xdim, ydim = fid.xdim, fid.ydim
THR = 0.25   # Ha et al. 2024, Appendix A
res = json.load(open("results.json"))

def make(tag):
    som = Lattice(xdim, ydim, fid.alpha_0, fid.train, sampling_type="uniform")
    som.lattice = np.load(f"{tag}_lattice.npy"); som.umat = np.load(f"{tag}_umat.npy")
    som.data_array = data; som.features_names = names
    return som

summary = {}
soms = {}
for tag in ("fiducial", "online", "batch"):
    som = make(tag)
    with contextlib.redirect_stdout(io.StringIO()):
        clusters = som.assign_cluster_to_lattice(smoothing=None, merge_cost=THR)
        labels, dist = som.predict(data, return_distance=True)
    np.save(f"{tag}_clusters_thr025.npy", clusters); np.save(f"{tag}_labels_thr025.npy", labels)
    ks = np.unique(clusters)
    rows = []
    for k in ks:
        m = labels == k
        rows.append(dict(cluster=int(k), fraction=float(m.mean()), mean=data[m].mean(0).round(3).tolist()))
    summary[tag] = dict(n_clusters=int(ks.size), qe=res[tag]["qe"], te=res[tag]["te"], train_s=res[tag].get("train_s"),
                        rows=rows, mean_bmu_dist=float(dist.mean()), p99_dist=float(np.percentile(dist, 99)))
    soms[tag] = (som, clusters, labels)
    print(tag, "clusters", ks.size, "QE", round(res[tag]["qe"], 4), "TE", round(res[tag]["te"], 5))
    for r in rows: print(f"   {r['cluster']}: {r['fraction']*100:5.1f}%  mean={r['mean']}")
json.dump(summary, open("summary_thr025.json", "w"), indent=1)

# Figure 1: U-matrices with cluster boundaries and centroids
fig, axs = plt.subplots(1, 3, figsize=(16, 3.6), dpi=150, constrained_layout=True)
titles = {"fiducial": "Fiducial lattice (Ha et al. 2024 pickle)", "online": "Online trainer, retrained (31 s)", "batch": "Batch trainer, 40 epochs"}
for ax, tag in zip(axs, titles):
    som, clusters, labels = soms[tag]
    im = ax.pcolormesh(som.umat.T, cmap="viridis", shading="auto")
    X, Y = np.meshgrid(np.arange(xdim) + 0.5, np.arange(ydim) + 0.5)
    ax.contour(X, Y, clusters.T, levels=np.arange(clusters.max() + 1) + 0.5, colors="white", linewidths=0.8)
    cents = som.get_unique_centroids(som.compute_centroids())
    ax.scatter(np.array(cents["position_x"]) + 0.5, np.array(cents["position_y"]) + 0.5, c="red", s=12, zorder=3)
    ax.set_aspect("equal"); ax.set_title(f"{titles[tag]}\n{np.unique(clusters).size} clusters, QE={summary[tag]['qe']:.4f}, TE={summary[tag]['te']:.4f}", fontsize=9)
    ax.set_xlabel("x node"); ax.set_ylabel("y node")
    plt.colorbar(im, ax=ax, fraction=0.03, label="U-matrix")
fig.savefig("fig_umatrices.png"); plt.close(fig)

# Figure 2: xy-slice of the cluster labels at mid z (128^3 cube, C order as in plasma.ipynb reshape (nz, ny, nx))
L = 128
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150, constrained_layout=True)
for ax, tag in zip(axs, titles):
    som, clusters, labels = soms[tag]
    cube = labels.reshape(L, L, L)
    nbg = np.argmax(np.bincount(labels))  # background = most populated cluster
    sl = cube[L // 2].astype(float); sl[sl == nbg] = np.nan
    ax.imshow(sl, origin="lower", cmap="tab10", vmin=-0.5, vmax=9.5, interpolation="nearest")
    jslice = data[:, 0].reshape(L, L, L)[L // 2]
    ax.contour(jslice, levels=[-1.5, 1.5], colors=["blue", "red"], linewidths=0.4, alpha=0.6)
    ax.set_title(f"{titles[tag]}\nnon-background labels at z = {L//2}; contours: j_par = ±1.5σ", fontsize=9)
    ax.set_xlabel("x cell"); ax.set_ylabel("y cell")
fig.savefig("fig_slices.png"); plt.close(fig)

# Figure 3: volume fractions per cluster, sorted by fraction, colour by sign of mean j_par
fig, axs = plt.subplots(1, 3, figsize=(16, 3.4), dpi=150, constrained_layout=True)
for ax, tag in zip(axs, titles):
    rows = sorted(summary[tag]["rows"], key=lambda r: -r["fraction"])
    fr = [r["fraction"] * 100 for r in rows]; lab = [f"c{r['cluster']}" for r in rows]
    cols = ["#9aa5b1" if f > 50 else ("#c0392b" if r["mean"][0] > 0.3 else ("#2e6db4" if r["mean"][0] < -0.3 else "#27766e")) for f, r in zip(fr, rows)]
    ax.bar(lab, fr, color=cols); ax.set_yscale("log"); ax.set_ylim(0.3, 100); ax.set_ylabel("volume fraction [%]")
    for i, f in enumerate(fr): ax.text(i, f * 1.15, f"{f:.1f}", ha="center", fontsize=8)
    ax.set_title(f"{titles[tag]}\nred: mean j_par > 0.3σ, blue: < −0.3σ, teal: |j_par| small, grey: background", fontsize=8)
fig.savefig("fig_fractions.png"); plt.close(fig)
print("figures written")
