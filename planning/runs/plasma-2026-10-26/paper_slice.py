"""xy-slice at z = 90 c/omega_p (index 27), the slice shown in Figure 1 of Ha et al. 2024, with the paper's colouring."""
import pickle, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

PKL = "/Users/tvh0021/git_repos/aweSOM/examples/plasma-turbulence/som_object.fiducial-5000-64-37-0.1-2097152-1u.pkl"
fid = pickle.load(open(PKL, "rb"))
L = 128; DX = 10 / 3; Z = 27          # z = 90 c/omega_p
data = fid.data_array
jpar = data[:, 0].reshape(L, L, L)     # (nz, ny, nx) as in plasma.ipynb
jrms = np.sqrt(np.mean(jpar ** 2))
ticks = np.arange(0, L, 30); ticklab = (ticks * DX).astype(int)
tab10 = plt.get_cmap("tab10")
LANDMARKS = [((90, 400), "single sheet"), ((350, 150), "double sheet")]   # paper, Section 4.1, in c/omega_p

def axes_setup(ax, title):
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    ax.set_xticks(ticks, ticklab); ax.set_yticks(ticks, ticklab)
    ax.set_xlabel(r"x ($c/\omega_p$)"); ax.set_ylabel(r"y ($c/\omega_p$)"); ax.grid(alpha=0.3)
    for (x, y), name in LANDMARKS:
        ax.add_patch(Circle((x / DX, y / DX), 9, fill=False, color="k", lw=1.2, ls="--"))
        ax.annotate(name, (x / DX, y / DX - 11), ha="center", va="top", fontsize=7)

def contours(ax):
    ax.contour(np.arange(L) + 0.5, np.arange(L) + 0.5, jpar[Z], levels=[-2 * jrms, 2 * jrms],
               colors=["red", "blue"], linewidths=0.5, linestyles="solid")

import json
summ = json.load(open("summary_thr025.json"))
fid_means = {r["cluster"]: np.array(r["mean"]) for r in summ["fiducial"]["rows"]}
PAPER_NAME = {0: "anti-aligned sheet", 1: "double-sheet (anti)", 2: "double-sheet (aligned)", 3: "e_par < 0", 4: "aligned sheet", 5: "e_par > 0", 6: "background"}
def paper_map(tag):
    """cluster id of `tag` -> paper cluster id (nearest fiducial mean; the largest cluster is background 6)."""
    rows = summ[tag]["rows"]; bg = max(rows, key=lambda r: r["fraction"])["cluster"]
    out = {}
    for r in rows:
        if r["cluster"] == bg: out[r["cluster"]] = 6; continue
        cand = {k: np.linalg.norm(np.array(r["mean"]) - v) for k, v in fid_means.items() if k != 6}
        out[r["cluster"]] = min(cand, key=cand.get)
    return out

maps = {"fiducial": "Fiducial SOM (paper pickle), 7 clusters\ncolours as paper Fig. 1c; cluster 6 (background) white",
        "online": "Online retrain, seed 42, 4 clusters, recoloured to nearest paper cluster",
        "batch": "Batch retrain, 40 epochs, 5 clusters, recoloured to nearest paper cluster"}

fig, axs = plt.subplots(2, 2, figsize=(14, 11.5), dpi=160, constrained_layout=True)
ax = axs[0, 0]
im = ax.pcolormesh(jpar[Z], cmap="RdBu", vmin=-1.5, vmax=1.5, rasterized=True)
fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03, label=r"$j_\parallel$ (standardized, $\sigma$ units)")
contours(ax)
axes_setup(ax, r"$j_\parallel$ at z = 90 $c/\omega_p$ (slice 27), paper Fig. 1b" + "\ncontours: red $j_\\parallel<-2j_{\\rm rms}$, blue $j_\\parallel>2j_{\\rm rms}$")
for ax, (tag, title) in zip(axs.flat[1:], maps.items()):
    lab = np.load(f"{tag}_labels_thr025.npy").reshape(L, L, L)[Z]
    pm = paper_map(tag); lut = np.full(max(pm) + 1, 6); 
    for k, v in pm.items(): lut[k] = v
    lab = lut[lab]                                      # relabel into paper cluster ids
    cols = [tab10(k) for k in range(7)]; cols[6] = (1, 1, 1, 1)
    im = ax.pcolormesh(lab, cmap=ListedColormap(cols), vmin=-0.5, vmax=6.5, rasterized=True)
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03, ticks=range(7), label="paper cluster id (colour)")
    cb.ax.set_yticklabels([f"{k}: {PAPER_NAME[k]}" for k in range(7)], fontsize=7)
    if tag != "fiducial":
        title += "\n" + ", ".join(f"c{k}→{v}" for k, v in sorted(pm.items()) if v != 6)
    contours(ax); axes_setup(ax, title)
fig.suptitle("xy-slice at z = 90 $c/\\omega_p$ as in Ha et al. 2024, Figure 1; dashed circles mark the single sheet (x=90, y=400) and double sheet (x=350, y=150) named in Section 4.1", fontsize=10)
fig.savefig("fig_paper_slice_z27.png")
print("fiducial fractions in this slice:", np.round(np.bincount(np.load("fiducial_labels_thr025.npy").reshape(L,L,L)[Z].ravel(), minlength=7) / L**2 * 100, 1))
