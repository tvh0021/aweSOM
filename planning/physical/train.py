"""Train one aweSOM lattice for the study and cluster it at every merge cost.

    python train.py --variant base5 --n 1000000 --lattice rule --alpha 0.5 \
        --steps 1 --seed 0 [--it 180224]

Rows: the first N rows of the shuffled training sample of iteration ``it``
(features taken from the grid column store, identical to the sample's own).
Scaling: (x - mean) / (2 sigma) of those rows (aweSOM manual scaling).
"""

import argparse
import contextlib
import io
import json
import subprocess
import time

import numpy as np

from common import COLS, MERGE_COSTS, MODELS, VARIANTS, col, limit_threads

limit_threads(2)
import aweSOM  # noqa: E402
from aweSOM import Lattice  # noqa: E402
from aweSOM.run_som import number_of_nodes  # noqa: E402


def lattice_dims(n, f, kind, ratio=0.7):
    nodes = number_of_nodes(n, f)
    if kind == "half":
        nodes //= 2
    elif kind == "quarter":
        nodes //= 4
    elif kind == "double":
        nodes *= 2
    xdim = int(np.sqrt(nodes / ratio))
    return xdim, int(nodes / xdim)


def tag_of(a):
    return f"{a.variant}-N{a.n:g}-{a.lattice}-a{a.alpha:g}-s{a.steps:g}-seed{a.seed}-it{a.it}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--lattice", default="rule", choices=["rule", "half", "quarter", "double"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--steps", type=float, default=1.0, help="training steps / N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--it", type=int, default=180224)
    a = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    tag = tag_of(a)
    names = VARIANTS[a.variant]
    t0 = time.time()
    rows = np.load(COLS / f"{a.it}_sample_to_grid.npy", mmap_mode="r")[: a.n]
    order = np.argsort(rows)
    x = np.empty((a.n, len(names)))
    for j, nm in enumerate(names):
        c = col(a.it, nm)
        x[order, j] = c[rows[order]]
    good = np.isfinite(x).all(1)
    n_dropped = int((~good).sum())
    if n_dropped:  # C_B is NaN where its 5^3 box leaves the covered region
        x = x[good]
    mean = x.mean(0)
    scale = 2 * x.std(0)
    xs = (x - mean) / scale
    t_load = time.time() - t0
    xdim, ydim = lattice_dims(a.n, len(names), a.lattice)
    steps = int(round(a.steps * a.n))
    lat = Lattice(xdim=xdim, ydim=ydim, alpha_0=a.alpha, train=steps, alpha_type="decay", sampling_type="uniform")
    lat.seed = a.seed
    up0 = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    t1 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        lat.train_lattice(xs, names, history=False)
    t_train = time.time() - t1
    t2 = time.time()
    ncs, nk = [], []
    with contextlib.redirect_stdout(io.StringIO()):
        for mc in MERGE_COSTS:
            cl = lat.assign_cluster_to_lattice(smoothing=None, merge_cost=mc)
            nc = np.asarray(cl).ravel(order="F").astype(np.int16)  # node index = x + y * xdim
            ncs.append(nc)
            nk.append(int(np.unique(nc).size))
    t_clust = time.time() - t2
    info = dict(
        tag=tag, variant=a.variant, names=names, n=a.n, n_dropped=n_dropped, lattice=a.lattice, xdim=xdim, ydim=ydim,
        alpha=a.alpha, steps=steps, seed=a.seed, it=a.it, merge_costs=list(MERGE_COSTS), n_clusters=nk,
        t_load=t_load, t_train=t_train, t_cluster=t_clust, uptime_start=up0, awesom_file=aweSOM.__file__,
        uptime_end=subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
    )
    np.savez(MODELS / f"{tag}.npz", weights=lat.lattice, umat=lat.umat, mean=mean, scale=scale,
             node_cluster=np.stack(ncs), info=json.dumps(info))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
