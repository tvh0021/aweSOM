"""Map every grid row of a som-features grid file to its SOM best-matching node,
then to cluster labels at several merge costs.

    PYTHONPATH=<worktree>/src python predict_grid.py --som som_object...pkl \
        --features features_bbh_0003-180224-N10000000.h5 --grid grid_LR_flux_180224.h5 \
        --train-labels-pattern '.../labels.bbh_0003_180224-91-64-0.5-10000000-1u{suffix}.npy' \
        --out-dir .

The BMU search runs once (``SOMModel.predict`` with node ids as "clusters",
backend "screen"); labels for each merge cost are a lookup of the node's
cluster. Each merge cost's node -> cluster map is checked against the saved
training labels on a subsample of the training rows.
"""

import argparse
import contextlib
import io
import json
import pickle
import time

import h5py
import numpy as np

import aweSOM
from aweSOM.model import Scaler, SOMModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--som", required=True)
    ap.add_argument("--features", required=True, help="training-set file with scaler/mean, scaler/sigma")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--merge-costs", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    ap.add_argument("--train-labels-pattern", default=None,
                    help="format string with {suffix}: '' for 0.2, '-mc<cost>' otherwise")
    ap.add_argument("--n-check", type=int, default=200_000)
    ap.add_argument("--scaler", choices=["training-float32", "stored"], default="training-float32",
                    help="'training-float32' recomputes manual_scaling's float32 mean/std from "
                         "the training features (what the lattice saw); 'stored' uses scaler/*")
    ap.add_argument("--chunk", type=int, default=1 << 20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="LR_flux_180224")
    args = ap.parse_args()
    print("aweSOM from", aweSOM.__file__, flush=True)
    t0 = time.time()

    with open(args.som, "rb") as f:
        lattice = pickle.load(f)
    xdim, ydim = int(lattice.xdim), int(lattice.ydim)
    weights = np.asarray(lattice.lattice, dtype=np.float64)
    print(f"lattice {xdim}x{ydim}, loaded in {time.time() - t0:.0f}s", flush=True)

    with h5py.File(args.features, "r") as f:
        stored_mean, stored_sigma = f["scaler/mean"][:], f["scaler/sigma"][:]
        names = [n.decode() for n in f["names"][:]]
        rng = np.random.default_rng(0)
        train = f["features"][:]
    n_train = train.shape[0]
    check_rows = np.sort(rng.choice(n_train, size=min(args.n_check, n_train), replace=False))
    train_check = train[check_rows]
    if args.scaler == "stored":
        mean, sigma = stored_mean, stored_sigma
    else:
        # train_som.py called run_som.manual_scaling on the float32 array, so
        # np.mean / np.std accumulated in float32; those are the statistics the
        # lattice was trained with (they differ from the stored float64 scaler
        # by up to 1% in sigma).
        mean = np.mean(train, axis=0).astype(np.float64)
        sigma = np.std(train, axis=0).astype(np.float64)
    data_array = getattr(lattice, "data_array", None)
    scaling_check = None
    if data_array is not None:
        k = min(100_000, n_train)
        mine = ((train[:k] - mean.astype(np.float32)) / (2.0 * sigma.astype(np.float32)))
        scaling_check = float(np.max(np.abs(mine - np.asarray(data_array[:k]))))
        print(f"scaler '{args.scaler}': max |scaled - lattice.data_array| on {k} rows "
              f"= {scaling_check:.3g}", flush=True)
    del train
    scaler = Scaler("manual", offset=mean, scale=2.0 * sigma)

    # node ids as "clusters" -> predict returns the BMU node
    node_model = SOMModel(xdim=xdim, ydim=ydim, weights=weights,
                          node_cluster=np.arange(xdim * ydim, dtype=np.int32),
                          feature_names=names, scaler=scaler,
                          node_dist_q99=np.full(xdim * ydim, np.nan), reject_distance=np.inf,
                          metadata={"note": "node-id model for flux-tube labels"})

    maps = {}
    check = {}
    train_nodes = node_model.predict(train_check, backend="screen")
    for mc in args.merge_costs:
        with contextlib.redirect_stdout(io.StringIO()):
            clusters = lattice.assign_cluster_to_lattice(smoothing=None, merge_cost=mc)
        node_cluster = np.asarray(clusters).ravel(order="F").astype(np.int32)
        maps[mc] = node_cluster
        if args.train_labels_pattern:
            suffix = "" if mc == 0.2 else f"-mc{mc:g}"
            saved = aweSOM.open_npy(args.train_labels_pattern.format(suffix=suffix), mmap_mode="r")
            agree = float(np.mean(node_cluster[train_nodes] == np.asarray(saved[check_rows])))
            check[f"{mc:g}"] = agree
            print(f"merge_cost {mc:g}: {len(np.unique(node_cluster))} clusters, "
                  f"agreement with saved training labels on {len(check_rows)} rows: {agree:.6f}",
                  flush=True)

    with h5py.File(args.grid, "r") as g:
        n = g["features"].shape[0]
        nodes = np.empty(n, dtype=np.int16)
        t1 = time.time()
        step = 8 * args.chunk
        for a in range(0, n, step):
            b = min(a + step, n)
            nodes[a:b] = node_model.predict(g["features"][a:b], chunk=args.chunk, backend="screen")
        t_pred = time.time() - t1
    print(f"predicted {n} rows in {t_pred:.0f}s", flush=True)
    np.save(f"{args.out_dir}/bmu_nodes_{args.tag}.npy", nodes)
    for mc, node_cluster in maps.items():
        np.save(f"{args.out_dir}/labels_{args.tag}_mc{mc:g}.npy", node_cluster[nodes])
        np.save(f"{args.out_dir}/node_cluster_mc{mc:g}.npy", node_cluster)
    with open(f"{args.out_dir}/predict_grid.json", "w") as f:
        json.dump({"som": args.som, "grid": args.grid, "n_rows": int(n), "backend": "screen",
                   "predict_seconds": t_pred, "total_seconds": time.time() - t0,
                   "training_label_agreement": check, "aweSOM": aweSOM.__file__,
                   "scaler": args.scaler, "scaling_check_max_abs": scaling_check,
                   "scaler_offset": mean.tolist(), "scaler_scale": (2 * sigma).tolist(),
                   "stored_mean": stored_mean.tolist(), "stored_sigma": stored_sigma.tolist()},
                  f, indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
