"""Freeze a trained GRMHD SOM into an aweSOM model file for chunked inference.

The model carries everything a chunk needs to be labelled the same way as
the training set: the lattice, the node clusters, the scaler, and the
feature thresholds (``b2_min``, ``s_v``, winsor limits, accuracy) copied from
the training file's attributes into the model's metadata
(:meth:`aweSOM.features_grmhd.FeatureThresholds.to_metadata`). Chunks never
recompute them.

Example (the BBH output-0003, it 180224, 10^7-point lattice)::

    python freeze_model.py \\
        --lattice /Volumes/ExternalNVMe/etk-analysis/som_results/som_object.bbh_0003_180224-91-64-0.5-10000000-1u.pkl \\
        --training /Volumes/ExternalNVMe/som-proto-f5/features_bbh_0003-180224-N10000000.h5 \\
        --merge-cost 0.1 -o model.h5

The scaler is refitted on the training features ("manual": mean and two
standard deviations, as ``run_som.manual_scaling`` did before training).
Sheet clusters are those whose training-set mean has ``log_chi_B`` above
``--sheet-chi`` and ``log_beta_inv`` below ``--sheet-beta``; they are
recorded as ``sheet_clusters`` in the metadata.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from aweSOM.features_grmhd import FEATURE_NAMES, FeatureThresholds
from aweSOM.model import Scaler, SOMModel


def sheet_clusters(features: np.ndarray, labels: np.ndarray, chi: float, beta: float):
    """Clusters whose mean log_chi_B > chi and mean log_beta_inv < beta, and every cluster's means."""
    jc, jb = FEATURE_NAMES.index("log_chi_B"), FEATURE_NAMES.index("log_beta_inv")
    means = {}
    for c in np.unique(labels):
        sel = labels == c
        m = np.nanmean(features[sel].astype(np.float64), axis=0)
        means[int(c)] = {"share": float(sel.mean()), **{n: float(v) for n, v in zip(FEATURE_NAMES, m)}}
    sheets = [c for c, m in means.items() if c >= 0 and m["log_chi_B"] > chi and m["log_beta_inv"] < beta]
    return sheets, means


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lattice", required=True, help="pickled aweSOM Lattice (run_som.save_som_object)")
    ap.add_argument("--training", required=True, help="etkplot training-set file the lattice was trained on")
    ap.add_argument("--merge-cost", type=float, required=True, help="centroid merge threshold for the clusters")
    ap.add_argument("--sheet-chi", type=float, default=0.3)
    ap.add_argument("--sheet-beta", type=float, default=0.5)
    ap.add_argument("--check-labels", help="optional .npy of labels of the training rows to compare with")
    ap.add_argument("-o", "--out", required=True, help="model file (.h5)")
    args = ap.parse_args(argv)
    t0 = time.time()

    with h5py.File(args.training, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["names"][()]]
        if tuple(names) != FEATURE_NAMES:
            raise SystemExit(f"{args.training}: features {names} are not {list(FEATURE_NAMES)}")
        raw = f["features"][()]
        thresholds = FeatureThresholds.from_h5_attrs(dict(f.attrs))
        iteration = int(f.attrs.get("iteration", -1))
    print(f"[freeze] {len(raw)} training rows, thresholds {thresholds}", flush=True)

    with open(args.lattice, "rb") as f:
        lattice = pickle.load(f)
    scaler = Scaler.fit(raw, "manual")
    scaled = scaler.transform(raw)
    with contextlib.redirect_stdout(io.StringIO()):
        clusters = lattice.assign_cluster_to_lattice(smoothing=None, merge_cost=args.merge_cost)
    lattice.lattice_assigned_clusters = clusters
    if getattr(lattice, "features_names", None) is None:
        lattice.features_names = list(FEATURE_NAMES)
    meta = {
        "dataset": Path(args.training).name,
        "training_iteration": str(iteration),
        "lattice_file": Path(args.lattice).name,
        "cluster_rule": f"assign_cluster_to_lattice(smoothing=None,merge_cost={args.merge_cost})",
        **thresholds.to_metadata(),
    }
    model = SOMModel.from_lattice(lattice, scaled, scaler, merge_cost=args.merge_cost, metadata=meta)
    del lattice, scaled
    print(f"[freeze] lattice {model.xdim}x{model.ydim}, {len(set(model.node_cluster.tolist()))} clusters, "
          f"{time.time() - t0:.0f}s", flush=True)

    labels = model.predict(raw)
    sheets, means = sheet_clusters(raw, labels, args.sheet_chi, args.sheet_beta)
    model.metadata["sheet_clusters"] = " ".join(map(str, sheets)) or "none"
    model.metadata["sheet_rule"] = f"mean_log_chi_B>{args.sheet_chi},mean_log_beta_inv<{args.sheet_beta}"
    report = {"sheet_clusters": sheets, "cluster_means": means, "seconds": None}
    if args.check_labels:
        ref = np.load(args.check_labels)
        n = min(len(ref), len(labels))
        report["agreement_with_check_labels"] = float(np.mean(ref[:n] == labels[:n]))
        print(f"[freeze] agreement with {args.check_labels}: {report['agreement_with_check_labels']:.6f}")
    model.save(args.out)
    report["seconds"] = time.time() - t0
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"[freeze] sheet clusters {sheets}; wrote {args.out}, {report['seconds']:.0f}s", flush=True)
    for c, m in sorted(means.items()):
        print(f"  c{c:<3d} " + " ".join(f"{k} {v:+.3f}" for k, v in m.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
