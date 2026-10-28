"""Freeze a trained aweSOM Lattice (pickle) into a pickle-free SOMModel for
the GRMHD BBH sheet comparison, and pick the current-sheet clusters.

The scaler is frozen from the training file. Two choices:

* ``--scaler training-float32`` (default): ``mean`` and ``2 std`` of the
  training features computed the way ``run_som.manual_scaling`` computed them
  when the lattice was trained (NumPy reductions on the float32 array). This
  reproduces the lattice's ``data_array`` to float32 rounding, so the model
  assigns the training rows the labels saved at training time.
* ``--scaler file``: the float64 ``scaler/mean`` and ``scaler/sigma`` stored in
  the training file. NumPy's float32 reductions along axis 0 accumulate in
  float32, so these differ from what the lattice saw (by up to 0.04 in scaled
  units for the 10^7-row BBH file); use only for a sensitivity check.

Sheet clusters are chosen by a rule on each cluster's mean features over the
training rows (default ``log_chi_B>0.3,log_beta_inv<0.5``, the criterion of
``etk-analysis/som_results/README.md`` section 9), or given with ``--sheet-clusters``.

    python build_model.py --lattice som_object.<tag>.pkl \
        --training-file features_bbh_0003-180224-N10000000.h5 --merge-cost 0.05 \
        --training-labels labels.<tag>-mc0.05.npy -o som_model.h5

A lattice trained with ``train_som.py --n-samples N`` saw only the first N rows of
the file; pass the same ``--n-samples N`` here so the scaler and the stored
``projection_1d`` refer to the same rows.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import time
from pathlib import Path

import h5py
import numpy as np

import aweSOM
from aweSOM.model import Scaler, SOMModel


def parse_rule(text):
    """'a>0.3,b<0.5' -> [(a, '>', 0.3), (b, '<', 0.5)]"""
    out = []
    for part in text.split(","):
        part = part.strip()
        for op in (">", "<"):
            if op in part:
                name, val = part.split(op)
                out.append((name.strip(), op, float(val)))
                break
        else:
            raise ValueError(f"cannot parse rule term {part!r}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lattice", required=True, type=Path, help="Lattice pickle from training")
    ap.add_argument("--training-file", required=True, type=Path, help="feature HDF5 the lattice was trained on")
    ap.add_argument("--merge-cost", type=float, required=True)
    ap.add_argument("--n-samples", type=int, default=None,
                    help="use only the first N rows of the training file, as train_som.py "
                         "--n-samples did when the lattice was trained (default: all rows)")
    ap.add_argument("--scaler", choices=["training-float32", "file"], default="training-float32")
    ap.add_argument("--sheet-rule", default="log_chi_B>0.3,log_beta_inv<0.5")
    ap.add_argument("--sheet-clusters", default=None, help="comma-separated ids (overrides the rule)")
    ap.add_argument("--training-labels", type=Path, default=None,
                    help="labels saved at training time with this merge cost, to check against")
    ap.add_argument("--name", default=None, help="dataset name stored in the metadata")
    ap.add_argument("-o", "--out", required=True, type=Path, help="output SOMModel (.h5)")
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    print(f"aweSOM from {aweSOM.__file__}", flush=True)

    with open(args.lattice, "rb") as fh:
        lat = pickle.load(fh)
    with h5py.File(args.training_file, "r") as h:
        n_use = h["features"].shape[0] if args.n_samples is None else min(args.n_samples, h["features"].shape[0])
        X = h["features"][:n_use]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in h["names"][()]]
        file_mean = h["scaler/mean"][:] if "scaler/mean" in h else None
        file_sigma = h["scaler/sigma"][:] if "scaler/sigma" in h else None
        weight = h["weight"][:n_use] if "weight" in h else np.ones(len(X))
        weight_volume = h["weight_volume"][:n_use] if "weight_volume" in h else np.ones(len(X))
        level = h["level"][:n_use] if "level" in h else np.zeros(len(X), np.int8)

    report = {"lattice": str(args.lattice), "training_file": str(args.training_file), "n_samples": int(n_use),
              "merge_cost": args.merge_cost, "scaler_mode": args.scaler, "feature_names": names}
    sc32 = Scaler("manual", X.mean(0).astype(np.float64), 2.0 * X.std(0).astype(np.float64))
    if file_mean is not None:
        scf = Scaler("manual", file_mean, 2.0 * file_sigma)
        report["scaler_float32_minus_file_offset"] = (sc32.offset - scf.offset).tolist()
        report["scaler_float32_over_file_scale"] = (sc32.scale / scf.scale).tolist()
    scaler = sc32 if args.scaler == "training-float32" else scf
    da = getattr(lat, "data_array", None)
    if da is not None and da.shape == X.shape:
        report["max_abs_scaled_minus_lattice_data_array"] = np.abs(scaler.transform(X) - da).max(0).tolist()
        print("scaled training data vs lattice.data_array, max abs per feature:",
              report["max_abs_scaled_minus_lattice_data_array"], flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        clusters = lat.assign_cluster_to_lattice(smoothing=None, merge_cost=args.merge_cost)
    node_cluster = clusters.ravel(order="F").astype(np.int64)
    n_clusters = int(np.unique(node_cluster).size)
    print(f"merge_cost {args.merge_cost}: {n_clusters} clusters", flush=True)

    proj = getattr(lat, "projection_1d", None)
    if proj is not None and len(proj) == len(X):
        lab = node_cluster[np.asarray(proj).ravel().astype(np.int64)]
    else:
        raise SystemExit("the lattice has no projection_1d of the training rows"
                         + ("" if proj is None else f" (lattice rows {len(proj)}, training rows {len(X)}; "
                            "pass --n-samples to match the subsample the lattice was trained on)"))
    if args.training_labels is not None:
        stored = np.load(args.training_labels)
        report["agreement_with_training_labels"] = float(np.mean(stored == lab))
        print(f"labels from the re-assigned lattice agree with {args.training_labels.name}: "
              f"{report['agreement_with_training_labels']:.6f}", flush=True)

    table = {}
    for c in range(n_clusters):
        m = lab == c
        table[c] = {
            "share_sample": float(m.mean()),
            "share_points": float(weight[m].sum() / weight.sum()),
            "share_volume": float(weight_volume[m].sum() / weight_volume.sum()),
            "share_sample_per_level": {int(L): float(m[level == L].mean()) for L in np.unique(level)},
            "mean": {n: float(X[m, k].mean()) for k, n in enumerate(names)},
        }
    if args.sheet_clusters:
        sheet = sorted(int(c) for c in args.sheet_clusters.split(","))
        rule_text = "given"
    else:
        rule = parse_rule(args.sheet_rule)
        sheet = [c for c, row in table.items()
                 if all((row["mean"][n] > v) if op == ">" else (row["mean"][n] < v) for n, op, v in rule)]
        rule_text = args.sheet_rule
    print(f"sheet clusters ({rule_text}): {sheet}", flush=True)
    for c, row in table.items():
        mark = "*" if c in sheet else " "
        print(f"{mark}{c:3d} sample {row['share_sample']:7.3%} points {row['share_points']:7.3%} volume "
              f"{row['share_volume']:7.3%}  " + " ".join(f"{n}={v:+.2f}" for n, v in row["mean"].items()), flush=True)

    t0 = time.perf_counter()
    meta = {
        "dataset": args.name or args.lattice.stem,
        "training_file": str(args.training_file),
        "lattice_pickle": str(args.lattice),
        "scaler_mode": args.scaler,
        "sheet_clusters": ",".join(map(str, sheet)),
        "sheet_rule": rule_text,
    }
    from aweSOM.features_grmhd import FeatureThresholds
    meta.update(FeatureThresholds.from_h5(args.training_file).to_metadata())
    model = SOMModel.from_lattice(lat, scaled_training_data=da, scaler=scaler,
                                  merge_cost=args.merge_cost, metadata=meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    report.update({"sheet_clusters": sheet, "sheet_rule": rule_text, "n_clusters": n_clusters,
                   "clusters": table, "model": str(args.out), "reject_distance": model.reject_distance,
                   "freeze_seconds": round(time.perf_counter() - t0, 1),
                   "total_seconds": round(time.perf_counter() - t_start, 1)})
    side = args.out.with_suffix(".json")
    side.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.out} and {side} ({report['total_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
