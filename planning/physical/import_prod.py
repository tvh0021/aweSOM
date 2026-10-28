"""Import the production 91x64 base5 lattice (etk-analysis som_results) as a study model.

    python import_prod.py

The production lattice was trained on float32-scaled inputs: its mean and
2*sigma came from aweSOM manual scaling in float32, which differ by up to 1%
from the float64 scaler stored with the training file. Applying the stored
float64 scaler changes 0.6-1% of labels, so the float32 statistics
(flux-tubes/training_scaler_float32.npz) are used here.
Writes models/prod-N1e+07-rule-a0.5-s1-seed0-it180224.npz in the train.py format.
"""

import contextlib
import io
import json
import pickle
from pathlib import Path

import numpy as np

from common import MERGE_COSTS, MODELS, VARIANTS

import aweSOM  # noqa: F401

SOM = Path("/Volumes/ExternalNVMe/etk-analysis/som_results/som_object.bbh_0003_180224-91-64-0.5-10000000-1u.pkl")
SCALER = Path("/Volumes/ExternalNVMe/awesom-results/2026-10-28/flux-tubes/training_scaler_float32.npz")


def main():
    with open(SOM, "rb") as fh:
        lat = pickle.load(fh)
    sc = np.load(SCALER)
    mean = np.asarray(sc["mean"], np.float32).astype(np.float64)
    scale = (2 * np.asarray(sc["sigma"], np.float32)).astype(np.float64)
    ncs, nk = [], []
    with contextlib.redirect_stdout(io.StringIO()):
        for mc in MERGE_COSTS:
            cl = lat.assign_cluster_to_lattice(smoothing=None, merge_cost=mc)
            nc = np.asarray(cl).ravel(order="F").astype(np.int16)
            ncs.append(nc)
            nk.append(int(np.unique(nc).size))
    tag = "prod-N1e+07-rule-a0.5-s1-seed0-it180224"
    info = dict(tag=tag, variant="prod", names=VARIANTS["base5"], n=10_000_000, n_dropped=0, lattice="rule",
                xdim=int(lat.xdim), ydim=int(lat.ydim), alpha=0.5, steps=10_000_000, seed=0, it=180224,
                merge_costs=list(MERGE_COSTS), n_clusters=nk, t_train=3134.19, source=str(SOM),
                scaler=str(SCALER), scaler_dtype="float32")
    np.savez(MODELS / f"{tag}.npz", weights=lat.lattice, umat=lat.umat, mean=mean, scale=scale,
             node_cluster=np.stack(ncs), info=json.dumps(info))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
