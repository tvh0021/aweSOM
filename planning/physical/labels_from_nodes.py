"""Write full-grid labels from the cached BMU nodes of score.py (no rescoring).

    python labels_from_nodes.py --it 135168 --mc 0.1 TAG [TAG ...]
"""

import argparse
import json

import numpy as np

from common import MODELS
from score import LABELS, NODES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--it", type=int, required=True)
    ap.add_argument("--mc", type=float, required=True)
    ap.add_argument("tags", nargs="+")
    a = ap.parse_args()
    for t in a.tags:
        model = np.load(MODELS / f"{t}.npz")
        mcs = json.loads(str(model["info"]))["merge_costs"]
        nc = np.append(model["node_cluster"][mcs.index(a.mc)], -1).astype(np.int16)
        nodes = np.load(NODES / f"{t}__it{a.it}.npy").astype(np.int32)
        np.save(LABELS / f"{t}__it{a.it}_mc{a.mc}.npy", nc[nodes])
        print(t, "K =", int(nc.max()) + 1)


if __name__ == "__main__":
    main()
