"""Replay the top-two sheet selection against the K run (jobs 8014/8047) and compare its first
pick with the previous first-candidates-only rule in all eight seed groups (4 seed sets x 2
lattices). Reads the K pool and labels only; writes nothing. The data are 1.6 GB of pool rows
plus the label files, so this is a script, not a test.

    python replay_top_two.py [K results directory]

Effects are recomputed on all pool rows with production_pool.rank_effects, candidates with
production_pool.sheet_candidates, and the group decision with production_pool.select_group.
"""
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('production_pool', HERE.parent/'production_pool.py')
pool = importlib.util.module_from_spec(spec); spec.loader.exec_module(pool)

K_RESULTS = Path('/Volumes/ExternalNVMe/awesom-results/cluster/HHR_flux_production_pool_k10_r8_8014/run/results')
SEEDS = (42, 43, 44)
GROUPS = ((42, 43), (42, 44), (43, 44), (42, 43, 44))


def main(results):
    with h5py.File(results/'pool.h5', 'r') as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f['names'][()]]
        raw = f['raw'][()]
    if names != list(pool.NAMES):
        raise ValueError(f'pool feature names {names} differ from production_pool.NAMES')
    rows = []
    for shape in ('91x64', '48x34'):
        labels = {s: np.load(results/f'{shape}-s{s}-labels.npy') for s in SEEDS}
        K = {s: int(labels[s].max())+1 for s in SEEDS}
        chi_col = names.index('log_chi_B')
        effects = {s: pool.rank_effects(raw, labels[s], K[s]) for s in SEEDS}
        orders = {s: pool.sheet_candidates(effects[s], names)['order'] for s in SEEDS}
        chi = {s: {c: float(effects[s][c, chi_col]) for c in range(K[s])} for s in SEEDS}
        jac = {(i, j): pool.jaccard_matrix(labels[i], labels[j], K[i], K[j]) for i, j in itertools.combinations(SEEDS, 2)}
        print(shape, 'candidate orders', orders, flush=True)
        for group in GROUPS:
            picks, audit = pool.select_group({s: orders[s] for s in group},
                                             {p: J for p, J in jac.items() if set(p) <= set(group)}, chi)
            prev = audit['approved_rule_outcome']
            prev_first = {s: p[0] for s, p in prev['selected'].items() if p} or None
            new_first = audit['first']['candidates'] if audit['first'] else None
            jaccard = ({k: round(v['jaccard'], 6) for k, v in audit['first']['pairs'].items()}
                       if audit['first'] else None)
            row = dict(lattice=shape, seeds=list(group), previous_status=prev['status'], previous_first=prev_first,
                       top_two_status=audit['status'], top_two_first=new_first, first_pick_jaccard=jaccard,
                       top_two_picks=picks, previous_picks=prev['selected'], same_first=prev_first == new_first)
            rows.append(row)
            print(json.dumps(pool.plain(row)), flush=True)
    same = sum(r['same_first'] for r in rows)
    print(f'same first pick in {same} of {len(rows)} seed groups')
    return rows


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else K_RESULTS)
