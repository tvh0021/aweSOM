"""Streaming support for proposal_figure's full AMR grid products."""
import json
from pathlib import Path
import h5py
import numpy as np


class FeatureColumn:
    """Read a single feature column only for the requested row slice."""
    def __init__(self, dataset, column):
        self.dataset, self.column = dataset, column

    def __getitem__(self, rows):
        return self.dataset[rows, self.column]


def label_statistics(args):
    """Reuse full-grid labels, counting volumes and histogram feature medians.

    Medians have bin width 0.001 in feature units; out-of-range values raise.
    This bounds memory independently of the total number of grid rows.
    """
    from aweSOM.npy_parts import ShardedNpyArray, npy_file_for, open_npy
    from proposal_figure import DX, FEATURES, grid_clip
    labels = open_npy(args.grid_labels, mmap_mode='r')
    try:
        out = Path(args.out)
        target_npy = out / f'labels_{args.tag}.npy'
        target_parts = out / f'labels_{args.tag}.parts'
        if isinstance(labels, ShardedNpyArray):
            src_path = labels.path.resolve()
            if target_npy.exists() or target_npy.is_symlink():
                raise FileExistsError(target_npy)
            if not (target_parts.exists() and target_parts.resolve() == src_path):
                if target_parts.exists() or target_parts.is_symlink():
                    raise FileExistsError(target_parts)
                target_parts.symlink_to(src_path, target_is_directory=True)
        else:
            src_path = npy_file_for(args.grid_labels).resolve()
            if not (target_npy.exists() and target_npy.resolve() == src_path):
                if target_npy.exists() or target_npy.is_symlink():
                    raise FileExistsError(target_npy)
                target_npy.symlink_to(src_path)
        clip = grid_clip(args.grid, args.train_features)
        with h5py.File(args.grid) as grid, h5py.File(args.train_features) as train:
            mismatch = {key: [np.asarray(train.attrs[key]).tolist(), np.asarray(grid.attrs[key]).tolist()]
                        for key in ('b2_min', 's_v', 'winsor_log_chi_B', 'winsor_log_grad_ln_b2', 'winsor_asinh_div_v')
                        if not np.allclose(train.attrs[key], grid.attrs[key], rtol=1e-9, atol=0)}
        hist, counts, volumes, per = {}, {}, {}, {}
        edges = np.linspace(-100, 100, 200001)
        cell_volume = np.zeros(max(DX)+1)
        for level, dx in DX.items():
            cell_volume[level] = dx**3
        total = 0
        with h5py.File(args.grid) as h:
            assert len(labels) == len(h['level'])
            for a in range(0, len(labels), 1000000):
                sl = slice(a,a+1000000)
                x = h['features'][sl]
                lev, lab = h['level'][sl], labels[sl]
                cv = cell_volume[lev]
                total += cv.sum()
                for c in np.unique(lab):
                    c = int(c); m = lab == c
                    hist.setdefault(c, np.zeros((len(FEATURES),len(edges)-1), np.int64))
                    counts[c] = counts.get(c,0)+int(m.sum())
                    volumes[c] = volumes.get(c,0)+float(cv[m].sum())
                    per.setdefault(c,{str(l):0 for l in DX})
                    for l in DX: per[c][str(l)] += int(np.sum(m & (lev==l)))
                    for j in range(len(FEATURES)):
                        v=x[m,j]
                        if not np.all((v>=edges[0]) & (v<edges[-1])): raise ValueError('histogram range')
                        hist[c][j] += np.histogram(v,edges)[0]
    finally:
        if isinstance(labels, ShardedNpyArray):
            labels.close()
    stats={}
    for c in counts:
        med=[float(edges[np.searchsorted(np.cumsum(row), (counts[c]+1)/2)]+.0005) for row in hist[c]]
        stats[c]={'n_points':counts[c], 'point_fraction':counts[c]/len(labels), 'volume_M3':volumes[c],
                  'volume_fraction':volumes[c]/total, 'median':dict(zip(FEATURES,med)), 'points_per_level':per[c]}
    sheet=max(stats,key=lambda c:stats[c]['median']['log_chi_B'])
    info={'sheet_cluster':sheet,'clusters':stats,'total_volume_M3':total,'grid':args.grid,
          'merge_cost':args.merge_cost,'som':args.som,'train_features':args.train_features,
          'feature_transform_mismatches':mismatch, 'labels_use':'supplied grid features without re-clipping',
          'median_bin_width':.001,'n_points':len(labels),'threshold_check':clip['check']}
    with open(out/f'clusters_{args.tag}.json','x') as f: json.dump(info,f,indent=2)
    print('sheet cluster',sheet,stats[sheet],flush=True)
