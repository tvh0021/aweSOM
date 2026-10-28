"""Chunked HHR cluster grouping and direct 2D reference overlap.

The grid contains no 3D B vectors. Components are candidate tubes, with a
separate geometric elongation cut; field alignment is explicitly unverified.
The finite grouping domain is the union of hole-centred xy boxes, |z| <= zmax.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree
from aweSOM import tubes
from tube_reference import references, plane_points, panel, public_summary


def overlap_metrics(predicted, reference, pixel_area):
    """Union coverage, false-positive area, and best component IoU per region.

    Background is -1 in predicted and 0 in reference. Empty reference recall
    is None. A component's union includes its whole area in the supplied box.
    """
    predicted, reference = np.asarray(predicted), np.asarray(reference)
    if predicted.shape != reference.shape:
        raise ValueError('shapes differ')
    fg, truth = predicted >= 0, reference > 0
    regions = []
    for label in np.unique(reference[truth]):
        m = reference == label
        best, best_id = 0., None
        for c in np.unique(predicted[m & fg]):
            p = predicted == c
            iou = np.count_nonzero(p & m) / np.count_nonzero(p | m)
            if iou > best:
                best, best_id = float(iou), int(c)
        regions.append(dict(reference_id=int(label), area_M2=float(m.sum()*pixel_area), covered_fraction=float(np.count_nonzero(m & fg)/m.sum()), best_component_id=best_id, best_jaccard=best))
    intersection = int(np.count_nonzero(fg & truth))
    union = int(np.count_nonzero(fg | truth))
    return dict(reference_area_M2=float(truth.sum()*pixel_area), predicted_area_M2=float(fg.sum()*pixel_area), false_positive_area_M2=float(np.count_nonzero(fg & ~truth)*pixel_area), reference_coverage=intersection/int(truth.sum()) if truth.any() else None, union_jaccard=intersection/union if union else None, regions=regions)


def sample_owned(points, xyz, levels, ids, dx):
    """Nearest owned cell centre on each level; finest containing cube wins.

    Query all owned cells, including background, to avoid extending a selected
    component into its neighbours. Missing points retain -1.
    """
    out = np.full(len(points), -1, np.int32)
    covered = np.zeros(len(points), bool)
    for lv in sorted(dx):
        rows = np.flatnonzero(levels == lv)
        if not len(rows):
            continue
        tree = cKDTree(xyz[rows])
        distance, idx = tree.query(points, p=np.inf, distance_upper_bound=dx[lv]/2 + 1e-6)
        good = np.isfinite(distance)
        out[good] = ids[rows[idx[good]]]
        covered[good] = True
        del tree
    return out, covered


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('grid', 'labels', 'raw', 'model', 'out-dir'):
        ap.add_argument('--'+key, required=True)
    ap.add_argument('--predict-labels', action='store_true', help='create the --labels file with the frozen model; refuse existing files')
    ap.add_argument('--clusters', type=int, nargs='+', help='override automatic cluster selection')
    ap.add_argument('--chunk-rows', type=int, default=1_000_000)
    ap.add_argument('--zmax', type=float, default=6.)
    ap.add_argument('--min-cells', type=int, default=100)
    ap.add_argument('--min-elongation', type=float, default=3.)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    if args.predict_labels:
        from aweSOM.model import SOMModel
        model = SOMModel.load(args.model)
        # Reserve exclusively before open_memmap, which itself would overwrite.
        with open(args.labels, 'xb'):
            pass
        with h5py.File(args.grid) as grid:
            target = np.lib.format.open_memmap(args.labels, mode='w+', dtype=np.uint8, shape=(len(grid['xyz']),))
            names = [s.decode() for s in grid['names'][:]]
            if names != list(model.feature_names):
                raise ValueError('model/grid feature names differ')
            for a in range(0, len(target), args.chunk_rows):
                f = grid['features'][a:a+args.chunk_rows]
                good = np.all(np.isfinite(f), axis=1)
                lab = np.full(len(f), 255, np.uint8)
                if good.any():
                    pred = model.predict(f[good], chunk=2**18, backend='screen')
                    if np.any((pred < 0) | (pred > 254)):
                        raise ValueError('model labels do not fit uint8')
                    lab[good] = pred
                target[a:a+len(f)] = lab
                print('predicted', a+len(f), flush=True)
            target.flush()
            del target
    labels = np.load(args.labels, mmap_mode='r')
    with np.load(args.raw) as raw:
        refs, pixel_area = references(raw)
        points = plane_points(raw).reshape(-1, 3)
        holes = raw['holes'].copy()
    with h5py.File(args.model) as model:
        names_model = [s.decode() for s in model['feature_names'][:]]
        model_clusters = model['node_cluster'][:]
    with h5py.File(args.grid) as grid:
        n = len(grid['xyz'])
        if labels.shape != (n,):
            raise ValueError('labels do not match grid rows')
        names = [s.decode() for s in grid['names'][:]]
        if names != names_model:
            raise ValueError('model/grid feature names differ')
        k = int(model_clusters.max())+1
        sums, counts = np.zeros((k, len(names))), np.zeros(k, np.int64)
        for a in range(0, n, args.chunk_rows):
            lab = np.asarray(labels[a:a+args.chunk_rows])
            f = grid['features'][a:a+args.chunk_rows]
            good = (lab >= 0) & (lab < k) & np.all(np.isfinite(f), axis=1)
            counts += np.bincount(lab[good], minlength=k)
            for j in range(len(names)):
                sums[:, j] += np.bincount(lab[good], weights=f[good, j], minlength=k)
        means = np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
        # Strong inverse beta (>10), smooth field direction (chi_B < 10^-0.5).
        candidates = args.clusters or np.flatnonzero((means[:, names.index('log_beta_inv')] > 1) & (means[:, names.index('log_chi_B')] < -.5)).tolist()
        print('cluster means', dict(zip(range(k), means.tolist())), flush=True)
        print('candidates', candidates, flush=True)
        xyz_parts, level_parts, mask_parts = [], [], []
        dx = {}
        for a in range(0, n, args.chunk_rows):
            xyz = grid['xyz'][a:a+args.chunk_rows]
            lv = grid['level'][a:a+args.chunk_rows]
            for l in np.unique(lv):
                if int(l) not in dx:
                    p = xyz[lv == l]
                    diffs = np.diff(np.unique(p[:, 2]))
                    if len(diffs):
                        dx[int(l)] = float(diffs.min())
            inside = np.zeros(len(xyz), bool)
            for hole in holes:
                inside |= np.all(abs(xyz[:, :2]-hole[:2]) <= 10.5, axis=1) & (abs(xyz[:, 2]-hole[2]) <= args.zmax)
            xyz_parts.append(xyz[inside])
            level_parts.append(lv[inside])
            mask_parts.append(np.isin(labels[a:a+args.chunk_rows][inside], candidates))
        xyz = np.concatenate(xyz_parts); del xyz_parts
        level = np.concatenate(level_parts); del level_parts
        mask = np.concatenate(mask_parts); del mask_parts
    print('cropped rows', len(xyz), 'selected', int(mask.sum()), 'dx', dx, flush=True)
    ids = tubes.group_cells(mask, xyz, level, dx, connectivity=26)
    del mask
    print('grouped', int(ids.max()+1), flush=True)
    desc = tubes.tube_descriptors(ids, xyz, level=level, dx_per_level=dx)
    shape_keep = (desc['n_cells'] >= args.min_cells) & (desc['elongation'] >= args.min_elongation)
    plane_ids, covered = sample_owned(points, xyz, level, ids, dx)
    shape_ids = plane_ids.copy()
    fg = shape_ids >= 0
    shape_ids[fg & ~np.isin(shape_ids, np.flatnonzero(shape_keep))] = -1
    summary = dict(command=sys.argv, grid_rows=n, cropped_rows=len(xyz), dx=dx, grouping_domain='union of hole-centred inertial xy boxes +/-10.5 M, z +/- zmax; cropped components may be truncated', zmax=args.zmax, cluster_means={str(c):{name: float(value) if np.isfinite(value) else None for name, value in zip(names, means[c])} for c in range(k)}, candidates=candidates, selection=('explicit --clusters override' if args.clusters else 'mean log_beta_inv > 1 and mean log_chi_B < -0.5; low directional gradient is a coherence proxy, not field alignment'), n_components=len(desc['id']), n_shape_selected=int(shape_keep.sum()), shape_selection=dict(min_cells=args.min_cells, min_elongation=args.min_elongation), field_alignment_verified=False, reference=public_summary(refs), matches={})
    for ref in refs:
        box = ref['box']
        bh = {}
        for name, sampled in [('all_candidates', plane_ids), ('elongated_candidates', shape_ids)]:
            pred = sampled.reshape(box.shape)
            bh[name] = {variant:overlap_metrics(pred[box], v['labels'][box], pixel_area) for variant, v in ref['variants'].items()}
            panel(ref, out/f"overlay_BH{ref['hole']}_{name}.png", overlay=pred >= 0)
        support = covered.reshape(box.shape)
        bh['reference_grid_support'] = {}
        for variant, v in ref['variants'].items():
            bh['reference_grid_support'][variant] = [dict(reference_id=int(label), covered_fraction=float(support[v['labels'] == label].mean())) for label in np.unique(v['labels'][v['labels'] > 0])]
        bh['supported_only_matches'] = {}
        supported = box & support
        for name, sampled in [('all_candidates', plane_ids), ('elongated_candidates', shape_ids)]:
            bh['supported_only_matches'][name] = {variant:overlap_metrics(sampled.reshape(box.shape)[supported], v['labels'][supported], pixel_area) for variant, v in ref['variants'].items()}
        bh['grid_coverage_fraction'] = float(covered.reshape(box.shape)[box].mean())
        summary['matches'][str(ref['hole'])] = bh
    with h5py.File(out/'components.h5', 'x') as f:
        f.create_dataset('grid_covered', data=covered.reshape(refs[0]['box'].shape))
        f.create_dataset('sampled_component_ids', data=plane_ids.reshape(refs[0]['box'].shape))
        f.create_dataset('shape_selected_ids', data=np.flatnonzero(shape_keep))
        for name in ('n_cells', 'volume', 'centroid', 'extents', 'elongation', 'axes'):
            f.create_dataset(name, data=desc[name])
    summary['seconds'] = time.monotonic()-start
    summary['peak_rss_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('matches', 'peak_rss_GiB', 'seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
