"""Michi's orbital-plane tube criterion using Eulerian coordinate B^i.

Samples retain their original co-rotating lattice; boxes and reported centroids
use quasi-inertial xy axes by default. Outputs are exclusive, never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy import ndimage


def criterion_mask(bx, by, bz, pressure, absolute=False):
    """Return strict criterion, magnetic/thermal ratio and signed cosine.

    Nonfinite fields, nonpositive pressure and zero field are excluded.
    The Euclidean coordinate norm is used, without a spatial metric.
    """
    bx, by, bz, pressure = np.broadcast_arrays(bx, by, bz, pressure)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ratio = bz**2 / (8 * np.pi * pressure)
        cosine = bz / np.sqrt(bx**2 + by**2 + bz**2)
    valid = (np.isfinite(bx) & np.isfinite(by) & np.isfinite(bz)
             & np.isfinite(pressure) & (pressure > 0)
             & np.isfinite(ratio) & np.isfinite(cosine))
    return valid & (ratio > 1) & ((np.abs(cosine) if absolute else cosine) > .7), ratio, cosine


def area_filter(mask, pixel_area, min_area, connectivity=4):
    """Label components inside the supplied crop; keep area strictly above cut."""
    if pixel_area <= 0 or min_area < 0 or connectivity not in (4, 8):
        raise ValueError('invalid area or connectivity')
    labels, _ = ndimage.label(mask, ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2))
    counts = np.bincount(labels.ravel())
    keep = counts * pixel_area > min_area
    keep[0] = False
    return np.where(keep[labels], labels, 0)


def plane_points(raw):
    a, b = np.meshgrid(raw['orbital_a'], raw['orbital_b'], indexing='ij')
    return raw['orbital_origin'] + a[..., None] * raw['orbital_e1'] + b[..., None] * raw['orbital_e2']


def references(raw, radius=.94, half_width=10., connectivity=4, frame='inertial'):
    points = plane_points(raw)
    area = float(abs(np.diff(raw['orbital_a']).mean() * np.diff(raw['orbital_b']).mean()))
    radii = np.broadcast_to(raw['horizon_radius'] if 'horizon_radius' in raw else radius, (len(raw['holes']),))
    result = []
    for h, hole in enumerate(raw['holes']):
        rel = points - hole
        x, y = rel[..., 0], rel[..., 1]
        if frame == 'corotating':
            x, y = rel @ raw['orbital_e1'], rel @ raw['orbital_e2']
        box = (abs(x) <= half_width) & (abs(y) <= half_width)
        outside = np.ones(box.shape, bool)
        for other, r in zip(raw['holes'], radii):
            outside &= np.sum((points - other)**2, axis=-1) > r*r
        variants = {}
        for absolute in (False, True):
            mask, ratio, cosine = criterion_mask(*(raw['orbital_'+k].astype(float) for k in ('Bx', 'By', 'Bz', 'P')), absolute=absolute)
            mask &= box & outside
            kept = area_filter(mask, area, .5*np.pi*radii[h]**2, connectivity)
            regions = []
            edge = box & ~ndimage.binary_erosion(box)
            for label in np.unique(kept[kept > 0]):
                m = kept == label
                regions.append(dict(id=int(label), area_M2=float(m.sum()*area), centroid_M=[float(x[m].mean()), float(y[m].mean())], mean_Bz2_8piP=float(ratio[m].mean()), mean_Bz_over_B=float(cosine[m].mean()), touches_crop_boundary=bool(np.any(m & edge))))
            variants['absolute' if absolute else 'signed'] = dict(labels=kept, mask=mask, regions=regions, count=len(regions), area_M2=sum(r['area_M2'] for r in regions))
        result.append(dict(hole=h+1, radius_M=float(radii[h]), x=x, y=y, box=box, variants=variants))
    return result, area


def public_summary(refs):
    return [{k: v for k, v in r.items() if k not in ('x', 'y', 'box', 'variants')} | {'variants': {name: {k: v for k, v in v.items() if k not in ('labels', 'mask')} for name, v in r['variants'].items()}} for r in refs]


def panel(ref, path, half_width=10., overlay=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout='constrained')
    for ax, (name, v) in zip(axes, ref['variants'].items()):
        values = np.where(v['labels'] > 0, 2, np.where(v['mask'], 1, 0)).astype(float)
        values[~ref['box']] = np.nan
        ax.pcolormesh(ref['x'], ref['y'], values, shading='nearest', cmap=ListedColormap(['#eeeeee', '#999999', '#71389c']), vmin=0, vmax=2, rasterized=True)
        if overlay is not None and np.any(overlay & ref['box']):
            ax.contour(ref['x'], ref['y'], overlay.astype(int), levels=[.5], colors=['#00a7a7'], linewidths=.7)
        ax.add_patch(plt.Circle((0, 0), ref['radius_M'], color='black'))
        ax.set(xlim=(-half_width, half_width), ylim=(-half_width, half_width), aspect='equal', xlabel='x - x_BH [M]', ylabel='y - y_BH [M]', title=f"BH{ref['hole']} {name}: {v['count']} kept, {v['area_M2']:.3f} M²")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--radius', type=float, default=.94)
    p.add_argument('--connectivity', type=int, choices=[4, 8], default=4)
    p.add_argument('--frame', choices=['inertial', 'corotating'], default='inertial')
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    with np.load(args.raw) as raw:
        refs, area = references(raw, args.radius, connectivity=args.connectivity, frame=args.frame)
        summary = dict(raw=args.raw, iteration=int(raw['it']), time=float(raw['time']), pixel_area_M2=area, frame=args.frame, connectivity=args.connectivity, field='Eulerian coordinate B^i; Euclidean norm', holes=public_summary(refs))
    for r in refs:
        panel(r, out / f"reference_BH{r['hole']}.png")
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
