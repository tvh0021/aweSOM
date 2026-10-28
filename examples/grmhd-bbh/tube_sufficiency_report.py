"""Render the exclusive Markdown report from a completed slab analysis."""
import argparse
import json
from pathlib import Path


def pair(folds):
    return ' / '.join('NA' if r['auc'] is None else f"{r['auc']:.4f}" for r in folds)


def report(s):
    lines=['# HHR flux-tube feature sufficiency on levels 5:8','',
    'This analysis applies the frozen levels 5:7 HHR lattices to the levels 5:8 grid. It does not train a new lattice or use the new training sample. The measurements below test information in the saved features and the transfer of the existing cluster selections.', '',
    '## Method and limitations','',
    'Read xyz and levels in chunks of one million rows. Before reading features, retain only the union of the quasi-inertial xy boxes ±12 M around the holes and the slab |z| < 2 dx(level) about the orbital plane. Retain original grid row indices and read features only for their contiguous runs. The level spacings are 0.25, 0.125, 0.0625 and 0.03125 M. Label finite selected rows with each frozen model using the exact screened BMU backend. No full-grid labels or feature array are allocated.', '',
    f"The grid has {s['grid_rows']:,} rows; the crop has {s['slab_rows']:,}, of which {s['finite_rows']:,} have five finite features. Per-level retained counts are {s['slab_rows_per_level']}. Peak process RSS is {s['peak_rss_GiB']:.3f} GiB; elapsed time is {s['seconds']:.1f} seconds.", '',
    "The reference is the previous pipeline's signed Eulerian-coordinate criterion Bz²/(8πP) > 1 and Bz/|B| > 0.7, Euclidean coordinate norm. Apply four-neighbour components inside each ±10 M inertial box, excluding horizons, and keep area > 0.5π(0.94 M)². The absolute-cosine sensitivity is also evaluated. This reuses the raw planes, so the previous discrepancy with Michi's quoted video regions remains unresolved. The rotated raw-plane footprint clips box corners; coverage below is relative to available raw-plane sample locations in each box, not the full geometric 400 M² box.", '',
    'Sample the finest containing owned grid cell at each raw-plane position; include background cells in this lookup. Use the original fixed cluster lists, rather than selecting clusters again from the cropped population. Group candidate unions with 26-neighbour AMR stitching inside the crop. The optional shape cut remains ≥100 cells and elongation ≥3, but these are thin-slab shapes and cannot establish 3D tube geometry or field alignment. Their counts and metrics are not directly comparable to the previous ±6 M grouping domain.', '',
    'For distributions, restrict to ±10 M boxes and finite five-feature and interpolated-direction values. Assign tube labels by nearest raw-plane retained-region pixel projected through the slab. Give tubes precedence over sheet clusters 4, 11 and 15 from mc0.05; background is everything else. This makes three disjoint classes. The sheet comparison concerns these clusters, not an independently verified sheet mask. Each owned cell has equal statistical weight, rather than equal area or volume. Bilinearly interpolate |Bz|/|B| from the raw plane onto the projected cell centres; this is a plane estimate, not a 3D field measurement.', '',
    'Fit balanced L2 logistic regression with training-only standardization, C=1, max_iter=500. Train on BH1 and test BH2, then reverse. Fit at most 100,000 deterministic random cells per training class and evaluate every eligible test cell. AUC direction comes only from the training hole; below-chance scores are not flipped using test labels. This tests linear separability; poor logistic AUC would not prove absence of nonlinear information. Adjacent slab cells share raw-plane labels and the direction feature repeats through z, so counts are not independent observations. Two holes in one snapshot do not establish temporal or resolution transfer.', '',
    'The sixth feature is part of the reference definition. Any gain is criterion-consistent and partly circular, not independent validation of physical tubes. Raw-plane sampling is 0.08 M, coarser than level-8 cells. Changes in extraction normalization or clipping between grids can also affect frozen-model labels.', '',
    '## Coverage and reference','',
    '| Hole | Grid support of observed box | Regions | Area M² | Region areas M² | Region support |',
    '|---|---:|---:|---:|---|---|']
    for r in s['reference']:
        h=str(r['hole']);v=r['variants']['signed'];c=s['coverage'][h]
        lines.append(f"| BH{h} | {100*c['coverage']:.3f}% | {v['count']} | {v['area_M2']:.4f} | "+', '.join(f"{rr['area_M2']:.4f}" for rr in v['regions'])+' | '+', '.join(f'{100*x:.2f}%' for x in c['reference_support'])+' |')
    lines += ['', '## Candidate matches','', '| Selection | Hole | Cut | Recall | Union IoU | False-positive area M² | Best per-region component IoU |','|---|---|---|---:|---:|---:|---|']
    for name,result in s['matches'].items():
        for h,bh in result['holes'].items():
            for cut in ('all','shape'):
                m=bh[cut]['signed']
                lines.append(f"| {name} | BH{h} | {cut} | {m['reference_coverage']:.4f} | {m['union_jaccard']:.5f} | {m['false_positive_area_M2']:.4f} | "+', '.join(f"{r['best_jaccard']:.5f}" for r in m['regions'])+' |')
    for name,result in s['matches'].items():
        lines += ['', f"{name} uses clusters {result['clusters']}, with {result['n_components']:,} cropped components and {result['n_shape_selected']:,} passing the shape cut."]
    lines += ['', '![Broad BH1 overlap](mc005-broad/overlay_BH1_all.png)', '![Broad BH2 overlap](mc005-broad/overlay_BH2_all.png)', '', '## Cell counts and feature distributions','', '| Hole | Tube | Sheet | Background |','|---|---:|---:|---:|']
    for h,c in s['counts'].items(): lines.append(f"| BH{h} | {c['tube']:,} | {c['sheet']:,} | {c['background']:,} |")
    lines += ['',f"Tube precedence reassigns {s['tube_sheet_overlap_before_priority']:,} cells that also have sheet-cluster labels. The statistics exclude {s['excluded_from_statistics']:,} retained crop rows outside the ±10 M boxes or lacking a finite feature/direction value. The tube sample sizes above must be interpreted as correlated cells, not independent tube examples.", '', '| Feature | Class | 16th percentile | Median | 84th percentile |','|---|---|---:|---:|---:|']
    for name,stats in s['distributions'].items():
        for c,v in stats.items():
            lines.append(f"| {name} | {c} | {v.get('p16',float('nan')):.5f} | {v.get('median',float('nan')):.5f} | {v.get('p84',float('nan')):.5f} |")
    for name in s['distributions']: lines += ['',f'![{name} distributions](hist_{name}.png)']
    lines += ['', '## Held-out AUC','', 'Each pair is train BH1 → test BH2 / train BH2 → test BH1. The augmented columns add the directional feature to the named feature or feature set. Direction alone has no separate augmented fit.', '', '| Inputs | Tube vs background | With direction | Tube vs sheet | With direction |','|---|---|---|---|---|']
    for name,comp in s['auc'].items():
        values=[]
        for c in ('tube_vs_background','tube_vs_sheet'):
            values += [pair(comp[c]['five_features_only']),pair(comp[c]['with_direction']) if 'with_direction' in comp[c] else 'same']
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    five=s['auc']['all_five']
    lines += ['', '## Verdict', '',
        f"The five features distinguish the reference tubes from sheet clusters, with cross-hole AUC {pair(five['tube_vs_sheet']['five_features_only'])}, but the combined linear classifier fails to transfer tube-versus-background discrimination, with AUC {pair(five['tube_vs_background']['five_features_only'])}.",
        'Against sheets, log_grad_ln_b2, log_beta_inv, log_chi_B and log_theta carry information in both held-out directions; asinh_div_v is weaker. Against background, log_chi_B and asinh_div_v have weak or hole-dependent signal, while the other single-feature fits transfer below chance. A below-chance AUC is evidence of a learned ranking that reverses across holes, not proof that the feature contains no information.',
        f"Adding |Bz|/|B| raises all-five tube-versus-background AUC to {pair(five['tube_vs_background']['with_direction'])} and tube-versus-sheet AUC to {pair(five['tube_vs_sheet']['with_direction'])}; direction alone gives {pair(s['auc']['direction_only']['tube_vs_background']['five_features_only'])} against background.",
        'A direction-aware sixth feature is strongly supported for a transferable tube selection under this reference. These results do not prove it is mathematically necessary: a nonlinear five-feature classifier, more independent snapshots and a physically independent tube reference remain untested. The present frozen SOM strict selections fail even with level 8 restored, so missing support no longer explains their zero overlap.', '']
    lines += ['', '## Next snapshots and provenance','', 'From the repository, the following commands read the local levels 5:8 inputs and raw planes alongside them, then write fresh directories it94208-rl5-8 and it98304-rl5-8. They refuse existing output directories or logs and reuse both frozen models.', '', '```zsh','cd /Users/tvh0021/git_repos/aweSOM','zsh examples/grmhd-bbh/run_tube_sufficiency.zsh 94208','zsh examples/grmhd-bbh/run_tube_sufficiency.zsh 98304','```','', 'The exact current Python invocation is recorded in summary.json under command. The wrapper exports PYTHONPATH, NUMBA_NUM_THREADS=4 and limits BLAS threads, then uses nice -n 10. All work is local. summary.json contains both reference variants, supported-only metrics, fold counts and standardized coefficients. slab.h5 preserves selected original row indices, features, coordinates, levels and both frozen-model labels. Each candidate directory contains component descriptors and plane labels.', '']
    return '\n'.join(lines)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory');args=ap.parse_args()
    out=Path(args.directory)
    with (out/'report.md').open('x') as f: f.write(report(json.loads((out/'summary.json').read_text())))
