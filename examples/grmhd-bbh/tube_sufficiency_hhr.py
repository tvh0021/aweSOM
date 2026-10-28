"""Local, slab-first frozen-SOM matching and cross-hole feature tests.

Outputs are exclusive. Grid features are read only for retained row indices.
The slab has |z-plane| < 2 dx(level); shape statistics describe this crop only.
"""
from __future__ import annotations
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aweSOM.model import SOMModel
from aweSOM import tubes
from tube_reference import references, plane_points, public_summary, panel, criterion_mask
from tube_match_hhr import sample_owned, overlap_metrics

DX = {5: .25, 6: .125, 7: .0625, 8: .03125}
SETS = {'mc005-strict': [2,3,6,12,14,16], 'mc01-strict': [1,2,4,8],
        'mc005-broad': [1,2,3,5,6,9,10,12,14,16]}


def slab_mask(xyz, levels, holes, origin, normal):
    distance = abs((xyz-origin) @ normal)
    spacing = np.array([DX[int(l)] for l in levels])
    box = np.zeros(len(xyz), bool)
    for hole in holes:
        box |= np.all(abs(xyz[:, :2]-hole[:2]) <= 12, axis=1)
    return box & (distance < 2*spacing)


def selected_features(dataset, rows):
    """Read only selected rows as contiguous runs, avoiding large fancy reads."""
    result = np.empty((len(rows), dataset.shape[1]), dataset.dtype)
    cuts = np.r_[0, np.flatnonzero(np.diff(rows) != 1)+1, len(rows)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a:
            result[a:b] = dataset[int(rows[a]):int(rows[b-1])+1]
    return result


def cross_hole_auc(X, classes, hole, negative, columns, seed=405504):
    """Balanced logistic fit on one hole, evaluate every cell of the other.

    Cap training at 100,000 cells per class with a fixed seed. Fit scaling on
    training only. Return null if a fold lacks either class.
    """
    folds = []
    for train_hole, test_hole in ((1,2),(2,1)):
        train = []
        rng = np.random.default_rng(seed)
        for c in (1, negative):
            rows = np.flatnonzero((hole == train_hole) & (classes == c))
            if len(rows) > 100000:
                rows = rng.choice(rows, 100000, replace=False)
            train.append(rows)
        test = np.flatnonzero((hole == test_hole) & np.isin(classes, [1,negative]))
        record = dict(train_hole=train_hole, test_hole=test_hole,
                      train_counts=[len(r) for r in train], test_counts=[int(np.sum(classes[test] == c)) for c in (1,negative)])
        if all(record['train_counts']) and all(record['test_counts']):
            tr = np.concatenate(train)
            scaler = StandardScaler().fit(X[tr][:,columns])
            clf = LogisticRegression(class_weight='balanced', max_iter=500, solver='lbfgs')
            clf.fit(scaler.transform(X[tr][:,columns]), classes[tr] == 1)
            scores = clf.decision_function(scaler.transform(X[test][:,columns]))
            record['auc'] = float(roc_auc_score(classes[test] == 1, scores))
            record['coefficients_standardized'] = clf.coef_[0].tolist()
        else:
            record['auc'] = None
        folds.append(record)
    return folds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('grid','raw','model005','model01','out-dir'):
        ap.add_argument('--'+key, required=True)
    ap.add_argument('--chunk-rows', type=int, default=1000000)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with np.load(args.raw) as raw:
        refs, area = references(raw)
        points = plane_points(raw).reshape(-1,3)
        holes = raw['holes'].copy(); origin = raw['orbital_origin'].copy()
        e1 = raw['orbital_e1'].copy(); e2 = raw['orbital_e2'].copy()
        normal = np.cross(e1,e2)
        a = raw['orbital_a'].copy(); b = raw['orbital_b'].copy()
        _, _, cos = criterion_mask(*(raw['orbital_'+k].astype(float) for k in ('Bx','By','Bz','P')))
        direction_interp = RegularGridInterpolator((a,b), abs(cos), bounds_error=False, fill_value=np.nan)
    for r in refs: panel(r,out/f"reference_BH{r['hole']}.png")
    xyz_parts=[]; level_parts=[]; row_parts=[]
    with h5py.File(args.grid) as grid:
        names = [s.decode() for s in grid['names'][:]]
        n = len(grid['xyz'])
        for lo in range(0,n,args.chunk_rows):
            xyz = grid['xyz'][lo:lo+args.chunk_rows]
            lv = grid['level'][lo:lo+args.chunk_rows]
            mask = slab_mask(xyz,lv,holes,origin,normal)
            rows = np.flatnonzero(mask)
            if len(rows):
                xyz_parts.append(xyz[rows]); level_parts.append(lv[rows]); row_parts.append(rows+lo)
            if lo % 20000000 == 0: print('xyz scan',lo,'/',n,flush=True)
        xyz = np.concatenate(xyz_parts); level = np.concatenate(level_parts); rows=np.concatenate(row_parts)
        del xyz_parts,level_parts,row_parts
        print('slab selected',len(rows),flush=True)
        features=selected_features(grid['features'],rows)
    finite=np.all(np.isfinite(features),axis=1)
    labs={}
    for tag,path in [('mc005',args.model005),('mc01',args.model01)]:
        model=SOMModel.load(path)
        if names != list(model.feature_names): raise ValueError('model feature names differ')
        lab=np.full(len(rows),-1,np.int32)
        lab[finite]=model.predict(features[finite],chunk=65536,backend='screen')
        labs[tag]=lab
        print('predicted',tag,flush=True)
    with h5py.File(out/'slab.h5','x') as f:
        for key,value in dict(row_indices=rows,xyz=xyz,level=level,features=features,**labs).items(): f.create_dataset(key,data=value,compression='gzip')
    summary=dict(command=sys.argv,grid_rows=n,slab_rows=len(rows),finite_rows=int(finite.sum()),
                 slab_rows_per_level={str(l):int(np.sum(level==l)) for l in np.unique(level)},
                 reference=public_summary(refs),matches={},coverage={})
    for selection,clusters in SETS.items():
        dest=out/selection; dest.mkdir()
        lab=labs['mc01' if selection.startswith('mc01-') else 'mc005']
        ids=tubes.group_cells(np.isin(lab,clusters),xyz,level,DX,connectivity=26)
        desc=tubes.tube_descriptors(ids,xyz,level=level,dx_per_level=DX)
        keep=(desc['n_cells']>=100)&(desc['elongation']>=3)
        sampled,covered=sample_owned(points,xyz,level,ids,DX)
        shaped=sampled.copy(); shaped[~np.isin(shaped,np.flatnonzero(keep))]=-1
        result=dict(clusters=clusters,n_components=len(keep),n_shape_selected=int(keep.sum()),holes={})
        for ref in refs:
            box=ref['box']; support=covered.reshape(box.shape)
            bh=dict(coverage=float(support[box].mean()),reference_support=[float(support[ref['variants']['signed']['labels']==r['id']].mean()) for r in ref['variants']['signed']['regions']])
            summary['coverage'][str(ref['hole'])]=bh.copy()
            for tag,p in [('all',sampled),('shape',shaped)]:
                pred=p.reshape(box.shape)
                bh[tag]={v:overlap_metrics(pred[box],rr['labels'][box],area) for v,rr in ref['variants'].items()}
                bh[tag+'_supported']={v:overlap_metrics(pred[box&support],rr['labels'][box&support],area) for v,rr in ref['variants'].items()}
                panel(ref,dest/f"overlay_BH{ref['hole']}_{tag}.png",overlay=pred>=0)
            result['holes'][str(ref['hole'])]=bh
        with h5py.File(dest/'components.h5','x') as f:
            f.create_dataset('sampled_component_ids',data=sampled.reshape(refs[0]['box'].shape))
            f.create_dataset('grid_covered',data=covered.reshape(refs[0]['box'].shape))
            f.create_dataset('shape_selected_ids',data=np.flatnonzero(keep))
            for key in ('n_cells','volume','centroid','extents','elongation'): f.create_dataset(key,data=desc[key])
        summary['matches'][selection]=result
        print('matched',selection,flush=True)
    # Reference classes use nearest raw-plane pixel, direction uses bilinear interpolation.
    uv=np.column_stack(((xyz-origin)@e1,(xyz-origin)@e2))
    direction=direction_interp(uv)
    hole=np.zeros(len(xyz),np.int8); truth=np.zeros(len(xyz),bool)
    for ref,h in zip(refs,holes):
        inside=np.all(abs(xyz[:,:2]-h[:2])<=10,axis=1)
        label_interp=RegularGridInterpolator((a,b),ref['variants']['signed']['labels'],method='nearest',bounds_error=False,fill_value=0)
        truth |= inside & (label_interp(uv)>0)
        hole[inside]=ref['hole']
    classes=np.where(truth,1,np.where(np.isin(labs['mc005'],[4,11,15]),2,0))
    valid=(hole>0)&finite&np.isfinite(direction)
    X=np.column_stack((features[valid],direction[valid])); classes=classes[valid]; hole=hole[valid]
    names6=names+['abs_Bz_over_B']; summary['distributions']={}; summary['counts']={}
    for h in (1,2): summary['counts'][str(h)]={c:int(np.sum((hole==h)&(classes==i))) for i,c in enumerate(['background','tube','sheet'])}
    summary['excluded_from_statistics']=int(np.sum(~valid))
    summary['tube_sheet_overlap_before_priority']=int(np.sum(truth & np.isin(labs['mc005'],[4,11,15]) & valid))
    for j,name in enumerate(names6):
        stats={}; fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
        lo,hi=np.percentile(X[:,j],[.1,99.9]); bins=np.linspace(lo,hi,81)
        for i,c in enumerate(['background','tube','sheet']):
            vals=X[classes==i,j]
            stats[c]=dict(n=len(vals),p16=float(np.percentile(vals,16)),median=float(np.median(vals)),p84=float(np.percentile(vals,84))) if len(vals) else dict(n=0)
            if len(vals): ax.hist(vals,bins=bins,density=True,histtype='step',label=f'{c} (n={len(vals):,})')
        ax.set(xlabel=name,ylabel='Density within plotted range',title='Slab cells, both holes; plot range 0.1–99.9 percentiles')
        ax.legend(); fig.savefig(out/f'hist_{name}.png',dpi=160); plt.close(fig)
        summary['distributions'][name]=stats
    auc={}
    for name,cols in [(n,[j]) for j,n in enumerate(names)]+[('all_five',list(range(5))),('direction_only',[5])]:
        auc[name]={}
        for comparison,negative in [('tube_vs_background',0),('tube_vs_sheet',2)]:
            auc[name][comparison]={'five_features_only':cross_hole_auc(X,classes,hole,negative,cols)}
            if name!='direction_only': auc[name][comparison]['with_direction']=cross_hole_auc(X,classes,hole,negative,cols+[5])
        print('auc',name,flush=True)
    summary['auc']=auc
    summary['seconds']=time.monotonic()-start
    summary['peak_rss_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('complete',summary['seconds'],summary['peak_rss_GiB'],flush=True)

if __name__=='__main__': main()
