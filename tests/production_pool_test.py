"""Production input refusals, sampling, label matching and real export/reload."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('production_pool', ROOT/'examples/grmhd-bbh/production_pool.py')
pool = importlib.util.module_from_spec(spec); spec.loader.exec_module(pool)
spec2 = importlib.util.spec_from_file_location('build_model', ROOT/'examples/grmhd-bbh/build_model.py')
build = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(build)


def inputs(tmp):
    paths=[]
    for it in [1,2]:
        d=tmp/str(it); d.mkdir()
        marker=d/'status'/'attempt'; marker.mkdir(parents=True)
        (marker/'final.tsv').write_text('VALIDATED\t0\tnow\n')
        p=d/'features.h5'
        with h5py.File(p,'w') as h:
            x=np.random.default_rng(it).normal(size=(120,7)).astype('f4')
            h['features']=x; h['names']=np.array(pool.NAMES,dtype='S')
            h['level']=np.tile([5,6,6],40); h['weight']=np.ones(120)*2; h['weight_volume']=np.ones(120)*.2
            g=h.create_group('level_counts'); g['owned']=[80,160]; g['level']=[5,6]; g['kept']=[40,80]
            h.attrs.update(iteration=it,pipeline='v2',b2_min=1.,s_v=1.,source='one-run',seed=0,
                           exponent=2.,accuracy=4,epsilon=.001,gamma_K=5/3,C_B_box=5,
                           direction_norm='Eulerian B dot B',horizon_mask=True,time=float(it))
        paths.append(p)
    return paths


def args(tmp,paths):
    return pool.parser().parse_args(['--inputs',*map(str,paths),'--anchor','1','--held-out','3',
                                    '--out',str(tmp/'output'),'--rows-per-snapshot','60',
                                    '--rows-per-epoch','60','--min-free-gb','0',
                                    '--lattices','4x3','3x2','--epochs','2','--chunk','29'])


@pytest.mark.parametrize('failure',['threshold','missing','status','failed_status','disk','names','pipeline','holdout','anchor'])
def test_refusals(tmp_path,monkeypatch,failure):
    paths=inputs(tmp_path); a=args(tmp_path,paths)
    if failure=='threshold':
        with h5py.File(paths[1],'r+') as h: h.attrs['s_v']=2.
    if failure=='missing': paths[1].unlink()
    if failure=='status':
        d=paths[1].parent/'status'/'new'; d.mkdir(); (d/'start.tsv').write_text('STARTED')
    if failure=='failed_status':
        (paths[1].parent/'status'/'attempt'/'final.tsv').write_text('FAILED\t1\tnow\n')
    if failure=='disk': monkeypatch.setattr(pool.shutil,'disk_usage',lambda p: SimpleNamespace(free=0))
    if failure=='names':
        with h5py.File(paths[1],'r+') as h: h['names'][0]=b'wrong'
    if failure=='pipeline':
        with h5py.File(paths[1],'r+') as h: h.attrs['pipeline']='v1'
    if failure=='holdout': a.held_out=['2']
    if failure=='anchor': a.anchor='9'
    with pytest.raises(ValueError): pool.inspect(a)
    assert not a.out.exists()


def test_override_and_strata(tmp_path):
    paths=inputs(tmp_path); a=args(tmp_path,paths)
    with h5py.File(paths[1],'r+') as h: h.attrs['s_v']=2.
    a.allow_independent_thresholds=True
    rec,quota,_,bad=pool.inspect(a); assert bad
    a.out.mkdir(); scaling=pool.pool(a,rec,quota,a.out)
    with h5py.File(a.out/'pool.h5') as h:
        source=pool.EpochSource(h['scaled'],scaling['strata'],60,42,17)
        assert sum(len(x) for x in source())==60
        assert source.quotas.tolist()==[10,20,10,20]
        assert h['scaled'].dtype==np.float64
    assert all(r['pool_quotas']=={5:20,6:40} for r in rec)


def test_medoid(tmp_path):
    a=np.array([0,0,1,1,2,2]); b=np.array([2,2,0,0,1,1]); c=np.array([0,1,0,1,2,2])
    assert pool.agreement(a,b)==1
    paths=[]
    for i,x in enumerate([a,b,c]):
        p=tmp_path/f'{i}.npy'; np.save(p,x); paths.append(p)
    idx,band=pool.medoid(paths,2)
    assert idx==0 and band['mean_agreement'][0]>band['mean_agreement'][2]


def test_sheet_median():
    table={0:{'mean':{'log_chi_B':4.,'log_beta_inv':0.},'median':{'log_chi_B':.5,'log_beta_inv':0.}},
           1:{'mean':{'log_chi_B':-4.,'log_beta_inv':0.},'median':{'log_chi_B':1.5,'log_beta_inv':0.}}}
    assert build.select_sheet_clusters(table,'log_chi_B>1,log_beta_inv<0.5')==[1]


def test_end_to_end(tmp_path):
    paths=inputs(tmp_path); a=args(tmp_path,paths)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(tmp_path/'cache'),
             NUMBA_NUM_THREADS='2',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(ROOT/'src'))
    cmd=[sys.executable,str(ROOT/'examples/grmhd-bbh/production_pool.py'),'--inputs',*map(str,paths),
         '--anchor','1','--held-out','3','--out',str(a.out),'--rows-per-snapshot','60',
         '--rows-per-epoch','60','--lattices','4x3','3x2','--epochs','2','--chunk','29','--min-free-gb','0']
    result=subprocess.run(cmd,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    m=json.loads((a.out/'results/freeze-manifest.json').read_text())
    assert len(m['models'])==6 and len(m['selection'])==2
    assert set(m)>={'feature_names','pipeline','etk_analysis_commit','physics','scaler','models',
                    'selection','sample_manifest','aweSOM_commit','input_hashes','verification'}
    for r in m['models']:
        assert r['verification']['reload_mismatches']==r['verification']['cluster_label_patch_screen_mismatches']==0
        assert r['verification']['scaler_max_abs_error']==0
        assert r['weights_dtype']=='float64'
    assert m['sample_manifest']['held_out_IDs']==['3']
    # reduced training settings are a rehearsal, never a production manifest
    assert m['diagnostic_only'] and any(r.startswith('epochs=2') for r in m['diagnostic_reasons'])


def test_diagnostic_reasons(tmp_path,monkeypatch):
    base=['--inputs','x.h5','--anchor','94208','--held-out','405504','--out',str(tmp_path/'o'),
          '--rows-per-snapshot','10000000','--rows-per-epoch','20000000']
    monkeypatch.setattr(pool,'APPROVED_INPUT_SHA256',{'94208':'a'*64,'98304':'b'*64})
    rec=lambda *ids: [dict(snapshot=i,sha256=pool.APPROVED_INPUT_SHA256.get(i,'c'*64)) for i in ids]
    reasons=lambda extra=(),records=rec('94208','98304'): pool.diagnostic_reasons(pool.parser().parse_args(base+list(extra)),records)
    assert reasons()==[]
    for extra,key in [(['--epochs','2'],'epochs'),(['--lattices','4x3'],'lattices'),(['--seeds','42'],'seeds'),
                      (['--rows-per-epoch','60'],'rows_per_epoch'),(['--allow-six-features'],'allow_six_features'),
                      (['--anchor','98304'],'anchor'),(['--held-out','405504','110592'],'held_out')]:
        assert any(r.startswith(key) for r in reasons(extra)),key
    # a different training set: reordered, added or swapped snapshots
    for other in (rec('98304','94208'),rec('94208','98304','110592'),rec('94208','110592')):
        assert any(r.startswith('snapshots') for r in reasons(records=other))
    # the approved iterations from another extraction or simulation
    other=rec('94208','98304'); other[1]['sha256']='d'*64
    assert [r for r in reasons(records=other) if r.startswith('input 98304')]
    # nothing is production before the inputs are pinned
    monkeypatch.setattr(pool,'APPROVED_INPUT_SHA256',{})
    assert len([r for r in reasons(records=other) if r.startswith('input')])==2


def test_rare_levels():
    data=np.arange(200)
    strata=[dict(snapshot=str(i//2), start=a, rows=n) for i,(a,n) in enumerate([(0,99),(99,1),(100,99),(199,1)])]
    a=pool.EpochSource(data,strata,20,42,7)
    b=pool.EpochSource(data,strata,20,42,7)
    seen=set()
    for _ in range(100):
        x=np.concatenate(list(a())); y=np.concatenate(list(b()))
        assert np.array_equal(x,y)
        assert len(np.unique(x))==20
        assert np.count_nonzero(x<100)==10
        seen.update(x)
    assert {99,199} <= seen
    with pytest.raises(ValueError,match='erase'): pool.allocate([99,1],10)
    with pytest.raises(ValueError,match='strata'): pool.EpochSource(data,strata,2,42,7)


@pytest.mark.parametrize('explained',[False,True])
def test_missing_extraction_level(tmp_path,explained):
    paths=inputs(tmp_path); a=args(tmp_path,paths)
    with h5py.File(paths[0],'r+') as h:
        h['level'][:]=5
        h['level_counts/kept'][:]=[120,0] if explained else [40,80]
    if explained:
        rec,*_=pool.inspect(a)
        assert rec[0]['observed_level_counts'][6]==0
    else:
        with pytest.raises(ValueError,match='observed count'): pool.inspect(a)


def test_export_statistics_exact(tmp_path):
    from aweSOM.model import distance_quantiles_fast
    rng=np.random.default_rng(12)
    raw=rng.normal(size=(1003,7)).astype('f4')
    nodes=rng.integers(0,9,len(raw),dtype='i4'); labels=nodes%3
    distances=rng.random(len(raw))
    q,r,medians=pool.export_statistics(raw,nodes,distances,labels,10,tmp_path,'test',29)
    expected,reject=distance_quantiles_fast(nodes,distances,10)
    assert np.array_equal(q,expected,equal_nan=True)
    assert r==reject
    for c in medians:
        assert np.array_equal(medians[c],np.median(raw[labels==c].astype('f8'),axis=0))
    assert not list(tmp_path.glob('*statistics*'))


@pytest.mark.parametrize('damage',['none','driver','source','missing'])
def test_non_git_kit(tmp_path,damage):
    import shutil
    paths=inputs(tmp_path); a=args(tmp_path,paths)
    kit=tmp_path/'kit'; kit.mkdir()
    driver=kit/'production_pool.py'; shutil.copy2(pool.__file__,driver)
    source=kit/'aweSOM-src/src/aweSOM'
    shutil.copytree(ROOT/'src/aweSOM',source,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
    manifest=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  files={str(p.relative_to(source)):pool.digest(p) for p in source.rglob('*') if p.is_file()},
                  driver_sha256=pool.digest(driver))
    if damage!='missing': (kit/'production_pool-source.json').write_text(json.dumps(manifest))
    if damage=='driver': driver.write_text(driver.read_text()+'\n# changed\n')
    if damage=='source':
        p=source/'__init__.py'; p.write_text(p.read_text()+'\n# changed\n')
    cmd=[sys.executable,str(driver),'--inputs',*map(str,paths),'--anchor','1','--held-out','3',
         '--out',str(a.out),'--rows-per-snapshot','60','--rows-per-epoch','60',
         '--lattices','3x2','--seeds','42','--epochs','1','--chunk','29','--min-free-gb','0']
    env=dict(os.environ,PYTHONPATH=str(source.parent),PYTHONDONTWRITEBYTECODE='1',
             NUMBA_CACHE_DIR=str(tmp_path/'cache'))
    result=subprocess.run(cmd,env=env,capture_output=True,text=True,cwd=kit)
    if damage=='none':
        assert result.returncode==0,result.stdout+result.stderr
        m=json.loads((a.out/'results/freeze-manifest.json').read_text())
        assert m['aweSOM_commit']==manifest['commit']
        v=m['models'][0]['verification']
        assert v['bmu_node_mismatches']==0 and v['bmu_distance_rtol_1e6']
        assert all(v['saved_components_equal'].values())
        assert not list((a.out/'results').glob('*-nodes.npy'))
        assert not list((a.out/'results').glob('*-distances.npy'))
    else:
        assert result.returncode!=0 and not a.out.exists()
        assert ('hash mismatch' if damage!='missing' else 'required outside') in result.stderr


def test_disk_scaler_quantiles(tmp_path):
    x=np.random.default_rng(93).normal(size=(1003,7)).astype('f4')
    with h5py.File(tmp_path/'pool.h5','w') as h:
        raw=h.create_dataset('raw',data=x)
        assert np.array_equal(pool.quantiles(raw),np.percentile(x.astype('f8'),[25,50,75],axis=0))
        assert np.array_equal(pool.quantiles(raw,13,752),np.percentile(x[13:752].astype('f8'),[25,50,75],axis=0))
    assert not (tmp_path/'scaler-column.npy').exists()
