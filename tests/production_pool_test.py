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
            h.create_group('level_counts')['owned']=[80,160]
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
        assert source.quotas.tolist()==[10,20,10,20]
        assert sum(len(x) for x in source())==60
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
        assert r['verification']['reload_mismatches']==r['verification']['patch_screen_mismatches']==0
        assert r['verification']['scaler_max_abs_error']==0
        assert r['weights_dtype']=='float64'
    assert m['sample_manifest']['held_out_IDs']==['3']
