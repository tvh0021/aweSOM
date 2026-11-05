"""Production input refusals, sampling, label matching and real export/reload."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import os
import signal
import subprocess
import sys
import time

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
    a.write_scaled=True
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
         '--rows-per-epoch','60','--lattices','4x3','3x2','--epochs','2','--chunk','29','--min-free-gb','0',
         '--target-clusters','1','--threads-per-seed','1','--seed-workers','2']
    result=subprocess.run(cmd,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    m=json.loads((a.out/'results/freeze-manifest.json').read_text())
    assert len(m['models'])==4 and len(m['selection'])==2
    assert [r['seed'] for r in m['models']]==[42,43,42,43]
    for choice in m['selection']:
        assert choice['sheet_selection']['audit']['status'] in ('selected','none','unavailable')
        assert choice['sheet_selection']['audit']['reason']
    assert set(m)>={'feature_names','pipeline','etk_analysis_commit','physics','scaler','models',
                    'selection','sample_manifest','aweSOM_commit','input_hashes','verification'}
    assert pool.APPROVED['target_clusters'] == 10
    from aweSOM import SOMModel
    for r in m['models']:
        saved = SOMModel.load(a.out/'results'/f"{r['tag']}.h5")
        sidecar = json.loads((a.out/'results'/f"{r['tag']}.json").read_text())
        assert r['execution']['numba_threads'] == r['training_info']['n_threads'] == 1
        assert set(r['phase_timings']) == {'training','clustering','labelling','statistics','export','verification'}
        assert len(r['training_info']['epoch_time']) == 2
        assert r['target_clusters'] == 1
        assert r['merge_cost_used'] >= .05
        assert r['n_clusters'] == len(np.unique(saved.node_cluster)) == 1
        for key in ('target_clusters', 'merge_cost_used', 'n_clusters'):
            assert float(saved.metadata[key]) == r[key] == sidecar[key]
        assert r['verification']['reload_mismatches']==r['verification']['cluster_label_patch_screen_mismatches']==0
        assert r['verification']['scaler_max_abs_error']==0
        assert r['weights_dtype']=='float64'
        # the group decision reached the model, its sidecar and the manifest alike
        assert r['sheet_selection_status'] == sidecar['sheet_selection_status'] != 'pending'
        assert saved.metadata['sheet_selection_status'] == r['sheet_selection_status']
        assert saved.metadata['selected_sheet_clusters'] == saved.metadata['sheet_clusters'] == ','.join(map(str,r['selected_sheet_clusters']))
        assert json.loads(saved.metadata['selection_audit']) == sidecar['selection_audit'] == r['selection_audit']
        assert list(json.loads(saved.metadata['seed_overlap'])) == ['42-43']
        assert np.array(r['cluster_effects']).shape == (r['n_clusters'], 7)
        assert 'sheet_rule' not in saved.metadata and 'sheet_rule' not in r
        assert r['legacy_threshold_rule_diagnostic']['role'].startswith('diagnostic only')
    assert m['sample_manifest']['held_out_IDs']==['3']
    # reduced training settings are a rehearsal, never a production manifest
    assert m['diagnostic_only'] and any(r.startswith('epochs=2') for r in m['diagnostic_reasons'])


def test_diagnostic_reasons(tmp_path,monkeypatch):
    base=['--inputs','x.h5','--anchor','94208','--held-out','405504','--out',str(tmp_path/'o'),
          '--rows-per-snapshot','10000000','--rows-per-epoch','20000000','--target-clusters','10','--r-min','8']
    monkeypatch.setattr(pool,'APPROVED_INPUT_SHA256',{'94208':'a'*64,'98304':'b'*64})
    rec=lambda *ids: [dict(snapshot=i,sha256=pool.APPROVED_INPUT_SHA256.get(i,'c'*64)) for i in ids]
    reasons=lambda extra=(),records=rec('94208','98304'): pool.diagnostic_reasons(pool.parser().parse_args(base+list(extra)),records)
    assert reasons()==[]
    # the per-lattice default radius (r5 on 48x34) is not the approved r8
    assert any(r.startswith('r_min') for r in pool.diagnostic_reasons(pool.parser().parse_args(
        [a for a in base if a not in ('--r-min','8')]),rec('94208','98304')))
    for extra,key in [(['--epochs','2'],'epochs'),(['--lattices','4x3'],'lattices'),(['--seeds','42'],'seeds'),
                      (['--rows-per-epoch','60'],'rows_per_epoch'),(['--allow-six-features'],'allow_six_features'),
                      (['--anchor','98304'],'anchor'),(['--target-clusters','12'],'target_clusters'),(['--r-min','5'],'r_min'),(['--held-out','405504','110592'],'held_out')]:
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
    raw[::7,5]=0  # ties
    strata=[dict(snapshot='1',start=0,rows=400),dict(snapshot='2',start=400,rows=300),dict(snapshot='1',start=700,rows=303)]
    q,r,medians,d=pool.export_statistics(raw,nodes,distances,labels,10,tmp_path,'test',29,
                                         names=pool.NAMES,strata=strata,n_clusters=4)
    expected,reject=distance_quantiles_fast(nodes,distances,10)
    assert np.array_equal(q,expected,equal_nan=True)
    assert r==reject
    for c in medians:
        assert np.array_equal(medians[c],np.median(raw[labels==c].astype('f8'),axis=0))
        np.testing.assert_allclose(d['cluster_quartiles'][c],np.percentile(raw[labels==c].astype('f8'),[25,50,75],axis=0),rtol=0,atol=1e-12)
    # cluster 3 is empty: NaN effects, zero rows
    assert d['cluster_effects'].shape==(4,7) and np.isnan(d['cluster_effects'][3]).all()
    assert d['cluster_rows']['all'].tolist()==np.bincount(labels,minlength=4).tolist()
    for c in range(3):
        for j in range(7):
            assert d['cluster_effects'][c,j]==pytest.approx(brute_effect(raw[:,j],labels==c),abs=1e-12)
    one=np.r_[np.arange(400),np.arange(700,1003)]
    assert d['cluster_rows']['1'].tolist()==np.bincount(labels[one],minlength=4).tolist()
    assert d['cluster_share']['2'].sum()==pytest.approx(1)
    med=np.median(raw.astype('f8'),axis=0); i=pool.NAMES.index
    fav=(raw[:,i('log_chi_B')]>med[i('log_chi_B')])&(raw[:,i('log_grad_ln_b2')]>med[i('log_grad_ln_b2')])&(raw[:,i('log_beta_inv')]<med[i('log_beta_inv')])
    np.testing.assert_allclose(d['cluster_joint_favorable_fraction'][:3],[fav[labels==c].mean() for c in range(3)])
    assert np.array_equal(d['pooled_medians'],med)
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
        # one seed cannot check reproduction: unavailable, never a selection
        assert m['selection'][0]['sheet_selection']['audit']['status']=='unavailable'
        assert m['models'][0]['selected_sheet_clusters']==[]
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


def test_thread_budget(monkeypatch):
    import numba
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '2')
    a = SimpleNamespace(threads_per_seed=2, seed_workers=1, seeds=[42,43])
    assert pool.thread_budget(a) == dict(seed_workers=1, threads_per_seed=2, allocated_cpus=2)
    a.seed_workers=2
    with pytest.raises(ValueError, match='budget'):
        pool.thread_budget(a)
    a.threads_per_seed=0
    with pytest.raises(ValueError): pool.thread_budget(a)


def test_phase_clock(capsys):
    clock = pool.PhaseClock('probe')
    clock.finish('training')
    assert clock.records['training']['wall_seconds'] >= 0
    assert clock.records['training']['cpu_seconds'] >= 0
    assert 'probe' in capsys.readouterr().out


PROBE = '''
import os, struct, time
class Unreadable:  # pickles in the sender, cannot be rebuilt in the parent
    def __reduce__(self): return (unreadable, ())
def unreadable(): raise MemoryError('SIMULATED_RECEIVER_ALLOCATION_FAILURE')
class Slow:  # takes two seconds to rebuild in the parent
    def __init__(self, text): self.text = text
    def __reduce__(self): return (slow, (self.text,))
def slow(text): time.sleep(2); return text
def probe(conn, mode):
    if mode == 'die': time.sleep(.5); os._exit(19)
    if mode == 'block': time.sleep(120)
    if mode == 'pid-block':
        with open(os.environ['PROBE_PIDS'], 'a') as f: f.write(f'{os.getpid()}\\n')
        time.sleep(120)
    if mode == 'partial':  # a framed message cut off mid-payload
        os.write(conn.fileno(), struct.pack('!i', 100) + b'1234567'); time.sleep(120)
    if mode == 'unreadable': conn.send(('ok', Unreadable())); conn.close(); return  # exits 0
    if mode == 'slow-error-die': conn.send(('error', Slow('TRACEBACK_SENTINEL'))); conn.close(); os._exit(19)
    if mode == 'garbled': conn.send('not a tuple'); conn.close(); return
    if mode == 'bad-status': conn.send(('weird', 1)); conn.close(); return
    if mode == 'double': conn.send(('ok', 'first')); conn.send(('ok', 'second')); conn.close(); return
    if mode == 'ok-die': conn.send(('ok', 'x')); conn.close(); os._exit(19)
    conn.send(('ok', mode if mode != 'linger' else b'x' * 2**23)); conn.close()
    if mode == 'linger': time.sleep(120)  # result sent, process never exits
'''


@pytest.fixture
def probe(tmp_path, monkeypatch):
    (tmp_path/'wave_probe.py').write_text(PROBE)
    monkeypatch.syspath_prepend(str(tmp_path))
    from wave_probe import probe
    return probe


def bounded(seconds, fn, *args, **kw):
    import multiprocessing as mp
    start = time.monotonic()
    try: return fn(*args, **kw)
    finally:
        assert time.monotonic()-start < seconds
        assert not mp.active_children()


def test_wave_failure_stops_siblings(probe):
    assert bounded(60, pool.run_wave, [('a',), ('b',)], target=probe) == ['a', 'b']
    with pytest.raises(RuntimeError, match='exited without result'):
        bounded(60, pool.run_wave, [('block',), ('die',)], target=probe)


@pytest.mark.parametrize('stall', ['linger', 'partial'])
def test_wave_failure_seen_while_sibling_stalls(probe, stall):
    with pytest.raises(RuntimeError):
        bounded(60, pool.run_wave, [(stall,), ('die',)], target=probe)


def test_wave_keeps_result_of_worker_that_hangs_on_exit(probe):
    out = bounded(60, pool.run_wave, [('linger',), ('a',)], target=probe, exit_grace=1)
    assert out == [b'x' * 2**23, 'a']


def test_wave_reader_exception_with_clean_exit_fails_fast(probe):
    with pytest.raises(RuntimeError, match='could not be read.*SIMULATED_RECEIVER_ALLOCATION_FAILURE'):
        bounded(30, pool.run_wave, [('unreadable',), ('block',)], target=probe)


def test_wave_keeps_traceback_sent_before_abnormal_exit(probe):
    with pytest.raises(RuntimeError, match='TRACEBACK_SENTINEL') as info:
        bounded(30, pool.run_wave, [('slow-error-die',), ('a',)], target=probe)
    assert 'exit code 19' in str(info.value)


@pytest.mark.parametrize('mode', ['garbled', 'bad-status'])
def test_wave_rejects_malformed_message(probe, mode):
    with pytest.raises(RuntimeError, match='could not be read'):
        bounded(30, pool.run_wave, [(mode,), ('block',)], target=probe)


def test_wave_reads_one_message_per_worker(probe):
    assert bounded(30, pool.run_wave, [('double',), ('a',)], target=probe) == ['first', 'a']


def test_wave_rejects_abnormal_exit_after_result(probe):
    with pytest.raises(RuntimeError, match='exit code 19 after its result'):
        bounded(30, pool.run_wave, [('ok-die',), ('a',)], target=probe)


def test_wave_leaks_no_descriptors(probe):
    fds = lambda: len(os.listdir('/proc/self/fd' if os.path.exists('/proc/self/fd') else '/dev/fd'))
    pool.run_wave([('a',)], target=probe)  # warm multiprocessing's own helpers
    before = fds()
    for _ in range(3):
        pool.run_wave([('a',), ('b',)], target=probe)
        with pytest.raises(RuntimeError): pool.run_wave([('block',), ('die',)], target=probe)
        pool.run_wave([('linger',)], target=probe, exit_grace=.5)
    assert fds() == before


@pytest.mark.parametrize('sig, codes', [(signal.SIGTERM, {143}), (signal.SIGINT, {-signal.SIGINT, 1, 130})])
def test_wave_parent_signal_cleans_up_children(probe, tmp_path, sig, codes):
    pids = tmp_path/'pids'; script = tmp_path/'driver.py'
    script.write_text(f'''
import importlib.util, sys
sys.path.insert(0, {str(tmp_path)!r})
spec = importlib.util.spec_from_file_location('production_pool', {pool.__file__!r})
pool = importlib.util.module_from_spec(spec); spec.loader.exec_module(pool)
from wave_probe import probe
if __name__ == '__main__': pool.run_wave([('pid-block',), ('pid-block',)], target=probe)
''')
    env = dict(os.environ, PROBE_PIDS=str(pids))
    proc = subprocess.Popen([sys.executable, str(script)], env=env)
    deadline = time.monotonic()+30
    while time.monotonic() < deadline and (not pids.exists() or len(pids.read_text().split()) < 2): time.sleep(.1)
    children = [int(p) for p in pids.read_text().split()]; assert len(children) == 2
    proc.send_signal(sig); assert proc.wait(timeout=30) in codes
    for _ in range(50):
        if all(not alive(pid) for pid in children): break
        time.sleep(.1)
    assert all(not alive(pid) for pid in children)


def alive(pid):
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    return True


@pytest.mark.parametrize('chunk', [1, 29, 1000])
def test_export_statistics_production_pool_bitwise(tmp_path, monkeypatch, chunk):
    from production_pool_statistics_reference import export_statistics as old_export
    paths = inputs(tmp_path); a = args(tmp_path, paths)
    records, quota, *_ = pool.inspect(a)
    a.out.mkdir(); pool.pool(a, records, quota, a.out)
    rng = np.random.default_rng(802)
    with h5py.File(a.out/'pool.h5') as h:
        raw = h['raw']
        assert raw.chunks is None and raw.dtype == np.dtype('f4')
        nodes = rng.integers(0, 9, len(raw), dtype='i4')
        labels = np.array([0, 3, 7], dtype='i4')[nodes % 3]
        distances = rng.random(len(raw))
        for name, data in [('nodes', nodes), ('labels', labels), ('distances', distances)]:
            np.save(a.out/f'{name}.npy', data)
        nodes = np.load(a.out/'nodes.npy', mmap_mode='r')
        labels = np.load(a.out/'labels.npy', mmap_mode='r')
        distances = np.load(a.out/'distances.npy', mmap_mode='r')
        old = old_export(raw, nodes, distances, labels, 10, a.out, 'old', chunk)
        # Statistics must never create a disk buffer or read individual columns.
        def no_scratch(*args, **kwargs):
            raise AssertionError('statistics scratch allocation')
        monkeypatch.setattr(np.lib.format, 'open_memmap', no_scratch)
        class RowsOnly:
            shape = raw.shape
            dtype = raw.dtype
            def __len__(self): return len(raw)
            def __getitem__(self, key):
                assert isinstance(key, slice)
                return raw[key]
        new = pool.export_statistics(RowsOnly(), nodes, distances, labels, 10, a.out, 'new', chunk)
    assert old[0].tobytes() == new[0].tobytes()
    assert np.float64(old[1]).tobytes() == np.float64(new[1]).tobytes()
    assert list(old[2]) == list(new[2])
    for c in old[2]:
        assert np.asarray(old[2][c]).tobytes() == np.asarray(new[2][c]).tobytes()


def brute_effect(x, inside):
    """2 AUC - 1 by pairwise comparison, ties counted as half."""
    a, b = x[inside].astype('f8')[:, None], x[~inside].astype('f8')[None, :]
    return 2*((a > b).mean() + .5*(a == b).mean()) - 1


def test_rank_effect_matches_brute_force():
    rng = np.random.default_rng(5)
    X = np.round(rng.normal(size=(301, 3)), 1).astype('f4')  # many ties
    labels = rng.integers(0, 4, len(X))
    E = pool.rank_effects(X, labels, 5)
    assert np.isnan(E[4]).all()
    for c in range(4):
        for j in range(3):
            assert E[c, j] == pytest.approx(brute_effect(X[:, j], labels == c), abs=1e-12)
    # the float64 (argsort) path agrees with the packed float32 sort, and -0.0 ties +0.0
    assert np.array_equal(pool.rank_effects(X.astype('f8'), labels, 5), E, equal_nan=True)
    z = np.array([[-0.], [0.], [1.], [-1.]], 'f4'); zl = np.array([0, 1, 1, 0])
    assert pool.rank_effects(z, zl, 2)[1, 0] == pytest.approx(brute_effect(z[:, 0], zl == 1))
    assert pool.rank_sums(z[:, 0], zl, 2)[1] == 0 and pool.rank_sums(np.array([3, -2, 1, 7], 'f4'), zl, 2)[1] == 2
    # a cluster holding exactly the largest values has E = 1
    x = np.arange(10.)[:, None]; lab = (x[:, 0] >= 7).astype(int)
    assert pool.rank_effects(x, lab, 2)[:, 0].tolist() == [-1., 1.]


def effects_table(rows):
    """K x 7 effects with (E_chi, E_grad, E_beta) per cluster in NAMES' columns."""
    E = np.zeros((len(rows), 7)); i = pool.NAMES.index
    for c, (chi, grad, beta) in enumerate(rows):
        E[c, [i('log_chi_B'), i('log_grad_ln_b2'), i('log_beta_inv')]] = chi, grad, beta
    return E


def test_candidates_strict_signs_and_order():
    E = effects_table([(.2, .1, -.1), (0., .5, -.5), (.6, .1, -.1), (.4, 0., -.1),
                       (.5, .2, 0.), (.3, .3, -.3), (np.nan,)*3, (.3, .1, -.2)])
    c = pool.sheet_candidates(E, pool.NAMES)
    # zeros never count; NaN (empty) fails; ordered by E_chi, ties by ID and recorded
    assert c['order'] == [2, 5, 7, 0]
    assert c['ties'] == [[5, 7]]
    assert [x['candidate'] for x in c['clusters']] == [True, False, True, False, False, True, False, True]
    assert c['clusters'][1]['E_log_chi_B_positive'] is False and c['clusters'][1]['E_log_grad_ln_b2_positive'] is True
    assert c['clusters'][4]['E_log_beta_inv_negative'] is False
    assert pool.sheet_candidates(effects_table([(.1, .1, .1)]), pool.NAMES)['order'] == []


def labelling(*blocks):
    return np.concatenate([np.full(n, c) for c, n in blocks])


def jaccards(labels):
    import itertools
    k = {s: int(l.max())+1 for s, l in labels.items()}
    return {(i, j): pool.jaccard_matrix(labels[i], labels[j], k[i], k[j], chunk=7)
            for i, j in itertools.combinations(labels, 2)}


def test_jaccard_matrix_and_mutual_best():
    a = np.array([0, 0, 1, 1, 2]); b = np.array([1, 1, 0, 2, 2])
    J = pool.jaccard_matrix(a, b, 3, 3, chunk=2)
    assert J[0, 1] == 1 and J[1, 0] == J[2, 2] == .5 and J[1, 2] == 1/3
    flags = pool.mutual_best(J)
    # cluster 1 of a prefers 0 of b; 2 of a and 2 of b prefer each other although 1 of a overlaps 2 of b
    assert flags[0, 1] and flags[1, 0] and flags[2, 2] and flags.sum() == 3
    J[1, 2] = .6  # now 2 of b prefers 1 of a, which prefers 2 of b: (1, 2) mutual, (1, 0) and (2, 2) not
    flags = pool.mutual_best(J)
    assert flags[0, 1] and flags[1, 2] and flags.sum() == 2


# Three seeds on 100 rows: seed 43 relabels seed 42, seed 44 relabels and shifts 5 rows.
SEEDS = {42: labelling((0, 40), (1, 30), (2, 20), (3, 10)),
         43: labelling((2, 40), (0, 30), (3, 20), (1, 10)),
         44: labelling((1, 38), (0, 32), (2, 20), (3, 10))}


def test_selection_first_and_second_reproduced():
    orders = {42: [1, 3], 43: [0, 1], 44: [0, 3]}
    picks, audit = pool.select_group(orders, jaccards(SEEDS))
    assert audit['status'] == 'selected' and picks == {42: [1, 3], 43: [0, 1], 44: [0, 3]}
    assert audit['first']['reproduced'] and audit['second']['reproduced']
    assert set(audit['first']['pairs']) == {'42-43', '42-44', '43-44'}


def test_selection_second_refused():
    # the second candidates do not correspond (cluster 3 of 42 vs cluster 3 of 43)
    picks, audit = pool.select_group({42: [1, 3], 43: [0, 3], 44: [0, 3]}, jaccards(SEEDS))
    assert audit['status'] == 'selected' and picks == {42: [1], 43: [0], 44: [0]}
    assert not audit['second']['reproduced'] and '42-43' in audit['reason']
    # one seed has no second candidate
    picks, audit = pool.select_group({42: [1, 3], 43: [0], 44: [0, 3]}, jaccards(SEEDS))
    assert picks == {42: [1], 43: [0], 44: [0]} and audit['second']['missing_in_seeds'] == [43]
    # a reproduced second without a reproduced first selects nothing
    picks, audit = pool.select_group({42: [0, 1], 43: [1, 0], 44: [1, 0]}, jaccards(SEEDS))
    assert audit['status'] == 'none' and all(p == [] for p in picks.values()) and audit['second'] is None


def test_selection_first_not_reproduced():
    # seeds 42 and 43 agree, 44 picks a non-matching cluster: unanimity fails for the whole group
    picks, audit = pool.select_group({42: [1], 43: [0], 44: [2]}, jaccards(SEEDS))
    assert audit['status'] == 'none' and picks == {42: [], 43: [], 44: []}
    assert audit['first']['pairs']['42-43']['mutual'] and not audit['first']['pairs']['42-44']['mutual']
    assert 'not mutual' in audit['reason']


def test_selection_empty_signature_and_single_seed():
    picks, audit = pool.select_group({42: [1], 43: [], 44: [0]}, jaccards(SEEDS))
    assert audit['status'] == 'none' and 'empty signature set' in audit['reason'] and '43' in audit['reason']
    assert all(p == [] for p in picks.values())
    picks, audit = pool.select_group({42: [1]}, {})
    assert audit['status'] == 'unavailable' and picks == {42: []} and 'fewer than two seeds' in audit['reason']


def test_selection_exports_serialize():
    J = jaccards(SEEDS)
    picks, audit = pool.select_group({42: [1, 3], 43: [0, 1], 44: [0, 3]}, J)
    overlap = pool.seed_overlap(J)
    assert overlap['42-43']['jaccard'].shape == (4, 4) and overlap['42-43']['mutual_best'].dtype == bool
    E = effects_table([(.2, .1, -.1), (np.nan,)*3])
    for x in (audit, overlap, pool.sheet_candidates(E, pool.NAMES), E):
        line = pool.one_line(x)
        assert '\n' not in line and json.loads(line) is not None
    assert json.loads(pool.one_line(E))[1][0] is None
