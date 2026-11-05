"""Scale-out driver regression checks."""
from production_pool_test import pool, inputs, args, ROOT
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import h5py
import numpy as np
import pytest


def test_raw_scaling_matches_stored(tmp_path):
    paths = inputs(tmp_path); a = args(tmp_path,paths); a.write_scaled = True
    records,quota,*_ = pool.inspect(a); a.out.mkdir()
    scaling = pool.pool(a,records,quota,a.out)
    from aweSOM import Scaler
    scaler = Scaler('median-iqr',np.array(scaling['center']),np.array(scaling['scale']))
    with h5py.File(a.out/'pool.h5') as h:
        raw = pool.EpochSource(h['raw'],scaling['strata'],60,42,17,
                               transform=lambda x: scaler.transform(x.astype('f8')))
        stored = pool.EpochSource(h['scaled'],scaling['strata'],60,42,17)
        plain = pool.EpochSource(h['raw'],scaling['strata'],60,42,17)
        def equal(actual,expected,unscaled):
            assert actual.tobytes() == expected.tobytes()
            assert actual.tobytes() == scaler.transform(unscaled.astype('f8')).tobytes()
        for _ in range(3):
            for a,b,c in zip(raw(),stored(),plain(),strict=True): equal(a,b,c)
            if callable(getattr(raw,'rows',None)):
                ix = np.array([59,0,7,7,21,42])
                equal(raw.rows(ix),stored.rows(ix),plain.rows(ix))
            if callable(getattr(raw,'lazy_chunks',None)):
                for a,b,c in zip(raw.lazy_chunks(),stored.lazy_chunks(),plain.lazy_chunks(),strict=True):
                    assert a[0] == b[0] == c[0]
                    equal(a[1](),b[1](),c[1]())
                    equal(a[1](),b[1](),c[1]())


@pytest.mark.parametrize('cores,workers,threads',[(32,2,16),(72,1,72),(144,1,144),(144,2,72)])
def test_wide_thread_budget(monkeypatch,cores,workers,threads):
    import numba
    monkeypatch.setenv('SLURM_CPUS_PER_TASK',str(cores))
    monkeypatch.setattr(numba.config,'NUMBA_NUM_THREADS',cores)
    a = SimpleNamespace(seed_workers=workers,seeds=[42,43],threads_per_seed=None)
    assert pool.thread_budget(a)['threads_per_seed'] == threads


def test_fit_collect_round_trip(tmp_path):
    paths = inputs(tmp_path)
    base = [sys.executable,str(ROOT/'examples/grmhd-bbh/production_pool.py'),
            '--inputs',*map(str,paths),'--anchor','1','--held-out','3',
            '--rows-per-snapshot','60','--rows-per-epoch','60','--lattices','3x2',
            '--seeds','42','43','--epochs','1','--chunk','29','--min-free-gb','0',
            '--target-clusters','1','--threads-per-seed','1']
    env = dict(os.environ,NUMBA_NUM_THREADS='2',NUMBA_CACHE_DIR=str(tmp_path/'cache'))
    def run(out,*extra):
        result = subprocess.run([*base,'--out',str(out),*map(str,extra)],env=env,capture_output=True,text=True)
        assert result.returncode == 0,result.stdout+result.stderr
        return out/'results'
    prepared = run(tmp_path/'prepared','--pool-only')
    with h5py.File(prepared/'pool.h5') as h: assert 'scaled' not in h
    (tmp_path/'scratch').mkdir()
    fits = [run(tmp_path/f'fit-{seed}','--reuse-pool',prepared,'--fit','3x2',seed,'--scratch',tmp_path/'scratch')
            for seed in [42,43]]
    before = [pool.digest(path/'3x2-s42.h5') if i==0 else pool.digest(path/'3x2-s43.h5') for i,path in enumerate(fits)]
    final = run(tmp_path/'collected','--reuse-pool',prepared,'--collect',*fits)
    manifest = json.loads((final/'freeze-manifest.json').read_text())
    assert len(manifest['models']) == 2 and len(manifest['selection']) == 1
    assert (final/'frozen-3x2.h5').exists()
    assert before == [pool.digest(path/f'3x2-s{seed}.h5') for seed,path in zip([42,43],fits)]


@pytest.mark.parametrize('threads',[1,2])
def test_cpu_no_checkpoint_baseline(threads):
    import numba
    from aweSOM.stream import train_batch_stream
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(threads)
        with np.load(ROOT/'tests/fixtures/scale_out_driver_cpu.npz') as fixture:
            weights,_ = train_batch_stream(fixture['raw'],12,8,epochs=3,chunk=2048,
                bmu='patch',seed=42,groups=64,r_min=2,final_pass='none')
            assert np.array_equal(weights,fixture['weights'])
    finally: numba.set_num_threads(previous)


def test_parallel_statistics_bitwise(tmp_path):
    rng = np.random.default_rng(17)
    values = np.round(rng.normal(size=(1003,7)),1).astype('f4')
    values[:5,0] = [-0.,0.,np.nextafter(np.float32(0),np.float32(1)),1.,-1.]
    nodes = rng.integers(0,9,len(values),dtype='i4')
    np.save(tmp_path/'labels.npy',nodes%3)
    labels = np.load(tmp_path/'labels.npy',mmap_mode='r')
    distances = rng.random(len(values))
    with h5py.File(tmp_path/'raw.h5','w') as h:
        h.create_dataset('raw',data=values)
    with h5py.File(tmp_path/'raw.h5','r') as h:
        raw = h['raw']
        serial = pool.export_statistics(raw,nodes,distances,labels,10,tmp_path,'serial',29,
                                        names=pool.NAMES,n_clusters=4,workers=1)
        parallel = pool.export_statistics(raw,nodes,distances,labels,10,tmp_path,'parallel',29,
                                          names=pool.NAMES,n_clusters=4,workers=2)
    assert pool.one_line(serial) == pool.one_line(parallel)


@pytest.mark.parametrize('low',['scratch','output','floor',None])
def test_reused_fit_disk_checks(tmp_path,monkeypatch,low):
    scratch = tmp_path/'scratch'; scratch.mkdir()
    a = SimpleNamespace(scratch=scratch,out=tmp_path/'fresh',threads_per_seed=7,
                        min_free_gb=1 if low == 'floor' else 0)
    calls = []
    def disk_usage(path):
        calls.append(path)
        free = 100 if (low == 'scratch' and path == scratch) or (low == 'output' and path == tmp_path) else 10000
        return SimpleNamespace(free=free)
    monkeypatch.setattr(pool.shutil,'disk_usage',disk_usage)
    if low:
        with pytest.raises(ValueError,match='Low .*disk'): pool.fit_disk_budget(a,10,7)
    else:
        assert pool.fit_disk_budget(a,10,7) == dict(scratch_bytes=560,output_bytes=160,shared_filesystem=True)
    assert calls == [scratch,tmp_path]
    assert not a.out.exists()


def test_collect_requires_reused_pool(tmp_path,capsys):
    with pytest.raises(SystemExit) as error:
        pool.main(['--inputs','missing.h5','--anchor','1','--held-out','3',
                   '--out',str(tmp_path/'fresh'),'--collect','missing-fit'])
    assert error.value.code == 2
    assert '--collect requires --reuse-pool' in capsys.readouterr().err
    assert not (tmp_path/'fresh').exists()
