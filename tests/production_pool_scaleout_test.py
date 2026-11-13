"""Scale-out driver regression checks."""
from production_pool_test import pool, inputs, args, ROOT
import json
import os
import subprocess
import sys
from pathlib import Path
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
            ix = np.array([59,0,7,7,21,42])
            equal(raw.rows(ix),stored.rows(ix),plain.rows(ix))
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


@pytest.mark.parametrize('statistics',['inline','later'])
def test_fit_collect_round_trip(tmp_path,statistics):
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
    fits = [run(tmp_path/f'fit-{seed}','--reuse-pool',prepared,'--fit','3x2',seed,'--scratch',tmp_path/'scratch','--statistics',statistics)
            for seed in [42,43]]
    before = [pool.digest(path/'3x2-s42.h5') if i==0 else pool.digest(path/'3x2-s43.h5') for i,path in enumerate(fits)]
    sources = list(fits)
    if statistics == 'later':
        refused = subprocess.run([*base,'--out',str(tmp_path/'refused'),'--reuse-pool',str(prepared),
                                  '--collect',*map(str,fits)],env=env,capture_output=True,text=True)
        assert refused.returncode != 0
        assert 'statistics are deferred' in refused.stderr
        fits = [run(tmp_path/f'complete-{seed}','--statistics-only',path)
                for seed,path in zip([42,43],fits)]
    final = run(tmp_path/'collected','--reuse-pool',prepared,'--collect',*fits)
    manifest = json.loads((final/'freeze-manifest.json').read_text())
    assert len(manifest['models']) == 2 and len(manifest['selection']) == 1
    assert (final/'frozen-3x2.h5').exists()
    assert before == [pool.digest(path/f'3x2-s{seed}.h5') for seed,path in zip([42,43],sources)]


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
                                        names=pool.SEVEN_FEATURE_NAMES,n_clusters=4,workers=1)
        parallel = pool.export_statistics(raw,nodes,distances,labels,10,tmp_path,'parallel',29,
                                          names=pool.SEVEN_FEATURE_NAMES,n_clusters=4,workers=2)
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



@pytest.mark.parametrize('full_run,pool_new,statistics,seed_workers,workers,output_per_row,scratch_bpr,output_bpr',[
    (False,False,'inline',1,7,16,56,16),
    (False,False,'later',1,7,16,0,16),
    (True,True,'inline',1,7,16,56,56),
    (True,True,'inline',2,7,16,112,68),
    (True,False,'inline',2,7,16,112,40),
    (True,True,'later',2,7,16,8,92),
    (False,False,'inline',1,1,4,16,4),
    (False,False,'inline',1,3,4,24,4),
    (False,False,'inline',1,7,4,56,4),
])
def test_fit_disk_budget_figures(tmp_path,monkeypatch,full_run,pool_new,statistics,
                                 seed_workers,workers,output_per_row,scratch_bpr,output_bpr):
    # Hand-counted B/row: pool 28, four labels 16, live nodes/distances 12/seed.
    # Deferred full runs retain all four 16-byte fits and need an 8-byte scaler column.
    # Statistics-only copies one 4-byte label array; one worker has a 16-byte floor.
    scratch = tmp_path/'scratch'; scratch.mkdir()
    output = tmp_path/'output'; output.mkdir()
    a = SimpleNamespace(scratch=scratch,out=output/'fresh',threads_per_seed=workers,
                        seed_workers=seed_workers,seeds=[42,43],lattices=['91x64','48x34'],
                        statistics=statistics,write_scaled=False,min_free_gb=0)
    monkeypatch.setattr(pool.shutil,'disk_usage',lambda path: SimpleNamespace(free=10**15))
    budget = pool.fit_disk_budget(a,10**9,7,itemsize=4,full_run=full_run,
                                  pool_new=pool_new,output_per_row=output_per_row)
    assert budget['scratch_bytes'] == scratch_bpr*10**9
    assert budget['output_bytes'] == output_bpr*10**9


@pytest.mark.parametrize('cluster,rc_path',[
    ('Tigris',None),('tigris',None),('SPORC',None),('sporc',None),
    ('','/shared/rc/awesom-test/fresh'),('','/home/thccrg/awesom-test/fresh'),('','/home/thccrg'),('','/.autofs/shared/rc/awesom-test/fresh'),
])
def test_fit_disk_budget_skips_free_space_on_tigris(tmp_path,monkeypatch,cluster,rc_path):
    # No filesystem free-space reads on RIT RC (Tigris or SPORC; user rule 2026-11-06, extended 2026-11-12).
    monkeypatch.setenv('SLURM_CLUSTER_NAME',cluster)
    def disk_usage(path): raise AssertionError(f'disk_usage({path}) called on RIT RC')
    monkeypatch.setattr(pool.shutil,'disk_usage',disk_usage)
    scratch = tmp_path/'scratch'; scratch.mkdir()
    out = Path(rc_path) if rc_path else tmp_path/'fresh'
    a = SimpleNamespace(scratch=scratch,out=out,threads_per_seed=7,min_free_gb=10**6)
    assert pool.on_tigris(a.out,scratch)
    assert pool.fit_disk_budget(a,10,7) == dict(scratch_bytes=560,output_bytes=160,shared_filesystem=None)
    monkeypatch.setenv('SLURM_CLUSTER_NAME','gp')
    assert not pool.on_tigris(tmp_path/'fresh',scratch)
    assert not pool.on_tigris(Path('/home/thccrg2/fresh'),scratch)

def test_collect_requires_reused_pool(tmp_path,capsys):
    with pytest.raises(SystemExit) as error:
        pool.main(['--inputs','missing.h5','--anchor','1','--held-out','3',
                   '--out',str(tmp_path/'fresh'),'--collect','missing-fit'])
    assert error.value.code == 2
    assert '--collect requires --reuse-pool' in capsys.readouterr().err
    assert not (tmp_path/'fresh').exists()


@pytest.mark.parametrize('dtype',['f4','f8'])
def test_merge_buffer_size_bitwise(tmp_path,monkeypatch,dtype):
    from aweSOM import pool_statistics as statistics
    rng = np.random.default_rng(81)
    values = np.round(rng.normal(size=1003),1).astype(dtype)
    labels = rng.integers(0,3,len(values),dtype='i4')
    counts = np.bincount(labels,minlength=4)
    results = []
    for name,cap in [('small',16),('large',2**26)]:
        directory = tmp_path/name; directory.mkdir()
        monkeypatch.setattr(statistics,'MERGE_BUFFER_RECORDS',cap)
        blocks = ((values[a:a+29],labels[a:a+29]) for a in range(0,len(values),29))
        results.append(statistics.summarize(blocks,counts,[.25,.5,.75],directory,29))
    for a,b in zip(*results,strict=True): assert a.tobytes() == b.tobytes()


def test_full_run_checks_scratch_before_pooling(tmp_path,monkeypatch):
    paths = inputs(tmp_path); a = args(tmp_path,paths)
    scratch = tmp_path/'scratch'; scratch.mkdir()
    usage = pool.shutil.disk_usage
    def disk_usage(path):
        return SimpleNamespace(free=0) if Path(path) == scratch else usage(path)
    from pathlib import Path
    monkeypatch.setattr(pool.shutil,'disk_usage',disk_usage)
    command = ['--inputs',*map(str,paths),'--anchor','1','--held-out','3',
               '--out',str(a.out),'--rows-per-snapshot','60','--rows-per-epoch','60',
               '--scratch',str(scratch),'--min-free-gb','0','--threads-per-seed','1']
    with pytest.raises(ValueError,match='Low scratch disk'): pool.main(command)
    assert not a.out.exists()


def test_deferred_statistics_matches_inline(tmp_path,monkeypatch):
    from aweSOM import SOMModel
    from aweSOM import stream
    paths = inputs(tmp_path); a = args(tmp_path,paths)
    a.threads_per_seed = 2; a.code_commit = '0'*40; a.bmu = 'patch'
    records,quota,*_ = pool.inspect(a)
    a.anchor_attrs = records[0]['attrs']
    a.out.mkdir(); scaling = pool.pool(a,records,quota,a.out)
    state = dict(records=records,quota=quota,disk={},mismatches=[],scaling=scaling,
                 settings={key:getattr(a,key) for key in pool.APPROVED},anchor=a.anchor,held_out=a.held_out)
    monkeypatch.setattr(pool,'execution_info',lambda:dict(test_execution=True))
    monkeypatch.setattr(stream,'peak_rss_bytes',lambda:123456)
    inline = tmp_path/'inline'; inline.mkdir()
    deferred = tmp_path/'deferred'; deferred.mkdir()
    for location in (inline,deferred):
        (location/'pool.h5').symlink_to(a.out/'pool.h5')
        pool.dump(location/'pool-state.json',state)
    a.statistics = 'inline'; expected = pool.fit(a,(3,2),42,scaling,inline)
    a.statistics = 'later'; pending = pool.fit(a,(3,2),42,scaling,deferred)
    assert pending['statistics'] == 'deferred'
    assert all(pending[key] is None for key in pool.STATISTIC_FIELDS)
    assert (deferred/'3x2-s42-nodes.npy').exists()
    assert (deferred/'3x2-s42-distances.npy').exists()
    deferred_model = SOMModel.load(deferred/'3x2-s42.h5')
    assert deferred_model.metadata['statistics'] == 'deferred'
    assert 'cluster_effects' not in deferred_model.metadata
    pool.dump(deferred/'fit-results.json',dict(models=[pending],settings=state['settings'],bmu=a.bmu))
    with pytest.raises(ValueError,match='statistics are deferred'):
        pool.group_selection(deferred,[pending],a.chunk)
    result = pool.main(['--statistics-only',str(deferred),'--out',str(tmp_path/'completed'),
                       '--threads-per-seed','2','--min-free-gb','0'])
    actual = result[0]
    assert actual['statistics'] == 'complete'
    for key in pool.STATISTIC_FIELDS:
        assert pool.one_line(actual[key]) == pool.one_line(expected[key])
    def without_timings(record):
        record = json.loads(pool.one_line(record))
        record.pop('phase_timings'); record.pop('wall_seconds')
        info = record['training_info']
        # stream.train_batch_stream documents exactly these elapsed-second fields
        # (per epoch, plus the per-rank lists of the scale-out trainer).
        # Unknown future fields stay in the comparison, even if named *time*.
        for key in ('epoch_time','io_wait','bmu_time','acc_time','update_time','final_time',
                    'reduce_time','checkpoint_time','rank_io_wait','rank_bmu','rank_acc',
                    'rank_pass_time'):
            info.pop(key)
        # label_rows records these elapsed seconds per rank, and the assembly and CRC check times.
        for key in ('crc_check_seconds','assemble_seconds','assemble_read_seconds',
                    'assemble_write_seconds','assemble_fsync_seconds'):
            record['labelling'].pop(key)
        for rank in record['labelling']['ranks']:
            for key in ('seconds','compute_seconds','io_wait_seconds','write_seconds',
                        'close_seconds','pwrite_seconds','fsync_seconds','wall_seconds'):
                rank.pop(key)
        return record
    assert without_timings(actual) == without_timings(expected)
    # Check serialized bytes too: dict equality ignores insertion order.
    assert pool.one_line(without_timings(actual)) == pool.one_line(without_timings(expected))
    finished = tmp_path/'completed/results'
    final_model = SOMModel.load(finished/'3x2-s42.h5')
    inline_model = SOMModel.load(inline/'3x2-s42.h5')
    assert final_model.node_dist_q99.tobytes() == inline_model.node_dist_q99.tobytes()
    assert final_model.reject_distance == inline_model.reject_distance
    assert final_model.metadata == inline_model.metadata
    assert 'statistics' not in final_model.metadata
    with h5py.File(finished/'3x2-s42.h5') as final_h5, h5py.File(inline/'3x2-s42.h5') as inline_h5:
        def compare(name, item):
            if isinstance(item,h5py.Dataset):
                other = inline_h5[name]
                assert item.dtype == other.dtype and item.shape == other.shape
                if item.dtype.hasobject:
                    assert item[()].tolist() == other[()].tolist()
                else:
                    assert item[()].tobytes() == other[()].tobytes()
        final_h5.visititems(compare)
    assert (finished/'3x2-s42-umat.npy').read_bytes() == (inline/'3x2-s42-umat.npy').read_bytes()
    assert (finished/'3x2-s42-labels.npy').read_bytes() == (inline/'3x2-s42-labels.npy').read_bytes()
    assert not (inline/'3x2-s42-nodes.npy').exists()
    assert not (inline/'3x2-s42-distances.npy').exists()
    for suffix in ('nodes','distances'):
        assert (deferred/f'3x2-s42-{suffix}.npy').exists()
        assert not (finished/f'3x2-s42-{suffix}.npy').exists()
    assert pool.digest(deferred/'3x2-s42-labels.npy') == pool.digest(finished/'3x2-s42-labels.npy')
