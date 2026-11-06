"""MPI path of production_pool.py --fit, with a fake communicator over spawned processes."""
from production_pool_test import pool, inputs, ROOT
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import pytest

DRIVER = ROOT/'examples/grmhd-bbh/production_pool.py'
COMMON = ['--anchor','1','--held-out','3','--rows-per-snapshot','60','--rows-per-epoch','60',
          '--epochs','3','--chunk','13','--min-free-gb','0','--target-clusters','1',
          '--threads-per-seed','1']


class FakeComm:
    """The mpi4py calls of the driver and train_batch_stream, over processes sharing a
    barrier and a manager dict. Collective k of rank r posts its payload under (k, r)."""

    def __init__(self, rank, size, barrier, slots):
        self.rank, self.size, self.barrier, self.slots, self.k = rank, size, barrier, slots, 0

    def Get_rank(self): return self.rank
    def Get_size(self): return self.size
    def Barrier(self): self.barrier.wait(120)

    def _exchange(self, obj):
        key = self.k; self.k += 1
        self.slots[(key, self.rank)] = obj
        self.barrier.wait(120)
        every = [self.slots[(key, r)] for r in range(self.size)]
        self.barrier.wait(120)
        del self.slots[(key, self.rank)]
        return every

    def Allgather(self, send, recv): recv[...] = np.stack(self._exchange(np.array(send)))
    def Bcast(self, buf, root=0): buf[...] = self._exchange(np.array(buf) if self.rank == root else None)[root]
    def bcast(self, obj, root=0): return self._exchange(obj if self.rank == root else None)[root]
    def allgather(self, obj): return self._exchange(obj)

    def gather(self, obj, root=0):
        every = self._exchange(obj)
        return every if self.rank == root else None


class StopFit(Exception):
    pass


def rank_main(conn, rank, size, barrier, slots, argv, watch, stop_after=None):
    """One fake rank: main(argv) with get_comm patched; reports writes under ``watch``.
    ``stop_after``: rank 0 raises StopFit after that many epochs (after the checkpoint)."""
    writes = []
    def audit(event, args):
        writing = event in ('os.mkdir', 'os.rename', 'os.symlink', 'os.link', 'os.remove') or (
            event == 'open' and (args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)
                                 or any(c in str(args[1] or '') for c in 'wax+')))
        if writing:
            paths = [os.fsdecode(a) for a in args if isinstance(a, (str, bytes, os.PathLike))]
            writes.extend(f'{event} {p}' for p in paths if p.startswith(watch))
    try:
        from aweSOM import distributed
        distributed.get_comm = lambda: FakeComm(rank, size, barrier, slots)
        if stop_after is not None:
            from aweSOM import stream
            train = stream.train_batch_stream
            def stop(e, w):
                if e + 1 == stop_after: raise StopFit(f'stopped after epoch {e + 1}')
            import functools  # partial keeps the signature, which the driver inspects
            stream.train_batch_stream = functools.partial(train, callback=stop)
        sys.addaudithook(audit)
        pool.main(argv)
        conn.send(('ok', writes))
    except BaseException:
        conn.send(('error', traceback.format_exc()))
    finally:
        conn.close()


def run_ranks(size, argv, watch, stop_after=None):
    ctx = mp.get_context('spawn')
    with ctx.Manager() as manager:
        barrier, slots = ctx.Barrier(size), manager.dict()
        pipes, procs = [], []
        for r in range(size):
            receive, send = ctx.Pipe(duplex=False)
            p = ctx.Process(target=rank_main, args=(send, r, size, barrier, slots, argv, str(watch), stop_after))
            p.start(); send.close(); pipes.append(receive); procs.append(p)
        results = [receive.recv() for receive in pipes]
        for p in procs:
            p.join(60)
    assert [p.exitcode for p in procs] == [0]*size
    if stop_after is not None:  # every rank stops: rank 0 on StopFit, the others on its broadcast
        assert all(status == 'error' for status, _ in results), results
        return [payload for _, payload in results]
    for status, payload in results:
        assert status == 'ok', payload
    return [payload for _, payload in results]


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('mpi-pool')
    paths = inputs(tmp)
    base = ['--inputs', *map(str, paths), *COMMON]
    env = dict(os.environ, NUMBA_NUM_THREADS='2', NUMBA_CACHE_DIR=str(tmp/'cache'))
    def run(out, *extra):
        result = subprocess.run([sys.executable, str(DRIVER), *base, '--out', str(out), *map(str, extra)],
                                env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout+result.stderr
        return out/'results'
    pool_dir = run(tmp/'prepared', '--pool-only')
    single = run(tmp/'single', '--reuse-pool', pool_dir, '--fit', '4x3', 42)
    return SimpleNamespace(tmp=tmp, base=base, pool=pool_dir, single=single)


def test_two_fake_ranks_match_single_process(prepared, monkeypatch):
    from aweSOM import SOMModel
    monkeypatch.setenv('NUMBA_CACHE_DIR', str(prepared.tmp/'cache'))
    out = prepared.tmp/'mpi2'
    argv = [*prepared.base, '--out', str(out), '--reuse-pool', str(prepared.pool), '--fit', '4x3', '42', '--mpi']
    writes = run_ranks(2, argv, out)
    assert writes[0] and not writes[1]  # rank 0 wrote every output, rank 1 nothing
    results = out/'results'
    a = SOMModel.load(results/'4x3-s42.h5').weights
    b = SOMModel.load(prepared.single/'4x3-s42.h5').weights
    rel = np.abs(a-b)/np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)
    print(f'two fake ranks vs single process: max relative weight difference {rel.max():.3e}')
    assert rel.max() < 1e-4
    rec = json.loads((results/'4x3-s42.json').read_text())
    ref = json.loads((prepared.single/'4x3-s42.json').read_text())
    assert rec['ranks'] == 2 and ref['ranks'] == 1
    assert rec['rank_hostnames'] == [os.uname().nodename]*2
    assert rec['rank_numba_threads'] == [1, 1] and rec['affinity_restored'] == [None, None]
    assert not {'rank_hostnames', 'rank_numba_threads', 'affinity_restored'} & set(ref)
    assert rec['training_info']['ranks'] == 2
    assert np.array(rec['training_info']['rank_rows']).sum(axis=1).tolist() == [60]*3
    assert min(np.array(rec['training_info']['rank_rows']).sum(axis=0)) > 0  # both ranks trained
    # The spawned worker keeps a per-process numba cache in results; MPI ranks use NUMBA_CACHE_DIR.
    names = lambda d: sorted(p.name for p in d.iterdir() if not p.name.startswith('numba-cache-'))
    assert names(results) == names(prepared.single)
    bundle = json.loads((results/'fit-results.json').read_text())
    assert bundle['models'] == [rec]
    # The README flow: a second seed on the MPI path, then --collect of both.
    other = prepared.tmp/'mpi2-s43'
    run_ranks(2, [*prepared.base, '--out', str(other), '--reuse-pool', str(prepared.pool), '--fit', '4x3', '43', '--mpi'], other)
    collected = prepared.tmp/'collected'
    result = subprocess.run([sys.executable, str(DRIVER), *prepared.base, '--lattices', '4x3', '--out', str(collected),
                             '--reuse-pool', str(prepared.pool), '--collect', str(results), str(other/'results')],
                            env=dict(os.environ, NUMBA_NUM_THREADS='2'), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    manifest = json.loads((collected/'results/freeze-manifest.json').read_text())
    assert [m['ranks'] for m in manifest['models']] == [2, 2] and (collected/'results/frozen-4x3.h5').exists()


def _max_rel(a, b):
    return (np.abs(a-b)/np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)).max()


@pytest.mark.parametrize('ranks', [1, 2])
def test_resume_continues_an_interrupted_fit(prepared, monkeypatch, ranks):
    """--resume on a fresh OUT continues the interrupted fit: 1 rank through the plain
    driver, 2 ranks through the MPI path. Another rank count is refused."""
    from aweSOM import SOMModel
    monkeypatch.setenv('NUMBA_CACHE_DIR', str(prepared.tmp/'cache'))
    fit = lambda out: [*prepared.base, '--out', str(out), '--reuse-pool', str(prepared.pool), '--fit', '4x3', '42']
    cut = prepared.tmp/f'cut-{ranks}'
    errors = run_ranks(ranks, [*fit(cut), '--mpi'], cut, stop_after=1)
    assert 'StopFit: stopped after epoch 1' in errors[0]
    checkpoint = cut/'results.partial/4x3-s42-checkpoint.npz'
    from aweSOM.stream import read_checkpoint
    assert read_checkpoint(str(checkpoint))[0]['epoch'] == 1 and read_checkpoint(str(checkpoint))[0]['ranks'] == ranks
    before = checkpoint.read_bytes()
    env = dict(os.environ, NUMBA_NUM_THREADS='2')
    plain = lambda out: subprocess.run([sys.executable, str(DRIVER), *fit(out), '--resume', str(checkpoint)],
                                       env=env, capture_output=True, text=True)
    out = prepared.tmp/f'resumed-{ranks}'
    if ranks == 1:
        result = plain(out)
        assert result.returncode == 0, result.stdout+result.stderr
    else:
        run_ranks(2, [*fit(out), '--mpi', '--resume', str(checkpoint)], out)
        result = plain(prepared.tmp/'resumed-wrong-ranks')
        assert result.returncode != 0
        assert 'checkpoint was written by a run with 2 rank(s), this run has 1' in result.stdout+result.stderr
    assert checkpoint.read_bytes() == before  # the source checkpoint is left unchanged
    rec = json.loads((out/'results/4x3-s42.json').read_text())
    assert rec['training_info']['resumed_from'] == 1 and rec['training_info']['ranks'] == ranks
    a = SOMModel.load(out/'results/4x3-s42.h5').weights
    b = SOMModel.load(prepared.single/'4x3-s42.h5').weights
    print(f'resumed on {ranks} rank(s) vs uninterrupted: max relative weight difference {_max_rel(a, b):.3e}')
    assert _max_rel(a, b) < 1e-4


def test_resume_requires_fit(prepared):
    with pytest.raises(ValueError, match='--resume requires --fit'):
        pool.main([*prepared.base, '--out', str(prepared.tmp/'resume-pool'), '--pool-only', '--resume', 'x'])


def _mpirun():
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return None
    return shutil.which('mpirun')


@pytest.mark.skipif(_mpirun() is None, reason='mpi4py or mpirun not available')
def test_real_mpirun_two_ranks(prepared):
    from aweSOM import SOMModel
    out = prepared.tmp/'mpirun2'
    env = dict(os.environ, NUMBA_NUM_THREADS='2', NUMBA_CACHE_DIR=str(prepared.tmp/'cache'), OMP_NUM_THREADS='1')
    cmd = [_mpirun(), '-n', '2', '--oversubscribe', sys.executable, '-m', 'mpi4py', str(DRIVER), *prepared.base,
           '--out', str(out), '--reuse-pool', str(prepared.pool), '--fit', '4x3', '42', '--mpi']
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-3000:]+result.stderr[-3000:]
    a = SOMModel.load(out/'results/4x3-s42.h5').weights
    b = SOMModel.load(prepared.single/'4x3-s42.h5').weights
    assert np.all(np.abs(a-b) <= 1e-4*np.maximum(np.abs(a), np.abs(b)))
    rec = json.loads((out/'results/4x3-s42.json').read_text())
    assert rec['ranks'] == 2 and len(rec['rank_hostnames']) == 2 and rec['rank_numba_threads'] == [1, 1]


def test_mpi_flag_without_communicator_fails(prepared, monkeypatch, capsys):
    from aweSOM import distributed
    monkeypatch.setattr(distributed, 'get_comm', lambda: None)
    out = prepared.tmp/'no-comm'
    with pytest.raises(SystemExit) as stop:
        pool.main([*prepared.base, '--out', str(out), '--reuse-pool', str(prepared.pool), '--fit', '4x3', '42', '--mpi'])
    assert stop.value.code != 0
    assert '--mpi: no MPI communicator with more than one rank' in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize('mode', [['--pool-only'], ['--collect', 'x'], ['--statistics-only', 'x']])
def test_mpi_refuses_other_modes(prepared, monkeypatch, capsys, mode):
    from aweSOM import distributed
    monkeypatch.setattr(distributed, 'get_comm', lambda: FakeComm(0, 2, None, None))
    monkeypatch.setenv('PMIX_RANK', '0')  # an MPI launch without --mpi takes the MPI path too
    out = prepared.tmp/'refused'
    with pytest.raises(SystemExit) as stop:
        pool.main([*prepared.base, '--out', str(out), '--reuse-pool', str(prepared.pool), *mode])
    assert stop.value.code != 0
    assert 'an MPI launch runs only --fit with --reuse-pool' in capsys.readouterr().err
    assert not out.exists()


def test_plain_run_never_initialises_mpi(monkeypatch):
    from aweSOM import distributed
    monkeypatch.setattr(distributed, 'get_comm', lambda: pytest.fail('get_comm called'))
    for key in pool.LAUNCHER_VARIABLES:
        monkeypatch.delenv(key, raising=False)
    assert pool.mpi_comm(SimpleNamespace(mpi=False)) == (None, None)
    monkeypatch.setattr(distributed, 'get_comm', lambda: None)
    monkeypatch.setenv('PMIX_RANK', '0')
    assert pool.mpi_comm(SimpleNamespace(mpi=False)) == (None, None)


def test_mpi_threads(monkeypatch):
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    monkeypatch.delenv('NUMBA_NUM_THREADS', raising=False)
    assert pool.mpi_threads(SimpleNamespace(threads_per_seed=None)) == cpus
    monkeypatch.setenv('NUMBA_NUM_THREADS', '1')
    assert pool.mpi_threads(SimpleNamespace(threads_per_seed=None)) == 1
    assert pool.mpi_threads(SimpleNamespace(threads_per_seed=cpus)) == cpus
    with pytest.raises(RuntimeError, match='affinity mask'):
        pool.mpi_threads(SimpleNamespace(threads_per_seed=cpus+1))


def test_restore_affinity():
    cpus = {'now': {0}}
    get = lambda pid: set(cpus['now'])
    def put(pid, saved): cpus['now'] = set(saved)
    assert pool.restore_affinity(None, get, put) is None
    assert pool.restore_affinity(frozenset({0}), get, put) is None
    rec = pool.restore_affinity(frozenset(range(32)), get, put)
    assert rec == dict(before=32, after_init=1, cpus_after_init=[0], after_restore=32)
