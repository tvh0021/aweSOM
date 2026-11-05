"""Scale-out features of aweSOM.stream: callable-source protocol, chunk ownership,
checkpoint/resume, the dev-parity fixture, and multi-process runs under mpirun."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import aweSOM.stream as stream
from aweSOM.inference import best_match_nodes
from aweSOM.stream import _epoch_chunks, _make_source, read_checkpoint, train_batch_stream

sys.path.insert(0, str(Path(__file__).parent))
import gmm_bbh7  # noqa: E402
import mpi_stream_worker as wk  # noqa: E402
from mpi_stream_worker import Windows  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
DEV_REF = FIXTURES / "stream_dev_82dee49_gmm_bbh7_200k_48x34.npz"
WORKER = Path(__file__).parent / "mpi_stream_worker.py"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class Chunks:
    """Stateless callable over an array (same chunk boundaries as an array source)."""

    def __init__(self, x, chunk, lazy=False, n_feat=None):
        self.x, self.chunk, self.reads, self.passes = x, chunk, 0, 0
        if n_feat is not None:
            self.n_feat = n_feat
        if lazy:
            self.lazy_chunks = self._lazy

    def _read(self, a):
        self.reads += 1
        return self.x[a : a + self.chunk]

    def _lazy(self):
        self.passes += 1
        for a in range(0, len(self.x), self.chunk):
            yield min(self.chunk, len(self.x) - a), (lambda a=a: self._read(a))

    def __call__(self):
        self.passes += 1
        for a in range(0, len(self.x), self.chunk):
            yield self._read(a)


@pytest.fixture(scope="module")
def gmm200k():
    x = gmm_bbh7.rows(200_000)
    ref = np.load(DEV_REF)
    if sha(x) != str(ref["rows_sha256"]):
        pytest.skip("gmm_bbh7 rows differ on this platform (numpy version or CPU); fixture not comparable")
    return x, ref


@pytest.fixture(scope="module")
def small():
    return gmm_bbh7.rows(30_000).astype(np.float64)


# ------------------------------------------------------------------ dev parity

def test_dev_parity_fixture_array(gmm200k):
    x, ref = gmm200k
    w, info = train_batch_stream(x, 48, 34, epochs=5, chunk=2**15, r_min=5, seed=42, final_pass="full")
    assert sha(w) == str(ref["weights_sha256"])
    np.testing.assert_array_equal(info["hits"], ref["hits"])
    np.testing.assert_array_equal(info["qe"], ref["qe"])


@pytest.mark.parametrize("lazy", [False, True])
def test_dev_parity_fixture_callable(gmm200k, lazy):
    x, ref = gmm200k
    src = Chunks(x, 2**15, lazy=lazy, n_feat=7)
    w, _ = train_batch_stream(src, 48, 34, epochs=5, r_min=5, seed=42, final_pass="full", n_samples=len(x))
    assert sha(w) == str(ref["weights_sha256"])


# ------------------------------------------------------------------ source protocol

def test_callable_known_shape_makes_no_counting_pass(small):
    src = Chunks(small, 4096, lazy=True)
    _make_source(src, len(small), n_features=7)
    assert src.passes == 0
    src = Chunks(small, 4096, lazy=True, n_feat=7)
    _make_source(src, len(small))
    assert src.passes == 0
    _make_source(src, None)
    assert src.passes == 1
    with pytest.raises(ValueError, match="columns"):
        _make_source(Chunks(small, 4096), None, n_features=5)


def test_callable_rows_reads_only_needed_chunks(small):
    src = Chunks(small, 1000, lazy=True, n_feat=7)
    s = _make_source(src, len(small))
    ix = np.array([5, 4999, 2100, 5])
    got = s.rows(ix)
    np.testing.assert_array_equal(got, small[ix])
    assert src.reads == 3  # chunks 0, 2, 4; stops after chunk 4


def test_callable_rows_method_is_used_and_keeps_state(small):
    src = Windows(small, 9000, 2048)
    w, _ = train_batch_stream(src, 6, 5, epochs=3, n_samples=9000, final_pass="none")
    assert src.calls == 5  # skipped counting pass (1), rows() (1), 3 epochs; no data pass for either
    ref = Windows(small, 9000, 2048)
    ix = np.random.default_rng(42).integers(0, 9000, 30)
    first = np.concatenate(list(ref()))
    np.testing.assert_array_equal(Windows(small, 9000, 2048).rows(ix), first[ix])


@pytest.mark.parametrize("kind", ["full", "sampled", "callable", "callable-sampled"])
@pytest.mark.parametrize("size", [2, 3, 7])
def test_ownership_partitions_the_rows(small, kind, size):
    callable_src = kind.startswith("callable")
    fraction = 0.4 if kind.endswith("sampled") else 1.0

    def chunks(rank, sz):
        src = _make_source(Chunks(small, 4000, lazy=True, n_feat=7) if callable_src else small, len(small))
        rng = np.random.default_rng([3, 1]) if fraction < 1 else None
        return list(_epoch_chunks(src, 4000, fraction, 500, rng, rank, sz))

    single = chunks(0, 1)
    parts = [chunks(r, size) for r in range(size)]
    assert sum(len(p) for p in parts) == len(single)
    for r, p in enumerate(parts):
        for j, c in enumerate(p):
            np.testing.assert_array_equal(c, single[r + j * size])


# ------------------------------------------------------------------ checkpoint / resume

class Stop(Exception):
    pass


def _stop_after(epoch):
    def cb(e, w):
        if e == epoch:
            raise Stop
    return cb


@pytest.mark.parametrize("kind", ["array", "array-sampled", "callable"])
def test_resume_is_bitwise(tmp_path, small, kind):
    kw = dict(epochs=5, chunk=4096, r_min=2, seed=3, final_pass="full")
    if kind == "array-sampled":
        kw.update(sample_fraction=0.5, sample_block=256, final_pass="sample")

    def src():
        return Windows(small, 12000, 3000) if kind == "callable" else small

    if kind == "callable":
        kw["n_samples"] = 12000
    w_ref, i_ref = train_batch_stream(src(), 8, 6, **kw)
    ck = str(tmp_path / "ck.npz")
    with pytest.raises(Stop):
        train_batch_stream(src(), 8, 6, checkpoint=ck, resume=True, callback=_stop_after(1), **kw)
    meta, arrays = read_checkpoint(ck)
    assert meta["epoch"] == 2 and arrays["weights"].shape == (48, 7)
    if kind == "callable":
        assert meta["calls"] == 4  # skipped counting pass, rows(), 2 epochs
    w, info = train_batch_stream(src(), 8, 6, checkpoint=ck, resume=True, **kw)
    assert info["resumed_from"] == 2
    assert sha(w) == sha(w_ref)
    for k in ("qe", "rows_per_epoch", "fallbacks", "hits"):
        np.testing.assert_array_equal(info[k], i_ref[k])
    assert info["qe_initial"] == i_ref["qe_initial"]
    assert read_checkpoint(ck)[0]["epoch"] == 5


def test_resume_without_file_starts_fresh_and_checks_settings(tmp_path, small):
    ck = str(tmp_path / "ck.npz")
    w0, _ = train_batch_stream(small, 5, 4, epochs=3, chunk=4096, final_pass="none")
    w1, info = train_batch_stream(small, 5, 4, epochs=3, chunk=4096, final_pass="none", checkpoint=ck, resume=True)
    assert info["resumed_from"] == 0 and sha(w0) == sha(w1)
    with pytest.raises(ValueError, match="xdim"):
        train_batch_stream(small, 4, 5, epochs=3, chunk=4096, checkpoint=ck, resume=True)
    with pytest.raises(ValueError, match="radius"):
        train_batch_stream(small, 5, 4, epochs=3, r_min=1, chunk=4096, checkpoint=ck, resume=True)
    with pytest.raises(ValueError, match="checkpoint path"):
        train_batch_stream(small, 5, 4, epochs=3, resume=True)


def test_import_does_not_load_mpi4py():
    code = "import sys, aweSOM.stream, aweSOM.distributed; print('mpi4py' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# ------------------------------------------------------------------ MPI

def _mpirun():
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return None
    return shutil.which("mpirun")


MPIRUN = _mpirun()
needs_mpi = pytest.mark.skipif(MPIRUN is None, reason="mpi4py or mpirun not available")


def run_worker(n, out, scenario):
    env = dict(os.environ, NUMBA_NUM_THREADS="2", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    cmd = [MPIRUN, "-n", str(n), "--oversubscribe", sys.executable, "-m", "mpi4py", str(WORKER),
           scenario, str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return [dict(np.load(p, allow_pickle=False)) for p in sorted(Path(out).glob("rank*.npz"))]


def rel_diff(a, b):
    """Max relative difference |a - b| / max(|a|, |b|), with a 1e-8 floor on the denominator."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)))


def _record(name, value):
    print(f"{name} = {value}")
    path = os.environ.get("AWESOM_MPI_TEST_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps({name: value}) + "\n")


@needs_mpi
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("ranks", [2, 4])
def test_mpi_array_source_matches_single_process(tmp_path, ranks, dtype):
    x = wk.data(dtype)
    outs = run_worker(ranks, tmp_path, "array" if dtype == "float32" else "array64")
    w_single, i_single, history = wk.run_array(x, None)
    for o in outs:  # every rank holds the same weights, bitwise
        assert sha(o["weights"]) == sha(outs[0]["weights"])
        assert int(o["ranks"]) == ranks
    o = outs[0]
    d = rel_diff(o["weights"], w_single)
    _record(f"array_{dtype}_P{ranks}_max_rel", d)
    assert np.all(np.abs(o["weights"] - w_single) <= 1e-4 * np.maximum(np.abs(o["weights"]), np.abs(w_single)) + 1e-12)
    np.testing.assert_array_equal(o["hits"], i_single["hits"])
    np.testing.assert_array_equal(o["rows_per_epoch"], i_single["rows_per_epoch"])
    assert o["rank_rows"].shape == (5, ranks) and (o["rank_rows"].sum(axis=1) == len(x)).all()
    # per-epoch BMUs of every row: the lattice entering each epoch under MPI vs single process
    flips = 0
    for e in range(5):
        a, _ = best_match_nodes(o["history"][e], x)
        b, _ = best_match_nodes(history[e], x)
        flips += int(np.count_nonzero(a != b))
    _record(f"array_{dtype}_P{ranks}_bmu_flips", flips)
    assert flips == 0


@needs_mpi
def test_mpi_fewer_chunks_than_ranks_and_sampling(tmp_path):
    x = wk.data()
    outs = run_worker(4, tmp_path, "uneven")
    w_single, i_single = wk.run_uneven(x, None)
    o = outs[0]
    assert (o["rank_rows"][:, 3] == 0).all()  # 3 chunks, rank 3 owns none
    d = rel_diff(o["weights"], w_single)
    _record("uneven_P4_max_rel", d)
    assert np.all(np.abs(o["weights"] - w_single) <= 1e-4 * np.maximum(np.abs(o["weights"]), np.abs(w_single)) + 1e-12)
    np.testing.assert_array_equal(o["rows_per_epoch"], i_single["rows_per_epoch"])
    s = outs[0]
    ws, is_ = wk.run_sampled(x, None)
    d = rel_diff(s["sampled_weights"], ws)
    _record("sampled_P4_max_rel", d)
    assert d < 1e-4
    np.testing.assert_array_equal(s["sampled_rows"], is_["rows_per_epoch"])


@needs_mpi
@pytest.mark.parametrize("ranks", [2, 4])
def test_mpi_callable_resume(tmp_path, ranks):
    outs = run_worker(ranks, tmp_path, "resume")
    o = outs[0]
    assert sha(o["resumed"]) == sha(o["uninterrupted"])  # bitwise for a fixed rank count
    assert int(o["resumed_from"]) == 2 and int(o["calls"]) == 7  # skipped counting pass, rows(), 5 epochs
    assert all(int(r["calls"]) == 7 for r in outs)
    w_single = wk.run_callable(wk.data(), None)
    d = rel_diff(o["uninterrupted"], w_single)
    _record(f"callable_P{ranks}_max_rel", d)
    assert d < 1e-4


@needs_mpi
def test_mpi_fixed_rank_count_is_bitwise_reproducible(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
    a = run_worker(3, tmp_path / "a", "array64")
    b = run_worker(3, tmp_path / "b", "array64")
    assert sha(a[0]["weights"]) == sha(b[0]["weights"])
