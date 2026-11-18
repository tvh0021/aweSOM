"""Single-process streaming features of aweSOM.stream: the callable-source protocol,
chunk ownership, checkpoint and resume, and the dev-parity fixture."""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import aweSOM.stream as stream
from aweSOM.stream import _epoch_chunks, _make_source, read_checkpoint, train_batch_stream

sys.path.insert(0, str(Path(__file__).parent))
import gmm_bbh7  # noqa: E402
from stream_windows import Stop, Windows  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
DEV_REF = FIXTURES / "stream_dev_82dee49_gmm_bbh7_200k_48x34.npz"


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
def rows200k():
    return gmm_bbh7.rows(200_000)


@pytest.fixture(scope="module")
def gmm200k(rows200k):
    ref = np.load(DEV_REF)
    if sha(rows200k) != str(ref["rows_sha256"]):
        msg = "gmm_bbh7 rows differ on this platform (numpy version or CPU); fixture not comparable"
        if os.environ.get("AWESOM_EXACT_FIXTURES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    return rows200k, ref


@pytest.fixture(scope="module")
def small():
    return gmm_bbh7.rows(30_000).astype(np.float64)


# ------------------------------------------------------------------ dev parity

@pytest.fixture(scope="module")
def dev_parity_array(rows200k):
    return train_batch_stream(rows200k, 48, 34, epochs=5, chunk=2**15, r_min=5, seed=42, final_pass="full")


def test_dev_parity_fixture_array(gmm200k, dev_parity_array):
    _, ref = gmm200k
    w, info = dev_parity_array
    # libm exp() in _h_table can round by 1 ULP across architectures (arm64 vs x86_64).
    np.testing.assert_allclose(w, ref["weights"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(info["hits"], ref["hits"])
    np.testing.assert_allclose(info["qe"], ref["qe"], rtol=1e-12, atol=1e-12)
    if os.environ.get("AWESOM_EXACT_FIXTURES") == "1":
        assert sha(w) == str(ref["weights_sha256"])
        np.testing.assert_array_equal(info["qe"], ref["qe"])


@pytest.mark.parametrize("lazy", [False, True])
def test_dev_parity_fixture_callable(rows200k, dev_parity_array, lazy):
    w_array, _ = dev_parity_array
    src = Chunks(rows200k, 2**15, lazy=lazy, n_feat=7)
    w, _ = train_batch_stream(src, 48, 34, epochs=5, r_min=5, seed=42, final_pass="full", n_samples=len(rows200k))
    assert sha(w) == sha(w_array)


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

def _stop_after(epoch):
    def cb(e, w):
        if e == epoch:
            raise Stop
    return cb


@pytest.mark.parametrize("kind", ["array", "array-sampled", "callable"])
def test_resume_is_bitwise(tmp_path, small, kind):
    kw = dict(epochs=5, chunk=4096, r_min=2, seed=3, final_pass="full", bmu="screen")
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
    with pytest.raises(ValueError, match="chunk"):
        train_batch_stream(small, 5, 4, epochs=3, chunk=2048, checkpoint=ck, resume=True)
    with pytest.raises(ValueError, match="groups"):
        train_batch_stream(small, 5, 4, epochs=3, chunk=4096, groups=32, checkpoint=ck, resume=True)
    meta, arrays = read_checkpoint(ck)
    meta["ranks"] = 2
    stream._write_checkpoint(ck, arrays, meta)
    with pytest.raises(ValueError, match="2 rank"):
        train_batch_stream(small, 5, 4, epochs=3, chunk=4096, checkpoint=ck, resume=True)
    meta["ranks"] = 1
    stream._write_checkpoint(ck, arrays, meta)
    with pytest.raises(ValueError, match="radius"):
        train_batch_stream(small, 5, 4, epochs=3, r_min=1, chunk=4096, checkpoint=ck, resume=True)
    with pytest.raises(ValueError, match="checkpoint path"):
        train_batch_stream(small, 5, 4, epochs=3, resume=True)
