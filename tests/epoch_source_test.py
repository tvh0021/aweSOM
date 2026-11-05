"""production_pool.EpochSource: lazy windows reproduce the historical generator; with the
trainer, the call numbering and weights equal those of the dev 82dee49 trainer and
EpochSource (which always made a counting pass) for every initialisation."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aweSOM.stream import train_batch_stream

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("production_pool_es", ROOT / "examples/grmhd-bbh/production_pool.py")
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)

STRATA = [dict(snapshot=str(i // 2), start=a, rows=n) for i, (a, n) in enumerate([(0, 300), (300, 200), (500, 250), (750, 250)])]


def old_call(self):
    """EpochSource.__call__ before the scale-out change (production_pool at 82dee49)."""
    rng = np.random.default_rng([self.seed, self.calls]); self.calls += 1
    for ix in self.groups:
        self.quotas[ix] = pool.randomized_allocate([self.strata[i]['rows'] for i in ix],
                                                   self.rows_per_call // len(self.groups), rng)
    for s, q in zip(self.strata, self.quotas):
        pos = int(rng.integers(s['rows']))
        while q:
            n = min(int(q), self.chunk, s['rows'] - pos)
            yield self.data[s['start'] + pos:s['start'] + pos + n]
            q -= n; pos = (pos + n) % s['rows']


def data():
    return np.random.default_rng(1).normal(size=(1000, 3))


def test_lazy_windows_match_historical_generator():
    a = pool.EpochSource(data(), STRATA, 400, 7, 37)
    b = pool.EpochSource(data(), STRATA, 400, 7, 37)
    reads = []
    a._read = lambda start, n, f=a._read: reads.append(n) or f(start, n)
    for _ in range(4):
        pairs = list(a.lazy_chunks())
        assert not reads  # nothing read until read() runs
        xa = [read() for _, read in pairs]
        reads.clear()
        xb = list(old_call(b))
        assert [m for m, _ in pairs] == [len(x) for x in xb]
        assert all(np.array_equal(u, v) for u, v in zip(xa, xb))
        assert np.array_equal(a.quotas, b.quotas) and a.calls == b.calls
    assert a.n_feat == 3


def test_rows_is_one_call():
    a = pool.EpochSource(data(), STRATA, 400, 7, 37)
    b = pool.EpochSource(data(), STRATA, 400, 7, 37)
    a.calls = 1  # as after the trainer skipped the counting pass
    ix = np.random.default_rng(0).integers(0, 400, 60)
    got = a.rows(ix)
    assert a.calls == 2
    list(old_call(b))
    np.testing.assert_array_equal(got, np.concatenate(list(old_call(b)))[ix])


def test_trainer_matches_the_old_scan_and_init_passes():
    """Old trainer: counting pass (call 0), initial rows (call 1), epochs (calls 2...)."""
    x = data()
    w, _ = train_batch_stream(pool.EpochSource(x, STRATA, 400, 7, 37), 4, 3, epochs=3, n_samples=400,
                              final_pass="none")
    b = pool.EpochSource(x, STRATA, 400, 7, 37)
    list(old_call(b))
    init_rows = np.concatenate(list(old_call(b)))
    w0 = init_rows[np.random.default_rng(42).integers(0, 400, 12)]

    def epochs():
        return old_call(b)

    w_ref, _ = train_batch_stream(epochs, 4, 3, epochs=3, n_samples=400, n_features=3, initial_weights=w0,
                                  final_pass="none")
    assert w.tobytes() == w_ref.tobytes()


DEV = "82dee49"


def _dev_module(name, path, tmp):
    """Module ``name`` built from ``git show DEV:path`` (skips when git or the commit is missing)."""
    try:
        text = subprocess.run(["git", "-C", str(ROOT), "show", f"{DEV}:{path}"], capture_output=True,
                              text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git show {DEV}:{path} unavailable")
    f = tmp / (name.rsplit(".", 1)[-1] + ".py")
    f.write_text(text)
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dev(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dev82dee49")
    stream_dev = _dev_module("aweSOM._dev_stream_82dee49", "src/aweSOM/stream.py", tmp)
    pool_dev = _dev_module("_dev_production_pool_82dee49", "examples/grmhd-bbh/production_pool.py", tmp)
    yield stream_dev, pool_dev
    sys.modules.pop("aweSOM._dev_stream_82dee49", None)
    sys.modules.pop("_dev_production_pool_82dee49", None)


W0 = np.random.default_rng(11).normal(size=(12, 3))
CASES = {
    "sampling-known-n": dict(n_samples=400),
    "sampling-n-omitted": dict(),
    "uniform-known-n": dict(n_samples=400, init="uniform"),
    "initial-weights-known-n": dict(n_samples=400, initial_weights=W0),
}


@pytest.mark.parametrize("case", list(CASES))
def test_trainer_and_epoch_source_match_dev(dev, case):
    """Weights and call counts of this trainer + EpochSource equal dev 82dee49's bitwise."""
    stream_dev, pool_dev = dev
    x = data()
    kw = dict(epochs=3, final_pass="none", **CASES[case])
    new = pool.EpochSource(x, STRATA, 400, 7, 37)
    old = pool_dev.EpochSource(x, STRATA, 400, 7, 37)
    w, _ = train_batch_stream(new, 4, 3, **kw)
    w_ref, _ = stream_dev.train_batch_stream(old, 4, 3, **kw)
    assert new.calls == old.calls
    assert w.tobytes() == w_ref.tobytes(), float(np.max(np.abs(w - w_ref)))
