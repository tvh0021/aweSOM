"""production_pool.EpochSource: lazy windows reproduce the historical generator, rows()
returns rows of call 1 and leaves the epochs on calls 2, 3, ... (as the old trainer did)."""
import importlib.util
from pathlib import Path

import numpy as np

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


def test_rows_is_call_one_and_epochs_start_at_call_two():
    a = pool.EpochSource(data(), STRATA, 400, 7, 37)
    b = pool.EpochSource(data(), STRATA, 400, 7, 37)
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
