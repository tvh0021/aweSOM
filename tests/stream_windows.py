"""Stateful callable data source for the streaming tests.

:class:`Windows` reads random row windows of an array, different on every call,
and keeps a call counter as a resumable callable source does. :class:`Stop` is
the exception it raises at ``fail_at_call``.
"""
import numpy as np


class Stop(Exception):
    pass


class Windows:
    """Stateful callable source: call k reads random windows of ``x``."""

    def __init__(self, x, rows, chunk, seed=7, fail_at_call=None):
        self.x, self.rows_per_call, self.chunk, self.seed = x, rows, chunk, seed
        self.calls = 0
        self.n_feat = x.shape[1]
        self.fail_at_call = fail_at_call

    def _windows(self, call):
        rng = np.random.default_rng([self.seed, call])
        out, left = [], self.rows_per_call
        while left:
            n = min(left, self.chunk)
            out.append((int(rng.integers(0, len(self.x) - n + 1)), n))
            left -= n
        return out

    def lazy_chunks(self):
        if self.calls == self.fail_at_call:
            raise Stop
        windows = self._windows(self.calls)
        self.calls += 1
        for a, n in windows:
            yield n, (lambda a=a, n=n: self.x[a : a + n])

    def __call__(self):
        for _, read in self.lazy_chunks():
            yield read()

    def rows(self, ix):
        """Rows ``ix`` of call ``calls``; counts as one call."""
        call = self.calls
        self.calls += 1
        stream_rows = np.concatenate([np.arange(a, a + n) for a, n in self._windows(call)])
        return self.x[stream_rows[ix]]
