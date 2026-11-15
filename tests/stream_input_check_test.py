"""Feature-count validation across streaming prediction and training routes."""

import numpy as np
import pytest

from aweSOM import _gpu_screen
from aweSOM.inference import predict
from aweSOM.stream import predict_stream, train_batch_stream

from fake_cupy import StandIn


def test_sol_repro_cpu_predict_stream_matches_predict():
    weights = [[0.0], [10.0]]
    clusters = [0, 1]
    rows = np.array([[0.0, 100.0], [10.0, -100.0]])
    msg = "data has 2 features but the lattice has 1"
    with pytest.raises(ValueError, match=f"^{msg}$"):
        predict(weights, clusters, rows, backend="numba")
    with pytest.raises(ValueError, match=f"^{msg}$"):
        predict_stream(weights, clusters, rows, backend="numba", prefetch=False)


@pytest.mark.parametrize("backend", ["numba", "screen", "patch"])
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("bad_feat", [2, 4, 0, 1, 6])
def test_predict_stream_cpu_mismatched_features(backend, prefetch, bad_feat):
    rng = np.random.default_rng(7)
    w = rng.normal(size=(6, 3))
    clusters = np.arange(6, dtype=np.int32) % 3
    x = np.zeros((11, bad_feat), dtype=np.float32)
    shape = (3, 2) if backend == "patch" else None
    msg = f"data has {bad_feat} features but the lattice has 3"
    with pytest.raises(ValueError, match=f"^{msg}$"):
        predict(w, clusters, x, backend=backend, lattice_shape=shape)
    with pytest.raises(ValueError, match=f"^{msg}$"):
        predict_stream(
            w, clusters, x, chunk=4, backend=backend, lattice_shape=shape, prefetch=prefetch
        )


@pytest.mark.parametrize("backend", ["numba", "screen", "patch"])
@pytest.mark.parametrize("bad_feat", [2, 4])
def test_predict_stream_empty_rows_mismatched_features(backend, bad_feat):
    w = np.zeros((6, 3))
    clusters = np.arange(6, dtype=np.int32)
    x = np.empty((0, bad_feat), dtype=np.float64)
    shape = (3, 2) if backend == "patch" else None
    msg = f"data has {bad_feat} features but the lattice has 3"
    with pytest.raises(ValueError, match=f"^{msg}$"):
        predict_stream(w, clusters, x, backend=backend, lattice_shape=shape, prefetch=False)


class _MidStreamCallable:
    """Callable chunk source with known metadata whose second chunk changes feature count."""

    def __init__(self, first_chunk: np.ndarray, second_chunk: np.ndarray):
        self._chunks = (first_chunk, second_chunk)
        self.n_feat = first_chunk.shape[1]

    def __call__(self):
        return iter(self._chunks)


class _StatefulMidStreamCallable:
    """Callable without n_feat/n_samples: counting pass is uniform, second pass changes shape."""

    def __init__(self, n_feat: int, bad_feat: int):
        self.n_feat_good = n_feat
        self.bad_feat = bad_feat
        self.calls = 0

    def __call__(self):
        self.calls += 1
        yield np.zeros((3, self.n_feat_good), dtype=np.float32)
        feat = self.n_feat_good if self.calls == 1 else self.bad_feat
        yield np.zeros((3, feat), dtype=np.float32)


@pytest.mark.parametrize("backend", ["numba", "screen", "patch"])
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("bad_feat", [2, 5])
def test_predict_stream_callable_changes_shape_mid_stream(backend, prefetch, bad_feat):
    rng = np.random.default_rng(11)
    w = rng.normal(size=(6, 3))
    clusters = np.arange(6, dtype=np.int32)
    shape = (3, 2) if backend == "patch" else None
    msg = f"data has {bad_feat} features but the lattice has 3"

    src_known = _MidStreamCallable(
        np.zeros((4, 3), dtype=np.float32),
        np.zeros((4, bad_feat), dtype=np.float32),
    )
    with pytest.raises(ValueError, match=f"^{msg}$"):
        predict_stream(
            w,
            clusters,
            src_known,
            n_samples=8,
            chunk=4,
            backend=backend,
            lattice_shape=shape,
            prefetch=prefetch,
        )

    src_stateful = _StatefulMidStreamCallable(3, bad_feat)
    with pytest.raises(ValueError, match=f"^{msg}$"):
        predict_stream(
            w,
            clusters,
            src_stateful,
            chunk=3,
            backend=backend,
            lattice_shape=shape,
            prefetch=prefetch,
        )


@pytest.mark.parametrize("bad_feat", [2, 5])
def test_predict_stream_queued_gpu_mismatched_and_mid_stream(monkeypatch, bad_feat):
    cancelled = []

    class TrackingStandIn(StandIn):
        def cancel(self):
            cancelled.append(True)

    monkeypatch.setattr(_gpu_screen, "GpuScreenSearcher", TrackingStandIn)
    w = np.zeros((6, 3))
    clusters = np.arange(6, dtype=np.int32)
    msg = f"data has {bad_feat} features but the lattice has 3"

    with pytest.raises(ValueError, match=f"^{msg}$"):
        predict_stream(
            w,
            clusters,
            np.zeros((5, bad_feat), dtype=np.float32),
            chunk=3,
            backend="cuda_screen",
            prefetch=False,
        )
    assert cancelled

    cancelled.clear()
    src = _MidStreamCallable(
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, bad_feat), dtype=np.float32),
    )
    with pytest.raises(ValueError, match=f"^{msg}$"):
        predict_stream(
            w,
            clusters,
            src,
            n_samples=6,
            chunk=3,
            backend="cuda_screen",
            prefetch=False,
        )
    assert cancelled


@pytest.mark.parametrize("bmu", ["numba", "screen", "patch", "auto"])
@pytest.mark.parametrize("bad_feat", [2, 5])
def test_train_batch_stream_callable_mismatched_features(bmu, bad_feat):
    msg = f"data has {bad_feat} features but the lattice has 3"
    src = _MidStreamCallable(
        np.zeros((4, 3), dtype=np.float32),
        np.zeros((4, bad_feat), dtype=np.float32),
    )
    with pytest.raises(ValueError, match=f"^{msg}$"):
        train_batch_stream(
            src,
            3,
            2,
            epochs=1,
            bmu=bmu,
            init="uniform",
            n_samples=8,
            final_pass="none",
            prefetch=False,
        )
