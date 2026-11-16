"""Verify GpuScreenSearcher retains the implicit construction CUDA device across calls."""

import sys
import types

import numpy as np
import pytest

from aweSOM import _gpu_screen
from aweSOM.inference import best_match_nodes
from aweSOM.stream import BMUSearcher

from fake_cupy import StandIn


class _RefStandIn(StandIn):
    """StandIn subclass used strictly for exact CPU kernel reference math inside fake RawKernel."""

    def __init__(self, weights):
        super().__init__(weights, len(weights), 1)


def _multi_device_fake_cupy(device_count: int = 2):
    state = {"current": 0}
    uploads = []
    allocs = []
    launches = []
    streams = []

    syncs = []

    class DeviceCtx:
        def __init__(self, device_id: int):
            self.id = int(device_id)
            self._prev = None

        def __enter__(self):
            if not 0 <= self.id < device_count:
                raise RuntimeError(f"invalid device ordinal {self.id}")
            self._prev = state["current"]
            state["current"] = self.id
            return self

        def __exit__(self, *exc):
            state["current"] = self._prev

    class Stream:
        current = None

        def __init__(self, **kwargs):
            self.device = state["current"]
            streams.append(self.device)

        def __enter__(self):
            self._prev, Stream.current = Stream.current, self
            return self

        def __exit__(self, *exc):
            Stream.current = self._prev

        def synchronize(self):
            assert state["current"] == self.device
            syncs.append(state["current"])

    class NullStream:
        def __init__(self):
            self.device = state["current"]

        def synchronize(self):
            assert state["current"] == self.device
            syncs.append(state["current"])

    class DeviceArray:
        def __init__(self, array: np.ndarray, device: int):
            self.a = array
            self.device = device

        @property
        def dtype(self):
            return self.a.dtype

        @property
        def shape(self):
            return self.a.shape

        def __getitem__(self, key):
            return DeviceArray(self.a[key], self.device)

        def set(self, host_arr, stream=None):
            assert state["current"] == self.device
            if stream is not None:
                assert stream.device == self.device
            uploads.append(state["current"])
            self.a[...] = host_arr

        def get(self, out, stream=None, blocking=False):
            if stream is not None:
                assert stream.device == self.device
            out[...] = self.a

    def asarray(a):
        arr = np.array(a, copy=True)
        dev = state["current"]
        uploads.append(dev)
        return DeviceArray(arr, dev)

    def empty(shape, dtype):
        dev = state["current"]
        allocs.append(dev)
        return DeviceArray(np.empty(shape, dtype), dev)

    def take(table, idx, out):
        assert state["current"] == table.device == idx.device == out.device
        out.a[...] = np.take(table.a, idx.a)

    def raw_kernel(source, name, options):
        def launch(grid, block, args, shared_mem, stream):
            w, x, idx, d2, n = args[:5]
            dev = state["current"]
            assert dev == w.device == x.device == idx.device == d2.device == stream.device
            launches.append(dev)
            ref = _RefStandIn(w.a)
            ref(x.a[: int(n)], idx.a[: int(n)], d2.a[: int(n)])

        return launch

    def set_device(d: int):
        d = int(d)
        if not 0 <= d < device_count:
            raise RuntimeError(f"invalid device ordinal {d}")
        state["current"] = d

    runtime = types.SimpleNamespace(
        getDeviceCount=lambda: device_count,
        getDevice=lambda: state["current"],
        setDevice=set_device,
        getDeviceProperties=lambda d: dict(sharedMemPerBlock=49152, maxThreadsPerBlock=1024),
    )
    cp = types.SimpleNamespace(
        asarray=asarray,
        empty=empty,
        take=take,
        RawKernel=raw_kernel,
        cuda=types.SimpleNamespace(
            runtime=runtime,
            Device=DeviceCtx,
            Stream=Stream,
            alloc_pinned_memory=lambda n: bytearray(n),
            get_current_stream=NullStream,
        ),
    )
    return cp, types.SimpleNamespace(
        state=state,
        uploads=uploads,
        allocs=allocs,
        launches=launches,
        streams=streams,
        syncs=syncs,
    )


@pytest.fixture
def multi_gpu_cupy(monkeypatch):
    StandIn.devices = []
    cp, log = _multi_device_fake_cupy(device_count=2)
    monkeypatch.setitem(sys.modules, "cupy", cp)
    _gpu_screen._kernels.cache_clear()
    yield cp, log
    _gpu_screen._kernels.cache_clear()


def test_implicit_device_zero_retained_after_switching_to_one(multi_gpu_cupy):
    cp, log = multi_gpu_cupy
    rng = np.random.default_rng(5)
    w = rng.normal(size=(12, 4))
    x = rng.normal(size=(19, 4)).astype(np.float32)
    clusters = np.arange(12, dtype=np.int32) % 4

    assert cp.cuda.runtime.getDevice() == 0
    search = _gpu_screen.GpuScreenSearcher(w, 4, 3, chunk=8)
    assert search.device == 0
    assert isinstance(search.device, int)
    assert log.uploads == [0]
    assert log.streams == [0, 0]

    with cp.cuda.Device(1):
        assert cp.cuda.runtime.getDevice() == 1
        table = search.cluster_map(clusters, dtype=np.uint8)
        assert cp.cuda.runtime.getDevice() == 1
        assert table.device == 0

        out = np.empty(len(x), dtype=np.uint8)
        d2 = np.empty(len(x), dtype=np.float64)
        tok = search.submit(x, out, d2, clusters=table)
        search.wait(tok)
        assert cp.cuda.runtime.getDevice() == 1

        idx = np.empty(len(x), dtype=np.int32)
        d2_sync = np.empty(len(x), dtype=np.float64)
        search(x, idx, d2_sync)
        assert cp.cuda.runtime.getDevice() == 1

    ref_idx, ref_d2 = best_match_nodes(w, x)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_array_equal(out, clusters[ref_idx])
    np.testing.assert_array_equal(d2.view(np.uint64), ref_d2.view(np.uint64))
    np.testing.assert_array_equal(d2_sync.view(np.uint64), ref_d2.view(np.uint64))

    assert log.uploads and set(log.uploads) == {0}
    assert log.allocs and set(log.allocs) == {0}
    assert log.launches and set(log.launches) == {0}
    assert log.syncs and set(log.syncs) == {0}


def test_implicit_device_inside_context_retained_after_exit(multi_gpu_cupy):
    cp, log = multi_gpu_cupy
    rng = np.random.default_rng(9)
    w = rng.normal(size=(10, 3))
    x = rng.normal(size=(15, 3)).astype(np.float32)
    clusters = np.arange(10, dtype=np.int32) % 5

    with cp.cuda.Device(1):
        search = _gpu_screen.GpuScreenSearcher(w, 5, 2, chunk=6)
        reusable = BMUSearcher(w, backend="cuda_screen")

    assert cp.cuda.runtime.getDevice() == 0
    assert search.device == 1
    assert reusable.device == 1

    table = search.cluster_map(clusters, dtype=np.uint8)
    out = np.empty(len(x), dtype=np.uint8)
    search.submit(x, out, clusters=table)
    search.wait()

    got_idx, got_d2 = reusable(x)
    assert cp.cuda.runtime.getDevice() == 0

    ref_idx, ref_d2 = best_match_nodes(w, x)
    np.testing.assert_array_equal(out, clusters[ref_idx])
    np.testing.assert_array_equal(got_idx, ref_idx)
    np.testing.assert_array_equal(got_d2.view(np.uint64), ref_d2.view(np.uint64))

    assert set(log.uploads) == {1}
    assert set(log.allocs) == {1}
    assert set(log.launches) == {1}
    assert log.syncs and set(log.syncs) == {1}


def test_cancel_with_pending_work_after_device_switch_clears_pending_and_restores_device(
    multi_gpu_cupy,
):
    cp, log = multi_gpu_cupy
    rng = np.random.default_rng(13)
    w = rng.normal(size=(12, 4))
    # 12 rows with chunk=8 splits into 2 subchunks (8 + 4), leaving both slots pending.
    x = rng.normal(size=(12, 4)).astype(np.float32)
    clusters = np.arange(12, dtype=np.int32) % 4

    assert cp.cuda.runtime.getDevice() == 0
    search = _gpu_screen.GpuScreenSearcher(w, 4, 3, chunk=8)
    assert search.device == 0
    table = search.cluster_map(clusters, dtype=np.uint8)

    sentinel_out = np.uint8(251)
    out = np.full(len(x), sentinel_out, dtype=np.uint8)
    d2 = np.full(len(x), -1.0, dtype=np.float64)
    tok = search.submit(x, out, d2, clusters=table)
    assert len(search.slots) == 2
    assert all(slot["pending"] is not None for slot in search.slots)
    assert np.all(out == sentinel_out)
    assert np.all(d2 == -1.0)

    with cp.cuda.Device(1):
        assert cp.cuda.runtime.getDevice() == 1
        syncs_before = len(log.syncs)
        search.cancel()
        assert cp.cuda.runtime.getDevice() == 1
        assert log.syncs[syncs_before:] == [0, 0]
        assert all(slot["pending"] is None for slot in search.slots)
        assert np.all(out == sentinel_out)
        assert np.all(d2 == -1.0)

        # Waiting on the cancelled token is a no-op and still preserves device 1.
        search.wait(tok)
        assert cp.cuda.runtime.getDevice() == 1
        assert np.all(out == sentinel_out)

        # Fresh submission after cancel still executes cleanly on retained device 0.
        out_after = np.empty(len(x), dtype=np.uint8)
        d2_after = np.empty(len(x), dtype=np.float64)
        tok2 = search.submit(x, out_after, d2_after, clusters=table)
        search.wait(tok2)
        assert cp.cuda.runtime.getDevice() == 1

    assert cp.cuda.runtime.getDevice() == 0
    ref_idx, ref_d2 = best_match_nodes(w, x)
    np.testing.assert_array_equal(out_after, clusters[ref_idx])
    np.testing.assert_array_equal(d2_after.view(np.uint64), ref_d2.view(np.uint64))
    assert log.syncs and set(log.syncs) == {0}

