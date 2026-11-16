"""CPU smoke test for planning/benchmarks/bench_label_path.py at minimal CLI size."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aweSOM._gpu_screen import cuda_available

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "planning/benchmarks/bench_label_path.py"

# Minimal CPU fake-CuPy module so bench_label_path.py runs all three layers
# (kernel, searcher, predict_stream) on CPU hosts where CuPy is not installed.
CPU_CUPY_SHIM = '''"""Minimal CPU fake-CuPy shim for bench_label_path.py smoke testing."""
import types
import numpy as np
from aweSOM.inference import _bmu_kernel


class _Stream:
    current = None

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        self._prev, _Stream.current = _Stream.current, self
        return self

    def __exit__(self, *args):
        _Stream.current = self._prev

    def synchronize(self):
        pass


class _Event:
    def record(self, stream=None):
        pass

    def synchronize(self):
        pass


class _DeviceArray:
    def __init__(self, arr):
        self.a = np.asarray(arr)

    @property
    def dtype(self):
        return self.a.dtype

    @property
    def shape(self):
        return self.a.shape

    def __getitem__(self, key):
        return _DeviceArray(self.a[key])

    def set(self, host_arr, stream=None):
        self.a[...] = host_arr

    def get(self, out=None, stream=None, blocking=True):
        if out is not None:
            out[...] = self.a
            return out
        return self.a.copy()


def asarray(a, dtype=None):
    return _DeviceArray(np.array(a, dtype=dtype, copy=True))


def empty(shape, dtype=np.float32):
    return _DeviceArray(np.empty(shape, dtype=dtype))


def take(table, idx, out=None):
    res = np.take(table.a, idx.a)
    if out is not None:
        out.a[...] = res
        return out
    return _DeviceArray(res)


class RawKernel:
    def __init__(self, source, name, options=()):
        self.name = name

    def __call__(self, grid, block, args, shared_mem=0, stream=None):
        w, x, idx, d2, n = args[:5]
        _bmu_kernel(w.a, x.a[: int(n)], idx.a[: int(n)], d2.a[: int(n)])


class _DeviceCtx:
    def __init__(self, dev=0):
        self.id = dev

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


cuda = types.SimpleNamespace(
    Stream=_Stream,
    Event=_Event,
    Device=_DeviceCtx,
    get_current_stream=lambda: _Stream(),
    get_elapsed_time=lambda begin, end: 1.0,
    alloc_pinned_memory=lambda nbytes: bytearray(nbytes),
    runtime=types.SimpleNamespace(
        getDeviceCount=lambda: 1,
        getDevice=lambda: 0,
        getDeviceProperties=lambda dev: {"sharedMemPerBlock": 65536, "maxThreadsPerBlock": 1024},
    ),
)
int32 = np.int32
float32 = np.float32
float64 = np.float64
uint8 = np.uint8
uint16 = np.uint16
'''


@pytest.mark.parametrize("out_dtype", ["int32", "uint8"])
def test_bench_label_path_cpu_smoke(tmp_path, out_dtype):
    assert BENCH.is_file(), f"missing benchmark script: {BENCH}"
    py_paths = [str(ROOT / "src")]
    if not cuda_available():
        shim_dir = tmp_path / "cpu_cupy_shim"
        shim_dir.mkdir()
        (shim_dir / "cupy.py").write_text(CPU_CUPY_SHIM)
        py_paths.insert(0, str(shim_dir))

    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(py_paths),
        NUMBA_NUM_THREADS="2",
    )
    cmd = [
        sys.executable,
        str(BENCH),
        "--rows",
        "64",
        "--pool",
        "64",
        "--chunk",
        "64",
        "--clusters",
        "4",
        "--out-dtype",
        out_dtype,
        "--layers",
        "kernel,searcher,predict_stream",
    ]
    res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    lines = [json.loads(line) for line in res.stdout.strip().splitlines() if line.strip()]
    assert [r["layer"] for r in lines] == ["kernel", "searcher", "predict_stream"]
    assert lines[-1]["labels_match_searcher"] is True
    assert lines[-1]["rows"] == 64
