"""Hardware checks of the CUDA screen against the CPU float64 reference, and training."""
import numpy as np
import pytest

cp = pytest.importorskip('cupy', reason='CuPy is not installed')
try:
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device', allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip('CUDA driver unavailable', allow_module_level=True)

from aweSOM._gpu_screen import GpuScreenSearcher
from aweSOM.inference import _bmu_kernel
from aweSOM.stream import train_batch_stream


@pytest.mark.parametrize('features', [1, 4, 7, 10, 32])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('case', ['random', 'ties', 'near', 'subnormal'])
def test_exact(features, dtype, case):
    rng = np.random.default_rng(42)
    w, x = rng.normal(size=(37, features)), rng.normal(size=(259, features)).astype(dtype)
    if case == 'ties':
        w[:] = w[0]
    elif case == 'near':
        w[:] = 1
        w[0, 0] = np.nextafter(1., 2.)
        x[:] = 0
    elif case == 'subnormal':
        w *= 1e-160
        x = (x.astype(np.float64) * 1e-160).astype(dtype)
    search = GpuScreenSearcher(w, 37, 1, chunk=113, tile=16)
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x))
    ri, rd = np.empty_like(idx), np.empty_like(d2)
    search(x, idx, d2)
    _bmu_kernel(w, x, ri, rd)
    np.testing.assert_array_equal(idx, ri)
    np.testing.assert_array_equal(d2.view(np.uint64), rd.view(np.uint64))


def test_training():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200_000, 7)).astype(np.float32)
    options = dict(xdim=48, ydim=34, epochs=5, chunk=2**18, final_pass='none',
                   n_threads=8, prefetch=False)
    cpu, _ = train_batch_stream(x, bmu='numba', **options)
    gpu, _ = train_batch_stream(x, bmu='cuda_screen', **options)
    np.testing.assert_array_equal(gpu.view(np.uint64), cpu.view(np.uint64))
