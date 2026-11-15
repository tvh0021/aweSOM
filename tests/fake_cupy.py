"""CPU stand-in for the CuPy searcher of ``aweSOM._gpu_screen``.

Tests patch ``_gpu_screen.GpuScreenSearcher`` with :class:`StandIn` to run the
``cuda_screen`` code paths without a GPU. The module imports
``aweSOM._gpu_screen`` only inside the methods that need it.
"""
import numpy as np


class StandIn:
    """CPU stand-in for GpuScreenSearcher: the float32-screen reference and its float64 recompute."""

    devices = []

    def __init__(self, weights, xdim, ydim, device=None, **kwargs):
        self.w = weights
        type(self).devices.append(device)

    def __call__(self, x, idx, d2):
        from aweSOM import _gpu_screen

        idx[:], d2[:], fb = _gpu_screen._cuda_screen_reference(self.w, x, return_fallbacks=True)
        return fb

    def cluster_map(self, clusters, dtype=np.int32):
        return np.asarray(clusters).astype(dtype)

    def submit(self, x, out, d2=None, clusters=None):  # synchronous: wait has nothing to do
        from aweSOM import _gpu_screen

        idx, dist = _gpu_screen._cuda_screen_reference(self.w, x)
        out[:] = idx if clusters is None else clusters[idx]
        if d2 is not None:
            d2[:] = dist
        return 0

    def wait(self, token=None):
        pass

    def cancel(self):
        pass
