"""Tests for the JAX batch SOM trainer (aweSOM.batch).

The module runs on 4 virtual CPU devices. The device count must be fixed before
the JAX backend initialises, so the flag is set before jax is imported. If
another test module already initialised JAX with one device, the tests still
run on one device and the multi-device comparison is an xfail.

The acceptance test against the online trainer is slow (about a minute) and
runs only with ``AWESOM_SLOW=1``.
"""

import os

_DEVICE_FLAG = "--xla_force_host_platform_device_count=4"
if _DEVICE_FLAG not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _DEVICE_FLAG).strip()

import contextlib
import io
import socket
import subprocess
import sys
import textwrap

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from aweSOM import Lattice  # noqa: E402
from aweSOM.batch import (  # noqa: E402
    neighbourhood_matrix,
    quantization_error,
    radius_schedule,
    topographic_error,
    train_batch,
)

N_DEVICES = jax.device_count()
CENTERS = np.array(
    [[0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0], [0.0, 3.0, 3.0, 3.0]]
)


def three_gaussians(n: int, seed: int) -> np.ndarray:
    """Return ``n`` samples from three isotropic Gaussians in 4 dimensions."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, len(CENTERS), n)
    return CENTERS[labels] + rng.normal(0.0, 0.5, (n, CENTERS.shape[1]))


@pytest.fixture(scope="module")
def data():
    return three_gaussians(20000, seed=0)


def test_import_contract():
    code = textwrap.dedent(
        """
        import sys
        import aweSOM
        assert "jax" not in sys.modules, "importing aweSOM must not import jax"
        sys.modules["jax"] = None  # simulate a missing jax
        try:
            import aweSOM.batch
        except ImportError as exc:
            assert "pip install 'aweSOM[gpu]'" in str(exc), str(exc)
        else:
            raise AssertionError("aweSOM.batch imported without jax")
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_radius_schedule_40_epochs():
    r = radius_schedule(40, 65, 8)
    assert r.shape == (40,)
    np.testing.assert_array_equal(r[:20], 65.0)
    np.testing.assert_array_equal(r[30:], 8.0)
    linear = r[20:30]
    assert np.all((linear < 65) & (linear > 8))
    assert np.all(np.diff(linear) < 0)
    np.testing.assert_allclose(np.diff(linear), np.diff(linear)[0])
    np.testing.assert_allclose(linear[0] - linear[1], (65 - 8) / 11)


def test_radius_schedule_short_runs():
    assert radius_schedule(1, 10, 2).tolist() == [10.0]
    r = radius_schedule(10, 13, 8)
    assert (r == 13).sum() == 5 and (r == 8).sum() == 2


@pytest.mark.parametrize("radius", [0.5, 3, 8, 13])
def test_neighbourhood_matrix_matches_gamma(radius):
    xdim, ydim = 12, 8
    nodes = xdim * ydim
    h = neighbourhood_matrix(xdim, ydim, radius)
    assert h.shape == (nodes, nodes) and h.dtype == np.float32
    np.testing.assert_array_equal(h, h.T)
    if radius != int(radius):
        assert np.all(np.diag(h) == 1.0)
        return
    m2ds = Lattice.coordinate(np.arange(nodes).reshape(-1, 1), xdim)
    for c in [0, 5, 37, nodes - 1]:
        expected = Lattice.Gamma(c, m2ds, 1.0, int(radius))
        np.testing.assert_allclose(h[c], expected, rtol=1e-6, atol=0)


def test_neighbourhood_matrix_radius_zero_is_identity():
    np.testing.assert_array_equal(neighbourhood_matrix(5, 4, 0), np.eye(20))


def test_train_batch_shape_and_qe_decreases(data):
    weights, info = train_batch(data, 12, 8, epochs=10)
    assert weights.shape == (96, 4) and weights.dtype == np.float64
    assert np.all(np.isfinite(weights))
    assert info["n_devices"] == N_DEVICES
    assert info["qe"].shape == (10,) and info["radius"].shape == (10,)
    assert info["qe"][-1] < info["qe"][0]
    assert info["hits"].sum() == len(data)
    # the in-pass QE diagnostic agrees with the exact float64 QE
    np.testing.assert_allclose(info["qe"][-1], quantization_error(weights, data), rtol=1e-5)


@pytest.mark.xfail(
    N_DEVICES < 4,
    reason="JAX initialised with fewer than 4 devices before this module set XLA_FLAGS",
    strict=False,
)
def test_sharded_equals_single_device(data):
    assert N_DEVICES >= 4
    w4, info4 = train_batch(data, 12, 8, epochs=10, devices=jax.devices()[:4])
    w1, info1 = train_batch(data, 12, 8, epochs=10, devices=jax.devices()[:1])
    assert info4["n_devices"] == 4 and info1["n_devices"] == 1
    np.testing.assert_allclose(w4, w1, rtol=1e-5, atol=1e-5 * np.abs(w1).max())


def test_padding_does_not_contribute():
    # 1001 rows do not divide evenly across devices or chunks
    x = three_gaussians(1001, seed=3)
    w_pad, info = train_batch(x, 5, 4, epochs=4, chunk=64, dtype=np.float64)
    w_ref, _ = train_batch(x, 5, 4, epochs=4, chunk=1001, devices=jax.devices()[:1], dtype=np.float64)
    assert info["n_samples"] == 1001 and info["hits"].sum() == 1001
    np.testing.assert_allclose(w_pad, w_ref, rtol=1e-12, atol=1e-12)


def test_float64_epoch_matches_numpy_reference(data):
    xdim, ydim, radius = 12, 8, 5.0
    nodes = xdim * ydim
    w0 = data[np.random.default_rng(42).integers(0, len(data), nodes)]
    weights, _ = train_batch(
        data, xdim, ydim, epochs=1, r_max=radius, r_min=radius, dtype=np.float64
    )
    d2 = ((data[:, None, :] - w0[None]) ** 2).sum(-1)
    bmu = d2.argmin(axis=1)
    s = np.zeros_like(w0)
    np.add.at(s, bmu, data)
    n = np.bincount(bmu, minlength=nodes).astype(float)
    h = neighbourhood_matrix(xdim, ydim, radius).astype(np.float64)
    # the trainer uses the float64 matrix; the float32 copy limits the tolerance
    expected = (h.T @ s) / (h.T @ n)[:, None]
    np.testing.assert_allclose(weights, expected, rtol=1e-6)


def test_degenerate_nodes_keep_weights(data):
    xdim, ydim = 6, 4
    nodes = xdim * ydim
    # all nodes far from the data on a line; only the nearest few get hits
    initial = np.zeros((nodes, 4))
    initial[:, 0] = 10.0 + np.arange(nodes)
    weights, info = train_batch(
        data, xdim, ydim, epochs=1, r_max=0, r_min=0, initial_weights=initial
    )
    assert np.all(info["radius"] == 0)
    d2 = ((data[:, None, :] - initial[None]) ** 2).sum(-1)
    bmu = d2.argmin(axis=1)
    hit = np.bincount(bmu, minlength=nodes) > 0
    assert 0 < hit.sum() < nodes
    np.testing.assert_array_equal(weights[~hit], initial[~hit])
    for c in np.flatnonzero(hit):
        np.testing.assert_allclose(weights[c], data[bmu == c].mean(axis=0), rtol=1e-6)


def test_quality_measures_small_example():
    # 2x2 lattice on a line: nodes 0 (0,0), 1 (1,0), 2 (0,1), 3 (1,1)
    w = np.array([[0.0], [1.0], [10.0], [3.0]])
    x = np.array([[0.1], [0.9], [9.0], [2.0]])
    assert quantization_error(w, x) == pytest.approx((0.1 + 0.1 + 1.0 + 1.0) / 4)
    # BMU pairs: (0,1) adjacent, (1,0) adjacent, (2,3) diagonal, tie 1/3 at 2.0 -> (1,3) adjacent
    assert topographic_error(w, x, 2, 2) == 0.0
    w5 = np.array([[0.0], [5.0], [5.0], [0.5], [9.0], [9.0]])  # 3x2 lattice
    # BMU 0 (0,0), second 3 (0,1): adjacent; x=8 -> 4 (1,1) then 5 (2,1): adjacent
    # x=2.7: 3 (0,1) then 1 (1,0): adjacent (diagonal); x=4.0 -> 1 (1,0), 2 (2,0)
    assert topographic_error(w5, np.array([[0.1], [8.0], [2.7], [4.0]]), 3, 2) == 0.0
    w_far = np.array([[0.0], [9.0], [9.0], [9.0], [9.0], [0.5]])
    # BMU 0 (0,0), second 5 (2,1): Chebyshev distance 2 -> error
    assert topographic_error(w_far, np.array([[0.2]]), 3, 2) == 1.0
    # three-way tie between nodes 0, 1 and 5: (distance, index) order picks 0 and 1
    w_tie = np.array([[1.0], [-1.0], [9.0], [9.0], [9.0], [1.0]])
    assert topographic_error(w_tie, np.array([[0.0]]), 3, 2) == 0.0


def _paired_run(seed: int) -> tuple[float, float]:
    """QE ratio - 1 and TE difference (batch - online) for one seed."""
    xdim, ydim = 20, 12
    x = three_gaussians(20000, seed=seed)
    lattice = Lattice(xdim, ydim, 0.3, train=20000)
    lattice.seed = seed
    with contextlib.redirect_stdout(io.StringIO()):
        lattice.train_lattice(x, ["a", "b", "c", "d"])
    w_online = lattice.lattice
    w_batch, _ = train_batch(x, xdim, ydim, epochs=40, seed=seed)
    q = quantization_error(w_batch, x) / quantization_error(w_online, x) - 1.0
    t = topographic_error(w_batch, x, xdim, ydim) - topographic_error(w_online, x, xdim, ydim)
    return q, t


@pytest.mark.skipif(os.environ.get("AWESOM_SLOW") != "1", reason="slow; set AWESOM_SLOW=1")
def test_acceptance_against_online_trainer():
    seeds = [1, 2, 3, 4, 5]
    pairs = np.array([_paired_run(s) for s in seeds])
    q, t = pairs[:, 0], pairs[:, 1]
    for s, (qs, ts) in zip(seeds, pairs):
        print(f"seed {s}: QE_batch/QE_online - 1 = {qs:+.5f}, TE_batch - TE_online = {ts:+.5f}")
    t975 = 2.7764  # t_{0.975, 4}
    q_ucb = q.mean() + t975 * q.std(ddof=1) / np.sqrt(len(q))
    t_ucb = t.mean() + t975 * t.std(ddof=1) / np.sqrt(len(t))
    print(f"UCB: QE ratio - 1 = {q_ucb:+.5f} (<= 0.03), TE diff = {t_ucb:+.5f} (<= 0.01)")
    assert q_ucb <= 0.03
    assert t_ucb <= 0.01


_MULTIPROCESS_WORKER = textwrap.dedent(
    """
    import sys
    import numpy as np
    import jax

    pid, port, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    jax.config.update("jax_cpu_collectives_implementation", "gloo")
    jax.distributed.initialize(f"localhost:{port}", num_processes=2, process_id=pid)
    from aweSOM.batch import train_batch

    x = np.random.default_rng(0).normal(size=(10001, 4))
    local = x[:6000] if pid == 0 else x[6000:]
    w, info = train_batch(local, 8, 6, epochs=6)
    assert info["n_samples"] == 10001 and info["hits"].sum() == 10001
    if pid == 0:
        np.save(out, w)
    """
)


@pytest.mark.skipif(os.environ.get("AWESOM_SLOW") != "1", reason="slow; set AWESOM_SLOW=1")
def test_two_process_cpu_matches_single_process(tmp_path):
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = str(sock.getsockname()[1])
    script = tmp_path / "worker.py"
    script.write_text(_MULTIPROCESS_WORKER)
    out = tmp_path / "w.npy"
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    procs = [
        subprocess.Popen([sys.executable, str(script), str(pid), port, str(out)], env=env)
        for pid in (0, 1)
    ]
    assert all(p.wait(timeout=300) == 0 for p in procs)
    x = np.random.default_rng(0).normal(size=(10001, 4))
    w_single, _ = train_batch(x, 8, 6, epochs=6)
    np.testing.assert_allclose(np.load(out), w_single, rtol=1e-12, atol=1e-12)
