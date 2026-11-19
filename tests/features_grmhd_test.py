"""Tests for aweSOM.features_grmhd: derivatives, the five features, thresholds and their metadata."""

import importlib
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from aweSOM import SOMModel, Scaler
from aweSOM.features_grmhd import (
    EPS,
    FEATURE_NAMES,
    INPUTS,
    FeatureThresholds,
    compute_features,
    derivative,
    finish_features,
    raw_features,
)

ETKPLOT = Path(os.environ.get("AWESOM_ETKPLOT_PATH", "/Volumes/ExternalNVMe/etk-analysis"))


def _grid(n=(20, 18, 16), dx=(0.1, 0.2, 0.25), origin=(-1.0, -1.5, -2.0)):
    ax = [o + d * np.arange(k) for o, d, k in zip(origin, dx, n)]
    return np.meshgrid(*ax, indexing="ij"), np.array(dx)


def _thresholds(**kw):
    base = dict(b2_min=1e-3, s_v=0.5, winsor={"log_chi_B": (-3.0, 1.0), "asinh_div_v": (-2.0, 2.0)})
    base.update(kw)
    return FeatureThresholds(**base)


@pytest.mark.parametrize("accuracy", [2, 4, 6])
def test_derivative_exact_on_polynomials_and_nan_at_edges(accuracy):
    (x, y, z), dx = _grid()
    f = x ** 3 - 2 * x * y + z ** 2  # degree 3: exact for accuracy >= 4, d/dx exact for 2 only up to quadratics
    w = accuracy // 2
    d = derivative(f, 0, dx[0], accuracy)
    assert np.all(np.isnan(d[:w])) and np.all(np.isnan(d[-w:]))
    inner = d[w:-w]
    want = (3 * x ** 2 - 2 * y)[w:-w]
    tol = 1e-10 if accuracy >= 4 else dx[0] ** 2 * 1.01  # accuracy 2 errs by f'''dx^2/6 = dx^2
    assert np.max(np.abs(inner - want)) <= tol
    dz = derivative(f, 2, dx[2], accuracy)[:, :, w:-w]
    assert np.allclose(dz, (2 * z)[:, :, w:-w], atol=1e-10)


def test_derivative_too_short_axis_is_all_nan():
    assert np.all(np.isnan(derivative(np.ones((3, 5, 5)), 0, 1.0, 4)))


def test_pointwise_features_and_known_fields():
    (x, y, z), dx = _grid()
    k = 1.3
    arrays = {
        "smallb2": np.exp(0.7 * x),  # |grad ln b2| = 0.7
        "P": 0.25 * np.exp(0.7 * x),
        "rho_b": np.full(x.shape, 2.0),
        # field rotating about z with wavenumber k: |grad b| = k, so chi_B = k
        "Bx": 3.0 * np.cos(k * z),
        "By": 3.0 * np.sin(k * z),
        "Bz": np.zeros(x.shape),
        "vel[0]": x,
        "vel[1]": 2 * y,
        "vel[2]": 3 * z,  # div v = 6
    }
    raw = raw_features(arrays, dx, 4)
    w = 2
    inner = (slice(w, -w),) * 3
    assert np.allclose(raw["log_beta_inv"], np.log10(arrays["smallb2"] / (2 * arrays["P"])))
    assert np.allclose(raw["log_theta"], np.log10(arrays["P"] / 2.0))
    assert np.allclose(raw["log_grad_ln_b2"][inner], np.log10(0.7 + EPS), atol=1e-4)
    assert np.allclose(raw["log_chi_B"][inner], np.log10(k + EPS), atol=2e-3)
    assert np.allclose(raw["asinh_div_v"][inner], 6.0)
    for n in ("log_chi_B", "log_grad_ln_b2", "asinh_div_v"):
        assert np.all(np.isnan(raw[n][:w])) and np.all(np.isnan(raw[n][:, -w:]))


def test_uniform_direction_field_has_floor_chi():
    (x, y, z), dx = _grid()
    B = (1 + x ** 2) * np.array([1.0, 2.0, -0.5])[:, None, None, None]  # direction fixed, magnitude varies
    arrays = {"Bx": B[0], "By": B[1], "Bz": B[2]}
    raw = raw_features(arrays, dx, 4, names=("log_chi_B",))
    assert np.allclose(raw["log_chi_B"][2:-2, 2:-2, 2:-2], np.log10(EPS), atol=1e-9)


def test_finish_features_floor_asinh_cast_clip():
    thr = _thresholds()
    raw = {
        "log_beta_inv": np.array([0.5, 1.0, np.nan, 2.0]),
        "log_theta": np.array([-1.0, -2.0, -3.0, -4.0]),
        "log_chi_B": np.array([0.5, 2.0, -1.0, np.nan]),
        "log_grad_ln_b2": np.array([0.0, 0.1, 0.2, 0.3]),
        "asinh_div_v": np.array([0.0, 10.0, -0.25, np.inf]),
    }
    b2 = np.array([1e-4, 1.0, 1.0, 1e-4])  # first and last below b2_min
    out = finish_features(raw, b2, thr)
    assert out.dtype == np.float32 and out.shape == (4, 5)
    chi = out[:, 2]
    assert chi[0] == np.float32(-3.0)  # floored
    assert chi[1] == np.float32(1.0)  # clipped
    assert chi[2] == np.float32(-1.0)
    assert np.isnan(chi[3])  # not floored: not finite
    dv = out[:, 4]
    assert dv[0] == 0 and dv[1] == np.float32(2.0)  # asinh(20) clipped
    assert dv[2] == np.float32(np.arcsinh(-0.5))
    assert np.isinf(dv[3])  # non-finite stays
    assert np.isnan(out[2, 0])  # unclipped feature passes NaN through


def test_compute_features_selects_points_in_c_order():
    rng = np.random.default_rng(1)
    shape = (9, 8, 7)
    arrays = {v: rng.uniform(0.5, 2.0, shape) for v in INPUTS}
    thr = _thresholds()
    sel = np.zeros(shape, bool)
    sel[2:-2, 2:-2, 2:-2] = rng.random((5, 4, 3)) < 0.5
    out = compute_features(arrays, (0.1, 0.1, 0.1), thr, select=sel)
    raw = raw_features(arrays, (0.1, 0.1, 0.1), thr.accuracy)
    want = finish_features({n: raw[n][sel] for n in FEATURE_NAMES}, arrays["smallb2"][sel], thr)
    assert np.array_equal(out, want, equal_nan=True)
    assert out.shape == (sel.sum(), 5) and np.all(np.isfinite(out))


def test_thresholds_metadata_roundtrip_bitwise(tmp_path):
    thr = FeatureThresholds(
        b2_min=2.2196638299940852e-08,
        s_v=0.042488324661616894,
        winsor={"log_chi_B": (-3.0, float(np.float32(1.5983762741))), "asinh_div_v": (-5.6042304039001465, 5.5)},
    )
    meta = thr.to_metadata()
    assert all(isinstance(v, str) for v in meta.values())
    back = FeatureThresholds.from_metadata(meta)
    assert back == thr
    rng = np.random.default_rng(0)
    model = SOMModel(
        xdim=3, ydim=2, weights=rng.normal(size=(6, 5)), node_cluster=np.arange(6) % 3,
        feature_names=list(FEATURE_NAMES), scaler=Scaler.identity(5), node_dist_q99=np.ones(6),
        reject_distance=1.0, metadata=meta,
    )
    for ext in ("h5", "txt"):
        p = tmp_path / f"m.{ext}"
        model.save(p) if ext == "h5" else model.save_text(p)
        assert FeatureThresholds.from_model(SOMModel.load(p)) == thr


def test_thresholds_from_h5_attrs_and_missing_metadata(tmp_path):
    p = tmp_path / "train.h5"
    with h5py.File(p, "w") as f:
        f.attrs.update(b2_min=1e-8, s_v=0.04, accuracy=4, epsilon=1e-3,
                       winsor_log_chi_B=np.array([-3.0, 1.6]), winsor_asinh_div_v=np.array([-5.0, 5.0]))
    thr = FeatureThresholds.from_h5(p)
    assert thr.b2_min == 1e-8 and thr.winsor["asinh_div_v"] == (-5.0, 5.0) and "log_grad_ln_b2" not in thr.winsor
    with pytest.raises(KeyError):
        FeatureThresholds.from_metadata({"dataset": "x"})
    with pytest.raises(ValueError):
        FeatureThresholds(b2_min=0, s_v=1.0, epsilon=1e-4)
    with pytest.raises(ValueError):
        FeatureThresholds(b2_min=0, s_v=1.0, accuracy=3)


def _etkplot():
    raw_env = os.environ.get("AWESOM_ETKPLOT_PATH")
    path = Path(raw_env) if raw_env is not None else ETKPLOT
    pkg_dir = path / "etkplot"
    if raw_env == "" or not pkg_dir.is_dir():
        if raw_env is not None:
            pytest.fail(f"AWESOM_ETKPLOT_PATH={raw_env!r} does not contain an etkplot package directory")
        pytest.skip("etkplot not available")
    sys.path.insert(0, str(path))
    try:
        from etkplot import features, thresholds
    except ImportError as exc:
        if raw_env is not None:
            pytest.fail(
                f"AWESOM_ETKPLOT_PATH={raw_env!r} failed to import etkplot.features and etkplot.thresholds: {exc}"
            )
        raise
    finally:
        sys.path.remove(str(path))
    expected_root = pkg_dir.resolve()
    for name, mod in (("etkplot.features", features), ("etkplot.thresholds", thresholds)):
        mod_file = getattr(mod, "__file__", None)
        if not mod_file or not Path(mod_file).resolve().is_relative_to(expected_root):
            source = (
                f"AWESOM_ETKPLOT_PATH={raw_env!r}"
                if raw_env is not None
                else f"etkplot path {str(path)!r} (AWESOM_ETKPLOT_PATH unset)"
            )
            pytest.fail(
                f"{source} did not resolve {name} under {str(expected_root)!r} (got {mod_file!r})"
            )
    return features, thresholds


def test_bitwise_equal_to_etkplot_on_random_block():
    etkf, etkt = _etkplot()
    rng = np.random.default_rng(3)
    shape = (12, 10, 9)
    arrays = {v: rng.lognormal(size=shape) for v in ("smallb2", "P", "rho_b")}
    arrays.update({v: rng.normal(size=shape) for v in ("Bx", "By", "Bz", "vel[0]", "vel[1]", "vel[2]")})
    arrays["Bx"][3, 4, 5] = np.nan
    dx = np.array([0.0625, 0.0625, 0.0625])
    ref = etkf.compute(list(FEATURE_NAMES), arrays, dx, None, finished=False)
    port = raw_features(arrays, dx, 4)
    for n in FEATURE_NAMES:
        assert np.array_equal(port[n], ref[n], equal_nan=True), n
    thr = _thresholds(b2_min=0.5, winsor={"log_chi_B": (-3.0, 0.5), "log_grad_ln_b2": (-1.0, 1.0), "asinh_div_v": (-1, 1)})
    et = etkt.Thresholds(b2_min=thr.b2_min, s_v=thr.s_v, limits=dict(thr.winsor))
    aux = {"smallb2": arrays["smallb2"].ravel()}
    want = np.stack([et.apply(n, ref[n].ravel(), aux, None) for n in FEATURE_NAMES], axis=1)
    got = finish_features({n: port[n].ravel() for n in FEATURE_NAMES}, aux["smallb2"], thr)
    assert np.array_equal(got, want, equal_nan=True)


def test_optional_sixth_feature_and_frozen_model():
    from aweSOM.features_grmhd import SIX_FEATURE_NAMES
    shape = (9, 9, 9)
    rng = np.random.default_rng(3)
    a = {n: rng.uniform(0.1, 2, shape) for n in INPUTS}
    a['Bx'][4, 4, 4] = a['By'][4, 4, 4] = a['Bz'][4, 4, 4] = 1e-6
    thr = _thresholds(b2_min=1e-3)
    actual = compute_features(a, (1, 1, 1), thr, names=SIX_FEATURE_NAMES)
    assert actual.reshape(*shape, 6)[4, 4, 4, 5] == 0
    model = type('Model', (), {'feature_names': SIX_FEATURE_NAMES, 'metadata': thr.to_metadata()})()
    assert FeatureThresholds.from_model(model) == thr


def test_optional_sixth_feature_matches_etkplot():
    from aweSOM.features_grmhd import SIX_FEATURE_NAMES
    from types import SimpleNamespace
    features, _ = _etkplot()
    shape = (9, 9, 9)
    rng = np.random.default_rng(3)
    a = {n: rng.uniform(0.1, 2, shape) for n in INPUTS}
    a['Bx'][4, 4, 4] = a['By'][4, 4, 4] = a['Bz'][4, 4, 4] = 1e-6
    thr = _thresholds(b2_min=1e-3)
    actual = compute_features(a, (1, 1, 1), thr, names=SIX_FEATURE_NAMES)
    expected = features.compute(SIX_FEATURE_NAMES, a, (1, 1, 1), SimpleNamespace(b2_min=thr.b2_min, s_v=thr.s_v, accuracy=4))
    for j, n in enumerate(SIX_FEATURE_NAMES):
        y = expected[n].astype(np.float32).ravel()
        if n in thr.winsor:
            y = np.clip(y, *thr.winsor[n])
        np.testing.assert_array_equal(actual[:, j], y)


def test_v2_smooth_floor_analytic_and_zero_field():
    from aweSOM.features_grmhd import SIX_FEATURE_NAMES
    (x, y, z), dx = _grid()
    a = dict(smallb2=np.ones_like(x), P=np.ones_like(x), rho_b=np.ones_like(x),
             Bx=x, By=np.zeros_like(x), Bz=np.ones_like(x))
    a.update({f'vel[{j}]': x*0 for j in range(3)})
    thr = FeatureThresholds(b2_min=0.5, s_v=1, pipeline='v2')
    select = (slice(2,-2),)*3
    got = compute_features(a, dx, thr, select=select, names=SIX_FEATURE_NAMES)
    norm2 = x*x+1+0.5
    # d(B/D)/dx = ((1.5)/D^3, 0, -x/D^3)
    chi = np.sqrt(1.5**2+x*x)/norm2**1.5
    np.testing.assert_allclose(got[:,2], np.log10(chi[select].ravel()+EPS), rtol=1e-6)
    np.testing.assert_allclose(got[:,5], (1/np.sqrt(norm2))[select].ravel(), rtol=1e-6)
    assert FeatureThresholds.from_metadata(thr.to_metadata()).pipeline == 'v2'
    # The exact zero field stays finite, without the legacy 0/0.
    for n in ('Bx','By','Bz'):
        a[n] = x*0
    zero = compute_features(a, dx, thr, select=select, names=SIX_FEATURE_NAMES)
    np.testing.assert_array_equal(zero[:,2], -3)
    np.testing.assert_array_equal(zero[:,5], 0)


def test_v2_smooth_floor_matches_etkplot():
    from aweSOM.features_grmhd import SIX_FEATURE_NAMES
    from types import SimpleNamespace
    features, _ = _etkplot()
    (x, y, z), dx = _grid()
    a = dict(smallb2=np.ones_like(x), P=np.ones_like(x), rho_b=np.ones_like(x),
             Bx=x, By=np.zeros_like(x), Bz=np.ones_like(x))
    a.update({f'vel[{j}]': x*0 for j in range(3)})
    thr = FeatureThresholds(b2_min=0.5, s_v=1, pipeline='v2')
    select = (slice(2,-2),)*3
    got = compute_features(a, dx, thr, select=select, names=SIX_FEATURE_NAMES)
    port = features.compute(SIX_FEATURE_NAMES, a, dx, SimpleNamespace(pipeline='v2', b2_min=.5, s_v=1, accuracy=4))
    np.testing.assert_array_equal(got, np.column_stack([port[n][select].ravel() for n in SIX_FEATURE_NAMES]).astype(np.float32))


def test_explicit_invalid_etkplot_path_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("AWESOM_ETKPLOT_PATH", str(tmp_path / "missing"))
    with pytest.raises(pytest.fail.Exception, match="AWESOM_ETKPLOT_PATH"):
        _etkplot()


def test_explicit_incomplete_etkplot_path_does_not_fall_back(monkeypatch, tmp_path):
    other = tmp_path / "other"
    other_pkg = other / "etkplot"
    other_pkg.mkdir(parents=True)
    (other_pkg / "features.py").write_text("# stub\n")
    (other_pkg / "thresholds.py").write_text("# stub\n")

    saved = {k: v for k, v in sys.modules.items() if k == "etkplot" or k.startswith("etkplot.")}

    def _clear_etkplot_modules():
        for key in [k for k in sys.modules if k == "etkplot" or k.startswith("etkplot.")]:
            del sys.modules[key]
        importlib.invalidate_caches()

    try:
        # Empty etkplot/ directory with no fallback on sys.path (or with any installed fallback).
        empty = tmp_path / "empty"
        (empty / "etkplot").mkdir(parents=True)
        _clear_etkplot_modules()
        monkeypatch.setenv("AWESOM_ETKPLOT_PATH", str(empty))
        with pytest.raises(pytest.fail.Exception, match="AWESOM_ETKPLOT_PATH"):
            _etkplot()

        # Empty etkplot/ directory with another regular etkplot package on sys.path.
        (other_pkg / "__init__.py").write_text("")
        monkeypatch.syspath_prepend(str(other))
        _clear_etkplot_modules()
        with pytest.raises(pytest.fail.Exception, match="AWESOM_ETKPLOT_PATH"):
            _etkplot()

        # Incomplete etkplot/ (features.py only) with another namespace etkplot providing thresholds.py.
        (other_pkg / "__init__.py").unlink()
        partial = tmp_path / "partial"
        (partial / "etkplot").mkdir(parents=True)
        (partial / "etkplot" / "features.py").write_text("# stub\n")
        _clear_etkplot_modules()
        monkeypatch.setenv("AWESOM_ETKPLOT_PATH", str(partial))
        with pytest.raises(pytest.fail.Exception, match="AWESOM_ETKPLOT_PATH"):
            _etkplot()
    finally:
        _clear_etkplot_modules()
        sys.modules.update(saved)


def test_unset_etkplot_path_skips_when_checkout_absent(monkeypatch, tmp_path):
    monkeypatch.delenv("AWESOM_ETKPLOT_PATH", raising=False)
    monkeypatch.setattr(sys.modules[__name__], "ETKPLOT", tmp_path / "missing")
    with pytest.raises(pytest.skip.Exception, match="etkplot not available"):
        _etkplot()
