"""The GRMHD current-sheet features, computed on one level block.

A numpy-only port of etkplot's ``features``/``derivatives``/``thresholds``
modules (``/Volumes/ExternalNVMe/etk-analysis/etkplot``), with five default feature columns, optional ``bz_frac`` in
``SIX_FEATURE_NAMES``, and ``pol_frac`` in ``SEVEN_FEATURE_NAMES``:

=================  =========================================================
``log_beta_inv``   log10(b^2 / (2 P))
``log_theta``      log10(P / rho)
``log_chi_B``      log10(chi_B + eps), chi_B = |d_j b^i|_F with b = B/|B|;
                   set to log10(eps) where b^2 < ``b2_min``
``log_grad_ln_b2`` log10(|grad b^2| / b^2 + eps)
``asinh_div_v``    asinh(div v / ``s_v``)
=================  =========================================================

``bz_frac`` is abs(Bz)/|B|, zero at zero norm or where |B| < b2_min.
``pol_frac`` is sqrt(B_R^2 + Bz^2)/|B| about the nearer hole in 3D,
with ties going to hole 1. On its z axis B_R^2 = (Bx^2 + By^2)/2.
Both fractions are zero at B = 0 and use no winsor clipping.

The default pipeline is v1. Opt in with ``FeatureThresholds(..., pipeline="v2")``.
V2 uses B/sqrt(B.B + b2_min) for chi and sqrt(|B|^2 + b2_min) for
both fraction denominators. It forbids winsor clipping and requires a frozen
median-IQR model scaler. Scaler center and scale belong to ``SOMModel.scaler``.
Pol_frac always needs explicit ``xyz`` axes and ``hole_centres`` at computation
time; saved training positions must never substitute for snapshot geometry.

eps = 1e-3. Derivatives are centred (Fornberg) first differences of
accuracy 2, 4 or 6 along the block's x, y, z axes (flat metric, coordinate
basis), each on the level's own grid. A point whose stencil reaches past the
block, or touches a NaN, comes out NaN. The arithmetic is the same as
etkplot's, operation for operation, so the values are bitwise equal to
etkplot's for the same inputs (``tests/features_grmhd_test.py`` pins this on
a synthetic block; the report records the check on a Carpet block).

The thresholds ``b2_min`` and ``s_v`` and the winsor limits of
``log_chi_B``, ``log_grad_ln_b2`` and ``asinh_div_v`` are set once, from the
training sample, and must never be recomputed per chunk: labels are only
comparable across chunks and snapshots if every chunk goes through the same
transform. :class:`FeatureThresholds` carries them; it is read from a
training-set or grid file's root attributes (:meth:`FeatureThresholds.from_h5_attrs`)
and stored in, and read back from, a frozen :class:`~aweSOM.model.SOMModel`'s
metadata (:meth:`FeatureThresholds.to_metadata`,
:meth:`FeatureThresholds.from_model`).

Inputs use etkplot's (IllinoisGRMHD/HydroBase) variable names. AsterX names
are mapped onto them by the readers (:mod:`aweSOM.io`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

__all__ = [
    "FEATURE_NAMES",
    "SIX_FEATURE_NAMES",
    "SEVEN_FEATURE_NAMES",
    "INPUTS",
    "EPS",
    "DEFAULT_ACCURACY",
    "FeatureThresholds",
    "derivative",
    "half_width",
    "raw_features",
    "finish_features",
    "compute_features",
]

FEATURE_NAMES: Tuple[str, ...] = (
    "log_beta_inv",
    "log_theta",
    "log_chi_B",
    "log_grad_ln_b2",
    "asinh_div_v",
)
SIX_FEATURE_NAMES = FEATURE_NAMES + ("bz_frac",)
SEVEN_FEATURE_NAMES = SIX_FEATURE_NAMES + ("pol_frac",)
INPUTS: Tuple[str, ...] = ("smallb2", "P", "rho_b", "Bx", "By", "Bz", "vel[0]", "vel[1]", "vel[2]")
EPS = 1e-3
ACCURACIES = (2, 4, 6)
DEFAULT_ACCURACY = 4
WINSORISED = ("log_chi_B", "log_grad_ln_b2", "asinh_div_v")
# features computed in float64 and finished (floor / asinh) before the cast to float32
_FINISHED = ("log_chi_B", "asinh_div_v")

# weight of offset +k, k = 1..w; offset -k has minus it (etkplot.derivatives._FIRST)
_FIRST = {
    2: (1.0 / 2,),
    4: (2.0 / 3, -1.0 / 12),
    6: (3.0 / 4, -3.0 / 20, 1.0 / 60),
}

_META_PREFIX = "feature_"


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def half_width(accuracy: int) -> int:
    """Points a centred stencil of ``accuracy`` reaches on each side."""
    if accuracy not in ACCURACIES:
        raise ValueError(f"accuracy must be one of {ACCURACIES}, not {accuracy!r}")
    return accuracy // 2


def _shifted(f: np.ndarray, axis: int, lo: int, hi: int) -> np.ndarray:
    idx = [slice(None)] * f.ndim
    idx[axis] = slice(lo, f.shape[axis] - hi)
    return f[tuple(idx)]


def derivative(f: np.ndarray, axis: int, dx: float, accuracy: int = DEFAULT_ACCURACY) -> np.ndarray:
    """d f / d x_axis by centred differences; NaN within the stencil's half-width
    of either end of ``axis``. Same operations, in the same order, as
    ``etkplot.derivatives.derivative``."""
    w = half_width(accuracy)
    out = np.full(f.shape, np.nan)
    n = f.shape[axis]
    if n <= 2 * w:
        return out
    acc = None
    for k, c in enumerate(_FIRST[accuracy], start=1):
        term = c * (_shifted(f, axis, w + k, w - k) - _shifted(f, axis, w - k, w + k))
        acc = term if acc is None else acc + term
    idx = [slice(None)] * f.ndim
    idx[axis] = slice(w, n - w)
    out[tuple(idx)] = acc / dx
    return out


def _log10(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(x)


def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


def _chi_B(B: Sequence[np.ndarray], dx, accuracy: int) -> np.ndarray:
    norm = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = [c / norm for c in B]
        chi2 = np.zeros(norm.shape)
        for j in range(3):
            dB = [derivative(c, j, dx[j], accuracy) for c in B]
            along = b[0] * dB[0] + b[1] * dB[1] + b[2] * dB[2]
            for i in range(3):
                chi2 += ((dB[i] - b[i] * along) / norm) ** 2
    return np.sqrt(chi2)


def smooth_chi(aux, floor):
    """Analytic gradient of B/sqrt(B.B + floor), with a spatially fixed floor."""
    B = [aux[k] for k in ("Bx", "By", "Bz")]
    norm = np.sqrt(sum(c*c for c in B) + floor)
    b = [c / norm for c in B]
    chi2 = np.zeros_like(norm)
    for j in range(3):
        dB = [aux[f"dB{i}{j}"] for i in range(3)]
        along = sum(b[i]*dB[i] for i in range(3))
        for i in range(3):
            chi2 += ((dB[i] - b[i]*along)/norm)**2
    return np.sqrt(chi2)


def _hole_centres(centres):
    """Validate explicitly supplied positions; never infer a geometry."""
    c = np.asarray(centres, dtype=float)
    if c.shape != (2, 3) or not np.all(np.isfinite(c)):
        raise ValueError("pol_frac needs hole_centres with shape (2, 3), all finite")
    return c


def _pol_frac(a, xyz, hole_centres):
    """sqrt(B_R^2 + Bz^2)/|B| about the nearer hole, before any floor. Kept
    <= 1 in floating point: B_R^2 is capped at Bx^2 + By^2 and the sums
    are formed in the same order as |B|^2's."""
    c = _hole_centres(hole_centres)
    if xyz is None:
        raise ValueError("pol_frac needs the point coordinates (xyz)")
    x, y, z = (np.asarray(u, dtype=float).reshape(
        [-1 if k == d else 1 for k in range(3)]) for d, u in enumerate(xyz))
    second = sum((u - c[1, k]) ** 2 for k, u in enumerate((x, y, z))) < \
        sum((u - c[0, k]) ** 2 for k, u in enumerate((x, y, z)))
    rx = np.where(second, x - c[1, 0], x - c[0, 0])  # nearer hole; ties to hole 1
    ry = np.where(second, y - c[1, 1], y - c[0, 1])
    rx, ry = np.broadcast_arrays(rx, ry)
    r = np.hypot(rx, ry)
    bx, by, bz = a["Bx"], a["By"], a["Bz"]
    bh2 = bx ** 2 + by ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        br = (rx * bx + ry * by) / r
    br2 = np.where(r > 0, np.minimum(br * br, bh2), 0.5 * bh2)
    norm = np.sqrt(bh2 + bz ** 2)
    return np.divide(np.sqrt(br2 + bz ** 2), norm, out=np.zeros_like(norm), where=norm != 0)


def raw_features(
    arrays: Mapping[str, np.ndarray],
    dx: Sequence[float],
    accuracy: int = DEFAULT_ACCURACY,
    names: Sequence[str] = FEATURE_NAMES,
    pipeline: str = "v1",
    *,
    xyz=None,
    hole_centres=None,
) -> Dict[str, np.ndarray]:
    """Each feature before its thresholds, float64, on the block's shape.

    ``log_chi_B`` is without its floor and ``asinh_div_v`` is div v itself
    (etkplot's "before finish" values).

    Args:
        arrays: input variable -> (nx, ny, nz) float array (NaN = no data).
        dx: grid spacing per axis.
        accuracy: order of the centred differences (2, 4 or 6).
        names: features to compute.
        pipeline: v1 (default) or v2, matching the finish thresholds.
        xyz: three 1D coordinate axes matching the block, required for pol_frac.
        hole_centres: explicit finite (2, 3) positions, required for pol_frac.

    Returns:
        dict: feature name -> float64 array of the block's shape.
    """
    if pipeline not in ("v1", "v2"):
        raise ValueError(f"unknown pipeline {pipeline!r}")
    dx = np.asarray(dx, dtype=float)
    out: Dict[str, np.ndarray] = {}
    for n in names:
        if n == "log_beta_inv":
            out[n] = _log10(_ratio(arrays["smallb2"], 2 * arrays["P"]))
        elif n == "log_theta":
            out[n] = _log10(_ratio(arrays["P"], arrays["rho_b"]))
        elif n == "log_chi_B":
            if pipeline == "v2":
                out.update({k: arrays[k] for k in ("Bx", "By", "Bz")})
                for i, k in enumerate(("Bx", "By", "Bz")):
                    for j in range(3):
                        out[f"dB{i}{j}"] = derivative(arrays[k], j, dx[j], accuracy)
                # Provisional floor tests stencil finiteness, including B=0.
                # finish_features uses these derivatives with the frozen floor.
                out[n] = _log10(smooth_chi(out, 1.0) + EPS)
                continue
            out[n] = _log10(_chi_B([arrays["Bx"], arrays["By"], arrays["Bz"]], dx, accuracy) + EPS)
        elif n == "log_grad_ln_b2":
            b2 = arrays["smallb2"]
            g2 = sum(derivative(b2, d, dx[d], accuracy) ** 2 for d in range(3))
            out[n] = _log10(_ratio(np.sqrt(g2), b2) + EPS)
        elif n == "pol_frac":
            out[n] = _pol_frac(arrays, xyz, hole_centres)
            out["B_norm"] = np.sqrt(arrays["Bx"] ** 2 + arrays["By"] ** 2 + arrays["Bz"] ** 2)
        elif n == "bz_frac":
            norm = np.sqrt(arrays["Bx"] ** 2 + arrays["By"] ** 2 + arrays["Bz"] ** 2)
            out[n] = np.divide(np.abs(arrays["Bz"]), norm, out=np.zeros_like(norm), where=norm != 0)
            out["B_norm"] = norm
            out["abs_Bz"] = np.abs(arrays["Bz"])
        elif n == "asinh_div_v":
            out[n] = sum(derivative(arrays[f"vel[{d}]"], d, dx[d], accuracy) for d in range(3))
        else:
            raise ValueError(f"unknown feature {n!r}; the features are {SEVEN_FEATURE_NAMES}")
    return out


@dataclass
class FeatureThresholds:
    """The global transform settings of the GRMHD features.

    Attributes:
        b2_min: log_chi_B is set to log10(eps) where smallb2 < b2_min (0: no floor).
        s_v: scale of div v in asinh_div_v.
        winsor: feature -> (lo, hi) clip limits, applied in float32 to finite values.
        accuracy: order of the centred differences.
        epsilon: eps of the log features (fixed at 1e-3; checked on load).
    """

    b2_min: float
    s_v: float
    winsor: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    accuracy: int = DEFAULT_ACCURACY
    epsilon: float = EPS
    pipeline: str = "v1"
    scaler_kind: str | None = None
    feature_names: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.pipeline not in ("v1", "v2"):
            raise ValueError(f"unknown pipeline {self.pipeline}")
        if self.pipeline == "v2" and self.winsor:
            raise ValueError("v2 does not allow winsor clipping")
        expected_scaler = "median-iqr" if self.pipeline == "v2" else "manual"
        if self.scaler_kind is None:
            self.scaler_kind = expected_scaler
        if self.scaler_kind != expected_scaler:
            raise ValueError("pipeline and scaler_kind disagree")
        self.feature_names = tuple(self.feature_names)
        if self.feature_names and self.feature_names not in (FEATURE_NAMES, SIX_FEATURE_NAMES, SEVEN_FEATURE_NAMES):
            raise ValueError("unsupported feature names or column order")
        self.b2_min = float(self.b2_min)
        if not np.isfinite(self.b2_min) or self.b2_min < 0:
            raise ValueError("b2_min must be finite and nonnegative")
        self.s_v = float(self.s_v)
        self.accuracy = int(self.accuracy)
        half_width(self.accuracy)
        if not (np.isfinite(self.s_v) and self.s_v > 0):
            raise ValueError(f"s_v must be positive, not {self.s_v}")
        if float(self.epsilon) != EPS:
            raise ValueError(f"epsilon {self.epsilon} differs from the port's {EPS}")
        self.winsor = {str(k): (float(v[0]), float(v[1])) for k, v in self.winsor.items()}
        unknown = [k for k in self.winsor if k not in FEATURE_NAMES]
        if unknown:
            raise ValueError(f"winsor limits for unknown features {unknown}")

    @property
    def halo(self) -> int:
        """Points needed on each side of a point along each axis."""
        return half_width(self.accuracy)

    # ---------------------------------------------------------------- I/O
    @classmethod
    def from_h5_attrs(cls, attrs: Mapping) -> "FeatureThresholds":
        """From the root attributes of an etkplot training-set or grid file
        (``b2_min``, ``s_v``, ``winsor_<feature>``, ``accuracy``, ``epsilon``)."""
        if _text(attrs.get("direction_norm", "Eulerian B dot B")) != "Eulerian B dot B":
            raise ValueError("unsupported direction_norm")
        winsor = {}
        for n in FEATURE_NAMES:
            key = f"winsor_{n}"
            if key in attrs:
                lo, hi = np.asarray(attrs[key], dtype=float).ravel()
                winsor[n] = (lo, hi)
        return cls(
            b2_min=float(attrs["b2_min"]),
            s_v=float(attrs["s_v"]),
            winsor=winsor,
            accuracy=int(attrs.get("accuracy", DEFAULT_ACCURACY)),
            epsilon=float(attrs.get("epsilon", EPS)),
            pipeline=_text(attrs.get("pipeline", "v1")),
            scaler_kind=_text(attrs["scaler_kind"]) if "scaler_kind" in attrs else None,
        )

    @classmethod
    def from_h5(cls, path) -> "FeatureThresholds":
        """From an etkplot training-set or grid file."""
        import h5py

        with h5py.File(path, "r") as f:
            thr = cls.from_h5_attrs(dict(f.attrs))
            if "names" in f:
                thr.feature_names = tuple(_text(n) for n in f["names"][:])
                thr.__post_init__()
            return thr

    def to_metadata(self) -> Dict[str, str]:
        """``SOMModel.metadata`` entries (``feature_*`` keys, floats as %.17g so
        they read back bitwise)."""
        fmt = lambda x: "%.17g" % x  # noqa: E731
        meta = {
            f"{_META_PREFIX}set": f"grmhd-etkplot-{self.pipeline}",
            f"{_META_PREFIX}pipeline": self.pipeline,
            f"{_META_PREFIX}b2_min": fmt(self.b2_min),
            f"{_META_PREFIX}s_v": fmt(self.s_v),
            f"{_META_PREFIX}accuracy": str(self.accuracy),
            f"{_META_PREFIX}epsilon": fmt(self.epsilon),
        }
        meta[f"{_META_PREFIX}scaler_kind"] = self.scaler_kind
        meta[f"{_META_PREFIX}direction_norm"] = "Eulerian B dot B"
        if self.feature_names:
            meta[f"{_META_PREFIX}columns"] = " ".join(self.feature_names)
        for n, (lo, hi) in self.winsor.items():
            meta[f"{_META_PREFIX}winsor_{n}"] = f"{fmt(lo)} {fmt(hi)}"
        return meta

    @classmethod
    def from_metadata(cls, meta: Mapping[str, str]) -> "FeatureThresholds":
        """Inverse of :meth:`to_metadata`. Raises KeyError if the model holds no thresholds."""
        p = _META_PREFIX
        if f"{p}b2_min" not in meta or f"{p}s_v" not in meta:
            raise KeyError(
                "the model's metadata holds no feature thresholds (feature_b2_min, feature_s_v); "
                "freeze it with FeatureThresholds.to_metadata()"
            )
        pipeline = _text(meta.get(f"{p}pipeline", "v1"))
        if meta.get(f"{p}set", f"grmhd-etkplot-{pipeline}") != f"grmhd-etkplot-{pipeline}":
            raise ValueError("feature set and pipeline disagree")
        if meta.get(f"{p}direction_norm", "Eulerian B dot B") != "Eulerian B dot B":
            raise ValueError("unsupported direction_norm")
        winsor = {}
        for n in FEATURE_NAMES:
            v = meta.get(f"{p}winsor_{n}")
            if v is not None:
                lo, hi = (float(t) for t in str(v).split())
                winsor[n] = (lo, hi)
        return cls(
            b2_min=float(meta[f"{p}b2_min"]),
            s_v=float(meta[f"{p}s_v"]),
            winsor=winsor,
            accuracy=int(meta.get(f"{p}accuracy", DEFAULT_ACCURACY)),
            epsilon=float(meta.get(f"{p}epsilon", EPS)),
            pipeline=pipeline,
            scaler_kind=meta.get(f"{p}scaler_kind"),
            feature_names=tuple(meta.get(f"{p}columns", "").split()),
        )

    @classmethod
    def from_model(cls, model) -> "FeatureThresholds":
        """From a frozen :class:`~aweSOM.model.SOMModel`; also checks that the
        model uses five, six or seven columns and matching transform metadata."""
        if tuple(model.feature_names) not in (FEATURE_NAMES, SIX_FEATURE_NAMES, SEVEN_FEATURE_NAMES):
            raise ValueError(f"model features {model.feature_names} are not {list(FEATURE_NAMES)}")
        thr = cls.from_metadata(model.metadata)
        if thr.feature_names and thr.feature_names != tuple(model.feature_names):
            raise ValueError("model and threshold feature names disagree")
        if thr.pipeline == "v2" and model.scaler.kind != thr.scaler_kind:
            raise ValueError("v2 requires the frozen median-iqr scaler")
        return thr


def finish_features(
    raw: Mapping[str, np.ndarray],
    smallb2: np.ndarray,
    thresholds: FeatureThresholds,
    names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    """Apply the thresholds to raw features at a set of points.

    Follows etkplot's ``Thresholds.apply``: the b2_min floor (log_chi_B) and
    asinh(x/s_v) (asinh_div_v) in float64, then the cast to float32, then the
    winsor clip in float32 of the finite values. Non-finite values stay so.

    Args:
        raw: feature -> float64 values at N points (from :func:`raw_features`).
        smallb2: float64 b^2 at the same N points (for the floor).
        thresholds: the frozen transform.
        names: column order.

    Returns:
        np.ndarray: float32 [N, len(names)].
    """
    first = raw[names[0]]
    out = np.empty((np.asarray(first).size, len(names)), np.float32)
    for j, n in enumerate(names):
        x = np.asarray(raw[n]).ravel()
        if n == "log_chi_B" and thresholds.pipeline == "v2":
            x = _log10(smooth_chi(raw, thresholds.b2_min).ravel() + EPS)
        elif n == "bz_frac" and thresholds.pipeline == "v2":
            denom = np.sqrt(np.asarray(raw["B_norm"]).ravel()**2 + thresholds.b2_min)
            x = np.divide(np.asarray(raw["abs_Bz"]).ravel(), denom, out=np.zeros_like(denom), where=denom != 0)
        elif n == "pol_frac" and thresholds.pipeline == "v2":
            norm = np.asarray(raw["B_norm"]).ravel()
            denom = np.sqrt(norm**2 + thresholds.b2_min)
            x = np.divide(x * norm, denom, out=np.zeros_like(norm), where=denom != 0)
        elif n == "log_chi_B" and thresholds.b2_min > 0:
            x = x.copy()
            with np.errstate(invalid="ignore"):
                x[(np.asarray(smallb2).ravel() < thresholds.b2_min) & np.isfinite(x)] = np.log10(EPS)
        elif n in ("bz_frac", "pol_frac"):
            x = np.where(np.asarray(raw["B_norm"]).ravel() < thresholds.b2_min, 0.0, x)
        elif n == "asinh_div_v":
            x = np.arcsinh(x / thresholds.s_v)
        y = x.astype(np.float32)
        if n in thresholds.winsor:
            lo, hi = (np.float32(v) for v in thresholds.winsor[n])
            ok = np.isfinite(y)
            y[ok] = np.clip(y[ok], lo, hi)
        out[:, j] = y
    return out


def compute_features(
    arrays: Mapping[str, np.ndarray],
    dx: Sequence[float],
    thresholds: FeatureThresholds,
    select=None,
    names: Sequence[str] = FEATURE_NAMES,
    *,
    xyz=None,
    hole_centres=None,
) -> np.ndarray:
    """Features after the thresholds at selected points of a block.

    Args:
        arrays: input variable -> (nx, ny, nz) float64 block including its halo.
        dx: spacing per axis.
        thresholds: the frozen transform (also gives the accuracy).
        select: boolean mask of the block's shape, or an index expression, picking
            the points to return (C order). None: every point.
        names: column order.
        xyz: three 1D axes for the full block, including its halo.
        hole_centres: explicit finite (2, 3) positions for this snapshot.

    Returns:
        np.ndarray: float32 [N, len(names)].
    """
    if thresholds.feature_names and tuple(names) != thresholds.feature_names:
        raise ValueError("requested and threshold feature names disagree")
    raw = raw_features(arrays, dx, thresholds.accuracy, names, thresholds.pipeline,
                       xyz=xyz, hole_centres=hole_centres)
    b2 = arrays["smallb2"]
    if select is None:
        pick = lambda a: np.asarray(a).ravel()  # noqa: E731
    else:
        pick = lambda a: np.asarray(a)[select]  # noqa: E731
    return finish_features({n: pick(v) for n, v in raw.items()}, pick(b2), thresholds, names)
