"""Benchmark original vs candidate feature kernels from examples/plasma-turbulence/obtain_features.py.

Run:
  PYTHONPATH=src:examples/plasma-turbulence python planning/opt/sce-post/bench_features.py [--quick]
Prints one JSON line with results. Synthetic float64 random fields (seed 0).
"""
import os, sys, json, time, contextlib, io
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "examples", "plasma-turbulence"))
import obtain_features as of  # original functions, imported unmodified

import numba as nb
from scipy.ndimage import zoom, convolve


# ---------------------------------------------------------------- candidates
def build_kernels(kn=19):
    """Verbatim copy of the kernelA/kernelS construction in obtain_features __main__ (kn=19)."""
    along_k, across_k = np.linspace(-1, 1, kn), np.linspace(-1, 1, kn)
    coord_kA = np.asarray(np.meshgrid(along_k, across_k))
    kernelA = np.zeros((kn, kn))
    for i in range(kn):
        for j in range(kn):
            if i != kn // 2 or j != kn // 2:
                hypot_ij = np.hypot(coord_kA[0, i, j], coord_kA[1, i, j])
                kernelA[i, j] = -coord_kA[0, i, j] / hypot_ij if hypot_ij <= 1 - 1 / kn else 0
    kernelA /= np.sum(np.abs(kernelA))
    kernelS = np.zeros((kn, kn))
    kernelS[kn // 2, :] = 1
    for i in range(kn):
        for j in range(kn):
            if i != kn // 2 or j != kn // 2:
                hypot_ij = np.hypot(coord_kA[0, i, j], coord_kA[1, i, j])
                kernelS[i, j] = np.abs(coord_kA[0, i, j] / hypot_ij) if hypot_ij <= 1 - 1 / kn else 0
    kernelS /= np.sum(np.abs(kernelS))
    return kernelA, kernelS


@nb.njit(cache=False, parallel=True, fastmath=False)
def _dirconv_numba(ext, direction, kA, kS, outA, outS):
    """Fused per-pixel kernel rotation + window dot product.

    Reimplements scipy.ndimage.rotate(K, angle, reshape=False, mode='nearest', order=1)
    (axes (0,1): in = R @ out + offset, R = [[c, s], [-s, c]], offset = cin - R @ cin;
    'nearest' clamps the source coordinate, then bilinear interpolation) inline, so the
    rotated kernels are never materialised. Window indexing matches convolve_by_slice:
    out[k, j, i] = sum ext[k+h, j:j+kn, i:i+kn] * rot(K) with direction[:, k, j, i].
    """
    nz, ny, nx = outA.shape
    kn = kA.shape[0]
    h = kn // 2
    cen = (kn - 1) / 2.0
    last = kn - 1
    for k in nb.prange(nz):
        sl = ext[k + h]
        for j in range(ny):
            for i in range(nx):
                n1 = direction[1, k, j, i]
                n2 = direction[2, k, j, i]
                ang = np.arctan2(n2, n1) * 180 / np.pi - 90
                rad = np.deg2rad(ang)
                c = np.cos(rad)
                s = np.sin(rad)
                off0 = cen - (c * cen + s * cen)
                off1 = cen - (-s * cen + c * cen)
                accA = 0.0
                accS = 0.0
                for p in range(kn):
                    for q in range(kn):
                        x0 = c * p + s * q + off0
                        x1 = -s * p + c * q + off1
                        if x0 < 0.0:
                            x0 = 0.0
                        elif x0 > last:
                            x0 = last
                        if x1 < 0.0:
                            x1 = 0.0
                        elif x1 > last:
                            x1 = last
                        f0 = np.floor(x0)
                        f1 = np.floor(x1)
                        i0 = int(f0)
                        i1 = int(f1)
                        w0 = x0 - f0
                        w1 = x1 - f1
                        i0b = i0 + 1 if i0 < last else last
                        i1b = i1 + 1 if i1 < last else last
                        a = ((1 - w0) * ((1 - w1) * kA[i0, i1] + w1 * kA[i0, i1b])
                             + w0 * ((1 - w1) * kA[i0b, i1] + w1 * kA[i0b, i1b]))
                        b = ((1 - w0) * ((1 - w1) * kS[i0, i1] + w1 * kS[i0, i1b])
                             + w0 * ((1 - w1) * kS[i0b, i1] + w1 * kS[i0b, i1b]))
                        v = sl[j + p, i + q]
                        accA += v * a
                        accS += v * b
                outA[k, j, i] = accA
                outS[k, j, i] = accS


def obtain_convolved_features_fast(feature_array, kernelA, kernelS, direction):
    """Drop-in for of.obtain_convolved_features (cubic arrays, like the original's [0,j,i] indexing assumes)."""
    kn = kernelA.shape[0]
    h = kn // 2
    ext = np.pad(feature_array, ((h, h), (h, h), (h, h)), mode="wrap")
    outA = np.empty(feature_array.shape)
    outS = np.empty(feature_array.shape)
    _dirconv_numba(ext, np.ascontiguousarray(direction), np.ascontiguousarray(kernelA),
                   np.ascontiguousarray(kernelS), outA, outS)
    return outA, outS


def pseudo_convolution_fast(variable, kernel, step_size):
    """Same math as of.pseudo_convolution (zoom order-3 spline downsample, NOT a block mean;
    wrap convolve; nearest-block upsample), with the triple Python loop replaced by np.repeat."""
    small = zoom(input=variable, zoom=1. / step_size)
    cw = convolve(small, kernel, mode="wrap", cval=0.0)
    up = cw
    for ax in range(3):
        up = np.repeat(up, step_size, axis=ax)
    out = np.zeros(variable.shape)
    s = tuple(slice(0, min(a, b)) for a, b in zip(variable.shape, up.shape))
    out[s] = up[s]
    return out


# vector-quantity features
def vec_features_orig(bx, by, bz, jx, jy, jz, ex, ey, ez):
    """Verbatim from obtain_features __main__ (guide='z')."""
    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    j_mag = np.sqrt(jx**2 + jy**2 + jz**2)
    e_mag = np.sqrt(ex**2 + ey**2 + ez**2)
    b_perp = np.sqrt(bx**2 + by**2)
    b_vec = np.array([bx, by, bz])
    j_vec = np.array([jx, jy, jz])
    e_vec = np.array([ex, ey, ez])
    b_dot_j = np.einsum('nijk,nijk->ijk', b_vec, j_vec)
    b_dot_e = np.einsum('nijk,nijk->ijk', b_vec, e_vec)
    e_dot_j = np.einsum('nijk,nijk->ijk', e_vec, j_vec)
    j_par = b_dot_j / b_mag
    j_par_abs = np.abs(j_par)
    j_perp2 = j_mag*j_mag - j_par*j_par
    j_perp2[j_perp2 < 0.] = 0.
    j_perp = np.sqrt(j_perp2)
    e_par = b_dot_e / b_mag
    e_perp = np.sqrt(e_mag*e_mag - e_par*e_par)
    return dict(b_mag=b_mag, j_mag=j_mag, e_mag=e_mag, b_perp=b_perp, b_dot_j=b_dot_j,
                e_dot_j=e_dot_j, j_par=j_par, j_par_abs=j_par_abs, j_perp=j_perp,
                e_par=e_par, e_perp=e_perp)


@nb.njit(parallel=True, fastmath=False)
def _vec_numba(bx, by, bz, jx, jy, jz, ex, ey, ez, o):
    n = bx.size
    bx = bx.ravel(); by = by.ravel(); bz = bz.ravel()
    jx = jx.ravel(); jy = jy.ravel(); jz = jz.ravel()
    ex = ex.ravel(); ey = ey.ravel(); ez = ez.ravel()
    for t in nb.prange(n):
        b2 = bx[t]*bx[t] + by[t]*by[t] + bz[t]*bz[t]
        j2 = jx[t]*jx[t] + jy[t]*jy[t] + jz[t]*jz[t]
        e2 = ex[t]*ex[t] + ey[t]*ey[t] + ez[t]*ez[t]
        bm = np.sqrt(b2); jm = np.sqrt(j2); em = np.sqrt(e2)
        bj = bx[t]*jx[t] + by[t]*jy[t] + bz[t]*jz[t]
        be = bx[t]*ex[t] + by[t]*ey[t] + bz[t]*ez[t]
        ej = ex[t]*jx[t] + ey[t]*jy[t] + ez[t]*jz[t]
        jp = bj / bm
        ep = be / bm
        jperp2 = jm*jm - jp*jp
        if jperp2 < 0.0:
            jperp2 = 0.0
        o[0, t] = bm; o[1, t] = jm; o[2, t] = em
        o[3, t] = np.sqrt(bx[t]*bx[t] + by[t]*by[t])
        o[4, t] = bj; o[5, t] = ej; o[6, t] = jp; o[7, t] = abs(jp)
        o[8, t] = np.sqrt(jperp2); o[9, t] = ep; o[10, t] = np.sqrt(em*em - ep*ep)


VEC_KEYS = ["b_mag", "j_mag", "e_mag", "b_perp", "b_dot_j", "e_dot_j", "j_par", "j_par_abs", "j_perp", "e_par", "e_perp"]


def vec_features_numpy_lean(bx, by, bz, jx, jy, jz, ex, ey, ez):
    """Same formulas, no stacked (3,N,N,N) copies; einsum dot -> elementwise multiply-add."""
    b_mag = np.sqrt(bx*bx + by*by + bz*bz)
    j_mag = np.sqrt(jx*jx + jy*jy + jz*jz)
    e_mag = np.sqrt(ex*ex + ey*ey + ez*ez)
    b_perp = np.sqrt(bx*bx + by*by)
    b_dot_j = bx*jx + by*jy + bz*jz
    b_dot_e = bx*ex + by*ey + bz*ez
    e_dot_j = ex*jx + ey*jy + ez*jz
    j_par = b_dot_j / b_mag
    j_perp = np.sqrt(np.maximum(j_mag*j_mag - j_par*j_par, 0.))
    e_par = b_dot_e / b_mag
    e_perp = np.sqrt(e_mag*e_mag - e_par*e_par)
    return dict(b_mag=b_mag, j_mag=j_mag, e_mag=e_mag, b_perp=b_perp, b_dot_j=b_dot_j,
                e_dot_j=e_dot_j, j_par=j_par, j_par_abs=np.abs(j_par), j_perp=j_perp,
                e_par=e_par, e_perp=e_perp)


def vec_features_numba(*f):
    o = np.empty((11, f[0].size), dtype=f[0].dtype)
    _vec_numba(*f, o)
    return {k: o[i].reshape(f[0].shape) for i, k in enumerate(VEC_KEYS)}


# ---------------------------------------------------------------- harness
def timeit(fn, reps):
    ts = []
    out = None
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), ts, out


def interleave(fa, fb, reps):
    ta, tb = [], []
    for _ in range(reps):
        t = time.perf_counter(); a = fa(); ta.append(time.perf_counter() - t)
        t = time.perf_counter(); b = fb(); tb.append(time.perf_counter() - t)
    return float(np.median(ta)), float(np.median(tb)), a, b


def diffs(a, b):
    a = np.asarray(a); b = np.asarray(b)
    d = np.abs(a - b)
    rel = d / np.maximum(np.abs(a), 1e-300)
    return dict(bitwise=bool(np.array_equal(a, b)), max_abs=float(d.max()),
                max_rel=float(rel.max()), max_abs_over_max_ref=float(d.max() / np.abs(a).max()))


def direction_of(field):
    g = np.gradient(field)
    g = np.array([g[2], g[1], g[0]])
    return g / np.linalg.norm(g, axis=0)


def quiet(fn):
    def w():
        with contextlib.redirect_stdout(io.StringIO()):
            return fn()
    return w


def main():
    quick = "--quick" in sys.argv
    rng = np.random.default_rng(0)
    res = {"numba_threads": nb.get_num_threads(), "kernel": "kn=19 (script default), kernelA/kernelS as in __main__"}
    kA, kS = build_kernels(19)

    # --- 0. rotate re-implementation vs scipy.ndimage.rotate, many angles
    from scipy.ndimage import rotate
    from numba import njit
    angs = np.concatenate([rng.uniform(-270, 90, 2000), [-90, 0, 90, -180, 45, -45, -270]])
    worst = 0.0
    for ang in angs:
        ref = rotate(kA, ang, reshape=False, mode="nearest", order=1)
        # one-pixel probe: window = delta at (p,q) picks rotated kernel entry -> use full eval with unit windows
        n = np.array([0.0, np.cos(np.deg2rad(ang + 90)), np.sin(np.deg2rad(ang + 90))])
        # direct check via fused kernel on a 1x1x1 output: ext=window, sum = <window, rot(K)>
        W = rng.standard_normal((19, 19))
        ext = np.zeros((19, 19, 19)); ext[9] = W
        oA = np.empty((1, 1, 1)); oS = np.empty((1, 1, 1))
        _dirconv_numba(ext, n.reshape(3, 1, 1, 1), kA, kS, oA, oS)
        refS = rotate(kS, ang, reshape=False, mode="nearest", order=1)
        worst = max(worst, abs(oA[0, 0, 0] - np.einsum("ji,ji", W, ref)), abs(oS[0, 0, 0] - np.einsum("ji,ji", W, refS)))
    res["rotate_reimpl_max_abs_diff_2007_angles"] = worst

    # --- 1. directional convolution: original (Pool over slices) vs numba, 64^3 (and 32^3)
    dirres = {}
    for N in ([32] if quick else [32, 64]):
        f = rng.standard_normal((N, N, N))
        d = direction_of(f)
        _ = obtain_convolved_features_fast(f[:8, :8, :8].copy(), kA, kS, direction_of(f[:8, :8, :8].copy()))  # JIT warm-up
        reps = 3
        tb, tc, (A0, S0), (A1, S1) = interleave(quiet(lambda: of.obtain_convolved_features(f, kA, kS, d)),
                                                lambda: obtain_convolved_features_fast(f, kA, kS, d), reps)
        nb.set_num_threads(1)
        t1, _, _ = timeit(lambda: obtain_convolved_features_fast(f, kA, kS, d), reps)
        nb.set_num_threads(res["numba_threads"])
        dirres[f"{N}^3"] = dict(orig_pool_s=tb, numba_parallel_s=tc, ratio=tb / tc, numba_1thread_s=t1,
                                ratio_vs_1thread=tb / t1,
                                asym=diffs(A0, A1), sym=diffs(S0, S1))
    for N in ([128] if quick else [128, 256]):
        f = rng.standard_normal((N, N, N))
        d = direction_of(f)
        t, ts, _ = timeit(lambda: obtain_convolved_features_fast(f, kA, kS, d), 3)
        dirres[f"{N}^3_numba_only"] = dict(numba_parallel_s=t, reps=ts)
        del f, d
    res["directional_conv"] = dirres

    # --- 2. pseudo_convolution (step 4, 3x3x3 ones/27, as in __main__)
    kiso = np.ones((3, 3, 3)) / 27
    pres = {}
    for N in ([64, 128] if quick else [64, 128, 256]):
        f = rng.standard_normal((N, N, N))
        tb, tc, a, b = interleave(lambda: of.pseudo_convolution(f, kiso, 4), lambda: pseudo_convolution_fast(f, kiso, 4), 3)
        tz, _, _ = timeit(lambda: zoom(f, 0.25), 3)
        pres[f"{N}^3"] = dict(orig_s=tb, cand_s=tc, ratio=tb / tc, zoom_only_s=tz, **diffs(a, b))
    # non-divisible shape check
    f = rng.standard_normal((66, 66, 66))
    pres["66^3_nondivisible"] = diffs(of.pseudo_convolution(f, kiso, 4), pseudo_convolution_fast(f, kiso, 4))
    res["pseudo_convolution"] = pres

    # --- 3. vector features
    vres = {}
    for N in ([128] if quick else [128, 256]):
        fl = [rng.standard_normal((N, N, N)) for _ in range(9)]
        _ = vec_features_numba(*[x[:4, :4, :4].copy() for x in fl])
        _ = vec_features_numba(*[x[:4, :4, :4].astype(np.float32) for x in fl])
        ta, _, A = timeit(lambda: vec_features_orig(*fl), 3)
        tl, _, L = timeit(lambda: vec_features_numpy_lean(*fl), 3)
        tn, _, B = timeit(lambda: vec_features_numba(*fl), 3)
        ta2, _, _ = timeit(lambda: vec_features_orig(*fl), 3)
        f32 = [x.astype(np.float32) for x in fl]
        to32, _, _ = timeit(lambda: vec_features_orig(*f32), 3)
        tn32, _, C = timeit(lambda: vec_features_numba(*f32), 3)
        worst_lean = {k: diffs(A[k], L[k])["max_rel"] for k in VEC_KEYS}
        worst_nb = {k: diffs(A[k], B[k])["max_rel"] for k in VEC_KEYS}
        f32err = {k: float(np.max(np.abs(A[k] - C[k])) / np.abs(A[k]).max()) for k in VEC_KEYS}
        vres[f"{N}^3"] = dict(orig_s=[ta, ta2], lean_numpy_s=tl, numba_fused_s=tn, ratio_numba=min(ta, ta2) / tn,
                              ratio_lean=min(ta, ta2) / tl, orig_f32_s=to32, numba_f32_s=tn32,
                              lean_max_rel=max(worst_lean.values()), numba_max_rel=max(worst_nb.values()),
                              f32_max_abs_over_max=max(f32err.values()),
                              lean_bitwise=all(np.array_equal(A[k], L[k]) for k in VEC_KEYS))
        del fl, f32, A, B, C, L
    res["vector_features"] = vres
    print(json.dumps(res))


if __name__ == "__main__":
    main()
