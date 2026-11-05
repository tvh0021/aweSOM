"""CPU float32 reference in the Metal kernel's exact operation order.

``bmu_f32`` rounds weights and rows to float32, then per row and node does a
float32 subtract, float32 multiply and float32 add in feature order, and keeps
the first minimum (strict ``<``) over nodes in index order. Numba compiles
without fastmath, so LLVM does not contract the multiply-add.

The builders make inputs whose answer depends on the tie rule or on FMA
contraction, so a kernel that breaks either rule fails a bitwise comparison.
"""
from fractions import Fraction

import numba
import numpy as np


@numba.njit(parallel=True, cache=True)
def _bmu_f32(w, x, idx, d2):
    rows, features = x.shape
    nodes = w.shape[0]
    for p in numba.prange(rows):
        best = np.float32(np.inf)
        best_i = 0
        for j in range(nodes):
            acc = np.float32(0.0)
            for f in range(features):
                d = w[j, f] - x[p, f]
                sq = d * d
                acc = acc + sq
            if acc < best:
                best = acc
                best_i = j
        idx[p] = best_i
        d2[p] = best


def bmu_f32(weights, data):
    """int32 indices and float32 squared distances, float32 arithmetic."""
    w = np.ascontiguousarray(weights, np.float32)
    x = np.ascontiguousarray(data, np.float32)
    idx, d2 = np.empty(len(x), np.int32), np.empty(len(x), np.float32)
    _bmu_f32(w, x, idx, d2)
    return idx, d2


def bmu_f32_numpy(weights, data):
    """Independent check of ``bmu_f32``: numpy ufuncs round every operation
    separately, and argmin returns the first minimum. O(rows * nodes) memory."""
    w = np.asarray(weights, np.float32)
    x = np.asarray(data, np.float32)
    acc = np.zeros((len(x), len(w)), np.float32)
    for f in range(w.shape[1]):
        d = w[None, :, f] - x[:, None, f]
        acc = acc + d * d
    idx = acc.argmin(axis=1).astype(np.int32)
    return idx, acc[np.arange(len(x)), idx]


def equidistant_pairs(features, pairs, rows_per_pair, extra_nodes, rng, span=3):
    """Rows exactly equidistant (in float32) to two nodes of their pair.

    Pair k has nodes m_k + v and m_k - v and rows m_k + u with u . v = 0, all
    small integers, so every float32 operation is exact and both distances
    equal |u|^2 + |v|^2 in any order. Pairs sit 1000 apart; ``extra_nodes``
    decoys sit farther still. Nodes are shuffled, so a pair can straddle tile
    boundaries in either order. Returns weights, rows and the expected index
    (lower node of the pair) per row.
    """
    if features < 2:
        raise ValueError('need at least two features for u orthogonal to v')
    w, x, expect = [], [], []
    for k in range(pairs):
        m = np.zeros(features)
        m[k % features] = 1000.0 * (1 + k // features)
        while True:
            v = rng.integers(-span, span + 1, features).astype(float)
            if v.any():
                break
        w += [m + v, m - v]
        for _ in range(rows_per_pair):
            u = rng.integers(-span, span + 1, features).astype(float)
            # Project out v with an integer correction on one coordinate where
            # v is nonzero: u_f -= (u . v) / v_f only when that is an integer.
            f = int(np.flatnonzero(v)[0])
            dot = u @ v
            if dot % v[f]:
                u[:] = 0
            else:
                u[f] -= dot / v[f]
            assert u @ v == 0
            x.append(m + u)
            expect.append(k)
    decoys = 1e5 + rng.integers(0, 100, (extra_nodes, features)).astype(float)
    w = np.vstack([np.array(w), decoys])
    order = rng.permutation(len(w))
    where = np.argsort(order)  # new position of old node i
    w = w[order]
    expect = np.array([min(where[2 * k], where[2 * k + 1]) for k in expect], np.int32)
    return w, np.array(x), expect


def fma_cases(count, rng, features=2, target=5.0):
    """Two-node cases where FMA contraction changes the answer.

    Row 0 (the origin, so d = w exactly). Node B = (1, 2, 0, ...) has distance
    exactly ``target`` = 5 either way. Node A = (a0, a1, 0, ...) is chosen so
    that round(round(a0^2) + round(a1^2)) == 5 (a tie with B, so the lower
    index wins), while round(round(a0^2) + a1^2), the contracted form, is not
    5. Returns a list of (weights, expected_idx, expected_d2, fused_idx,
    fused_d2) with A placed at index 0 and B at index 1.
    """
    f32 = np.float32
    out = []
    while len(out) < count:
        a0 = f32(rng.uniform(0.5, 2.0))
        p0 = f32(a0 * a0)
        a1 = f32(np.sqrt(f32(target) - p0))
        for step in rng.permutation(np.arange(-8, 9)):
            c = f32(np.nextafter(a1, f32(np.inf if step > 0 else -np.inf)))
            for _ in range(abs(int(step)) - 1):
                c = f32(np.nextafter(c, f32(np.inf if step > 0 else -np.inf)))
            if step == 0:
                c = a1
            plain = f32(p0 + f32(c * c))
            exact = float(p0) + float(c) ** 2
            assert Fraction(exact) == Fraction(float(p0)) + Fraction(float(c)) ** 2
            fused = f32(exact)
            if plain == f32(target) and fused != plain:
                w = np.zeros((2, features), np.float32)
                w[0, :2] = a0, c
                w[1, :2] = 1, 2
                fused_idx = 0 if fused < plain else 1
                out.append((w, 0, plain, fused_idx, min(fused, plain)))
                break
    return out


U32 = 2.0 ** -24  # unit roundoff of float32


def certified_rows(weights, data, chunk=4096):
    """Rows whose float32 argmin provably equals the exact float64 argmin.

    Per row and node, with float64 weights w, input x and d = w - x, the
    float32 kernel computes d_hat with |d_hat - d| <= e = 1.01 u (|w| + |d|
    (+ |x| when x is float64 and gets rounded)), then squares and sums F terms
    in float32 with relative error at most gamma_{F+1} = (F+1)u / (1-(F+1)u).
    So |D32 - D| <= B = gamma (D + S) + S with S = sum(2|d|e + e^2). A row is
    certified when the float64 best node j beats every other node k by
    D_k - B_k > D_j + B_j, or when k's weights equal j's exactly (then the
    float32 distances are equal too and both searches take the lower index).

    The bound assumes normal float32 arithmetic: no overflow and no subnormal
    intermediate. Metal flushes float32 subnormals to zero, so a row whose
    distances fall below about 1e-38 can be certified here and still differ
    on the GPU. No real-data input of this package comes near that range.

    Returns (certified mask, float64 argmin, min over rows of the slack
    (D_k - B_k) - (D_j + B_j) relative to D_k, ignoring exact duplicates).
    """
    w = np.asarray(weights, np.float64)
    x_in = np.asarray(data)
    rounded_x = x_in.dtype == np.float64
    x = x_in.astype(np.float64)
    F = w.shape[1]
    gamma = (F + 1) * U32 / (1 - (F + 1) * U32)
    ok = np.empty(len(x), bool)
    arg = np.empty(len(x), np.int64)
    worst = np.inf
    for s in range(0, len(x), chunk):
        xc = x[s:s + chunk]
        d = w[None] - xc[:, None]
        e = 1.01 * U32 * (np.abs(w)[None] + np.abs(d) + (np.abs(xc)[:, None] if rounded_x else 0))
        D = (d * d).sum(-1)
        S = (2 * np.abs(d) * e + e * e).sum(-1)
        B = gamma * (D + S) + S
        j = D.argmin(axis=1)
        r = np.arange(len(xc))
        hi = D[r, j] + B[r, j]
        slack = (D - B) - hi[:, None]
        same = (w[None] == w[j][:, None]).all(-1)  # includes k == j
        slack[same] = np.inf
        ok[s:s + chunk] = (slack > 0).all(axis=1)
        arg[s:s + chunk] = j
        worst = min(worst, float((slack / np.where(same, 1.0, D)).min()))
    return ok, arg, worst
