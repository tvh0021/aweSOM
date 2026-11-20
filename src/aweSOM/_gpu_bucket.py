"""Private bucketed exact BMU search inside the ``cuda_screen`` backend.

The dense ``cuda_screen`` kernel screens every node for every row. This path
screens each row against a short candidate list instead, and proves that the
nodes it skipped cannot win. Rows it cannot prove go to an exact fallback.
Every row gets the dense kernel's node index and float64 distance, bit for
bit (benchmark ``nvidia-10x-bucket-bench-v2``, Tigris job 1239799: about
2 ns/row against 7.5 to 12.6 for the dense kernel on 100x100 lattices).

One chunk of rows runs as seven kernels on the caller's stream:

1. ``seed``: float32 argmin over the P group centroids. It only picks the
   bucket and is never trusted.
2. ``bucket_hist``, ``bucket_scan``, ``bucket_scatter``: a counting sort that
   builds ``perm`` (position -> original row). Rows do not move; every result
   is written to its original row, so the caller sees rows in input order.
3. ``bpass``: one block per tile of one bucket. Every row of bucket p screens
   the same candidate list (the members of the first K_p groups in p's order)
   with the shipped float32 screen and top-two certificate. A row whose
   winner j1 passes the certificate gets the reference distance ``a`` of j1.
   It is certified only if every group left out of the list is provably
   farther than ``a`` (the exclusion test in :func:`_pass2_rows`). Certified
   rows get ``(j1, a)``; every other row is appended to a device list.
4. ``fb_screen``: the shipped float32 screen over all nodes, one warp per
   listed row, then the shipped certificate. Rows it cannot certify go to a
   second list.
5. ``fb_exact``: the float64 reference distance to every node, one warp per
   listed row, ties to the lowest index, and the dense default ``(0, inf)``
   when no distance is finite.

Why the results equal the dense kernel's: a certified row's j1 is the strict
float64-reference minimum over its candidates (the shipped proof), and the
exclusion test shows every skipped node has a larger computed distance than
``a`` (so neither a smaller distance nor an equal one at a lower index is
lost). ``a`` is computed with the shipped unfused float64 code. Fallback rows
take either the shipped screen and certificate over all nodes, or the full
reference scan. The candidate lists, the pilot rows that size them and the
bucket order inside ``perm`` change speed only, never results.

Per-lattice setup (:class:`BucketEngine`, one per searcher, so once per
epoch in training and once per label call):

* feature-space groups of at most ``LEAF`` nodes with certified radii, built
  from the weights alone;
* a pilot: up to ``PILOT_ROWS`` rows taken evenly from the first chunk. The
  dense kernel gives their exact distances and the seed kernel their
  buckets, on the device. Each bucket's list length K_p is the one that
  minimises a cost model on its pilot rows. A bucket with fewer than
  ``RARE`` pilot rows screens the whole lattice (K = P), which costs about a
  dense row but never falls back for lack of candidates;
* packed float32 tables, uploaded once.

If a chunk sends more than ``RETUNE_FRACTION`` of its rows to the fallback or
to whole-lattice buckets (input whose distribution moves along the file, such
as a grid labelled in order), a sample of that chunk joins the pilot and the
lists are rebuilt before a later chunk (at most ``RETUNE_MAX`` times per
searcher, at least ``RETUNE_GAP`` chunks apart). Each chunk's counters reach
the host through an asynchronous copy that is read only after the caller's
own wait on that chunk, so there is no extra synchronisation per chunk.

The module switch ``aweSOM._gpu_screen._BUCKET`` selects this path; the dense
kernel stays as the control.
"""
import time
import weakref
from functools import lru_cache

import numpy as np
from numba import njit, prange

from ._gpu_screen import _fma32, _screen_parameters as screen_parameters
from .inference import FEATURE_ETA, _U32, _feature_patches

# Launch configuration measured best in the benchmark (b128R4, X32 exclusion, s32r4 fallback).
BLOCK = 128  # threads per bucketed-pass block
ROWS = 4  # rows per thread in the seed and bucketed passes
CT = 256  # candidate nodes (or excluded groups) per shared-memory tile
SEED_BLOCK = 128
HIST_BLOCK = 256
SCATTER_BLOCK = 256
SCATTER_ROWS = 8
FT = 256  # nodes per shared tile in the fallback kernels (a multiple of 32)
FB_BLOCK = 256  # threads per fallback block (8 warps)
FB_ROWS = 4  # listed rows per warp in the fallback kernels
FB_GRID = 1056  # fixed fallback grid (8 blocks per SM of a 132-SM H100); grid-stride, any GPU works
OPTIONS = ('--fmad=false', '--ftz=false', '--std=c++11')

# Setup and tuning.
LEAF = 64  # nodes per feature-space group
MAX_GROUPS = 1024  # the bucket scan runs in one block of 1024 threads
MAX_FEATURES = 10  # register arrays of ROWS x F floats per thread
PILOT_ROWS = 2**15
RARE = 8  # buckets with fewer pilot rows screen the whole lattice
COST_EXCLUDED = 1.4  # cost of one excluded group, in candidate nodes
COST_FALLBACK = 12000.0  # cost of one fallback row, in candidate nodes
RETUNE_FRACTION = 0.08
RETUNE_GAP = 4  # chunks
RETUNE_MAX = 16
RETUNE_MIN_ROWS = 1024  # smaller chunks never trigger a rebuild
SAMPLE_ROWS = 4096  # rows kept per chunk for a rebuild
RESERVOIR = 2**16  # most pilot rows kept (oldest dropped first)

ETA = FEATURE_ETA  # 2**-1000


def _feature_eps(F):
    return (F + 2) * 2.0**-44


def _pads(F):
    """float32 stride of a node (a multiple of 4, for float4 loads) and of an excluded centroid plus radius."""
    return -(-F // 4) * 4, -(-(F + 1) // 4) * 4


# ----------------------------------------------------------------------------- groups and tables (host)
def build_groups(w):
    """Feature-space groups of ``w`` (float64, finite): members, centroids, certified radii and, for
    every seed group p, the order in which p's candidate list takes the groups.

    ``order[p]`` starts with p, then sorts the other groups by ``|c_p - c_q| - R_q``.
    ``csize[p, K]`` is the number of nodes in the first K groups of that order.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        _, perm, offs, centT, rad, _ = _feature_patches(w, leaf_size=LEAF)
    cent = np.ascontiguousarray(centT.T)
    P = len(offs) - 1
    sizes = np.diff(offs)
    with np.errstate(invalid="ignore", over="ignore"):  # non-finite lattices: modelled, never searched
        cc = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))
        gap = cc - rad[None, :]
    np.fill_diagonal(gap, -np.inf)  # the seed group first; the others keep their stable order
    order = np.argsort(gap, axis=1, kind="stable")
    pos = np.empty_like(order)
    pos[np.arange(P)[:, None], order] = np.arange(P)[None, :]
    csize = np.concatenate([np.zeros((P, 1), np.int64), np.cumsum(sizes[order], axis=1)], axis=1)
    # _feature_patches sorts each group's members, so perm[offs[q]:offs[q + 1]] is ascending.
    return dict(P=P, cent=cent, rad=rad, perm=np.asarray(perm, np.int64), offs=np.asarray(offs, np.int64),
                sizes=sizes, order=order, pos=pos, csize=csize)


def screen_scalars(w, grp):
    """safe, wmax (as the dense kernel), cmax >= every |c_q|, G1 >= 1 + g as float32, R32up >= rad as float32."""
    safe, wmax = screen_parameters(w)
    cent, rad = grp["cent"], grp["rad"]
    safe = bool(safe and np.all(np.isfinite(cent)) and np.all(np.isfinite(rad)) and np.max(np.abs(cent)) <= 1e15)
    cmax = float(np.sqrt((cent * cent).sum(axis=1)).max() * (1.0 + 1e-12)) if safe else 0.0
    F = w.shape[1]
    g = (F + 2) * _U32 * 1.01
    G1 = np.float32(1.0 + g)
    if np.float64(G1) < 1.0 + g:
        G1 = np.nextafter(G1, np.float32(np.inf))
    r32 = rad.astype(np.float32)
    r32 = np.where(r32.astype(np.float64) < rad, np.nextafter(r32, np.float32(np.inf)), r32).astype(np.float32)
    return dict(safe=int(safe), wmax=float(wmax), cmax=cmax, G1=np.float32(G1), R32up=r32)


@njit(parallel=True, cache=True)
def _geometric_rank(x, cent, rad, best, seed, pos):
    """Smallest K whose first K groups (in bucket seed's order) hold every group whose ball can reach
    the row's best distance (with a 1e-6 relative + 1e-9 margin for the certificate's slack)."""
    n, F = x.shape
    P = cent.shape[0]
    out = np.zeros(n, np.int64)
    for i in prange(n):
        s = seed[i]
        r = 0
        lim = best[i] * (1.0 + 1e-6) + 1e-9
        for q in range(P):
            d2 = 0.0
            for f in range(F):
                d = cent[q, f] - np.float64(x[i, f])
                d2 += d * d
            if np.sqrt(d2) - rad[q] <= lim:
                k = pos[s, q] + 1
                if k > r:
                    r = k
        out[i] = r if np.isfinite(best[i]) else P
    return out


def pilot_rank(grp, x, d2, seed):
    """Per pilot row, the list length its bucket needs to cover the row (``P`` for no finite best)."""
    return _geometric_rank(np.ascontiguousarray(x, np.float64), grp["cent"], grp["rad"],
                           np.sqrt(np.asarray(d2, np.float64)), np.asarray(seed, np.int64), grp["pos"])


def tune_k(grp, seed, rank):
    """Per bucket: the K minimising ``members(K) + COST_EXCLUDED (P - K) + COST_FALLBACK fallback(K)``
    on its pilot rows, and the mask of rare buckets (fewer than ``RARE`` pilot rows), which get K = P."""
    P, csize = grp["P"], grp["csize"]
    seed = np.asarray(seed, np.int64)
    rank = np.minimum(np.asarray(rank, np.int64), P)
    hist = np.bincount(seed * (P + 1) + rank, minlength=P * (P + 1)).reshape(P, P + 1).astype(np.float64)
    cnt = hist.sum(axis=1)
    covered = np.cumsum(hist, axis=1)[:, 1:]  # covered[p, K - 1]: pilot rows of p with rank <= K
    Ks = np.arange(1, P + 1)
    miss = 1.0 - covered / np.maximum(cnt, 1.0)[:, None]
    cost = csize[:, 1:] + COST_EXCLUDED * (P - Ks)[None, :] + COST_FALLBACK * miss
    K = Ks[np.argmin(cost, axis=1)]
    rare = cnt < RARE
    K[rare] = P
    return K.astype(np.int64), rare


def pack(w, grp, sc, K, rare):
    """Host tables for the kernels.

    ``cw``/``cid``: float32 candidate weights (rounded to nearest, as the dense kernel converts them) and
    their node ids, bucket p's list at ``[cand_lo[p], cand_hi[p])``. Rare buckets share one segment of
    every node in ascending order and exclude nothing. ``ex32``: excluded groups' float32 centroids with
    the radius rounded up in column F, bucket p's at ``[ex_lo[p], ex_hi[p])``.
    """
    P, F, nodes = grp["P"], w.shape[1], w.shape[0]
    pad, px = _pads(F)
    order, sizes, perm, offs = grp["order"], grp["sizes"], grp["perm"], grp["offs"]
    ranks = np.arange(P)
    keep = ~rare[:, None]
    take = (ranks[None, :] < K[:, None]) & keep
    groups = order[take]  # row-major: bucket by bucket, each in its list order
    lens = sizes[groups]
    first = np.concatenate(([0], np.cumsum(lens)[:-1])) if len(lens) else np.zeros(0, np.int64)
    cid = perm[np.repeat(offs[groups] - first, lens) + np.arange(int(lens.sum()))]
    per_bucket = np.where(rare, 0, (take * sizes[order]).sum(axis=1))
    cand_lo = np.concatenate(([0], np.cumsum(per_bucket)[:-1]))
    cand_hi = cand_lo + per_bucket
    if rare.any():
        cand_lo = np.where(rare, len(cid), cand_lo)
        cand_hi = np.where(rare, len(cid) + nodes, cand_hi)
        cid = np.concatenate([cid, np.arange(nodes)])
    exg = order[(ranks[None, :] >= K[:, None]) & keep]
    ex_n = np.where(rare, 0, P - K)
    ex_lo = np.concatenate(([0], np.cumsum(ex_n)[:-1]))
    ex_hi = ex_lo + ex_n
    cw = np.zeros((max(1, len(cid)), pad), np.float32)
    cw[:len(cid), :F] = w[cid].astype(np.float32)
    ex32 = np.zeros((max(1, len(exg)), px), np.float32)
    ex32[:len(exg), :F] = grp["cent"][exg].astype(np.float32)
    ex32[:len(exg), F] = sc["R32up"][exg]
    c32 = np.zeros((P, pad), np.float32)
    c32[:, :F] = grp["cent"].astype(np.float32)
    i32 = lambda a: np.ascontiguousarray(a, np.int32)  # noqa: E731
    return dict(P=P, F=F, c32=c32, cw=cw, cid=i32(np.resize(cid, max(1, len(cid)))),
                cand_lo=i32(cand_lo), cand_hi=i32(cand_hi), ex32=ex32, ex_lo=i32(ex_lo), ex_hi=i32(ex_hi),
                K=K, rare=rare, cand_len=per_bucket + rare * nodes)


# ----------------------------------------------------------------------------- bit-exact CPU models
@njit(cache=True)
def _ru32(v):
    """float64 -> float32 rounded upward (``__double2float_ru``)."""
    r = np.float32(v)
    if np.float64(r) < v:
        r = np.nextafter(r, np.float32(np.inf))
    return r


@njit(cache=True)
def _fadd_ru(a, b):
    """``__fadd_ru`` of float32 values: exact sum = s + err (TwoSum), then the float32 at or above it."""
    a64 = np.float64(a)
    b64 = np.float64(b)
    s = a64 + b64
    r = np.float32(s)
    if not np.isfinite(s):
        return r
    rr = np.float64(r)
    if rr < s:
        return np.nextafter(r, np.float32(np.inf))
    if rr > s:
        return r
    bb = s - a64
    err = (a64 - (s - bb)) + (b64 - bb)
    if err > 0.0:
        return np.nextafter(r, np.float32(np.inf))
    return r


@njit(cache=True)
def _fmul_ru(a, b):
    """``__fmul_ru`` of float32 values: the float64 product is exact."""
    p = np.float64(a) * np.float64(b)
    r = np.float32(p)
    if np.float64(r) < p:
        r = np.nextafter(r, np.float32(np.inf))
    return r


@njit(parallel=True, cache=True)
def _seed_rows(c32, x, out):
    """The seed kernel: float32 FMA argmin over centroids, strict < (lowest group on ties)."""
    n = x.shape[0]
    P, F = c32.shape[0], x.shape[1]
    inf32 = np.float32(np.inf)
    for q in prange(n):
        x32 = np.empty(F, np.float32)
        for f in range(F):
            x32[f] = np.float32(np.float64(x[q, f]))
        s1 = inf32
        js = 0
        for g in range(P):
            acc = np.float32(0.0)
            for f in range(F):
                d = c32[g, f] - x32[f]
                acc = _fma32(d, d, acc)
            if acc < s1:
                js = g
            s1 = np.fmin(s1, acc)
        out[q] = js


@njit(parallel=True, cache=True)
def _pass2_rows(w, cw, cid, cand_lo, cand_hi, ex32, ex_lo, ex_hi, x, rows, bucket, wmax, cmax, g1, safe,
                code, j1o, ao):
    """The bucketed pass for ``rows`` of ``x`` in buckets ``bucket``: code 0 certified (j1o, ao = the
    reference's bits), 1 top-two test failed (or unsafe), 2 exclusion failed.

    Exclusion: every member k of an excluded group q has computed squared distance ``a_k > a`` if a
    lower bound on the true centroid distance exceeds ``R_q + s'``, ``s' = sqrt((a + eta) / (1 - eps))``,
    ``eps = (F + 2) 2**-44``, ``eta = 2**-1000``: then ``D_k >= (|x - c_q| - R_q)**2 > (a + eta)/(1 - eps)``
    and ``a_k >= D_k (1 - eps) - eta > a``. The float32 centroid screen ``S`` satisfies
    ``S <= |x32 - c32|**2 (1 + g)``, and ``|x - c| >= |x32 - c32| - u|x| - u|c|`` is covered by
    ``ec = 2.02 u (cmax + |x|) + 1e-18``. With every rounding directed upward, ``t = up(s') + ec`` as
    float32, ``rr = fadd_ru(R32up_q, t)`` and ``thr = fmul_ru(fmul_ru(rr, rr), G1)``: ``S > thr`` (finite)
    gives ``|x32 - c32| > R_q + t``, hence ``|x - c_q| > R_q + s'``. A float64 step rounded upward is
    round-to-nearest followed by ``nextafter(+inf)``, so kernel and model decide identically.
    """
    n = rows.shape[0]
    F = w.shape[1]
    u = _U32
    g = (F + 2) * u * 1.01
    eps = (F + 2) * 2.0**-44
    one_m_eps = 1.0 - eps
    eta = 2.0**-1000
    inf32 = np.float32(np.inf)
    for t in prange(n):
        q = rows[t]
        p = bucket[t]
        x32 = np.empty(F, np.float32)
        rw = np.empty(F)
        for f in range(F):
            rw[f] = np.float64(x[q, f])
            x32[f] = np.float32(rw[f])
        s1 = inf32
        s2 = inf32
        jj = 0
        if safe:
            for k in range(cand_lo[p], cand_hi[p]):
                acc = np.float32(0.0)
                for f in range(F):
                    d = cw[k, f] - x32[f]
                    acc = _fma32(d, d, acc)
                s2 = np.fmin(s2, np.fmax(s1, acc))
                if acc < s1:
                    jj = cid[k]
                s1 = np.fmin(s1, acc)
        c = 1
        a = np.inf
        if safe:
            xn2 = 0.0
            for f in range(F):
                xn2 += rw[f] * rw[f]
            e = 2.0 * u * (wmax + np.sqrt(xn2)) * 1.01 + 1e-18
            lo2 = np.sqrt(np.float64(s2) / (1.0 + g)) - e
            hi1 = (np.sqrt(np.float64(s1) / (1.0 - g)) + e) * (1.0 + 1e-12)
            if lo2 > hi1 and s2 < inf32:
                acc64 = 0.0
                for f in range(F):
                    dd = w[jj, f] - rw[f]
                    acc64 += dd * dd
                a = acc64
                ok = True
                qq = np.nextafter(a + eta, np.inf)
                qq = np.nextafter(qq / one_m_eps, np.inf)
                sp = np.nextafter(np.sqrt(qq), np.inf)
                ec = 2.0 * u * (cmax + np.sqrt(xn2)) * 1.01 + 1e-18
                t32 = _ru32(np.nextafter(sp + ec, np.inf))
                for k in range(ex_lo[p], ex_hi[p]):
                    S = np.float32(0.0)
                    for f in range(F):
                        d = ex32[k, f] - x32[f]
                        S = _fma32(d, d, S)
                    rr = _fadd_ru(ex32[k, F], t32)
                    thr = _fmul_ru(_fmul_ru(rr, rr), g1)
                    if not (S > thr and S < inf32):
                        ok = False
                        break
                c = 0 if ok else 2
        code[t] = c
        j1o[t] = jj
        ao[t] = a


LANES = 32


@njit(parallel=True, cache=True)
def _fb_exact_rows(w, x, rows, idx_out, d2_out):
    """The fb_exact kernel: lane l scans the nodes j % 32 == l in ascending order with strict ``<`` from
    (inf, sentinel); the butterfly keeps the smaller (value, index) pair, so the result is the lowest index
    of the minimum, as the dense kernel's ascending strict-``<`` scan. No finite value: (0, +inf)."""
    n = rows.shape[0]
    nodes, F = w.shape
    for t in prange(n):
        q = rows[t]
        rw = np.empty(F)
        for f in range(F):
            rw[f] = np.float64(x[q, f])
        best = np.full(LANES, np.inf)
        bi = np.full(LANES, 2147483647, np.int64)
        for j in range(nodes):
            ln = j % LANES
            acc = 0.0
            for f in range(F):
                d = w[j, f] - rw[f]
                acc = acc + d * d
            if acc < best[ln]:
                best[ln] = acc
                bi[ln] = j
        off = 16
        while off > 0:
            nb = best.copy()
            ni = bi.copy()
            for ln in range(LANES):
                o = ln ^ off
                if best[o] < best[ln] or (best[o] == best[ln] and bi[o] < bi[ln]):
                    nb[ln] = best[o]
                    ni[ln] = bi[o]
            best = nb
            bi = ni
            off //= 2
        idx_out[t] = bi[0] if best[0] < np.inf else 0
        d2_out[t] = best[0]


@njit(parallel=True, cache=True)
def _fb_screen_rows(w, x, rows, wmax, safe, cert_out, j1_out, a_out):
    """The fb_screen kernel: the shipped float32 FMA screen over every node per lane, a butterfly merge
    (the other lane wins on a smaller s1 or an equal s1 with a lower j1; s2 = min of the loser's s1 and
    the winner's s2), then the shipped certificate. Certified rows (cert_out 1) get (j1, reference
    distance of j1); the others go on to fb_exact."""
    n = rows.shape[0]
    nodes, F = w.shape
    u = _U32
    g = (F + 2) * u * 1.01
    inf32 = np.float32(np.inf)
    for t in prange(n):
        q = rows[t]
        rw = np.empty(F)
        x32 = np.empty(F, np.float32)
        for f in range(F):
            rw[f] = np.float64(x[q, f])
            x32[f] = np.float32(rw[f])
        s1 = np.full(LANES, inf32)
        s2 = np.full(LANES, inf32)
        j1 = np.zeros(LANES, np.int64)
        if safe:
            for j in range(nodes):
                ln = j % LANES
                acc = np.float32(0.0)
                for f in range(F):
                    d = np.float32(w[j, f]) - x32[f]
                    acc = _fma32(d, d, acc)
                s2[ln] = np.fmin(s2[ln], np.fmax(s1[ln], acc))
                if acc < s1[ln]:
                    j1[ln] = j
                s1[ln] = np.fmin(s1[ln], acc)
        off = 16
        while off > 0:
            n1 = s1.copy()
            n2 = s2.copy()
            nj = j1.copy()
            for ln in range(LANES):
                o = ln ^ off
                if s1[o] < s1[ln] or (s1[o] == s1[ln] and j1[o] < j1[ln]):
                    n1[ln] = s1[o]
                    nj[ln] = j1[o]
                    n2[ln] = np.fmin(s2[o], s1[ln])
                else:
                    n2[ln] = np.fmin(s2[ln], s1[o])
            s1 = n1
            s2 = n2
            j1 = nj
            off //= 2
        c = 0
        a = np.inf
        if safe:
            xn2 = 0.0
            for f in range(F):
                xn2 += rw[f] * rw[f]
            e = 2.0 * u * (wmax + np.sqrt(xn2)) * 1.01 + 1e-18
            lo2 = np.sqrt(np.float64(s2[0]) / (1.0 + g)) - e
            hi1 = (np.sqrt(np.float64(s1[0]) / (1.0 - g)) + e) * (1.0 + 1e-12)
            if lo2 > hi1 and s2[0] < inf32:
                acc64 = 0.0
                for f in range(F):
                    dd = w[j1[0], f] - rw[f]
                    acc64 += dd * dd
                a = acc64
                c = 1
        cert_out[t] = c
        j1_out[t] = j1[0]
        a_out[t] = a


def build_tables(w, pilot_x, pilot_d2=None, pilot_seed=None, grp=None):
    """Groups, scalars, K from pilot rows and packed tables, all on the host (tests and the reference).

    ``pilot_d2`` defaults to the reference distances, ``pilot_seed`` to the seed model's buckets."""
    from .inference import best_match_nodes

    w = np.ascontiguousarray(w, np.float64)
    grp = build_groups(w) if grp is None else grp
    sc = screen_scalars(w, grp)
    px = np.ascontiguousarray(pilot_x, np.float64)
    if pilot_d2 is None:
        pilot_d2 = best_match_nodes(w, px)[1]
    if pilot_seed is None:
        pilot_seed = np.empty(len(px), np.int64)
        _seed_rows(grp["cent"].astype(np.float32), px, pilot_seed)
    K, rare = tune_k(grp, pilot_seed, pilot_rank(grp, px, pilot_d2, pilot_seed))
    T = pack(w, grp, sc, K, rare)
    T.update(sc=sc, grp=grp)
    return T


def _bucket_reference(w, x, T):
    """CPU model of the whole device path with tables ``T``: ``(idx, d2, info)``.

    ``info``: per-row bucket and pass code, and the rows fb_screen left to fb_exact."""
    w = np.ascontiguousarray(w, np.float64)
    x = np.ascontiguousarray(x)
    n, F = x.shape
    sc = T["sc"]
    seed = np.empty(n, np.int64)
    _seed_rows(T["c32"][:, :F], x, seed)
    code = np.empty(n, np.int8)
    j1 = np.empty(n, np.int64)
    a = np.empty(n, np.float64)
    _pass2_rows(w, T["cw"], T["cid"], T["cand_lo"], T["cand_hi"], T["ex32"], T["ex_lo"], T["ex_hi"], x,
                np.arange(n, dtype=np.int64), seed, sc["wmax"], sc["cmax"], sc["G1"], sc["safe"], code, j1, a)
    idx = j1.astype(np.int32)
    d2 = a.copy()
    fb = np.nonzero(code != 0)[0].astype(np.int64)
    left = np.zeros(0, np.int64)
    if len(fb):
        safe, wmax = screen_parameters(w)
        cert = np.empty(len(fb), np.int8)
        fj = np.empty(len(fb), np.int64)
        fa = np.empty(len(fb), np.float64)
        _fb_screen_rows(w, x, fb, wmax, int(safe), cert, fj, fa)
        idx[fb] = fj
        d2[fb] = fa
        left = fb[cert == 0]
        if len(left):
            ei = np.empty(len(left), np.int64)
            ed = np.empty(len(left), np.float64)
            _fb_exact_rows(w, x, left, ei, ed)
            idx[left] = ei
            d2[left] = ed
    return idx, d2, dict(seed=seed, code=code, screen_left=left)


# ----------------------------------------------------------------------------- CUDA source
def _source(F):
    pad, px = _pads(F)
    head = ('#define AWESOM_F %d\n#define AWESOM_R %d\n#define AWESOM_PAD %d\n#define AWESOM_PX %d\n'
            '#define SR %d\n#define FT %d\n#define FB_R %d\n' % (F, ROWS, pad, px, SCATTER_ROWS, FT, FB_ROWS))
    return head + r'''
    __device__ __forceinline__ double inf_d() { return __longlong_as_double(0x7ff0000000000000LL); }

    // Reference squared distance, as the dense kernel: acc = 0; acc += d * d, ascending features.
    __device__ double exact(const double* w, const double* row) {
        double acc = 0.0;
        #pragma unroll
        for (int f = 0; f < AWESOM_F; ++f) {
            double d = __dsub_rn(w[f], row[f]);
            acc = __dadd_rn(acc, __dmul_rn(d, d));
        }
        return acc;
    }
    // The same arithmetic, inlined (fallback kernels).
    __device__ __forceinline__ double exact_inl(const double* w, const double* row) {
        double acc = 0.0;
        #pragma unroll
        for (int f = 0; f < AWESOM_F; ++f) {
            double d = __dsub_rn(w[f], row[f]);
            acc = __dadd_rn(acc, __dmul_rn(d, d));
        }
        return acc;
    }

    // ------------------------------------------------------------------ seed pass
    // float32 FMA argmin over the P group centroids; block 0 also zeroes zero[0:nzero].
    template <typename T>
    __device__ void seed_body(const float* c32, const T* x, int* seed, int rows, int P, int* zero, int nzero) {
        extern __shared__ float4 smem[];
        if (blockIdx.x == 0)
            for (int k = threadIdx.x; k < nzero; k += blockDim.x) zero[k] = 0;
        const float4* c4 = (const float4*)c32;
        for (int k = threadIdx.x; k < P * (AWESOM_PAD / 4); k += blockDim.x) smem[k] = c4[k];
        __syncthreads();
        const int first = blockIdx.x * blockDim.x * AWESOM_R + threadIdx.x;
        const float inff = (float)inf_d();
        float xr[AWESOM_R][AWESOM_F];
        float s1[AWESOM_R];
        int js[AWESOM_R];
        #pragma unroll
        for (int r = 0; r < AWESOM_R; ++r) {
            const int p = first + r * blockDim.x;
            #pragma unroll
            for (int f = 0; f < AWESOM_F; ++f)
                xr[r][f] = p < rows ? __double2float_rn((double)x[p * AWESOM_F + f]) : 0.0f;
            s1[r] = inff;
            js[r] = 0;
        }
        for (int g = 0; g < P; ++g) {
            float cv[AWESOM_PAD];
            #pragma unroll
            for (int q = 0; q < AWESOM_PAD / 4; ++q) {
                const float4 v = smem[g * (AWESOM_PAD / 4) + q];
                cv[4 * q] = v.x; cv[4 * q + 1] = v.y; cv[4 * q + 2] = v.z; cv[4 * q + 3] = v.w;
            }
            #pragma unroll
            for (int r = 0; r < AWESOM_R; ++r) {
                float acc = 0.0f;
                #pragma unroll
                for (int f = 0; f < AWESOM_F; ++f) {
                    const float d = __fsub_rn(cv[f], xr[r][f]);
                    acc = __fmaf_rn(d, d, acc);
                }
                js[r] = acc < s1[r] ? g : js[r];
                s1[r] = fminf(s1[r], acc);
            }
        }
        #pragma unroll
        for (int r = 0; r < AWESOM_R; ++r) {
            const int p = first + r * blockDim.x;
            if (p < rows) seed[p] = js[r];
        }
    }
    extern "C" __global__ void seed_f32(const float* c32, const float* x, int* seed, int rows, int P,
                                        int* zero, int nzero) { seed_body(c32, x, seed, rows, P, zero, nzero); }
    extern "C" __global__ void seed_f64(const float* c32, const double* x, int* seed, int rows, int P,
                                        int* zero, int nzero) { seed_body(c32, x, seed, rows, P, zero, nzero); }

    // ------------------------------------------------------------------ histogram, scan, scatter
    extern "C" __global__ void bucket_hist(const int* seed, int rows, int P, int* counts) {
        extern __shared__ int sh[];
        for (int k = threadIdx.x; k < P; k += blockDim.x) sh[k] = 0;
        __syncthreads();
        const int lane = threadIdx.x & 31;
        for (int base = blockIdx.x * blockDim.x; base < rows; base += gridDim.x * blockDim.x) {
            const int i = base + threadIdx.x;
            const int b = i < rows ? seed[i] : -1;
            const unsigned peers = __match_any_sync(0xffffffffu, b);
            if (b >= 0 && lane == __ffs(peers) - 1) atomicAdd(&sh[b], __popc(peers));
        }
        __syncthreads();
        for (int k = threadIdx.x; k < P; k += blockDim.x)
            if (sh[k]) atomicAdd(&counts[k], sh[k]);
    }

    // One block of 1024 threads: bucket offsets, scatter cursors and tile offsets (tiles of rb rows).
    extern "C" __global__ void bucket_scan(const int* counts, int P, int rb, int* bucket_off, int* cursor,
                                           int* tile_off) {
        __shared__ int sc[1024];
        __shared__ int st[1024];
        const int t = threadIdx.x;
        const int c = t < P ? counts[t] : 0;
        sc[t] = c;
        st[t] = (c + rb - 1) / rb;
        __syncthreads();
        for (int off = 1; off < 1024; off <<= 1) {
            const int a = t >= off ? sc[t - off] : 0;
            const int b = t >= off ? st[t - off] : 0;
            __syncthreads();
            sc[t] += a;
            st[t] += b;
            __syncthreads();
        }
        if (t < P) {
            bucket_off[t + 1] = sc[t];
            tile_off[t + 1] = st[t];
            cursor[t] = sc[t] - c;
        }
        if (t == 0) { bucket_off[0] = 0; tile_off[0] = 0; }
    }

    // perm[packed position] = original row. The order inside a bucket is arbitrary; results do not depend on it.
    extern "C" __global__ void bucket_scatter(const int* seed, int rows, int P, int* cursor, int* perm) {
        extern __shared__ int sh[];
        int* cnt = sh;
        int* basep = sh + P;
        for (int k = threadIdx.x; k < P; k += blockDim.x) cnt[k] = 0;
        __syncthreads();
        const int lane = threadIdx.x & 31;
        const unsigned lt = (1u << lane) - 1u;
        int loc[SR], bk[SR];
        #pragma unroll
        for (int r = 0; r < SR; ++r) {
            const int i = blockIdx.x * blockDim.x * SR + r * blockDim.x + threadIdx.x;
            const int b = i < rows ? seed[i] : -1;
            const unsigned peers = __match_any_sync(0xffffffffu, b);
            const int leader = __ffs(peers) - 1;
            int basev = 0;
            if (b >= 0 && lane == leader) basev = atomicAdd(&cnt[b], __popc(peers));
            basev = __shfl_sync(0xffffffffu, basev, leader);
            bk[r] = b;
            loc[r] = basev + __popc(peers & lt);
        }
        __syncthreads();
        for (int k = threadIdx.x; k < P; k += blockDim.x)
            if (cnt[k]) basep[k] = atomicAdd(&cursor[k], cnt[k]);
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < SR; ++r) {
            const int i = blockIdx.x * blockDim.x * SR + r * blockDim.x + threadIdx.x;
            if (bk[r] >= 0) perm[basep[bk[r]] + loc[r]] = i;
        }
    }

    // ------------------------------------------------------------------ bucketed pass
    // One block per tile of one bucket; rows are read through perm and results go to the original rows.
    template <typename T>
    __device__ void bpass(const double* w, const T* x, const int* perm, const int* bucket_off,
                          const int* tile_off, const float* cw, const int* cid, const int* cand_lo,
                          const int* cand_hi, const float* ex32, const int* ex_lo, const int* ex_hi, int* idx,
                          double* d2, int* fb_rows, int* counters, int P, int ct, double wmax, double cmax,
                          float g1, int safe, double eta, double one_m_eps) {
        extern __shared__ float4 smem[];
        __shared__ int s_p, s_cnt, s_cnt1, s_base;
        const double inf = inf_d();
        const float inff = (float)inf;
        if (threadIdx.x == 0) {
            const int b = blockIdx.x;
            const int total = tile_off[P];
            int p = -1;
            if (b < total) {  // tile_off[lo] <= b < tile_off[hi]
                int lo = 0, hi = P;
                while (hi - lo > 1) {
                    const int mid = (lo + hi) >> 1;
                    if (tile_off[mid] <= b) lo = mid; else hi = mid;
                }
                p = lo;
            }
            s_p = p; s_cnt = 0; s_cnt1 = 0;
        }
        __syncthreads();
        const int p = s_p;
        if (p < 0) return;  // the whole block: past the last tile
        const int start = bucket_off[p] + (blockIdx.x - tile_off[p]) * blockDim.x * AWESOM_R;
        const int end = bucket_off[p + 1];
        int src[AWESOM_R];
        float xr[AWESOM_R][AWESOM_F];
        float s1[AWESOM_R], s2[AWESOM_R];
        int j1[AWESOM_R];
        #pragma unroll
        for (int r = 0; r < AWESOM_R; ++r) {
            const int pos = start + threadIdx.x + r * blockDim.x;
            src[r] = pos < end ? perm[pos] : -1;
            #pragma unroll
            for (int f = 0; f < AWESOM_F; ++f)
                xr[r][f] = src[r] >= 0 ? __double2float_rn((double)x[src[r] * AWESOM_F + f]) : 0.0f;
            s1[r] = inff; s2[r] = inff; j1[r] = 0;
        }
        // Phase 1: the shipped float32 screen over the bucket's candidate list.
        if (safe) {
            const int c0 = cand_lo[p], c1 = cand_hi[p];
            int* sid = (int*)(smem + ct * (AWESOM_PAD / 4));
            const float4* cw4 = (const float4*)cw;
            for (int base = c0; base < c1; base += ct) {
                const int count = min(ct, c1 - base);
                for (int k = threadIdx.x; k < count * (AWESOM_PAD / 4); k += blockDim.x)
                    smem[k] = cw4[base * (AWESOM_PAD / 4) + k];
                for (int k = threadIdx.x; k < count; k += blockDim.x) sid[k] = cid[base + k];
                __syncthreads();
                for (int j = 0; j < count; ++j) {
                    float wv[AWESOM_PAD];
                    #pragma unroll
                    for (int q = 0; q < AWESOM_PAD / 4; ++q) {
                        const float4 v = smem[j * (AWESOM_PAD / 4) + q];
                        wv[4 * q] = v.x; wv[4 * q + 1] = v.y; wv[4 * q + 2] = v.z; wv[4 * q + 3] = v.w;
                    }
                    const int node = sid[j];
                    #pragma unroll
                    for (int r = 0; r < AWESOM_R; ++r) {
                        float acc = 0.0f;
                        #pragma unroll
                        for (int f = 0; f < AWESOM_F; ++f) {
                            const float d = __fsub_rn(wv[f], xr[r][f]);
                            acc = __fmaf_rn(d, d, acc);
                        }
                        s2[r] = fminf(s2[r], fmaxf(s1[r], acc));
                        j1[r] = acc < s1[r] ? node : j1[r];
                        s1[r] = fminf(s1[r], acc);
                    }
                }
                __syncthreads();
            }
        }
        // Phase 2: the shipped top-two certificate; reference distance a of j1; exclusion slack t32.
        int st[AWESOM_R];  // 0 pending exclusion, 1 top-two failed (or unsafe), 3 no row
        double a[AWESOM_R];
        float t32[AWESOM_R];
        bool ok[AWESOM_R];
        #pragma unroll
        for (int r = 0; r < AWESOM_R; ++r) {
            st[r] = 3; a[r] = inf; t32[r] = 0.0f;
            if (src[r] >= 0) {
                st[r] = 1;
                if (safe) {
                    double row[AWESOM_F];
                    #pragma unroll
                    for (int f = 0; f < AWESOM_F; ++f) row[f] = (double)x[src[r] * AWESOM_F + f];
                    double xn2 = 0.0;
                    #pragma unroll
                    for (int f = 0; f < AWESOM_F; ++f) xn2 = __dadd_rn(xn2, __dmul_rn(row[f], row[f]));
                    const double u = 5.9604644775390625e-8;
                    double g = (AWESOM_F + 2) * u * 1.01;
                    double e = 2.0 * u * (wmax + __dsqrt_rn(xn2)) * 1.01 + 1e-18;
                    double lo2 = __dsqrt_rn(__ddiv_rn((double)s2[r], 1.0 + g)) - e;
                    double hi1 = (__dsqrt_rn(__ddiv_rn((double)s1[r], 1.0 - g)) + e) * (1.0 + 1e-12);
                    if (lo2 > hi1 && s2[r] < inff) {
                        a[r] = exact(w + j1[r] * AWESOM_F, row);
                        st[r] = 0;
                        double qq = nextafter(__dadd_rn(a[r], eta), inf);
                        qq = nextafter(__ddiv_rn(qq, one_m_eps), inf);
                        const double sp = nextafter(__dsqrt_rn(qq), inf);
                        const double ec = 2.0 * u * (cmax + __dsqrt_rn(xn2)) * 1.01 + 1e-18;
                        t32[r] = __double2float_ru(nextafter(__dadd_rn(sp, ec), inf));
                    }
                }
            }
            ok[r] = st[r] == 0;
        }
        // Phase 3: every excluded group must be provably farther than a (see _pass2_rows).
        const int e0 = ex_lo[p], e1 = ex_hi[p];
        for (int base = e0; base < e1; base += ct) {
            int any = 0;
            #pragma unroll
            for (int r = 0; r < AWESOM_R; ++r) any |= ok[r] ? 1 : 0;
            if (!__syncthreads_or(any)) break;
            const int count = min(ct, e1 - base);
            const float4* e4 = (const float4*)ex32;
            for (int k = threadIdx.x; k < count * (AWESOM_PX / 4); k += blockDim.x)
                smem[k] = e4[base * (AWESOM_PX / 4) + k];
            __syncthreads();
            for (int j = 0; j < count; ++j) {
                float ev[AWESOM_PX];
                #pragma unroll
                for (int q = 0; q < AWESOM_PX / 4; ++q) {
                    const float4 v = smem[j * (AWESOM_PX / 4) + q];
                    ev[4 * q] = v.x; ev[4 * q + 1] = v.y; ev[4 * q + 2] = v.z; ev[4 * q + 3] = v.w;
                }
                #pragma unroll
                for (int r = 0; r < AWESOM_R; ++r) {
                    if (ok[r]) {
                        float S = 0.0f;
                        #pragma unroll
                        for (int f = 0; f < AWESOM_F; ++f) {
                            const float d = __fsub_rn(ev[f], xr[r][f]);
                            S = __fmaf_rn(d, d, S);
                        }
                        const float rr = __fadd_ru(ev[AWESOM_F], t32[r]);
                        const float thr = __fmul_ru(__fmul_ru(rr, rr), g1);
                        ok[r] = S > thr && S < inff;
                    }
                }
            }
            __syncthreads();
        }
        // Phase 4: certified rows get (j1, a); the rest are appended to the fallback list.
        int lpos[AWESOM_R];
        #pragma unroll
        for (int r = 0; r < AWESOM_R; ++r) {
            lpos[r] = -1;
            if (st[r] != 3) {
                if (ok[r]) {
                    idx[src[r]] = j1[r];
                    d2[src[r]] = a[r];
                } else {
                    lpos[r] = atomicAdd(&s_cnt, 1);
                    if (st[r] == 1) atomicAdd(&s_cnt1, 1);
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            s_base = s_cnt ? atomicAdd(&counters[0], s_cnt) : 0;
            if (s_cnt1) atomicAdd(&counters[1], s_cnt1);
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < AWESOM_R; ++r)
            if (lpos[r] >= 0) fb_rows[s_base + lpos[r]] = src[r];
    }

    #define BPASS_SIG(T) (const double* w, const T* x, const int* perm, const int* bucket_off, \
        const int* tile_off, const float* cw, const int* cid, const int* cand_lo, const int* cand_hi, \
        const float* ex32, const int* ex_lo, const int* ex_hi, int* idx, double* d2, int* fb_rows, \
        int* counters, int P, int ct, double wmax, double cmax, float g1, int safe, double eta, \
        double one_m_eps)
    #define BPASS_CALL(T) bpass<T>(w, x, perm, bucket_off, tile_off, cw, cid, cand_lo, cand_hi, ex32, \
        ex_lo, ex_hi, idx, d2, fb_rows, counters, P, ct, wmax, cmax, g1, safe, eta, one_m_eps)
    extern "C" __global__ void bpass_f32 BPASS_SIG(float) { BPASS_CALL(float); }
    extern "C" __global__ void bpass_f64 BPASS_SIG(double) { BPASS_CALL(double); }

    // ------------------------------------------------------------------ fallback
    // fb_exact: float64 reference distance of each listed row to every node, one warp per FB_R rows.
    // Lane l scans the nodes j % 32 == l in ascending order with strict <, from (+inf, INT_MAX), so it
    // keeps its lowest index among equal values and never takes NaN or +inf. The butterfly keeps the
    // smaller (value, index) pair: the lowest index of the minimum, as the dense kernel's ascending
    // strict-< scan. No finite value anywhere: the dense default (0, +inf). The list length is read on
    // the device; a fixed grid strides over it.
    template <typename T>
    __device__ void fb_exact_body(const double* w, const T* x, const int* list, const int* count_p,
                                  int* idx, double* d2, int nodes) {
        extern __shared__ double fsd[];  // SoA tile: fsd[f * FT + j]
        const int count = *count_p;
        const int lane = threadIdx.x & 31;
        const int per_block = (blockDim.x >> 5) * FB_R;
        const int wrow = (threadIdx.x >> 5) * FB_R;
        const double inf = inf_d();
        // g0 < count is the same for every thread of the block, so the barriers below are uniform.
        for (int g0 = blockIdx.x * per_block; g0 < count; g0 += gridDim.x * per_block) {
            int src[FB_R];
            double row[FB_R][AWESOM_F];
            double best[FB_R];
            int bi[FB_R];
            #pragma unroll
            for (int r = 0; r < FB_R; ++r) {
                const int p = g0 + wrow + r;
                src[r] = p < count ? list[p] : -1;
                #pragma unroll
                for (int f = 0; f < AWESOM_F; ++f)
                    row[r][f] = src[r] >= 0 ? (double)x[src[r] * AWESOM_F + f] : 0.0;
                best[r] = inf;
                bi[r] = 0x7fffffff;
            }
            for (int base = 0; base < nodes; base += FT) {
                const int cnt = min(FT, nodes - base);
                __syncthreads();  // the previous tile is consumed
                for (int k = threadIdx.x; k < cnt * AWESOM_F; k += blockDim.x)
                    fsd[(k % AWESOM_F) * FT + k / AWESOM_F] = w[base * AWESOM_F + k];
                __syncthreads();
                for (int j = lane; j < cnt; j += 32) {
                    double wv[AWESOM_F];
                    #pragma unroll
                    for (int f = 0; f < AWESOM_F; ++f) wv[f] = fsd[f * FT + j];
                    #pragma unroll
                    for (int r = 0; r < FB_R; ++r) {
                        const double acc = exact_inl(wv, row[r]);
                        if (acc < best[r]) { best[r] = acc; bi[r] = base + j; }
                    }
                }
            }
            #pragma unroll
            for (int r = 0; r < FB_R; ++r) {
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    const double ob = __shfl_xor_sync(0xffffffffu, best[r], off);
                    const int oi = __shfl_xor_sync(0xffffffffu, bi[r], off);
                    if (ob < best[r] || (ob == best[r] && oi < bi[r])) { best[r] = ob; bi[r] = oi; }
                }
                if (lane == 0 && src[r] >= 0) {
                    idx[src[r]] = best[r] < inf ? bi[r] : 0;
                    d2[src[r]] = best[r];
                }
            }
        }
    }

    // fb_screen: the shipped float32 FMA screen over every node (per lane: the shipped branch-free top
    // two), a butterfly merge (the other lane wins on a smaller s1, or an equal s1 with a lower j1; s2
    // becomes the min of the loser's s1 and the winner's s2), then the shipped certificate. Certified
    // rows get (j1, reference distance of j1); the others are appended to list2 (count2) for fb_exact.
    template <typename T>
    __device__ void fb_screen_body(const double* w, const T* x, const int* list, const int* count_p,
                                   int* idx, double* d2, int* list2, int* count2, int nodes, double wmax,
                                   int safe) {
        extern __shared__ float fsf[];  // SoA tile: fsf[f * FT + j]
        const int count = *count_p;
        const int lane = threadIdx.x & 31;
        const int per_block = (blockDim.x >> 5) * FB_R;
        const int wrow = (threadIdx.x >> 5) * FB_R;
        const float inff = (float)inf_d();
        for (int g0 = blockIdx.x * per_block; g0 < count; g0 += gridDim.x * per_block) {
            int src[FB_R];
            float xr[FB_R][AWESOM_F];
            float s1[FB_R], s2[FB_R];
            int j1[FB_R];
            #pragma unroll
            for (int r = 0; r < FB_R; ++r) {
                const int p = g0 + wrow + r;
                src[r] = p < count ? list[p] : -1;
                #pragma unroll
                for (int f = 0; f < AWESOM_F; ++f)
                    xr[r][f] = src[r] >= 0 ? __double2float_rn((double)x[src[r] * AWESOM_F + f]) : 0.0f;
                s1[r] = inff;
                s2[r] = inff;
                j1[r] = 0;
            }
            if (safe) {  // a kernel argument: uniform
                for (int base = 0; base < nodes; base += FT) {
                    const int cnt = min(FT, nodes - base);
                    __syncthreads();
                    for (int k = threadIdx.x; k < cnt * AWESOM_F; k += blockDim.x)
                        fsf[(k % AWESOM_F) * FT + k / AWESOM_F] = __double2float_rn(w[base * AWESOM_F + k]);
                    __syncthreads();
                    for (int j = lane; j < cnt; j += 32) {
                        float wv[AWESOM_F];
                        #pragma unroll
                        for (int f = 0; f < AWESOM_F; ++f) wv[f] = fsf[f * FT + j];
                        const int node = base + j;
                        #pragma unroll
                        for (int r = 0; r < FB_R; ++r) {
                            float acc = 0.0f;
                            #pragma unroll
                            for (int f = 0; f < AWESOM_F; ++f) {
                                const float d = __fsub_rn(wv[f], xr[r][f]);
                                acc = __fmaf_rn(d, d, acc);
                            }
                            s2[r] = fminf(s2[r], fmaxf(s1[r], acc));
                            j1[r] = acc < s1[r] ? node : j1[r];
                            s1[r] = fminf(s1[r], acc);
                        }
                    }
                }
            }
            #pragma unroll
            for (int r = 0; r < FB_R; ++r) {
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    const float o1 = __shfl_xor_sync(0xffffffffu, s1[r], off);
                    const float o2 = __shfl_xor_sync(0xffffffffu, s2[r], off);
                    const int oj = __shfl_xor_sync(0xffffffffu, j1[r], off);
                    if (o1 < s1[r] || (o1 == s1[r] && oj < j1[r])) {
                        s2[r] = fminf(o2, s1[r]);
                        s1[r] = o1;
                        j1[r] = oj;
                    } else {
                        s2[r] = fminf(s2[r], o1);
                    }
                }
                if (lane == 0 && src[r] >= 0) {
                    double row[AWESOM_F];
                    #pragma unroll
                    for (int f = 0; f < AWESOM_F; ++f) row[f] = (double)x[src[r] * AWESOM_F + f];
                    bool certified = false;
                    if (safe) {
                        double xn2 = 0.0;
                        #pragma unroll
                        for (int f = 0; f < AWESOM_F; ++f)
                            xn2 = __dadd_rn(xn2, __dmul_rn(row[f], row[f]));
                        // The shipped certificate (aweSOM._gpu_screen), unchanged.
                        const double u = 5.9604644775390625e-8;
                        double g = (AWESOM_F + 2) * u * 1.01;
                        double e = 2.0 * u * (wmax + __dsqrt_rn(xn2)) * 1.01 + 1e-18;
                        double lo2 = __dsqrt_rn(__ddiv_rn((double)s2[r], 1.0 + g)) - e;
                        double hi1 = (__dsqrt_rn(__ddiv_rn((double)s1[r], 1.0 - g)) + e) * (1.0 + 1e-12);
                        certified = lo2 > hi1 && s2[r] < inff;
                    }
                    if (certified) {
                        idx[src[r]] = j1[r];
                        d2[src[r]] = exact_inl(w + j1[r] * AWESOM_F, row);
                    } else {
                        list2[atomicAdd(count2, 1)] = src[r];
                    }
                }
            }
        }
    }

    extern "C" __global__ void fb_exact_f32(const double* w, const float* x, const int* list,
        const int* count_p, int* idx, double* d2, int nodes) {
        fb_exact_body(w, x, list, count_p, idx, d2, nodes);
    }
    extern "C" __global__ void fb_exact_f64(const double* w, const double* x, const int* list,
        const int* count_p, int* idx, double* d2, int nodes) {
        fb_exact_body(w, x, list, count_p, idx, d2, nodes);
    }
    extern "C" __global__ void fb_screen_f32(const double* w, const float* x, const int* list,
        const int* count_p, int* idx, double* d2, int* list2, int* count2, int nodes, double wmax, int safe) {
        fb_screen_body(w, x, list, count_p, idx, d2, list2, count2, nodes, wmax, safe);
    }
    extern "C" __global__ void fb_screen_f64(const double* w, const double* x, const int* list,
        const int* count_p, int* idx, double* d2, int* list2, int* count2, int nodes, double wmax, int safe) {
        fb_screen_body(w, x, list, count_p, idx, d2, list2, count2, nodes, wmax, safe);
    }
    '''


KERNEL_NAMES = ("seed_f32", "seed_f64", "bucket_hist", "bucket_scan", "bucket_scatter", "bpass_f32", "bpass_f64",
                "fb_exact_f32", "fb_exact_f64", "fb_screen_f32", "fb_screen_f64")


@lru_cache(maxsize=16)
def _kernels(F):
    import cupy as cp

    source = _source(F)
    return {name: cp.RawKernel(source, name, options=OPTIONS) for name in KERNEL_NAMES}


def shared_bytes(F, P):
    """Dynamic shared memory per launch: seed, bucketed pass, fb_screen, fb_exact."""
    pad, px = _pads(F)
    return dict(seed=P * pad * 4, bpass=CT * max(pad * 4 + 4, px * 4), fb_screen=FT * F * 4, fb_exact=FT * F * 8,
                hist=P * 4, scatter=2 * P * 4)


def eligible(nodes, features, safe, min_nodes):
    """Whether a lattice can use the bucketed path (before grouping)."""
    return bool(safe) and 1 <= features <= MAX_FEATURES and nodes >= max(min_nodes, 2)


# ----------------------------------------------------------------------------- device engine
class Unsupported(ValueError):
    """The lattice cannot use the bucketed search (the searcher keeps the dense kernel)."""


# Sums of BucketEngine.stats over every engine of this process (benchmarks read this after a run).
TOTALS = dict(engines=0, setup_seconds=0.0, builds=0, rows=0, fallback_rows=0, top_two_failed=0,
              rare_rows=0)


def reset_totals():
    for key in TOTALS:
        TOTALS[key] = 0.0 if key == "setup_seconds" else 0


class _Scratch:
    """Device buffers for one chunk of at most ``cap`` rows, used by one stream at a time.

    Each search queues a copy of its counters into a pinned buffer; ``records`` lists the searches
    whose counters the host has not read yet (one per chunk queued since the last wait)."""

    def __init__(self, cp, cap, P, pinned):
        self.cap, self.P, self.cp, self.pinned = int(cap), P, cp, pinned
        self.seed = cp.empty(cap, np.int32)
        self.perm = cp.empty(cap, np.int32)
        self.fb_rows = cp.empty(cap, np.int32)
        self.list2 = cp.empty(cap, np.int32)
        self.zbuf = cp.empty(4 + P, np.int32)  # counters[0:4] (zeroed by the seed kernel), counts[4:]
        self.bucket_off = cp.empty(P + 1, np.int32)
        self.tile_off = cp.empty(P + 1, np.int32)
        self.cursor = cp.empty(P, np.int32)
        self.buffers = []  # pinned (owner, int32[4 + P]) pairs, reused
        self.records = []  # (rows, table generation, sample or None, pinned buffer index)
        self.stream = None  # the stream of the last search

    def buffer(self, i):
        while len(self.buffers) <= i:
            self.buffers.append(self.pinned(self.cp, (4 + self.P,), np.int32))
        return self.buffers[i][1]


class BucketEngine:
    """Bucketed search of one lattice on one device, owned by a ``GpuScreenSearcher``.

    ``dense(x_dev, idx_dev, d2_dev, n, stream, f64)`` launches the dense kernel; the pilot uses it for
    exact distances. Tables are built at the first :meth:`search` from that chunk's rows, and replaced
    only after every stream that used them has finished (see ``RETUNE_*``). ``stats`` (seconds and row
    counts) is for benchmarks; ``setup_seconds`` includes grouping, compilation, every build and
    rebuild, and is also added to :data:`TOTALS`.
    """

    def __init__(self, cp, w, w_dev, dense, pinned, shared_limit):
        t0 = time.perf_counter()
        self.cp, self.w, self.w_dev, self.dense, self.pinned = cp, w, w_dev, dense, pinned
        self.nodes, self.F = w.shape
        self.grp = build_groups(w)
        self.P = self.grp["P"]
        self.sc = screen_scalars(w, self.grp)
        self.shared = shared_bytes(self.F, self.P)
        if self.P > MAX_GROUPS or not self.sc["safe"] or max(self.shared.values()) > shared_limit:
            raise Unsupported("lattice not supported by the bucketed search")
        safe, wmax = screen_parameters(w)
        self.fb_args = (np.int32(self.nodes), np.float64(wmax), np.int32(safe))
        eps = _feature_eps(self.F)
        self.scal = (np.int32(self.P), np.int32(CT), np.float64(self.sc["wmax"]), np.float64(self.sc["cmax"]),
                     np.float32(self.sc["G1"]), np.int32(self.sc["safe"]), np.float64(ETA), np.float64(1.0 - eps))
        self.c32 = np.zeros((self.P, _pads(self.F)[0]), np.float32)
        self.c32[:, :self.F] = self.grp["cent"].astype(np.float32)
        self.k = _kernels(self.F)
        self.home = cp.cuda.get_current_stream()  # allocations and uploads (see _TrainRing.rows)
        self.stream = cp.cuda.Stream(non_blocking=True)  # pilot work
        self.t = None  # device tables
        self.gen = 0
        self.pilot = None  # host (x float64, d2, seed, rank) of every pilot row kept
        self.pending = []  # host samples waiting for a rebuild
        self.since = 0  # chunks searched since the last build
        self.scratches = weakref.WeakSet()
        self.stats = dict(setup_seconds=0.0, builds=0, rows=0, fallback_rows=0, top_two_failed=0, rare_rows=0,
                          pilot_rows=0)
        TOTALS["engines"] += 1
        self._setup(time.perf_counter() - t0)

    def _setup(self, seconds):
        self.stats["setup_seconds"] += seconds
        TOTALS["setup_seconds"] += seconds

    def scratch(self, cap):
        """Buffers for one stream's chunks of at most ``cap`` rows (allocated on the home stream)."""
        with self.home:
            s = _Scratch(self.cp, cap, self.P, self.pinned)
        self.scratches.add(s)
        return s

    # ------------------------------------------------------------------ setup
    def _pilot_device(self, x):
        """Exact distances (dense kernel) and seed buckets (seed kernel) of host rows ``x`` (float64)."""
        cp, n, s = self.cp, len(x), self.stream
        with self.home:
            xd = cp.empty(x.shape, np.float64)
            idx = cp.empty(n, np.int32)
            d2 = cp.empty(n, np.float64)
            seed = cp.empty(n, np.int32)
            c32 = cp.asarray(self.c32)
        self.home.synchronize()
        hd2, hseed = np.empty(n, np.float64), np.empty(n, np.int32)
        xd.set(x, stream=s)
        self.dense(xd, idx, d2, n, s, True)
        self.k["seed_f64"](((n + SEED_BLOCK * ROWS - 1) // (SEED_BLOCK * ROWS),), (SEED_BLOCK,),
                           (c32, xd, seed, np.int32(n), np.int32(self.P), seed, np.int32(0)),
                           shared_mem=self.shared["seed"], stream=s)
        d2.get(out=hd2, stream=s, blocking=False)
        seed.get(out=hseed, stream=s, blocking=False)
        s.synchronize()
        return hd2, hseed.astype(np.int64)

    def _build(self, first=None):
        """Build the device tables from ``first`` (rows of the first chunk) or rebuild them with the
        pending samples added to the pilot."""
        t0 = time.perf_counter()
        if first is not None:
            step = max(1, len(first) // PILOT_ROWS)
            new = np.array(first[::step][:PILOT_ROWS], np.float64)
        else:
            new = np.concatenate(self.pending)
        self.pending = []
        d2, seed = self._pilot_device(new)
        rank = pilot_rank(self.grp, new, d2, seed)
        if self.pilot is None:
            self.pilot = (new, d2, seed, rank)
        else:
            self.pilot = tuple(np.concatenate([a, b])[-RESERVOIR:] for a, b in zip(self.pilot, (new, d2, seed, rank)))
        K, rare = tune_k(self.grp, self.pilot[2], self.pilot[3])
        T = pack(self.w, self.grp, self.sc, K, rare)
        # Chunks queued on other streams may still read the old tables.
        for s in list(self.scratches):
            if s.stream is not None:
                s.stream.synchronize()
        self.t = None
        with self.home:
            self.t = {k: self.cp.asarray(T[k]) for k in ("c32", "cw", "cid", "cand_lo", "cand_hi", "ex32",
                                                          "ex_lo", "ex_hi")}
        self.home.synchronize()  # the searches run on streams that do not order against home
        self.rare, self.K = rare, K
        self.gen += 1
        self.since = 0
        self.stats["builds"] += 1
        TOTALS["builds"] += 1
        self.stats["pilot_rows"] = len(self.pilot[0])
        self._setup(time.perf_counter() - t0)

    # ------------------------------------------------------------------ per chunk
    def search(self, s, stream, x, idx, d2, n, x_host):
        """Queue the search of the first ``n`` (>= 1) rows of device ``x`` on ``stream`` with scratch
        ``s``; node indices and squared distances go to ``idx`` and ``d2`` in row order. ``x_host``
        holds the same rows on the host (read for the pilot and rebuild samples only)."""
        if self.t is None:
            self._build(first=x_host[:n])
        elif self.pending and self.since >= RETUNE_GAP:
            self._build()
        t, k = self.t, self.k
        sfx = "f64" if x_host.dtype == np.float64 else "f32"  # device x has the host rows' dtype
        P = np.int32(self.P)
        nn = np.int32(n)
        rb = BLOCK * ROWS
        counts = s.zbuf[4:]
        sample = None
        if self.stats["builds"] <= RETUNE_MAX and n >= RETUNE_MIN_ROWS:
            step = max(1, n // SAMPLE_ROWS)
            sample = np.array(x_host[:n:step][:SAMPLE_ROWS], np.float64)
        k["seed_" + sfx](((n + SEED_BLOCK * ROWS - 1) // (SEED_BLOCK * ROWS),), (SEED_BLOCK,),
                         (t["c32"], x, s.seed, nn, P, s.zbuf, np.int32(4 + self.P)),
                         shared_mem=self.shared["seed"], stream=stream)
        k["bucket_hist"]((min((n + HIST_BLOCK - 1) // HIST_BLOCK, 4096),), (HIST_BLOCK,),
                         (s.seed, nn, P, counts), shared_mem=self.shared["hist"], stream=stream)
        k["bucket_scan"]((1,), (1024,), (counts, P, np.int32(rb), s.bucket_off, s.cursor, s.tile_off),
                         shared_mem=0, stream=stream)
        k["bucket_scatter"](((n + SCATTER_BLOCK * SCATTER_ROWS - 1) // (SCATTER_BLOCK * SCATTER_ROWS),),
                            (SCATTER_BLOCK,), (s.seed, nn, P, s.cursor, s.perm),
                            shared_mem=self.shared["scatter"], stream=stream)
        # sum over buckets of ceil(count / rb) <= ceil(n / rb) + P tiles; extra blocks exit at once.
        k["bpass_" + sfx](((n + rb - 1) // rb + self.P,), (BLOCK,),
                          (self.w_dev, x, s.perm, s.bucket_off, s.tile_off, t["cw"], t["cid"], t["cand_lo"],
                           t["cand_hi"], t["ex32"], t["ex_lo"], t["ex_hi"], idx, d2, s.fb_rows, s.zbuf) + self.scal,
                          shared_mem=self.shared["bpass"], stream=stream)
        k["fb_screen_" + sfx]((FB_GRID,), (FB_BLOCK,),
                              (self.w_dev, x, s.fb_rows, s.zbuf[0:1], idx, d2, s.list2, s.zbuf[2:3]) + self.fb_args,
                              shared_mem=self.shared["fb_screen"], stream=stream)
        k["fb_exact_" + sfx]((FB_GRID,), (FB_BLOCK,),
                             (self.w_dev, x, s.list2, s.zbuf[2:3], idx, d2, self.fb_args[0]),
                             shared_mem=self.shared["fb_exact"], stream=stream)
        i = len(s.records)
        s.zbuf.get(out=s.buffer(i), stream=stream, blocking=False)
        s.records.append((n, self.gen, sample, i))
        s.stream = stream
        self.since += 1

    def collect(self, s):
        """Read the counters of ``s``'s queued searches; call only after waiting for its stream."""
        records, s.records = s.records, []
        for n, gen, sample, i in records:
            st = s.buffers[i][1]
            fb = int(st[0])
            rare = int(st[4:][self.rare].sum()) if gen == self.gen else 0
            for key, v in (("rows", n), ("fallback_rows", fb), ("top_two_failed", int(st[1])), ("rare_rows", rare)):
                self.stats[key] += v
                TOTALS[key] += v
            if (sample is not None and gen == self.gen and fb + rare > RETUNE_FRACTION * n
                    and self.stats["builds"] < RETUNE_MAX and len(self.pending) < 8):
                self.pending.append(sample)

    @staticmethod
    def discard(s):
        """Forget ``s``'s queued searches (their stream was cancelled)."""
        s.records = []
