"""Lead 1: JAX CPU float64 BMU variants vs numba, on the real plasma data.

Usage: python jax_lead1_cpu_f64.py [ndev] [reps]
ndev sets --xla_force_host_platform_device_count before jax is imported.
All variants are timed interleaved (round-robin over reps), warm-up excluded,
median reported. Prints one JSON line.
"""
import json
import os
import sys
import time

NDEV = int(sys.argv[1]) if len(sys.argv) > 1 else 10
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NDEV}"

import pickle

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from aweSOM.inference import best_match_nodes, best_match_nodes_jax

lat = pickle.load(open("/Users/tvh0021/git_repos/aweSOM/examples/plasma-turbulence/"
                       "som_object.fiducial-5000-64-37-0.1-2097152-1u.pkl", "rb"))
X = np.ascontiguousarray(lat.data_array, dtype=np.float64)
F = X.shape[1]
W = np.ascontiguousarray(np.asarray(lat.lattice, dtype=np.float64).reshape(-1, F))
N, M = X.shape[0], W.shape[0]

ref_idx, ref_d2 = best_match_nodes(W, X)
devs = jax.local_devices()


def make_variants():
    def dist_rows(block, w, zero):  # [t, nodes], same arithmetic as library
        d = block[:, None, 0] - w[None, :, 0]
        acc = d * d + zero
        for f in range(1, F):
            d = block[:, None, f] - w[None, :, f]
            acc = acc + (d * d + zero)
        return jnp.where(jnp.isnan(acc), jnp.inf, acc)

    def dist_T(block, w, zero):  # [nodes, t]
        d = w[:, 0, None] - block[None, :, 0]
        d = -d  # x - w, same as library sign (squares equal anyway)
        acc = d * d + zero
        for f in range(1, F):
            d = block[None, :, f] - w[:, f, None]
            acc = acc + (d * d + zero)
        return jnp.where(jnp.isnan(acc), jnp.inf, acc)

    def transposed(w, block, zero):
        acc = dist_T(block, w, zero)
        return jnp.argmin(acc, axis=0).astype(jnp.int32), jnp.min(acc, axis=0)

    def variadic(w, block, zero):
        acc = dist_rows(block, w, zero)
        iota = lax.broadcasted_iota(jnp.int32, acc.shape, 1)

        def red(a, b):
            av, ai = a
            bv, bi = b
            take_b = (bv < av) | ((bv == av) & (bi < ai))
            return jnp.where(take_b, bv, av), jnp.where(take_b, bi, ai)

        v, i = lax.reduce((acc, iota), (jnp.array(jnp.inf, acc.dtype), jnp.int32(M)), red, (1,))
        return i, v

    def nodeblock(w, block, zero, nb=128):
        # scan over node blocks in ascending order; strict < keeps lowest index
        mp = -(-M // nb) * nb
        wp = jnp.concatenate([w, jnp.full((mp - M, F), jnp.inf, w.dtype)], 0).reshape(-1, nb, F)

        def body(carry, args):
            best, bidx = carry
            wb, off = args
            acc = dist_rows(block, wb, zero)
            bi = jnp.argmin(acc, axis=1).astype(jnp.int32)
            bv = jnp.min(acc, axis=1)
            upd = bv < best
            return (jnp.where(upd, bv, best), jnp.where(upd, bi + off, bidx)), None

        offs = jnp.arange(wp.shape[0], dtype=jnp.int32) * nb
        init = (jnp.full(block.shape[0], jnp.inf, block.dtype), jnp.zeros(block.shape[0], jnp.int32))
        (b, i), _ = lax.scan(body, init, (wp, offs))
        return i, b

    return {"transposed": jax.jit(transposed), "variadic": jax.jit(variadic),
            "nodeblock128": jax.jit(nodeblock)}


def run_custom(fn, chunk, devices):
    idx = np.empty(N, np.int32)
    d2 = np.empty(N, np.float64)
    wd = [jax.device_put(W, d) for d in devices]
    zd = [jax.device_put(np.zeros((), np.float64), d) for d in devices]
    pad = np.zeros((chunk, F))
    pend = []
    for k, s in enumerate(range(0, N, chunk)):
        e = min(s + chunk, N)
        b = X[s:e]
        if e - s < chunk:
            pad[: e - s] = b
            pad[e - s:] = 0
            b = pad
        dv = k % len(devices)
        out = fn(wd[dv], jax.device_put(b, devices[dv]), zd[dv])
        pend.append((s, e, *out))
        if len(pend) > len(devices):
            s0, e0, i0, v0 = pend.pop(0)
            idx[s0:e0] = np.asarray(i0)[: e0 - s0]
            d2[s0:e0] = np.asarray(v0)[: e0 - s0]
    for s0, e0, i0, v0 in pend:
        idx[s0:e0] = np.asarray(i0)[: e0 - s0]
        d2[s0:e0] = np.asarray(v0)[: e0 - s0]
    return idx, d2


with jax.enable_x64(True):
    V = make_variants()
    cases = {"numba": lambda: best_match_nodes(W, X)}
    for c in (2048, 4096, 14169, 32768):
        cases[f"lib_1dev_chunk{c}"] = (lambda c=c: best_match_nodes_jax(W, X, chunk=c, devices=devs[:1]))
    if len(devs) > 1:
        for c in (2048, 4096, 14169):
            cases[f"lib_{len(devs)}dev_chunk{c}"] = (lambda c=c: best_match_nodes_jax(W, X, chunk=c, devices=devs))
    for name, fn in V.items():
        cases[f"{name}_1dev_chunk4096"] = (lambda fn=fn: run_custom(fn, 4096, devs[:1]))
        if len(devs) > 1:
            cases[f"{name}_{len(devs)}dev_chunk4096"] = (lambda fn=fn: run_custom(fn, 4096, devs))

    res = {}
    for name, fn in cases.items():  # warm-up + correctness
        i, d = fn()
        res[name] = {"label_mismatch": int(np.count_nonzero(i != ref_idx)),
                     "d2_mismatch": int(np.count_nonzero(d != ref_d2)), "t": []}
    for _ in range(REPS):
        for name, fn in cases.items():
            t0 = time.perf_counter()
            fn()
            res[name]["t"].append(time.perf_counter() - t0)

tb = float(np.median(res["numba"]["t"]))
base = float(np.median(res["lib_1dev_chunk14169"]["t"]))
out = {"lead": 1, "device": str(devs[0].platform), "ndev": len(devs), "jax": jax.__version__,
       "N": N, "F": F, "nodes": M, "reps": REPS, "results": {}}
for name, r in res.items():
    t = float(np.median(r["t"]))
    out["results"][name] = {"pts_per_s": round(N / t), "vs_numba": round(tb / t, 3),
                            "vs_lib_default": round(base / t, 3),
                            "label_mismatch": r["label_mismatch"], "d2_mismatch": r["d2_mismatch"]}
print(json.dumps(out))
