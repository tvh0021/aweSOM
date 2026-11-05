"""Compare the final weights of multi-rank runs with the 1-rank run of the same setup.

Runs are the directories holding ``run.json`` + ``weights.npy`` under the given
roots. Runs with equal (lattice, n, epochs, seed, data_seed, r_min, chunk, groups)
form a group; the reference is its 1-rank run (or the run under ``--ref``).
Runs whose generated data differ from the reference (``block0_sha256``, e.g. a
different CPU or numpy) are reported as errors, not compared.
Contract (brief-common): |a - b| <= 1e-4 * max(|a|, |b|) + 1e-12 for every weight.
Writes ``parity.json`` and ``parity.md`` to ``--out``; exit 1 if any run fails or a
group has no reference.

Usage: python parity.py --out DIR ROOT [ROOT ...] [--ref REFROOT ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

RTOL, ATOL = 1e-4, 1e-12


def runs(roots):
    for root in roots:
        for p in sorted(Path(root).rglob("run.json")):
            if (p.parent / "weights.npy").exists():
                yield p.parent, json.loads(p.read_text())


def key(r):
    s = r["settings"]
    return (s["lattice"], int(s["n"]), s["epochs"], s["seed"], s["data_seed"], s["r_min"], s["chunk"], s["groups"])


def compare(a, b):
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = float(np.max(diff / np.maximum(scale, 1e-300))) if diff.any() else 0.0
    ok = bool(np.all(diff <= RTOL * scale + ATOL))
    return dict(max_rel=rel, max_abs=float(diff.max()), within_tolerance=ok, bitwise=bool(not diff.any()))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ref", nargs="*", default=[])
    p.add_argument("roots", nargs="+")
    a = p.parse_args(argv)
    refs = {}
    for d, r in list(runs(a.ref)) + list(runs(a.roots)):
        if r["ranks"] == 1 and key(r) not in refs:
            refs[key(r)] = (d, r)
    rows, failed = [], False
    for d, r in runs(a.roots):
        if r["ranks"] == 1:
            continue
        k = key(r)
        if k not in refs:
            rows.append(dict(run=str(d), lattice=k[0], n=k[1], ranks=r["ranks"], error="no 1-rank reference"))
            failed = True
            continue
        if r.get("block0_sha256") != refs[k][1].get("block0_sha256"):
            rows.append(dict(run=str(d), reference=str(refs[k][0]), lattice=k[0], n=k[1], ranks=r["ranks"],
                             error="generated data differ from the reference (block0_sha256)"))
            failed = True
            continue
        c = compare(np.load(d / "weights.npy"), np.load(refs[k][0] / "weights.npy"))
        rows.append(dict(run=str(d), reference=str(refs[k][0]), lattice=k[0], n=k[1], ranks=r["ranks"],
                         nodes=r["n_nodes"], **c))
        failed |= not c["within_tolerance"]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "parity.json").write_text(json.dumps(dict(rtol=RTOL, atol=ATOL, failed=failed, runs=rows), indent=1))
    lines = ["| lattice | n | ranks | nodes | max rel diff | bitwise | within 1e-4 | run |", "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        if "error" in x:
            lines.append(f"| {x['lattice']} | {x['n']:.0e} | {x['ranks']} | | | | {x['error']} | {x['run']} |")
        else:
            lines.append(f"| {x['lattice']} | {x['n']:.0e} | {x['ranks']} | {x['nodes']} | {x['max_rel']:.2e} | "
                         f"{x['bitwise']} | {x['within_tolerance']} | {Path(x['run']).name} |")
    text = "\n".join(lines) + "\n"
    (a.out / "parity.md").write_text(text)
    print(text + ("PARITY FAILED" if failed else "parity ok"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
