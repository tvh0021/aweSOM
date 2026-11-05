"""Markdown summary of scale-out runs (run.json), I/O probes (io-*.json) and parity.

Measured columns: rows/s, rows/s per thread, thread efficiency (per-thread rate vs
the run with the fewest threads, same lattice and n, one rank), node efficiency
(rate / (nodes * 1-node 32-thread rate), same lattice and n), epoch-1 and
later-epoch rates, reduction time, io_wait, peak RSS. The projection section scales
the measured 1-node 32-thread per-core rate to 72 and 144 cores at 1x, 1.5x and 2x
per-core speed; it is arithmetic, not a measurement.

Writes ``summary.md`` and ``gate.json`` (node efficiency at 1e8 rows, 2 and 4
nodes, >= 0.70 for every lattice: the condition for submitting nodes_1e9.sbatch).

Usage: python summarize.py --out DIR ROOT [ROOT ...]
"""
import argparse
import json
import sys
from pathlib import Path

GATE = 0.70


def load(roots):
    runs, io = [], []
    for root in roots:
        for p in sorted(Path(root).rglob("run.json")):
            r = json.loads(p.read_text())
            r["dir"] = str(p.parent)
            runs.append(r)
        for p in sorted(Path(root).rglob("io-*.json")):
            io.append(dict(json.loads(p.read_text()), file=str(p)))
    return runs, io


def setup(r):
    s = r["settings"]
    return s["lattice"], int(s["n"])


def fmt(x, spec):
    return "" if x is None else format(x, spec)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("roots", nargs="+")
    a = p.parse_args(argv)
    runs, io = load(a.roots)
    base = {}  # 1 node, 1 rank, 32 threads
    fewest = {}  # 1 rank, fewest threads
    for r in runs:
        k = setup(r)
        if r["ranks"] == 1 and r["threads_per_rank"] == 32:
            base.setdefault(k, r)
        if r["ranks"] == 1 and (k not in fewest or r["threads_per_rank"] < fewest[k]["threads_per_rank"]):
            fewest[k] = r
    out = ["# Scale-out runs", "", "Measured. Rates are rows processed / training wall time (5 epochs, all rows "
           "every epoch, final_pass none). Thread eff = per-thread rate / per-thread rate of the fewest-thread "
           "1-rank run; node eff = rate / (nodes x 1-node 32-thread rate).", "",
           "| lattice | n | nodes | ranks x threads | wall s | Mrows/s | krows/s/thread | thread eff | node eff | "
           "epoch-1 Mrows/s | later Mrows/s | reduce s | io_wait s | max RSS GB | max/min rank pass s | dir |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    gate = {}
    for r in sorted(runs, key=lambda r: (setup(r), r["n_nodes"], r["ranks"], r["threads_per_rank"])):
        k = setup(r)
        rate = r["rows_per_second"]
        per_thread = rate / r["total_threads"]
        teff = neff = None
        if r["ranks"] == 1 and k in fewest:
            f = fewest[k]
            teff = per_thread / (f["rows_per_second"] / f["total_threads"])
        if (r["n_nodes"] > 1 or r["ranks"] > 1) and k in base:
            neff = rate / (r["n_nodes"] * base[k]["rows_per_second"])
            if k[1] == int(1e8) and r["n_nodes"] in (2, 4) and r["ranks"] == r["n_nodes"]:
                gate.setdefault(k[0], {})[r["n_nodes"]] = neff
        imb = ""
        passes = [e.get("rank_pass_seconds") for e in r["epochs"]]
        if passes and all(passes) and r["ranks"] > 1:
            imb = f"{sum(max(x) for x in passes):.0f}/{sum(min(x) for x in passes):.0f}"
        out.append(f"| {k[0]} | {k[1]:.0e} | {r['n_nodes']} | {r['ranks']} x {r['threads_per_rank']} | "
                   f"{r['training_wall_seconds']:.0f} | {rate / 1e6:.2f} | {per_thread / 1e3:.1f} | "
                   f"{fmt(teff, '.2f')} | {fmt(neff, '.2f')} | {r['first_epoch_rows_per_second'] / 1e6:.2f} | "
                   f"{fmt(r['later_epoch_rows_per_second'] and r['later_epoch_rows_per_second'] / 1e6, '.2f')} | "
                   f"{r['reduce_seconds_total']:.3f} | {r['io_wait_total']:.2f} | "
                   f"{max(r['peak_rss_bytes_per_rank']) / 2**30:.1f} | {imb} | {Path(r['dir']).name} |")
    if io:
        out += ["", "# Lagoon and local-disk I/O probe (measured)", "",
                "| test | role | host | GB | seconds | GB/s | note |", "|---|---|---|---|---|---|---|"]
        for x in io:
            out.append(f"| {x.get('test')} | {x.get('role')} | {x.get('host')} | {x.get('gb', 0):.1f} | "
                       f"{x.get('seconds', 0):.1f} | {x.get('gb_per_s', 0):.3f} | {x.get('note', '')} |")
    out += ["", "# PROJECTION (not measured): 72 and 144 Arm cores", "",
            "Per-core rate = measured 1-node 32-thread rate / 32, times cores, times a per-core speed factor "
            "(1x, 1.5x, 2x of a Xeon Gold 6130 core). Assumes linear thread scaling beyond 32 threads; compare "
            "with the measured thread efficiency above. Hours = 40 epochs x N rows / rate.", "",
            "| lattice | n measured | cores | factor | projected Mrows/s | 40 epochs at 1e9, h | 40 epochs at 1e10, h |",
            "|---|---|---|---|---|---|---|"]
    for k, r in sorted(base.items()):
        per_core = r["rows_per_second"] / 32
        for cores in (72, 144):
            for factor in (1.0, 1.5, 2.0):
                rate = per_core * cores * factor
                out.append(f"| {k[0]} | {k[1]:.0e} | {cores} | {factor:g}x | {rate / 1e6:.2f} | "
                           f"{40 * 1e9 / rate / 3600:.2f} | {40 * 1e10 / rate / 3600:.1f} |")
    ok = bool(gate) and all(len(v) == 2 and min(v.values()) >= GATE for v in gate.values())
    out += ["", f"# Gate for nodes_1e9.sbatch: node efficiency >= {GATE} at 1e8 rows on 2 and 4 nodes", "",
            json.dumps(gate), "", "PASS" if ok else "NOT MET (or 1e8 node sweep missing)"]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.md").write_text("\n".join(out) + "\n")
    (a.out / "gate.json").write_text(json.dumps(dict(threshold=GATE, efficiency=gate, passed=ok), indent=1) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
