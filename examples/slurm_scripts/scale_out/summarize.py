"""Markdown summary of scale-out runs (run.json), I/O probes (io-*.json) and parity.

Measured columns: rows/s, rows/s per thread, thread efficiency (per-thread rate vs
the run with the fewest threads, same lattice and n, one rank), node efficiency
(rate / (nodes * 1-node 32-thread rate), same lattice and n), epoch-1 and
later-epoch rates, reduction time, io_wait, peak RSS. The projection section scales
the measured 1-node 32-thread per-core rate to 72 and 144 cores at 1x, 1.5x and 2x
per-core speed; it is arithmetic, not a measurement.

Writes ``summary.md`` and ``gate.json`` (node efficiency at 1e8 rows, 2 and 4
nodes, >= 0.70 for every lattice: the condition for submitting nodes_1e9.sbatch).
``--check-gate`` re-evaluates a gate.json for the lattices a 1e9 job will run.

Usage: python summarize.py --out DIR ROOT [ROOT ...]
       python summarize.py --check-gate gate.json --lattices 91x64 160x100
"""
import argparse
import json
import sys
from pathlib import Path

GATE = 0.70
GATE_NODES = (2, 4)
LATTICES = ("91x64", "160x100")


def gate_check(efficiency, lattices=LATTICES, threshold=GATE, nodes=GATE_NODES):
    """``(passed, reasons)``: every lattice needs a measured 1e8 node efficiency >= threshold
    at every node count in ``nodes``. ``efficiency`` maps lattice -> {nodes: efficiency}."""
    reasons = []
    for lat in lattices:
        got = {int(k): v for k, v in (efficiency.get(lat) or {}).items()}
        for nn in nodes:
            if got.get(nn) is None:
                reasons.append(f"{lat}: no {nn}-node run at 1e8 rows")
            elif not got[nn] >= threshold:
                reasons.append(f"{lat}: {nn}-node efficiency {got[nn]:.3f} < {threshold}")
    return not reasons and bool(lattices), reasons


def check_gate_file(path, lattices):
    g = json.loads(Path(path).read_text())
    ok, reasons = gate_check(g.get("efficiency", {}), lattices, g.get("threshold", GATE))
    print(f"gate {'PASS' if ok else 'NOT MET'} for {' '.join(lattices)}" + "".join(f"\n  {r}" for r in reasons))
    return 0 if ok else 1


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


def baseline_label(r):
    """CPU model, partition and node of a run, as recorded in its run.json."""
    env = r.get("environment", {})
    cpu = env.get("lscpu", {}).get("Model name") or env.get("platform") or "unknown CPU"
    part = env.get("slurm", {}).get("SLURM_JOB_PARTITION") or "no partition"
    nodes = ",".join(r.get("nodes", [])) or "unknown node"
    return f"{cpu}, {part}, {nodes}"


def fmt(x, spec):
    return "" if x is None else format(x, spec)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--check-gate", type=Path, help="exit 0 if this gate.json passes for --lattices")
    p.add_argument("--lattices", nargs="+", default=list(LATTICES))
    p.add_argument("roots", nargs="*")
    a = p.parse_args(argv)
    if a.check_gate is not None:
        return check_gate_file(a.check_gate, a.lattices)
    if a.out is None or not a.roots:
        p.error("--out and at least one ROOT are required")
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
            "(1x, 1.5x, 2x of one core of the baseline run, whose CPU, partition and node are in the last "
            "column). Assumes linear thread scaling beyond 32 threads; compare with the measured thread "
            "efficiency above. Hours = 40 epochs x N rows / rate.", "",
            "| lattice | n measured | cores | factor | projected Mrows/s | 40 epochs at 1e9, h | 40 epochs at 1e10, h "
            "| baseline CPU, partition, node |",
            "|---|---|---|---|---|---|---|---|"]
    for k, r in sorted(base.items()):
        per_core = r["rows_per_second"] / 32
        label = baseline_label(r)
        for cores in (72, 144):
            for factor in (1.0, 1.5, 2.0):
                rate = per_core * cores * factor
                out.append(f"| {k[0]} | {k[1]:.0e} | {cores} | {factor:g}x | {rate / 1e6:.2f} | "
                           f"{40 * 1e9 / rate / 3600:.2f} | {40 * 1e10 / rate / 3600:.1f} | {label} |")
    ok, reasons = gate_check(gate)
    out += ["", f"# Gate for nodes_1e9.sbatch: node efficiency >= {GATE} at 1e8 rows on 2 and 4 nodes, "
            f"lattices {', '.join(LATTICES)}", "", json.dumps(gate), "", "PASS" if ok else "NOT MET"] + reasons
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.md").write_text("\n".join(out) + "\n")
    (a.out / "gate.json").write_text(json.dumps(dict(threshold=GATE, nodes=list(GATE_NODES), lattices=list(LATTICES),
                                                         efficiency=gate, passed=ok, reasons=reasons), indent=1) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
