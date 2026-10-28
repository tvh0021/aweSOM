"""Print markdown tables from the JSON files that the real_*.py benchmarks write.

usage: python planning/benchmarks/real_report.py [results_dir]
"""

import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "/Volumes/ExternalNVMe/awesom-results/2026-10-28/validate")
CASES = ["plasma", "bbh-29x20", "bbh-51x36", "bbh-91x64"]
BACKENDS = ["numba", "screen", "kdtree", "jax"]


def ld(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def l1(t):
    """1-minute load average from an 'a b c' load string."""
    return t["load"].split()[0]


def inference():
    rows = [(c, ld(f"inference-{c}.json")) for c in CASES]
    rows = [(c, d) for c, d in rows if d is not None]

    print("### A. Inference, all threads (points/s, min of 3, [1-min load])\n")
    print("| Case | N | F | nodes | baseline (first 1,048,576 pts) | `Lattice.best_match` (same pts) | "
          + " | ".join(BACKENDS) + " |")
    print("|---" * (6 + len(BACKENDS)) + "|")
    for c, d in rows:
        b, lb = d["baseline"], d["lattice_best_match_new"]
        row = [c, f"{d['N']:,}", str(d["F"]), str(d["nodes"]),
               f"{b['pts_per_s']:.3g} [{l1(b)}]", f"{lb['pts_per_s']:.3g} [{l1(lb)}]"]
        row += [f"{d['backends'][k]['pts_per_s']:.3g} [{l1(d['backends'][k])}]" for k in BACKENDS]
        print("| " + " | ".join(row) + " |")

    print("\n### A. Equality with the baseline\n")
    print("| Case | baseline nodes == pickle | baseline labels == ref | "
          + " | ".join(f"{k}: labels equal (max abs dist diff)" for k in BACKENDS)
          + " | screen float64 fallbacks | `assign_cluster_to_lattice` equal |")
    print("|---" * (5 + len(BACKENDS)) + "|")
    for c, d in rows:
        b = d["baseline"]
        row = [c, f"{b['node_equal_pickle']:,}/{b['n']:,}", f"{b['label_equal_ref']:,}/{b['n']:,}"]
        for k in BACKENDS:
            x = d["backends"][k]
            row.append(f"{x['label_equal']:,}/{d['N']:,} ({x['max_abs_dist_diff']:g})")
        row.append(f"{d['backends']['screen'].get('float64_fallbacks', '-'):,}")
        row.append(str(d.get("assign_cluster_to_lattice", {}).get("baseline_equals_new", "-")))
        print("| " + " | ".join(row) + " |")

    print("\n### A. One thread, and 1e10 points on 48 cores (a projection from the per-core rate, not a measurement)\n")
    one = ["baseline", "numba", "screen", "kdtree"]
    print("| Case | " + " | ".join(f"{k}: pts/s/core [load] -> 1e10 on 48 cores" for k in one) + " |")
    print("|---" * (1 + len(one)) + "|")
    for c, d in rows:
        row = [c]
        for k in one:
            x = d["one_thread"][k]
            row.append(f"{x['pts_per_s']:.3g} [{l1(x)}] -> {x['projected_1e10_on_48_cores_s'] / 60:.1f} min")
        print("| " + " | ".join(row) + " |")

    print("\n### A. `SOMModel.predict` on raw float32 features (`Scaler` works in float64)\n")
    print("| Case | max abs diff vs float32-scaled training data | labels equal / N | pts/s [load] |")
    print("|---|---|---|---|")
    for c, d in rows:
        s = d.get("somodel_raw")
        if s:
            print(f"| {c} | {s['scaler_vs_pickle_max_abs']:.3g} | {s['label_equal']:,}/{d['N']:,} | "
                  f"{s['pts_per_s']:.3g} [{l1(s)}] |")


def train():
    cases = [(c, ld(f"train-{c}.json")) for c in ["29x20", "51x36"]]
    cases = [(c, d) for c, d in cases if d is not None]
    print("\n### B. Online training (baseline shuffle patched to `default_rng(42)`, same initial lattice; min of 3)\n")
    print("| Case | steps | baseline s [load] | new, history=True s [load] | new, history=False s [load] | speedup | bitwise |")
    print("|---|---|---|---|---|---|---|")
    for c, d in cases:
        t, bw = d["timing"], d["bitwise"]
        flags = [v for k, v in bw.items() if isinstance(v, bool)]
        print(f"| {c} | {d['steps']:,} | {t['baseline']['s_min']:.2f} [{l1(t['baseline'])}] | "
              f"{t['new_history_true']['s_min']:.2f} [{l1(t['new_history_true'])}] | "
              f"{t['new_history_false']['s_min']:.2f} [{l1(t['new_history_false'])}] | "
              f"{t['new_history_true']['speedup']:.1f}x / {t['new_history_false']['speedup']:.1f}x | "
              f"{all(flags)}; max abs {bw['lattice_max_abs_diff']:g}; labels {bw['labels_equal']:,}/{bw['labels_total']:,} |")
    print("\n### B. Unpatched baseline vs new trainer with seeds 42-44 (QE, TE, clusters at merge 0.2)\n")
    print("| Case | stored pickle QE / TE | baseline QE | new QE | baseline TE | new TE | clusters baseline / new |")
    print("|---|---|---|---|---|---|---|")
    for c, d in cases:
        s, p = d["statistical"], d["stored_pickle"]
        nb = ",".join(str(r["n_clusters"]) for r in s["baseline"])
        nn = ",".join(str(r["n_clusters"]) for r in s["new"])
        print(f"| {c} | {p['qe']:.4f} / {p['te']:.4f} | {s['baseline_qe_mean']:.4f} +- {s['baseline_qe_std']:.4f} | "
              f"{s['new_qe_mean']:.4f} +- {s['new_qe_std']:.4f} | {s['baseline_te_mean']:.4f} +- {s['baseline_te_std']:.4f} | "
              f"{s['new_te_mean']:.4f} +- {s['new_te_std']:.4f} | {nb} / {nn} |")


def sce():
    d = ld("sce.json")
    if d is None:
        return
    print("\n### C. SCE on the plasma run labels (128^3 cells)\n")
    print("| Maps | clusters | baseline s [load] | new s [load] | new, write_masks=False s [load] | speedup | output files identical | multimap_mappings identical |")
    print("|---|---|---|---|---|---|---|---|")
    for k in ["small", "full"]:
        x = d.get(k)
        if not x:
            continue
        print(f"| {x['maps']} | {x['clusters_total']} | {x['baseline']['s_min']:.2f} [{l1(x['baseline'])}] | "
              f"{x['new']['s_min']:.2f} [{l1(x['new'])}] | {x['new_no_masks']['s_min']:.3f} [{l1(x['new_no_masks'])}] | "
              f"{x['new']['speedup']:.0f}x / {x['new_no_masks']['speedup']:.0f}x | "
              f"{x['all_files_identical']} ({x['output_files']} files) | "
              f"{x['multimap_identical']} / {x['multimap_identical_no_masks']} |")
    cb = d["combine"]
    print(f"\n`combine_separated_clusters` from masks {cb['from_masks']['s_min']:.3f} s, "
          f"`combine_separated_clusters_from_labels` {cb['from_labels']['s_min']:.3f} s "
          f"[{l1(cb['from_labels'])}], bitwise identical: {cb['bitwise_identical']}")


def clumps():
    d = ld("clumps.json")
    if d is None:
        return
    print(f"\n### C. Clumps on `examples/plasma-turbulence/merged_sheets` (cc3d installed: {d['cc3d_installed']})\n")
    print("| Array | input | conn | clumps | identical | baseline `get_clumps` s [load] | `label_clumps` s [load] | speedup |")
    print("|---|---|---|---|---|---|---|---|")
    for x in d["label_clumps"]:
        print(f"| {x['array']} | {x['input']} | {x['conn']} | {x['clumps']:,} | {x['identical']} | "
              f"{x['baseline']['s_min']:.2f} [{l1(x['baseline'])}] | {x['new']['s_min']:.4f} [{l1(x['new'])}] | "
              f"{x['speedup']:.0f}x |")
    print("\n| `merge_double_sheets` sub-cube n | double / positive / negative clumps | merged | identical | baseline s | fast s | speedup [load] |")
    print("|---|---|---|---|---|---|---|")
    for x in d["merge"]:
        print(f"| {x['n']} | {x['double']} / {x['positive']} / {x['negative']} | {x['merged']} | {x['identical']} | "
              f"{x['baseline']['s_min']:.3f} | {x['new']['s_min']:.4f} | {x['speedup']:.0f}x [{l1(x['new'])}] |")


if __name__ == "__main__":
    inference()
    train()
    sce()
    clumps()
