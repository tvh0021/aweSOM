"""examples/slurm_scripts/scale_out: summarize.py (1e9 gate, projection label, under-pinned and
unfinished runs) and the CPU affinity guard of scale_out.py."""
import importlib.util
import json
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1] / "examples/slurm_scripts/scale_out"
spec = importlib.util.spec_from_file_location("scale_out_summarize", KIT / "summarize.py")
summarize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize)
gate_check = summarize.gate_check


def test_gate_needs_both_node_counts_for_every_lattice():
    both = {"91x64": {2: 0.9, 4: 0.8}, "160x100": {2: 0.75, 4: 0.71}}
    assert gate_check(both) == (True, [])
    one_lattice = {"91x64": {2: 0.9, 4: 0.8}}
    ok, reasons = gate_check(one_lattice)
    assert not ok and reasons == ["160x100: no 2-node run at 1e8 rows", "160x100: no 4-node run at 1e8 rows"]
    assert gate_check(one_lattice, ["91x64"]) == (True, [])
    ok, reasons = gate_check({"91x64": {2: 0.9}}, ["91x64"])
    assert not ok and reasons == ["91x64: no 4-node run at 1e8 rows"]
    ok, reasons = gate_check({"91x64": {2: 0.9, 4: 0.69}}, ["91x64"])
    assert not ok and reasons == ["91x64: 4-node efficiency 0.690 < 0.7"]
    assert gate_check({}, []) == (False, [])


def test_gate_file_from_one_lattice_sweep_fails_a_two_lattice_job(tmp_path):
    g = tmp_path / "gate.json"
    # JSON turns the node counts into strings; a one-lattice sweep that "passed" for itself.
    g.write_text(json.dumps(dict(threshold=0.7, efficiency={"91x64": {"2": 0.8, "4": 0.8}}, passed=True)))
    assert summarize.main(["--check-gate", str(g), "--lattices", "91x64", "160x100"]) == 1
    assert summarize.main(["--check-gate", str(g), "--lattices", "91x64"]) == 0


def test_projection_label_is_the_recorded_baseline():
    r = dict(environment=dict(lscpu={"Model name": "Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz"},
                              slurm={"SLURM_JOB_PARTITION": "gp-default"}), nodes=["gp-n0013"])
    assert summarize.baseline_label(r) == "Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, gp-default, gp-n0013"
    r = dict(environment=dict(lscpu={}, slurm={}, platform="macOS-arm64"), nodes=[])
    assert summarize.baseline_label(r) == "macOS-arm64, no partition, unknown node"


spec_so = importlib.util.spec_from_file_location("scale_out_run", KIT / "scale_out.py")
scale_out = importlib.util.module_from_spec(spec_so)
spec_so.loader.exec_module(scale_out)


class FakeComm:
    """allgather over ranks whose (rank, host, cpus) tuples are fixed in advance."""

    def __init__(self, every):
        self.every = every

    def allgather(self, mine):
        assert mine in self.every
        return list(self.every)


def test_affinity_check_passes_with_full_cpu_set(monkeypatch):
    monkeypatch.setattr(scale_out.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    assert scale_out.affinity_check(32, 0) == 32
    host = scale_out.platform.node()
    assert scale_out.affinity_check(32, 0, FakeComm([(0, host, 32), (1, "gp-n0016", 32)])) == 32


def test_affinity_check_fails_on_one_cpu(monkeypatch):
    # Job 8413: every rank bound to CPU 0 with 32 numba threads.
    monkeypatch.setattr(scale_out.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    with pytest.raises(RuntimeError, match=r"rank 0 on .*: affinity 1 CPUs < 32 threads"):
        scale_out.affinity_check(32, 0)


def test_affinity_check_fails_on_every_rank_when_another_rank_is_short(monkeypatch):
    # This rank is fine; rank 1 on another node is not. All ranks see the gathered list and stop.
    monkeypatch.setattr(scale_out.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    host = scale_out.platform.node()
    comm = FakeComm([(0, host, 32), (1, "gp-n0016", 1)])
    with pytest.raises(RuntimeError, match=r"rank 1 on gp-n0016: affinity 1 CPUs < 32 threads") as e:
        scale_out.affinity_check(32, 0, comm)
    assert "rank 0" not in str(e.value)


def test_affinity_check_skipped_without_sched_getaffinity(monkeypatch):
    monkeypatch.delattr(scale_out.os, "sched_getaffinity", raising=False)
    assert scale_out.affinity_check(32, 0) == -1


def _run(tmp_path, name, ranks, nodes, affinity, rate, lattice="91x64"):
    d = tmp_path / "runs" / name
    d.mkdir(parents=True)
    r = dict(settings=dict(lattice=lattice, n=int(1e8)), ranks=ranks, n_nodes=nodes, threads_per_rank=32,
             total_threads=32 * ranks, rows_per_second=rate, training_wall_seconds=5e8 / rate,
             first_epoch_rows_per_second=rate, later_epoch_rows_per_second=rate, reduce_seconds_total=0.0,
             io_wait_total=0.0, peak_rss_bytes_per_rank=[2**30] * ranks, epochs=[dict(rank_pass_seconds=None)],
             per_rank=[dict(affinity=list(range(affinity)))] * ranks, environment={}, nodes=[])
    (d / "run.json").write_text(json.dumps(r))
    (d / "progress.log").write_text("")


def test_summary_excludes_under_pinned_runs_and_lists_unfinished(tmp_path):
    # Job 8413 shape: 1-node baseline fine, 2- and 4-node ranks pinned to one CPU, a killed run.
    _run(tmp_path, "91x64-n1e8-nodes1", 1, 1, 32, 2.7e6)
    _run(tmp_path, "91x64-n1e8-nodes2", 2, 2, 1, 0.26e6)
    _run(tmp_path, "91x64-n1e8-nodes4", 4, 4, 32, 9.7e6)
    killed = tmp_path / "runs" / "160x100-n1e8-nodes2"
    killed.mkdir()
    (killed / "progress.log").write_text("epoch 1/5\n")
    assert summarize.main(["--out", str(tmp_path / "summary"), str(tmp_path)]) == 0
    md = (tmp_path / "summary" / "summary.md").read_text()
    gate = json.loads((tmp_path / "summary" / "gate.json").read_text())
    assert "| 2 x 32 | 1 UNDER-PINNED |" in md
    assert "Unfinished (progress.log but no run.json, e.g. killed by the time limit): 160x100-n1e8-nodes2" in md
    assert set(gate["efficiency"]["91x64"]) == {"4"}
    assert abs(gate["efficiency"]["91x64"]["4"] - 9.7 / (4 * 2.7)) < 1e-12
    assert "91x64: no 2-node run at 1e8 rows" in gate["reasons"]
