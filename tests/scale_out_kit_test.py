"""examples/slurm_scripts/scale_out: the 1e9 gate and the projection label of summarize.py."""
import importlib.util
import json
from pathlib import Path

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
