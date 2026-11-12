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


def test_projection_label_takes_the_cpu_from_the_option():
    # run.json records no CPU model; old files still carry lscpu/platform, which are ignored.
    old = dict(environment=dict(lscpu={"Model name": "Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz"},
                                platform="Linux-x86_64", slurm={"SLURM_JOB_PARTITION": "gp-default"}),
               nodes=["gp-n0013"])
    assert summarize.baseline_label(old) == "unknown CPU, gp-default, gp-n0013"
    assert summarize.baseline_label(old, "Xeon Gold 6130") == "Xeon Gold 6130, gp-default, gp-n0013"
    new = dict(environment=dict(slurm={}), nodes=[])
    assert summarize.baseline_label(new) == "unknown CPU, no partition, unknown node"


def test_node_name_comes_from_slurm_only(monkeypatch):
    monkeypatch.setenv("SLURMD_NODENAME", "test-node")
    assert scale_out.node_name() == "test-node"
    monkeypatch.delenv("SLURMD_NODENAME")
    assert scale_out.node_name() == "local"


class NamesComm:
    """Get_size and allgather over fixed per-rank SLURMD_NODENAME values (None = unset)."""

    def __init__(self, names):
        self.names = names

    def Get_size(self):
        return len(self.names)

    def allgather(self, mine):
        return list(self.names)


def test_multi_rank_run_needs_slurm_node_names_on_every_rank():
    scale_out.require_node_names(None)                      # serial run: "local" is fine
    scale_out.require_node_names(NamesComm([None]))         # one rank: fine
    scale_out.require_node_names(NamesComm(["gg-a-001", "gg-a-002"]))
    with pytest.raises(RuntimeError, match=r"SLURMD_NODENAME is unset on ranks \[1\]"):
        scale_out.require_node_names(NamesComm(["gg-a-001", None]))


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
    host = scale_out.node_name()
    assert scale_out.affinity_check(32, 0, FakeComm([(0, host, 32), (1, "gp-n0016", 32)])) == 32


def test_affinity_check_fails_on_one_cpu(monkeypatch):
    # Job 8413: every rank bound to CPU 0 with 32 numba threads.
    monkeypatch.setattr(scale_out.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    with pytest.raises(RuntimeError, match=r"rank 0 on .*: affinity 1 CPUs < 32 threads"):
        scale_out.affinity_check(32, 0)


def test_affinity_check_fails_on_every_rank_when_another_rank_is_short(monkeypatch):
    # This rank is fine; rank 1 on another node is not. All ranks see the gathered list and stop.
    monkeypatch.setattr(scale_out.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    host = scale_out.node_name()
    comm = FakeComm([(0, host, 32), (1, "gp-n0016", 1)])
    with pytest.raises(RuntimeError, match=r"rank 1 on gp-n0016: affinity 1 CPUs < 32 threads") as e:
        scale_out.affinity_check(32, 0, comm)
    assert "rank 0" not in str(e.value)


def test_affinity_check_skipped_without_sched_getaffinity(monkeypatch):
    monkeypatch.delattr(scale_out.os, "sched_getaffinity", raising=False)
    assert scale_out.affinity_check(32, 0) == -1


def _run(tmp_path, name, ranks, nodes, affinity, rate, lattice="91x64", restored=None):
    d = tmp_path / "runs" / name
    d.mkdir(parents=True)
    r = dict(settings=dict(lattice=lattice, n=int(1e8)), ranks=ranks, n_nodes=nodes, threads_per_rank=32,
             total_threads=32 * ranks, rows_per_second=rate, training_wall_seconds=5e8 / rate,
             first_epoch_rows_per_second=rate, later_epoch_rows_per_second=rate, reduce_seconds_total=0.0,
             io_wait_total=0.0, peak_rss_bytes_per_rank=[2**30] * ranks, epochs=[dict(rank_pass_seconds=None)],
             per_rank=[dict(affinity=list(range(affinity)), affinity_restored=restored)] * ranks,
             environment={}, nodes=[])
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


class FakeAffinity:
    """sched_getaffinity / sched_setaffinity of one thread; ``allowed`` is the cgroup."""

    def __init__(self, now, allowed):
        self.now, self.allowed, self.calls = set(now), set(allowed), []

    def get(self, pid):
        assert pid == 0
        return set(self.now)

    def put(self, pid, cpus):
        self.calls.append(set(cpus))
        if not set(cpus) <= self.allowed:
            raise OSError(22, "Invalid argument")
        self.now = set(cpus)


def test_restore_affinity_gives_back_the_saved_set(capsys):
    # Job 8444: 32 CPUs before MPI_Init, CPU 0 after; the cgroup still allows all 32.
    fake = FakeAffinity({0}, range(32))
    rec = scale_out.restore_affinity(frozenset(range(32)), 1, get=fake.get, put=fake.put)
    assert rec == dict(before=32, after_init=1, cpus_after_init=[0], after_restore=32)
    assert fake.now == set(range(32)) and fake.calls == [set(range(32))]
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "rank 1 on" in err and "from 32 to 1 CPUs [0]; restored to 32" in err


def test_restore_affinity_does_nothing_if_the_set_did_not_shrink(capsys):
    for now in (range(32), range(16)):  # unchanged; or a half-node rank whose saved set was 16
        fake = FakeAffinity(now, range(32))
        assert scale_out.restore_affinity(frozenset(now), 0, get=fake.get, put=fake.put) is None
        assert fake.calls == []
    assert scale_out.restore_affinity(None, 0, get=fake.get, put=fake.put) is None
    assert capsys.readouterr().err == ""


def test_restore_affinity_records_a_refused_set_and_affinity_check_then_fails(monkeypatch):
    fake = FakeAffinity({0}, {0})  # a cgroup that no longer allows the saved CPUs
    rec = scale_out.restore_affinity(frozenset(range(32)), 0, get=fake.get, put=fake.put)
    assert rec["after_restore"] == 1 and rec["error"].startswith("OSError")
    monkeypatch.setattr(scale_out.os, "sched_getaffinity", fake.get, raising=False)
    with pytest.raises(RuntimeError, match="after MPI_Init and the restore"):
        scale_out.affinity_check(32, 0)


def test_summary_marks_restored_runs_and_keeps_healthy_numbers(tmp_path):
    _run(tmp_path, "91x64-n1e8-nodes1", 1, 1, 32, 2.7e6)
    _run(tmp_path, "91x64-n1e8-nodes2", 2, 2, 32, 5.3e6,
         restored=dict(before=32, after_init=1, after_restore=32, cpus_after_init=[0]))
    assert summarize.main(["--out", str(tmp_path / "summary"), str(tmp_path)]) == 0
    md = (tmp_path / "summary" / "summary.md").read_text()
    assert "| 1 x 32 | 32 |" in md
    assert "| 2 x 32 | 32 (restored) |" in md
    gate = json.loads((tmp_path / "summary" / "gate.json").read_text())
    assert abs(gate["efficiency"]["91x64"]["2"] - 5.3 / (2 * 2.7)) < 1e-12


def _bash(script, *args):
    import subprocess

    return subprocess.run(["bash", "-c", f'. "{KIT}/common.sh"; {script}', *args], capture_output=True, text=True)


def test_launch_cmd_turns_psm_affinity_off_and_keeps_the_slurm_binding():
    out = _bash('LAUNCH=srun-pmix; launch_cmd 2 1 32 none; printf "%s\\n" "${CMD[@]}"')
    assert out.returncode == 0, out.stderr
    cmd = out.stdout.splitlines()
    srun = cmd.index("srun")
    assert cmd[0] == "env"
    assert {"HFI_NO_CPUAFFINITY=1", "PSM3_NO_CPUAFFINITY=1", "IPATH_NO_CPUAFFINITY=1"} <= set(cmd[:srun])
    assert not any("binding_policy" in x for x in cmd)
    assert cmd[srun + 1:] == ["--mpi=pmix", "--nodes=2", "--ntasks=2", "--ntasks-per-node=1", "--cpus-per-task=32",
                              "--distribution=block:block", "--cpu-bind=verbose,cores"]
    out = _bash('LAUNCH=mpirun; MPIRUN=mpirun; NODES=(h1 h2); launch_cmd 2 1 32 none; printf "%s\\n" "${CMD[@]}"')
    assert out.returncode == 0, out.stderr
    cmd = out.stdout.splitlines()
    assert "HFI_NO_CPUAFFINITY=1" in cmd[:cmd.index("mpirun")]
    for v in ("HFI_NO_CPUAFFINITY", "PSM3_NO_CPUAFFINITY", "IPATH_NO_CPUAFFINITY"):
        assert cmd[cmd.index(v) - 1] == "-x"


def test_affinity_report_warns_but_passes_when_the_restore_fired(tmp_path):
    a = "AFFINITY rank={r} size=2 host=gp-n001{r} before_init=32 after_init={a} restored={x}\n"
    f = tmp_path / "affinity-nodes2.txt"
    f.write_text(a.format(r=0, a=32, x=0) + a.format(r=1, a=32, x=0))
    out = _bash('affinity_report nodes2 "$0" 2 32', str(f))
    assert out.returncode == 0 and "WARNING" not in out.stdout + out.stderr
    assert out.stdout.strip() == "affinity pre-flight nodes2: 2 rank(s), each >= 32 CPUs after MPI_Init"
    f.write_text(a.format(r=0, a=1, x=32) + a.format(r=1, a=32, x=0))
    out = _bash('affinity_report nodes2 "$0" 2 32', str(f))
    assert out.returncode == 0
    warn = [x for x in out.stdout.splitlines() if x.startswith("WARNING")]
    assert len(warn) == 1 and "on 1 rank(s)" in warn[0] and "HFI_NO_CPUAFFINITY=1" in warn[0]
    f.write_text(a.format(r=0, a=1, x=1) + a.format(r=1, a=32, x=0))
    out = _bash('affinity_report nodes2 "$0" 2 32', str(f))
    assert out.returncode == 1 and "FAILED" in out.stderr


def _verdict(tmp_path, text, nr, cpus):
    import subprocess

    f = tmp_path / "affinity.txt"
    f.write_text(text)
    out = subprocess.run(["bash", "-c", f'. "{KIT}/common.sh"; affinity_verdict "$0" "$1" "$2"', str(f), str(nr),
                          str(cpus)], capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    assert lines[-1].startswith("RESTORED ")
    return lines[:-1], int(lines[-1].split()[1])


def test_preflight_verdict_accepts_restored_ranks_and_flags_short_ones(tmp_path):
    a = "AFFINITY rank={r} size=2 host=gp-n001{r} before_init=32 after_init={a} restored={x}\n"
    srun_noise = "cpu-bind=MASK - gp-n0015, task  0  0 [2245488]: mask 0xffffffff set\n"
    # 8444 with the workaround: both ranks dropped to 1 CPU and got 32 back.
    assert _verdict(tmp_path, srun_noise + a.format(r=0, a=1, x=32) + a.format(r=1, a=1, x=32), 2, 32) == ([], 2)
    # Nothing to restore.
    assert _verdict(tmp_path, a.format(r=0, a=32, x=0) + a.format(r=1, a=32, x=0), 2, 32) == ([], 0)
    # The restore fired but came back short; a missing rank.
    bad, fired = _verdict(tmp_path, a.format(r=0, a=1, x=1) + a.format(r=1, a=32, x=0), 2, 32)
    assert bad == ["rank 0 on gp-n0010: 1 CPUs after MPI_Init and the restore (32 before), needs 32"] and fired == 1
    bad, _ = _verdict(tmp_path, a.format(r=0, a=32, x=0), 2, 32)
    assert bad == ["1 of 2 ranks reported"]
    # Old lines without restored= (kit 0f41635) still parse.
    old = "AFFINITY rank=0 size=1 host=h before_init=32 after_init=1\n"
    assert _verdict(tmp_path, old, 1, 32) == (["rank 0 on h: 1 CPUs after MPI_Init (32 before), needs 32"], 0)


def test_gp_only_surveys_skip_or_refuse_on_rit_rc(tmp_path, monkeypatch):
    import subprocess
    import sys

    spec_p = importlib.util.spec_from_file_location("scale_out_probe", KIT / "probe.py")
    probe = importlib.util.module_from_spec(spec_p)
    spec_p.loader.exec_module(probe)

    rc_envs = [
        {"SLURM_CLUSTER_NAME": "tigris"},
        {"SLURM_CLUSTER_NAME": "SPORC"},
        {"SLURM_CLUSTER_NAME": "sporc"},
        {"SLURM_CLUSTER_NAME": "", "OUT": "/shared/rc/awesom-grmhd/out"},
        {"SLURM_CLUSTER_NAME": "", "KIT": "/home/thccrg/kit"},
        {"SLURM_CLUSTER_NAME": "", "SLURM_SUBMIT_DIR": "/home/thccrg"},
        {"SLURM_CLUSTER_NAME": "", "OUT": "/.autofs/shared/rc/awesom-grmhd/out"},
    ]
    base_env = {k: v for k, v in scale_out.os.environ.items()
                if k not in {"SLURM_CLUSTER_NAME", "SLURM_SUBMIT_DIR", "KIT", "OUT"}}

    for overrides in rc_envs:
        env = {**base_env, **overrides}
        assert subprocess.run(["bash", "-c", f'. "{KIT}/common.sh"; on_tigris'],
                              cwd=tmp_path, env=env).returncode == 0
        for sbatch in ("probe.sbatch", "gpu_probe.sbatch", "bind_probe.sbatch"):
            res = subprocess.run(["bash", str(KIT / sbatch)], cwd=tmp_path, env=env,
                                 capture_output=True, text=True)
            assert res.returncode == 2 and "refusing to run on RIT RC" in res.stderr
        bp = subprocess.run([sys.executable, str(KIT / "bind_probe.py"), "--no-mpi"],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
        assert bp.returncode != 0 and "refusing to run on RIT RC" in bp.stderr
        for k in ("SLURM_CLUSTER_NAME", "SLURM_SUBMIT_DIR", "KIT", "OUT"):
            monkeypatch.delenv(k, raising=False)
        for k, v in overrides.items():
            monkeypatch.setenv(k, v)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit, match="refusing to run on RIT RC"):
            probe.refuse_on_tigris()

    for k in ("SLURM_CLUSTER_NAME", "SLURM_SUBMIT_DIR", "KIT", "OUT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "gp")
    with pytest.raises(SystemExit, match="refusing to run on RIT RC"):
        probe.main(["env", "--json", "/home/thccrg/probe-env.json"])

    gp_env = {**base_env, "SLURM_CLUSTER_NAME": "gp", "OUT": "/home/thccrg2/out"}
    assert subprocess.run(["bash", "-c", f'. "{KIT}/common.sh"; on_tigris'],
                          cwd=tmp_path, env=gp_env).returncode == 1
    monkeypatch.setenv("OUT", "/home/thccrg2/out")
    probe.refuse_on_tigris()
    monkeypatch.setenv("PWD", "/home/thccrg/work")
    with pytest.raises(SystemExit, match="refusing to run on RIT RC"):
        probe.refuse_on_tigris()
