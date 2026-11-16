"""Tests for examples/grmhd-bbh/scale-out/replay_top_two.py against production_pool selection."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from aweSOM import SOMModel, Scaler
from aweSOM.npy_parts import part_filename, parts_dir_for, write_npy_manifest
from aweSOM.stream import NpyWriter

ROOT = Path(__file__).resolve().parents[1]
REPLAY_PATH = ROOT / "examples/grmhd-bbh/scale-out/replay_top_two.py"
POOL_PATH = ROOT / "examples/grmhd-bbh/production_pool.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


replay = _load("replay_top_two", REPLAY_PATH)
pool = _load("production_pool_for_replay", POOL_PATH)


def _write_sharded_labels(base_npy: Path, arr: np.ndarray) -> None:
    pdir = parts_dir_for(base_npy)
    pdir.mkdir(parents=True, exist_ok=False)
    mid = len(arr) // 2
    entries = []
    for r, (a, b) in enumerate(((0, mid), (mid, len(arr)))):
        fname = part_filename(r, 2)
        mm = np.lib.format.open_memmap(pdir / fname, mode="w+", dtype=arr.dtype, shape=(b - a,))
        del mm
        w = NpyWriter(pdir / fname)
        w[0 : b - a] = arr[a:b]
        w.close(sync=False)
        entries.append(
            dict(
                rank=r,
                file=fname,
                start=a,
                stop=b,
                rows=b - a,
                bytes=int((b - a) * arr.dtype.itemsize),
                crc32=int(w.crc32),
            )
        )
    write_npy_manifest(pdir, dtype=arr.dtype, shape=arr.shape, parts=entries, sync=False)


def _build_synthetic_fixture(results: Path, seeds: tuple[int, ...]) -> tuple[np.ndarray, list[str], dict]:
    """Create pool.h5 + label files (and fit .h5/.json records for group_selection comparison).

    Row blocks (120 rows total):
      block A (0..30):  strongest sheet signature (log_chi_B=4.0, log_grad_ln_b2=3.0, log_beta_inv=-3.0)
      block B (30..60): second sheet signature    (log_chi_B=2.5, log_grad_ln_b2=2.0, log_beta_inv=-2.0)
      block C (60..90): non-sheet background      (log_chi_B=-2.0, log_grad_ln_b2=-2.0, log_beta_inv=2.0)
      block D (90..120): non-sheet background     (log_chi_B=-3.0, log_grad_ln_b2=-3.0, log_beta_inv=3.0)

    On 91x64:
      seed 42 splits block A across clusters 0 and 1 (15 rows each), puts block B in cluster 2 (30 rows),
      so E(log_chi_B) ranks 0, 1, 2 with 0 highest, while seed 43 puts block A in background-mixed cluster 0
      (non-mutual with 42:0) and puts block B cleanly in cluster 1 -> top-1 fails mutual check, top-2 succeeds
      on 42:2 <-> 43:1!
    On 48x34:
      every seed maps block A -> cluster 1, block B -> cluster 2, blocks C,D -> 0,3 -> top-1 (cluster 1)
      and top-2 (cluster 2) both reproduce identically under both rules.
    """
    results.mkdir(parents=True, exist_ok=True)
    names = list(pool.SEVEN_FEATURE_NAMES)
    rng = np.random.default_rng(109)
    n = 120
    raw = rng.normal(scale=0.05, size=(n, len(names))).astype(np.float32)

    chi_idx = names.index("log_chi_B")
    grad_idx = names.index("log_grad_ln_b2")
    beta_idx = names.index("log_beta_inv")

    # Block A: rows 0..30
    raw[0:30, chi_idx] += 4.0
    raw[0:30, grad_idx] += 3.0
    raw[0:30, beta_idx] -= 3.0
    # Block B: rows 30..60
    raw[30:60, chi_idx] += 2.5
    raw[30:60, grad_idx] += 2.0
    raw[30:60, beta_idx] -= 2.0
    # Block C & D: rows 60..120
    raw[60:90, chi_idx] -= 2.0
    raw[60:90, grad_idx] -= 2.0
    raw[60:90, beta_idx] += 2.0
    raw[90:120, chi_idx] -= 3.0
    raw[90:120, grad_idx] -= 3.0
    raw[90:120, beta_idx] += 3.0

    with h5py.File(results / "pool.h5", "w") as f:
        f.create_dataset("names", data=np.array(names, dtype="S"))
        f.create_dataset("raw", data=raw)

    lab_by_shape: dict[str, dict[int, np.ndarray]] = {"91x64": {}, "48x34": {}}

    # 91x64: top-two rescues non-mutual first candidate
    l42 = np.empty(n, dtype=np.uint8)
    l42[0:10] = 0    # top-1 candidate on seed 42 (highest E_log_chi_B), only 10 rows
    l42[10:30] = 3   # diluted with C below so cluster 3 is NOT a candidate
    l42[30:60] = 2   # second candidate on seed 42 (30 rows of block B)
    l42[60:95] = 3
    l42[95:120] = 1
    lab_by_shape["91x64"][42] = l42

    l43 = np.empty(n, dtype=np.uint8)
    l43[0:10] = 2    # mixed with block D below so 43's top-1 is elsewhere
    l43[10:22] = 0   # top-1 candidate on seed 43 (rows 10..22 of block A -> disjoint from 42's cluster 0!)
    l43[22:30] = 2
    l43[30:60] = 1   # second candidate on seed 43 (exact match with 42's cluster 2!)
    l43[60:90] = 3
    l43[90:120] = 2
    lab_by_shape["91x64"][43] = l43

    if 44 in seeds:
        l44 = np.empty(n, dtype=np.uint8)
        l44[0:10] = 3
        l44[10:20] = 3
        l44[20:30] = 0  # top-1 candidate on seed 44 (rows 20..30 -> disjoint from 42:0 and 43:0!)
        l44[30:60] = 2  # second candidate on seed 44 (exact match with 42:2 and 43:1!)
        l44[60:90] = 1
        l44[90:120] = 3
        lab_by_shape["91x64"][44] = l44

    # 48x34: mutual first and second picks across all seeds
    for idx_s, s in enumerate(seeds):
        lab = np.empty(n, dtype=np.uint8)
        c_first = (1 + idx_s) % 4
        c_second = (2 + idx_s) % 4
        c_bg1 = (3 + idx_s) % 4
        c_bg2 = idx_s % 4
        lab[0:30] = c_first
        lab[30:60] = c_second
        lab[60:90] = c_bg1
        lab[90:120] = c_bg2
        lab_by_shape["48x34"][s] = lab

    recs_by_shape: dict[str, dict[int, dict]] = {"91x64": {}, "48x34": {}}
    scaler = Scaler("manual", np.zeros(len(names)), np.ones(len(names)))
    for shape in ("91x64", "48x34"):
        for s in seeds:
            arr = lab_by_shape[shape][s]
            base = results / f"{shape}-s{s}-labels.npy"
            # Mix plain .npy and sharded .parts to verify open_npy integration
            if shape == "91x64" and s == 43:
                _write_sharded_labels(base, arr)
            else:
                np.save(base, arr)
            k = int(arr.max()) + 1
            eff = pool.rank_effects(raw, arr, k)
            cand = pool.sheet_candidates(eff, names)
            tag = f"{shape}-s{s}"
            model = SOMModel(
                xdim=2,
                ydim=2,
                weights=np.zeros((4, len(names))),
                node_cluster=np.arange(4, dtype=np.int32) % k,
                feature_names=names,
                scaler=scaler,
                node_dist_q99=np.ones(4),
                reject_distance=10.0,
                metadata={"tag": tag},
            )
            model.save(results / f"{tag}.h5")
            rec = dict(
                tag=tag,
                seed=s,
                lattice=shape,
                cluster_effects=eff.tolist(),
                sheet_candidates=cand,
            )
            (results / f"{tag}.json").write_text(json.dumps(pool.plain(rec)))
            recs_by_shape[shape][s] = rec

    return raw, names, recs_by_shape


def test_replay_top_two_two_seed_matches_production_pool(tmp_path, monkeypatch):
    results = tmp_path / "k_two_seed"
    _, _, recs_by_shape = _build_synthetic_fixture(results, seeds=(42, 43))
    monkeypatch.setattr(replay, "SEEDS", (42, 43))
    monkeypatch.setattr(replay, "GROUPS", ((42, 43),))

    rows = replay.main(results)
    assert len(rows) == 2
    by_shape = {r["lattice"]: r for r in rows}

    # On 91x64, previous rule fails ('none') while top-two selects 42:2 <-> 43:1
    assert by_shape["91x64"]["previous_status"] == "none"
    assert by_shape["91x64"]["top_two_status"] == "selected"
    assert by_shape["91x64"]["top_two_picks"] == {42: [2], 43: [1]}
    assert by_shape["91x64"]["same_first"] is False

    # On 48x34, both rules agree on first and second picks
    assert by_shape["48x34"]["previous_status"] == "selected"
    assert by_shape["48x34"]["top_two_status"] == "selected"
    assert by_shape["48x34"]["same_first"] is True

    # Compare against production_pool.group_selection on the exact same fit records
    for shape in ("91x64", "48x34"):
        recs = [dict(recs_by_shape[shape][s]) for s in (42, 43)]
        gs = pool.group_selection(results, recs, chunk=64)
        got = pool.plain(by_shape[shape])
        assert got["top_two_picks"] == gs["picks"]
        assert got["top_two_status"] == gs["audit"]["status"]
        assert got["previous_status"] == gs["audit"]["approved_rule_outcome"]["status"]
        assert got["previous_picks"] == gs["audit"]["approved_rule_outcome"]["selected"]


def test_replay_top_two_default_cli_and_bad_names(tmp_path):
    results = tmp_path / "k_three_seed"
    _, _, recs_by_shape = _build_synthetic_fixture(results, seeds=(42, 43, 44))

    # Run via CLI entrypoint (__main__)
    env = dict(PYTHONPATH=str(ROOT / "src"))
    env.update({k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"})
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, str(REPLAY_PATH), str(results)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
         check=True,
    )
    assert "same first pick in 4 of 8 seed groups" in proc.stdout

    rows = replay.main(results)
    assert len(rows) == 8
    for row in rows:
        shape = row["lattice"]
        grp = tuple(row["seeds"])
        # Fresh copy of fit records for this group
        recs = [json.loads(json.dumps(pool.plain(recs_by_shape[shape][s]))) for s in grp]
        # Re-save fresh h5/json per group check in a sub-dir or verify via select_group + group_selection
        if grp == (42, 43, 44):
            gs = pool.group_selection(results, recs, chunk=64)
            assert pool.plain(row)["top_two_picks"] == gs["picks"]
            assert row["top_two_status"] == gs["audit"]["status"]

    # Check feature-name mismatch guard
    bad_dir = tmp_path / "bad_names"
    bad_dir.mkdir()
    with h5py.File(bad_dir / "pool.h5", "w") as f:
        f.create_dataset("names", data=np.array([b"wrong"]))
        f.create_dataset("raw", data=np.zeros((4, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="SEVEN_FEATURE_NAMES"):
        replay.main(bad_dir)
