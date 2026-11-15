"""Regression tests for drivers and scripts when ``resource`` / ``peak_rss_bytes()`` is absent."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import h5py
import numba
import numpy as np

from aweSOM import Lattice, SOMModel, stream
from production_pool_test import inputs, pool

ROOT = Path(__file__).resolve().parents[1]


def _load_fresh(name: str, path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_train_stream_lagoon_tolerates_none_peak_rss(tmp_path, monkeypatch, capsys):
    lagoon = _load_fresh(
        "_train_stream_lagoon_no_rss",
        ROOT / "examples/slurm_scripts/train_stream_lagoon.py",
        monkeypatch,
    )
    monkeypatch.setattr(lagoon, "peak_rss_bytes", lambda: None)

    feat_path = tmp_path / "features.h5"
    rng = np.random.default_rng(7)
    with h5py.File(feat_path, "w") as h:
        h["features"] = rng.normal(size=(48, 3)).astype(np.float32)
        h["names"] = np.array(["f0", "f1", "f2"], dtype="S")

    out_dir = tmp_path / "out"
    for k, v in {
        "AWESOM_FEATURES": str(feat_path),
        "AWESOM_OUT": str(out_dir),
        "XDIM": "3",
        "YDIM": "2",
        "EPOCHS": "2",
        "SAMPLE_ROWS": "24",
        "SAMPLE_BLOCK": "8",
        "CHUNK": "16",
        "SEED": "42",
    }.items():
        monkeypatch.setenv(k, v)

    lagoon.main()
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip().startswith("{")]
    assert [rec["stage"] for rec in lines] == ["open", "sample", "train", "labels", "done"]
    assert all(rec["peak_rss_gb"] is None for rec in lines)
    assert (out_dir / "model.h5").exists()
    assert (out_dir / "labels.h5").exists()


def test_production_pool_tolerates_none_peak_rss(tmp_path, monkeypatch):
    paths = inputs(tmp_path)
    monkeypatch.setattr(stream, "peak_rss_bytes", lambda: None)
    monkeypatch.setattr(pool, "execution_info", lambda: dict(test_execution=True))
    monkeypatch.setattr(pool, "run_wave", lambda jobs: [pool.fit(*j) for j in jobs])

    base = [
        "--inputs", *map(str, paths),
        "--anchor", "1",
        "--held-out", "3",
        "--rows-per-snapshot", "60",
        "--rows-per-epoch", "60",
        "--lattices", "3x2",
        "--seeds", "42", "43",
        "--epochs", "1",
        "--chunk", "29",
        "--min-free-gb", "0",
        "--target-clusters", "2",
        "--threads-per-seed", "1",
    ]
    prev_threads = numba.get_num_threads()
    try:
        pool.main([*base, "--out", str(tmp_path / "prepared"), "--pool-only"])
        prepared = tmp_path / "prepared" / "results"
        fit_dirs = []
        for seed in (42, 43):
            out = tmp_path / f"fit-{seed}"
            pool.main([*base, "--out", str(out), "--reuse-pool", str(prepared), "--fit", "3x2", str(seed)])
            res = out / "results"
            fit_rec = json.loads((res / "fit-results.json").read_text())["models"][0]
            assert fit_rec["peak_rss_bytes"] is None
            fit_dirs.append(res)

        collected = tmp_path / "collected"
        pool.main([*base, "--out", str(collected), "--reuse-pool", str(prepared), "--collect", *map(str, fit_dirs)])
        manifest = json.loads((collected / "results" / "freeze-manifest.json").read_text())
        assert manifest["parent_peak_rss_bytes"] is None
        assert manifest["concurrent_peak_rss_upper_bound_bytes"] is None
        assert all(m["peak_rss_bytes"] is None for m in manifest["models"])

        # Also verify mixed provenance: parent has RSS, collected fits have None.
        monkeypatch.setattr(stream, "peak_rss_bytes", lambda: 500_000)
        mixed = tmp_path / "collected-mixed"
        pool.main([*base, "--out", str(mixed), "--reuse-pool", str(prepared), "--collect", *map(str, fit_dirs)])
        mixed_manifest = json.loads((mixed / "results" / "freeze-manifest.json").read_text())
        assert mixed_manifest["parent_peak_rss_bytes"] == 500_000
        assert mixed_manifest["concurrent_peak_rss_upper_bound_bytes"] is None
    finally:
        numba.set_num_threads(prev_threads)


def test_scale_out_imports_and_runs_without_resource(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "resource", None)
    fresh = _load_fresh(
        "_scale_out_without_resource",
        ROOT / "examples/slurm_scripts/scale_out/scale_out.py",
        monkeypatch,
    )
    assert fresh.resource is None
    assert fresh.peak_rss() is None

    monkeypatch.setattr(fresh, "GEN_BLOCK", 64)
    out = tmp_path / "so_run"
    prev_threads = numba.get_num_threads()
    try:
        rc = fresh.main([
            "--out", str(out),
            "--n", "128",
            "--lattice", "4x3",
            "--epochs", "1",
            "--threads", "1",
            "--r-min", "1",
            "--chunk", "64",
            "--warmup", "0",
        ])
    finally:
        numba.set_num_threads(prev_threads)
    assert rc == 0
    rank0 = json.loads((out / "rank0.json").read_text())
    run = json.loads((out / "run.json").read_text())
    assert rank0["peak_rss_bytes"] is None
    assert rank0["rss_after_generate_bytes"] is None
    assert run["peak_rss_bytes_per_rank"] == [None]


def test_label_grid_stream_imports_and_runs_without_resource(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "resource", None)
    fresh = _load_fresh(
        "_label_grid_stream_without_resource",
        ROOT / "examples/grmhd-bbh/label_grid_stream.py",
        monkeypatch,
    )
    assert fresh.resource is None

    rng = np.random.default_rng(11)
    lat = Lattice(6, 5)
    lat.lattice = rng.normal(size=(30, 3))
    lat.umat = lat.compute_umat()
    with contextlib.redirect_stdout(io.StringIO()):
        lat.assign_cluster_to_lattice(target_clusters=2)
    model = SOMModel.from_lattice(lat, scaled_training_data=lat.lattice)
    model_path = tmp_path / "model.h5"
    model.save(model_path)

    x = lat.lattice.astype(np.float32)
    grid_path = tmp_path / "grid.h5"
    with h5py.File(grid_path, "w") as h:
        h["features"] = x
        h["level"] = np.full(len(x), 5, dtype=np.int32)
        h["names"] = np.array(model.feature_names, dtype="S")

    out_path = tmp_path / "labels.npy"
    monkeypatch.setattr(
        sys,
        "argv",
        ["label_grid_stream.py", "--grid", str(grid_path), "--model", str(model_path), "--output", str(out_path)],
    )
    fresh.main()
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["peak_GB"] is None
    np.testing.assert_array_equal(np.load(out_path), model.predict(x, backend="patch"))
