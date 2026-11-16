"""End-to-end 1-rank and 2-rank MPI tests for examples/slurm_scripts/scale_out/scale_out.py."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/slurm_scripts/scale_out/scale_out.py"

spec = importlib.util.spec_from_file_location("scale_out_mod", SCRIPT)
scale_out = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scale_out)


def _mpirun() -> str | None:
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return None
    return shutil.which("mpirun")


MPIRUN = _mpirun()
needs_mpi = pytest.mark.skipif(MPIRUN is None, reason="mpi4py or mpirun not available")


def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Max relative difference |a - b| / max(|a|, |b|, 1e-8)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)
    return float(np.max(np.abs(a - b) / denom))


def test_launcher_rank_env_precedence(monkeypatch):
    for key in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID"):
        monkeypatch.delenv(key, raising=False)
    assert scale_out._launcher_rank() == 0

    monkeypatch.setenv("SLURM_PROCID", "3")
    assert scale_out._launcher_rank() == 3
    monkeypatch.setenv("PMIX_RANK", "not-a-digit")
    assert scale_out._launcher_rank() == 3
    monkeypatch.setenv("PMIX_RANK", "2")
    assert scale_out._launcher_rank() == 2
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    assert scale_out._launcher_rank() == 1


def test_gmm_and_owned_rows_unit(tmp_path, monkeypatch):
    gmm = scale_out.GMM(SCRIPT.parent / "gmm_bbh7.npz")
    b0 = gmm.block(20261103, 0)
    b1 = gmm.block(20261103, 1)
    assert b0.shape == (scale_out.GEN_BLOCK, 7) and b0.dtype == np.float32
    ix = np.array([0, 17, scale_out.GEN_BLOCK - 1, scale_out.GEN_BLOCK + 5], dtype=np.int64)
    got = gmm.rows(20261103, ix, threads=2)
    expected = np.vstack([b0[[0, 17, scale_out.GEN_BLOCK - 1]], b1[[5]]])
    np.testing.assert_array_equal(got, expected)

    with pytest.raises(ValueError, match="multiple"):
        scale_out.OwnedRows(gmm, 1024, 1024, 20261103, rank=0, size=1, threads=1)

    n = scale_out.GEN_BLOCK + 512
    src0 = scale_out.OwnedRows(gmm, n, scale_out.GEN_BLOCK, 20261103, rank=0, size=2, threads=2)
    src1 = scale_out.OwnedRows(gmm, n, scale_out.GEN_BLOCK, 20261103, rank=1, size=2, threads=2)
    assert src0.owned == [0] and src0.owned_rows == scale_out.GEN_BLOCK
    assert src1.owned == [1] and src1.owned_rows == 512
    with pytest.raises(RuntimeError, match="not owned"):
        src0._read(1)
    np.testing.assert_array_equal(src0.rows(np.array([0, 5])), b0[[0, 5]])
    np.testing.assert_array_equal(src1.rows(np.array([0, 5])), np.zeros((2, 7), dtype=np.float32))

    src_all = scale_out.OwnedRows(gmm, 512, scale_out.GEN_BLOCK, 20261103, rank=0, size=1, threads=1)
    chunks = list(src_all())
    assert len(chunks) == 1 and chunks[0].shape == (512, 7)

    # source_info with a kit-manifest.json present
    fake_kit = tmp_path / "kit"
    fake_kit.mkdir()
    (fake_kit / "kit-manifest.json").write_text(
        json.dumps({"commit": "deadbeef", "branch": "scale-out", "tree_sha256": "abc", "files": {}})
    )
    monkeypatch.setattr(scale_out, "KIT", fake_kit)
    info = scale_out.source_info()
    assert info["commit"] == "deadbeef" and info["package_matches_manifest"] is False


def _run_scale_out(out_dir: Path, *, ranks: int, n: int, epochs: int, warmup: int, resume: bool = False) -> dict:
    cache_root = out_dir.parent / f"numba_cache_r{ranks}"
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        NUMBA_NUM_THREADS="2",
        OMP_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        SLURMD_NODENAME="local-test-node",
        SCALE_CACHE_ROOT=str(cache_root),
    )
    cli_args = [
        "--out",
        str(out_dir),
        "--tag",
        f"test-r{ranks}",
        "--n",
        str(n),
        "--lattice",
        "10x10",
        "--epochs",
        str(epochs),
        "--threads",
        "2",
        "--r-min",
        "3",
        "--chunk",
        str(scale_out.GEN_BLOCK),
        "--groups",
        "4",
        "--warmup",
        str(warmup),
    ]
    if resume:
        cli_args.append("--resume")
    if ranks == 1:
        cmd = [sys.executable, str(SCRIPT), *cli_args]
    else:
        cmd = [
            MPIRUN,
            "-n",
            str(ranks),
            "--oversubscribe",
            # Threaded ranks: keep the full CPU mask (Open MPI binds 2 ranks to one core each).
            "--bind-to",
            "none",
            sys.executable,
            "-m",
            "mpi4py",
            str(SCRIPT),
            "--mpi",
            *cli_args,
        ]
    res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, f"cmd={cmd}\nstdout:\n{res.stdout[-3000:]}\nstderr:\n{res.stderr[-3000:]}"
    weights = np.load(out_dir / "weights.npy")
    run_json = json.loads((out_dir / "run.json").read_text())
    rank_jsons = [json.loads((out_dir / f"rank{r}.json").read_text()) for r in range(ranks)]
    return {"weights": weights, "run": run_json, "ranks": rank_jsons}


@needs_mpi
@pytest.mark.parametrize(
    "n,epochs,warmup",
    [
        (20_000, 3, 1024),
        (2 * scale_out.GEN_BLOCK + 16_384, 2, 0),
    ],
)
def test_scale_out_main_mpi_matches_single_rank(tmp_path, n, epochs, warmup):
    single = _run_scale_out(tmp_path / "r1", ranks=1, n=n, epochs=epochs, warmup=warmup)
    mpi = _run_scale_out(tmp_path / "r2", ranks=2, n=n, epochs=epochs, warmup=warmup)

    assert single["weights"].shape == (100, 7)
    assert mpi["weights"].shape == (100, 7)
    assert single["run"]["ranks"] == 1 and mpi["run"]["ranks"] == 2
    assert len(mpi["ranks"]) == 2
    assert mpi["ranks"][0]["weights_sha256"] == mpi["ranks"][1]["weights_sha256"]
    assert mpi["run"]["weights_sha256"] == hashlib.sha256(mpi["weights"].tobytes()).hexdigest()
    assert (tmp_path / "r2" / "checkpoint.npz").exists()
    assert (tmp_path / "r2" / "progress.log").exists()

    diff = rel_diff(mpi["weights"], single["weights"])
    print(f"scale_out_n{n}_e{epochs}_max_rel_diff = {diff:.6e}")
    assert diff < 1e-4, f"relative max diff {diff} >= 1e-4"
