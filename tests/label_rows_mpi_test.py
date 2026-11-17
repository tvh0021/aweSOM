"""The production_pool driver labelling on every rank: label_rows under mpirun
and through spawned fake ranks, backend resolution, and the .parts outputs."""
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aweSOM import distributed
from aweSOM.inference import best_match_nodes, best_match_nodes_patch

import mpi_label_rows_worker as rows_worker
import mpi_label_worker as worker
import output_guard
from fake_comm import FakeComm
from fake_cupy import StandIn
from production_pool_mpi_test import prepared  # noqa: F401  (module fixture)
from production_pool_test import pool

HERE = Path(__file__).parent


def bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


def _mpirun():
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return None
    return shutil.which("mpirun")


# ------------------------------------------------------- label_rows under mpirun


def single_files(tmp_path, tag, n):
    """np.save of the single-process labels, nodes and distances: today's one-rank bytes."""
    lab, _ = worker.label(worker.data(n), "patch", n_samples=n)
    nodes, d2 = best_match_nodes(worker.lattice(), worker.data(n))
    ref = {"labels": lab, "nodes": nodes.astype(worker.DTYPES["nodes"]), "distances": np.sqrt(d2)}
    paths = {}
    for k, a in ref.items():
        paths[k] = tmp_path / f"np-save-{tag}-{k}.npy"
        np.save(paths[k], a)
    return paths


def test_label_rows_one_rank_bytes(tmp_path):
    """One rank (no comm): the final files are np.save's bytes and no .parts directory is made."""
    pool = rows_worker.load_driver()
    for tag, n in rows_worker.CASES.items():
        rec = rows_worker.label_case(pool, tmp_path, tag, n, None)
        assert rec["layout"] == "single" and rec["manifest_seconds"] == 0.0 and rec["directory_fsync"] is None
        assert [r["part_files"] for r in rec["ranks"]] == [False]
        for k, path in single_files(tmp_path, tag, n).items():
            assert (tmp_path / f"{tag}-{k}.npy").read_bytes() == path.read_bytes(), (tag, k)
    assert not list(tmp_path.glob("*.part*"))


@pytest.mark.skipif(_mpirun() is None, reason="mpi4py or mpirun not available")
@pytest.mark.parametrize("ranks", [2, 3])
def test_mpirun_label_rows_one_writer_per_file(tmp_path, ranks):
    """The driver's label_rows on real MPI: sharded .parts/ outputs open lazily equal to one
    rank's bit for bit (uneven and empty ranges), stitch_npy writes single .npy files
    byte-identical to np.save, and no two ranks ever hold one output file open at once."""
    from aweSOM.npy_parts import ShardedNpyArray, open_npy, stitch_npy, verify_npy_parts

    env = dict(__import__("os").environ, NUMBA_NUM_THREADS="2", OMP_NUM_THREADS="1",
               PYTHONPATH=str(HERE.parent / "src"))
    cmd = [_mpirun(), "-n", str(ranks), "--oversubscribe", sys.executable, "-m", "mpi4py",
           str(HERE / "mpi_label_rows_worker.py"), str(tmp_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    report = json.loads((tmp_path / "report.json").read_text())
    for tag, n in rows_worker.CASES.items():
        rec = report["records"][tag]
        assert rec["layout"] == "sharded" and rec["manifest_seconds"] >= 0
        assert isinstance(rec["directory_fsync"], bool)
        assert [(r["start"], r["stop"]) for r in rec["ranks"]] == [distributed.row_range(n, r, ranks) for r in range(ranks)]
        assert [r["part_files"] for r in rec["ranks"]] == [True] * ranks
        for k, path in single_files(tmp_path, tag, n).items():
            assert not (tmp_path / f"{tag}-{k}.npy").exists()
            pdir = tmp_path / f"{tag}-{k}.parts"
            assert (pdir / "manifest.json").is_file()
            sharded = open_npy(tmp_path / f"{tag}-{k}.npy", mmap_mode="r")
            assert isinstance(sharded, ShardedNpyArray)
            try:
                bitwise(np.asarray(sharded), np.load(path))
            finally:
                sharded.close()
            assert verify_npy_parts(pdir)["total_rows"] == n
            stitch_npy(pdir)
            assert (tmp_path / f"{tag}-{k}.npy").read_bytes() == path.read_bytes(), (tag, k)
    for r in report["ranks"]:
        non_control = [p for p in r["opened"] if not p.endswith("control-labels.npy")]
        finals = [p for p in non_control if output_guard.is_final(p)]
        assert not finals
        if r["rank"]:
            assert r["caught"] is True
            assert {f"{Path(p).parent.name}/{Path(p).name}" for p in non_control} == {
                f"{tag}-{k}.parts/part-{r['rank']:04d}.npy"
                for tag in rows_worker.CASES
                for k in ("labels", "nodes", "distances")
            }
        else:
            assert {f"{Path(p).parent.name}/{Path(p).name}" for p in non_control} == {
                f"{tag}-{k}.parts/part-{rk:04d}.npy"
                for tag in rows_worker.CASES
                for k in ("labels", "nodes", "distances")
                for rk in range(ranks)
            }
    print(f"{ranks} mpirun ranks: driver label_rows .parts + stitched .npy byte-identical to one rank, one writer per file")


# ------------------------------------------------------------- the driver


def test_resolve_label_backend(monkeypatch):
    import aweSOM.stream as stream
    monkeypatch.setattr(stream, "cuda_available", lambda: False)
    assert pool.resolve_label_backend("auto") == "patch"  # CPU-only: auto falls back to patch
    assert pool.resolve_label_backend("patch") == "patch"
    with pytest.raises(RuntimeError, match=r"no usable CUDA device .* rank\(s\) \[0\]"):
        pool.resolve_label_backend("cuda_screen")
    monkeypatch.setattr(stream, "cuda_available", lambda: True)
    assert pool.resolve_label_backend("auto") == "cuda_screen"
    assert pool.resolve_label_backend("patch") == "patch"
    one_without = SimpleNamespace(allgather=lambda ok: [True, ok, False])
    assert pool.resolve_label_backend("auto", one_without) == "patch"  # every rank agrees on patch
    with pytest.raises(RuntimeError, match=r"rank\(s\) \[2\]"):
        pool.resolve_label_backend("cuda_screen", one_without)
    with pytest.raises(ValueError):
        pool.resolve_label_backend("cuda")


def test_auto_falls_back_to_patch_without_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert pool.resolve_label_backend("auto") == "patch"


def driver_rank(conn, rank, size, barrier, slots, argv, mode):
    """One fake rank of production_pool.main, with every open of an output file during
    label_rows guarded (output_guard). mode: 'cpu', 'gpu' (fake CuPy with two devices and the CPU stand-in
    searcher), 'no-shared-out' (rank 1 cannot open OUT files), 'bad-header' (rank 1 hits a
    malformed .npy header), 'truncated-part' (rank 1 truncates a byte off its labels part after
    closing), or 'bad-crc' (rank 1 flips a payload byte of its labels part after closing).
    Sends (status, payload, guarded opens)."""
    try:
        output_guard.install()
        from aweSOM import distributed as dist, _gpu_screen as screen, stream
        dist.get_comm = lambda: FakeComm(rank, size, barrier, slots)
        dist._shared_split_type = lambda: "shared"
        if mode == "gpu":
            runtime = SimpleNamespace(getDeviceCount=lambda: 2)
            sys.modules["cupy"] = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
            screen.GpuScreenSearcher = StandIn
            StandIn.devices = []
        else:  # a CPU-only machine, also on a GPU node
            sys.modules["cupy"] = None
        if mode == "no-shared-out" and rank == 1:
            def missing(path):
                raise FileNotFoundError(2, "No such file or directory", str(path))
            stream.NpyWriter = missing
        if mode == "bad-header" and rank == 1:
            def malformed(path):
                raise ValueError(f"{path}: malformed .npy header")
            stream.NpyWriter = malformed
        if mode == "truncated-part" and rank == 1:
            close = stream.NpyWriter.close

            def truncate_part(self, sync=True):
                close(self, sync)
                if self.path.endswith("-labels.parts/part-0001.npy"):
                    with open(self.path, "r+b") as fh:
                        fh.truncate(os.fstat(fh.fileno()).st_size - 1)
            stream.NpyWriter.close = truncate_part
        if mode == "bad-crc" and rank == 1:
            close = stream.NpyWriter.close

            def corrupt(self, sync=True):
                close(self, sync)
                if self.path.endswith("-distances.parts/part-0001.npy"):
                    with open(self.path, "r+b") as fh:
                        fh.seek(self.offset)
                        byte = fh.read(1)
                        fh.seek(self.offset)
                        fh.write(bytes([byte[0] ^ 1]))
            stream.NpyWriter.close = corrupt
        pool.label_rows = output_guard.locking_during(pool.label_rows)
        pool.main(argv)
        conn.send(("ok", StandIn.devices, output_guard.opened))
    except BaseException:
        conn.send(("error", traceback.format_exc(), output_guard.opened))
    finally:
        conn.close()


def run_driver(size, argv, mode):
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        barrier, slots = ctx.Barrier(size), manager.dict()
        pipes, procs = [], []
        for r in range(size):
            receive, send = ctx.Pipe(duplex=False)
            p = ctx.Process(target=driver_rank, args=(send, r, size, barrier, slots, argv, mode))
            p.start()
            send.close()
            pipes.append(receive)
            procs.append(p)
        results = [receive.recv() for receive in pipes]
        for p in procs:
            p.join(60)
    return results


def relabel(results, tag="4x3-s42"):
    """Single-process patch labels of the run's own saved weights over the whole pool."""
    import h5py
    from aweSOM import SOMModel
    model = SOMModel.load(results / f"{tag}.h5")
    with h5py.File(results / "pool.h5") as h:
        raw = h["raw"][:]
    nodes, d2 = best_match_nodes_patch(model.weights, model.scaler.transform(raw.astype("f8")), model.xdim, model.ydim)
    return np.asarray(model.node_cluster).ravel()[nodes], nodes, np.sqrt(d2)


@pytest.mark.parametrize("size,mode", [(2, "cpu"), (3, "cpu"), (2, "gpu"), (3, "gpu")])
def test_driver_ranks_label_their_rows(prepared, monkeypatch, size, mode):  # noqa: F811
    from aweSOM.npy_parts import open_npy, stitch_npy

    monkeypatch.setenv("NUMBA_CACHE_DIR", str(prepared.tmp / "cache"))
    out = prepared.tmp / f"label-{mode}-{size}"
    argv = [*prepared.base, "--out", str(out), "--reuse-pool", str(prepared.pool), "--fit", "4x3", "42",
            "--mpi", "--statistics", "later"]
    results = run_driver(size, argv, mode)
    assert all(status == "ok" for status, _, _ in results), results
    res = out / "results"
    for k in ("labels", "nodes", "distances"):
        assert not (res / f"4x3-s42-{k}.npy").exists()
        assert (res / f"4x3-s42-{k}.parts" / "manifest.json").is_file()
    for r, (_, _, opened) in enumerate(results):
        assert not any(output_guard.is_final(p) for p in opened)
        if r:  # every non-zero rank opens only its own part files
            assert {f"{Path(p).parent.name}/{Path(p).name}" for p in opened} == {
                f"4x3-s42-{k}.parts/part-{r:04d}.npy" for k in ("labels", "nodes", "distances")
            }
    lab, nodes, dist = relabel(res)
    bitwise(open_npy(res / "4x3-s42-labels.npy", mmap_mode=None), lab.astype(np.uint8))  # narrow: 12 nodes, few clusters
    bitwise(open_npy(res / "4x3-s42-nodes.npy", mmap_mode=None), nodes.astype(np.uint8))
    bitwise(open_npy(res / "4x3-s42-distances.npy", mmap_mode=None), dist)
    for k, expected in (("labels", lab.astype(np.uint8)), ("nodes", nodes.astype(np.uint8)), ("distances", dist)):
        stitch_npy(res / f"4x3-s42-{k}.parts")
        ref_path = prepared.tmp / f"ref-{mode}-{size}-{k}.npy"
        np.save(ref_path, expected)
        assert (res / f"4x3-s42-{k}.npy").read_bytes() == ref_path.read_bytes()
    rec = json.loads((res / "4x3-s42.json").read_text())["labelling"]
    n = len(lab)
    assert rec["requested"] == "auto" and rec["backend"] == ("cuda_screen" if mode == "gpu" else "patch")
    assert rec["layout"] == "sharded" and rec["manifest_seconds"] >= 0 and isinstance(rec["directory_fsync"], bool)
    assert [(r["start"], r["stop"]) for r in rec["ranks"]] == [distributed.row_range(n, r, size) for r in range(size)]
    for r in rec["ranks"]:
        assert r["part_files"] is True and r["bytes_written"] == 10 * r["rows"]
        assert r["pwrite_seconds"] >= 0 and r["fsync_seconds"] >= 0
    devices = [payload for _, payload, _ in results]
    if mode == "gpu":
        assert [r["device"] for r in rec["ranks"]] == [r % 2 for r in range(size)]
        assert devices == [[r % 2] for r in range(size)]  # one searcher per rank, on its device
    else:
        assert all(r["device"] is None for r in rec["ranks"])


def test_driver_refuses_out_other_ranks_cannot_open(prepared, monkeypatch):  # noqa: F811
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(prepared.tmp / "cache"))
    out = prepared.tmp / "label-node-local"
    argv = [*prepared.base, "--out", str(out), "--reuse-pool", str(prepared.pool), "--fit", "4x3", "42", "--mpi"]
    results = run_driver(2, argv, "no-shared-out")
    assert [status for status, _, _ in results] == ["error", "error"]
    for _, message, _ in results:
        assert "multi-rank labelling needs OUT on a file system every rank can open" in message
        assert "rank 1: cannot open" in message
    assert not (out / "results").exists()


def test_driver_header_error_stops_every_rank(prepared, monkeypatch):  # noqa: F811
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(prepared.tmp / "cache"))
    out = prepared.tmp / "label-bad-header"
    argv = [*prepared.base, "--out", str(out), "--reuse-pool", str(prepared.pool), "--fit", "4x3", "42", "--mpi"]
    results = run_driver(2, argv, "bad-header")
    assert [status for status, _, _ in results] == ["error", "error"]
    for _, message, _ in results:
        assert "rank 1: cannot open the label files: ValueError" in message


def test_driver_manifest_check_failure_stops_every_rank(prepared, monkeypatch):  # noqa: F811
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(prepared.tmp / "cache"))
    out = prepared.tmp / "label-truncated-part"
    argv = [*prepared.base, "--out", str(out), "--reuse-pool", str(prepared.pool), "--fit", "4x3", "42",
            "--mpi", "--statistics", "later"]
    results = run_driver(2, argv, "truncated-part")
    assert [status for status, _, _ in results] == ["error", "error"]
    for _, message, _ in results:
        assert "labelling manifest check on rank 0 failed" in message
        assert "part-0001.npy: file is" in message
    pdir = out / "results.partial" / "4x3-s42-labels.parts"
    assert (pdir / "part-0001.npy").exists()  # kept for inspection
    assert not (pdir / "manifest.json").exists()  # completion marker never published


def test_corrupted_part_caught_by_verify_and_stitch(prepared, monkeypatch):  # noqa: F811
    from aweSOM.npy_parts import stitch_npy, verify_npy_parts

    monkeypatch.setenv("NUMBA_CACHE_DIR", str(prepared.tmp / "cache"))
    out = prepared.tmp / "label-bad-crc"
    argv = [*prepared.base, "--out", str(out), "--reuse-pool", str(prepared.pool), "--fit", "4x3", "42",
            "--mpi", "--statistics", "later"]
    results = run_driver(2, argv, "bad-crc")
    assert [status for status, _, _ in results] == ["ok", "ok"], results
    pdir = out / "results" / "4x3-s42-distances.parts"
    with pytest.raises(ValueError, match="CRC-32 .* differs from manifest"):
        verify_npy_parts(pdir)
    with pytest.raises(ValueError, match="CRC-32 .* differs from manifest"):
        stitch_npy(pdir)
    assert not (out / "results" / "4x3-s42-distances.npy").exists()
