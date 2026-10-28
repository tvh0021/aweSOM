"""End-to-end tests of examples/grmhd-bbh/analyze_chunks.py on synthetic Carpet and CarpetX runs."""

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from aweSOM import SOMModel, Scaler
from aweSOM.features_grmhd import FEATURE_NAMES, INPUTS, FeatureThresholds, compute_features
from aweSOM.io import CarpetHDF5Reader

sys.path.insert(0, str(Path(__file__).parent))
from grmhd_fixtures import write_carpet, write_silo  # noqa: E402

pytest.importorskip("matplotlib")
SCRIPT = Path(__file__).parents[1] / "examples" / "grmhd-bbh" / "analyze_chunks.py"
spec = importlib.util.spec_from_file_location("analyze_chunks", SCRIPT)
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)

THR = FeatureThresholds(b2_min=1.0, s_v=0.5, winsor={"log_chi_B": (-3.0, 1.0), "asinh_div_v": (-2.0, 2.0)})
ROWS = {"carpet": 15 ** 3 - 9 ** 3 + 17 ** 3, "silo": 16 ** 3 - 6 ** 3 + 12 ** 3}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    rng = np.random.default_rng(7)
    meta = THR.to_metadata()
    meta.update(sheet_clusters="1", dataset="synthetic")
    m = SOMModel(xdim=3, ydim=2, weights=rng.normal(size=(6, 5)) * 0.5, node_cluster=np.array([0, 0, 1, 1, 2, 2]),
                 feature_names=list(FEATURE_NAMES), scaler=Scaler.identity(5), node_dist_q99=np.ones(6),
                 reject_distance=10.0, metadata=meta)
    p = d / "model.h5"
    m.save(p)
    return p


@pytest.fixture(scope="module")
def carpet_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("carpet")
    write_carpet(d, it=0)
    return d


@pytest.fixture(scope="module")
def silo_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("silo")
    write_silo(d, it=0)
    return d


def _run(source, model, out, *extra):
    return ac.main(["--source", str(source), "--model", str(model), "-o", str(out), "--slab", "4", "--save-features",
                    "--midplane-z", "0.25", *extra])


def _read(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in ("labels", "xyz", "level", "finite", "features", "midplane")}, dict(f.attrs)


def test_carpet_end_to_end(carpet_run, model_path, tmp_path):
    out = tmp_path / "out"
    assert _run(carpet_run, model_path, out, "--reader", "carpet", "--iterations", "0") == 0
    rows, attrs = _read(out / "it00000000.h5")
    assert attrs["complete"] and len(rows["labels"]) == ROWS["carpet"]
    model = SOMModel.load(model_path)
    assert np.array_equal(rows["labels"], model.predict(rows["features"]))
    # the rows are the reader's blocks, in order, with the frozen thresholds
    reader = CarpetHDF5Reader(carpet_run, backend="native")
    want, xyz, lev = [], [], []
    for b in reader.iter_level_blocks(INPUTS, 0, halo=THR.halo, slab=4):
        want.append(compute_features(b.data, b.spacing, THR, select=b.full_mask()))
        xyz.append(b.xyz())
        lev.append(np.full(b.n_owned, b.level))
    assert np.array_equal(rows["features"], np.concatenate(want), equal_nan=True)
    assert np.array_equal(rows["xyz"], np.concatenate(xyz).astype(np.float32))
    assert np.array_equal(rows["level"], np.concatenate(lev))
    assert np.array_equal(rows["finite"], np.all(np.isfinite(rows["features"]), axis=1))
    assert 0 < rows["finite"].sum() < len(rows["finite"])  # patch edges lack the stencil
    stats = json.loads((out / "it00000000_stats.json").read_text())
    assert stats["complete"] and set(stats["levels"]) == {"0", "1"}
    for rl, s in stats["levels"].items():
        lab = rows["labels"][(rows["level"] == int(rl)) & rows["finite"]]
        assert s["cells"] == int((rows["level"] == int(rl)).sum())
        assert sum(s["volume_fraction"].values()) == pytest.approx(1.0)
        assert s["volume_fraction"].get("1", 0.0) == pytest.approx(np.mean(lab == 1))
        assert s["sheet_cells"] == int(np.sum(lab == 1))
        f = rows["features"][(rows["level"] == int(rl)) & rows["finite"]]
        for j, n in enumerate(FEATURE_NAMES):
            assert abs(s["feature_median"][n] - np.median(f[:, j])) <= 0.0051
    assert stats["run"]["rows_total"] == ROWS["carpet"]
    assert (out / "it00000000_quicklook.png").stat().st_size > 1000
    mid = rows["midplane"]
    assert len(mid) > 0 and set(np.unique(mid[:, 4]).astype(int)) == {0, 1}
    # a second run finds the iteration complete and changes nothing
    assert _run(carpet_run, model_path, out, "--reader", "carpet", "--iterations", "0") == 0
    again, _ = _read(out / "it00000000.h5")
    assert all(np.array_equal(again[k], rows[k], equal_nan=True) for k in rows)


class _Interrupt(Exception):
    pass


class _StopAfter:
    """A reader that raises after ``n`` blocks."""

    def __init__(self, reader, n):
        self.reader, self.n, self.cache = reader, n, reader.cache

    def __getattr__(self, name):  # levels, reference_var, _cat, ... for the layout signature
        return getattr(self.reader, name)

    def iter_level_blocks(self, *a, **kw):
        for i, b in enumerate(self.reader.iter_level_blocks(*a, **kw)):
            if i == self.n:
                raise _Interrupt
            yield b


def test_resume_after_interruption(carpet_run, model_path, tmp_path):
    full, part = tmp_path / "full", tmp_path / "part"
    assert _run(carpet_run, model_path, full, "--reader", "carpet", "--iterations", "0") == 0
    model = SOMModel.load(model_path)
    args = ac.main.__globals__["argparse"].Namespace(
        out=str(part), source=str(carpet_run), levels=None, slab=4, save_features=True, midplane_z=0.25,
        predict_backend="screen", verbose=False, reader="carpet", carpet_backend="native", fill="coarse",
        model=str(model_path))
    part.mkdir()
    reader = _StopAfter(CarpetHDF5Reader(carpet_run, backend="native"), 5)
    with pytest.raises(_Interrupt):
        ac.process_iteration(reader, model, THR, 0, args, [1])
    with h5py.File(part / "it00000000.h5", "a") as f:
        assert not f.attrs["complete"] and len(f["blocks"]) == 5
        n = f["labels"].shape[0]
        for k in ("labels", "xyz", "level", "finite", "features"):  # rows of a block cut short
            f[k].resize(n + 7, axis=0)
        f["blocks"].create_group("rl9_p0_x00000")  # a block record cut short
    assert _run(carpet_run, model_path, part, "--reader", "carpet", "--iterations", "0") == 0
    a, _ = _read(full / "it00000000.h5")
    b, attrs = _read(part / "it00000000.h5")
    assert attrs["complete"]
    for k in ("labels", "xyz", "level", "finite", "features"):
        assert np.array_equal(a[k], b[k], equal_nan=True), k
    sa = json.loads((full / "it00000000_stats.json").read_text())["levels"]
    sb = json.loads((part / "it00000000_stats.json").read_text())["levels"]
    assert sa == sb


def test_watch_carpetx_silo(silo_run, model_path, tmp_path):
    out = tmp_path / "watch"
    rc = _run(silo_run, model_path, out, "--watch", "--poll", "0.05", "--settle", "0", "--max-idle", "0.2")
    assert rc == 0
    rows, attrs = _read(out / "it00000000.h5")
    assert attrs["complete"] and len(rows["labels"]) == ROWS["silo"]
    assert np.array_equal(rows["labels"], SOMModel.load(model_path).predict(rows["features"]))
    assert rows["finite"].all()  # Silo ghosts and coarse fill give every halo cell a value


def test_resume_refuses_a_different_slab_width(carpet_run, model_path, tmp_path):
    out = tmp_path / "part"
    model = SOMModel.load(model_path)
    args = ac.main.__globals__["argparse"].Namespace(
        out=str(out), source=str(carpet_run), levels=None, slab=4, save_features=True, midplane_z=0.25,
        predict_backend="screen", verbose=False, reader="carpet", carpet_backend="native", fill="coarse",
        model=str(model_path))
    out.mkdir()
    reader = _StopAfter(CarpetHDF5Reader(carpet_run, backend="native"), 3)
    with pytest.raises(_Interrupt):
        ac.process_iteration(reader, model, THR, 0, args, [1])
    # block keys encode only the slab start: a wider slab would skip rows the narrow blocks never held
    with pytest.raises(SystemExit, match="different configuration"):
        _run(carpet_run, model_path, out, "--reader", "carpet", "--iterations", "0", "--slab", "8")
    with pytest.raises(SystemExit, match="different configuration"):
        _run(carpet_run, model_path, out, "--reader", "carpet", "--iterations", "0", "--levels", "0")
    assert _run(carpet_run, model_path, out, "--reader", "carpet", "--iterations", "0") == 0
    full = tmp_path / "full"
    assert _run(carpet_run, model_path, full, "--reader", "carpet", "--iterations", "0") == 0
    a, _ = _read(full / "it00000000.h5")
    b, _ = _read(out / "it00000000.h5")
    assert np.array_equal(a["xyz"], b["xyz"])


def test_watch_waits_for_shards_and_extends_a_grown_snapshot(tmp_path, model_path, monkeypatch):
    import grmhd_fixtures

    src = tmp_path / "silo"
    out = tmp_path / "watch"
    two = grmhd_fixtures.carpetx_levels()
    # 1. the metafile is there but the data file it refers to is not yet: nothing is processed
    grmhd_fixtures.write_silo(src, it=0)
    shard = src / "run.it00000000.silo.dir" / "run.it00000000.p000000.silo"
    hidden = shard.with_suffix(".hidden")
    shard.rename(hidden)
    assert _run(src, model_path, out, "--watch", "--poll", "0.05", "--settle", "0", "--max-idle", "0.2") == 0
    assert not (out / "it00000000.h5").exists()
    # 2. only level 0 written: processed and marked complete for that layout
    hidden.rename(shard)
    monkeypatch.setattr(grmhd_fixtures, "carpetx_levels", lambda: {0: two[0]})
    grmhd_fixtures.write_silo(src, it=0)
    assert _run(src, model_path, out, "--watch", "--poll", "0.05", "--settle", "0", "--max-idle", "0.2") == 0
    rows, attrs = _read(out / "it00000000.h5")
    assert attrs["complete"] and set(rows["level"]) == {0}
    n0 = len(rows["labels"])
    # 3. level 1 appears later: level 0 loses the cells it covers, so the file is redone, not left as it was
    monkeypatch.setattr(grmhd_fixtures, "carpetx_levels", lambda: two)
    grmhd_fixtures.write_silo(src, it=0)
    assert _run(src, model_path, out, "--watch", "--poll", "0.05", "--settle", "0", "--max-idle", "0.2") == 0
    rows, attrs = _read(out / "it00000000.h5")
    assert attrs["complete"] and len(rows["labels"]) == ROWS["silo"] > n0
    assert set(rows["level"]) == {0, 1}
    assert np.array_equal(rows["labels"], SOMModel.load(model_path).predict(rows["features"]))
    assert len(np.unique(rows["xyz"], axis=0)) == len(rows["xyz"])
