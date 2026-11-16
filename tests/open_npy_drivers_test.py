"""Verify example driver stages that switched from np.load(..., mmap_mode='r') to aweSOM.open_npy.

Each stage runs once against a monolithic `.npy` label input and once against a sharded
`<stem>.parts/` directory written via aweSOM's own sharded writer (`NpyWriter` + `write_npy_manifest`),
asserting identical outputs.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import subprocess
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from aweSOM import Lattice, SOMModel, Scaler
from aweSOM.features_grmhd import SEVEN_FEATURE_NAMES
from aweSOM.inference import best_match_nodes_screen
from aweSOM.npy_parts import part_filename, parts_dir_for, write_npy_manifest
from aweSOM.stream import NpyWriter

ROOT = Path(__file__).resolve().parents[1]
BBH_DIR = ROOT / "examples/grmhd-bbh"
FT_DIR = ROOT / "examples/grmhd-flux-tubes"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_label_layout(base_npy: Path, arr: np.ndarray, layout: str) -> Path:
    """Write ``arr`` either as monolithic ``base_npy`` or sharded ``<stem>.parts/``."""
    base_npy.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    if layout == "npy":
        np.save(base_npy, arr)
        return base_npy
    assert layout == "parts"
    pdir = parts_dir_for(base_npy)
    pdir.mkdir(parents=True, exist_ok=False)
    n = len(arr)
    edges = [0, n // 3, (2 * n) // 3, n]
    entries = []
    ranks = len(edges) - 1
    for r, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        fname = part_filename(r, ranks)
        part_path = pdir / fname
        mm = np.lib.format.open_memmap(part_path, mode="w+", dtype=arr.dtype, shape=(b - a, *arr.shape[1:]))
        del mm
        writer = NpyWriter(part_path)
        writer[0 : b - a] = arr[a:b]
        writer.close(sync=False)
        row_bytes = int(np.prod(arr.shape[1:], dtype=np.int64)) * arr.dtype.itemsize
        entries.append(
            dict(
                rank=r,
                file=fname,
                start=a,
                stop=b,
                rows=b - a,
                bytes=(b - a) * row_bytes,
                crc32=int(writer.crc32),
            )
        )
    write_npy_manifest(pdir, dtype=arr.dtype, shape=arr.shape, parts=entries, sync=False)
    return pdir


@pytest.fixture(scope="module")
def synth_env(tmp_path_factory):
    """Build a compact, self-consistent synthetic BBH + flux-tube dataset once per module."""
    base = tmp_path_factory.mktemp("open_npy_drivers_fixture")
    names = list(SEVEN_FEATURE_NAMES)
    rng = np.random.default_rng(20261115)

    # Level 5: 6 x 6 x 4 = 144 cells (dx = 0.25), centered at origin
    xs5 = np.arange(6, dtype=np.float32) * 0.25 - 0.625
    ys5 = np.arange(6, dtype=np.float32) * 0.25 - 0.625
    zs5 = np.arange(4, dtype=np.float32) * 0.25 - 0.375
    g5 = np.stack(np.meshgrid(xs5, ys5, zs5, indexing="ij"), axis=-1).reshape(-1, 3)

    # Level 6: 4 x 4 x 2 = 32 cells (dx = 0.125), centered at origin ( >=2 z values for dx inference )
    xs6 = np.arange(4, dtype=np.float32) * 0.125 - 0.1875
    ys6 = np.arange(4, dtype=np.float32) * 0.125 - 0.1875
    zs6 = np.arange(2, dtype=np.float32) * 0.125 - 0.0625
    g6 = np.stack(np.meshgrid(xs6, ys6, zs6, indexing="ij"), axis=-1).reshape(-1, 3)

    xyz = np.concatenate([g5, g6], axis=0).astype(np.float32)
    level = np.concatenate([np.full(len(g5), 5, np.int32), np.full(len(g6), 6, np.int32)])
    n_grid = len(xyz)  # 176 rows

    # Construct 3 distinct clusters in feature space so SOM clustering produces >= 2 clusters cleanly
    feats = rng.normal(scale=0.15, size=(n_grid, len(names))).astype(np.float32)
    # Strip 0 (elongated along x at iy=2, iz=1 on level 5 -> indices 2*4+1 + ix*24): high log_beta_inv, low log_chi_B
    tube_rows = np.array([ix * 24 + 2 * 4 + 1 for ix in range(6)], dtype=np.int64)
    feats[tube_rows, names.index("log_beta_inv")] += 3.0
    feats[tube_rows, names.index("log_chi_B")] -= 2.0
    # Sheet slab (iz=2 plane on level 5): high log_chi_B, high log_grad_ln_b2, low log_beta_inv
    sheet_rows = np.flatnonzero((level == 5) & np.isclose(xyz[:, 2], 0.125))
    feats[sheet_rows, names.index("log_chi_B")] += 3.5
    feats[sheet_rows, names.index("log_grad_ln_b2")] += 3.0
    feats[sheet_rows, names.index("log_beta_inv")] -= 1.5

    holes = np.array([[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    hr = np.array([0.94, 0.94], dtype=np.float64)

    grid_h5 = base / "grid.h5"
    with h5py.File(grid_h5, "w") as h:
        h.create_dataset("xyz", data=xyz)
        h.create_dataset("level", data=level)
        h.create_dataset("features", data=feats)
        h.create_dataset("names", data=np.array(names, dtype="S"))
        h.attrs["horizon_centres"] = holes
        h.attrs["horizon_radius"] = hr
        h.attrs["iteration"] = 180224
        h.attrs["time"] = 1000.0
        h.attrs["b2_min"] = 1e-8
        h.attrs["s_v"] = 0.04
        h.attrs["pipeline"] = "v1"
        h.attrs["scaler_kind"] = "manual"

    # Magnetic field file B.h5 aligned along x-axis for flux-tube coherence
    b_h5 = base / "B.h5"
    B_vec = np.tile(np.array([[1.0, 0.02, 0.01]], dtype=np.float32), (n_grid, 1))
    with h5py.File(b_h5, "w") as h:
        h.create_dataset("B", data=B_vec)

    # Training subset (80 rows taken directly from grid rows so check_labels finds exact matches on both levels)
    tr_idx = np.r_[np.arange(0, 64), np.arange(144, 160)]
    tr_raw = feats[tr_idx].copy()
    tr_xyz = xyz[tr_idx].copy()
    tr_lev = level[tr_idx].copy()
    dx_arr = np.where(tr_lev == 5, 0.25, 0.125)
    mean32 = np.mean(tr_raw, axis=0)
    std32 = np.std(tr_raw, axis=0)
    mean64 = tr_raw.astype(np.float64).mean(axis=0)
    std64 = tr_raw.astype(np.float64).std(axis=0)
    scaled32 = ((tr_raw - mean32) / (2.0 * std32)).astype(np.float64)

    train_h5 = base / "training.h5"
    with h5py.File(train_h5, "w") as h:
        h.create_dataset("features", data=tr_raw)
        h.create_dataset("xyz", data=tr_xyz)
        h.create_dataset("level", data=tr_lev)
        h.create_dataset("weight", data=np.ones(len(tr_idx), dtype=np.float64))
        h.create_dataset("weight_volume", data=(dx_arr**3).astype(np.float64))
        h.create_dataset("names", data=np.array(names, dtype="S"))
        g = h.create_group("scaler")
        g.attrs["kind"] = "manual"
        g.create_dataset("mean", data=mean64)
        g.create_dataset("sigma", data=std64)
        g.create_dataset("scale_unweighted", data=2.0 * std64)
        g.create_dataset("feature_weights", data=np.ones(len(names), dtype=np.float64))
        h.attrs["iteration"] = 180224
        h.attrs["b2_min"] = 1e-8
        h.attrs["s_v"] = 0.04
        h.attrs["pipeline"] = "v1"
        h.attrs["scaler_kind"] = "manual"

    # Build a 4x3 Lattice pickle trained/seeded on scaled32
    lat = Lattice(4, 3)
    lat.lattice = scaled32[:12].copy()
    lat.data_array = scaled32
    lat.features_names = names
    lat.umat = lat.compute_umat()
    lat.projection_1d = best_match_nodes_screen(lat.lattice, scaled32)[0]
    clusters_mc01 = lat.assign_cluster_to_lattice(smoothing=None, merge_cost=0.1)
    lat.lattice_assigned_clusters = clusters_mc01
    som_pkl = base / "lattice.pkl"
    with som_pkl.open("wb") as f:
        pickle.dump(lat, f)

    scaler = Scaler("manual", mean32.astype(np.float64), 2.0 * std32.astype(np.float64))
    model = SOMModel.from_lattice(
        lat,
        scaled_training_data=scaled32,
        scaler=scaler,
        merge_cost=0.1,
        metadata={"dataset": "synth", "sheet_clusters": "0", "selected_sheet_clusters": "0"},
    )
    model_h5 = base / "model.h5"
    model.save(model_h5)

    train_labels = model.predict(tr_raw).astype(np.uint8)
    grid_labels = model.predict(feats).astype(np.uint8)
    # Ensure tube_rows carry label 0 and sheet_rows carry label 1 for deterministic component tests
    grid_labels[tube_rows] = 0
    grid_labels[sheet_rows] = 1
    train_labels = grid_labels[tr_idx]

    # Raw planes npz for proposal_figure.stage_sample and tube_match_hhr.main
    axis_vals = np.linspace(-6.0, 6.0, 25, dtype=np.float64)
    shape2d = (len(axis_vals), len(axis_vals))
    raw_planes = base / "raw_planes.npz"
    plane_dict = dict(
        holes=holes,
        horizon_radius=hr,
        time=np.float64(1000.0),
        it=np.int64(180224),
    )
    for kind, e2 in (("orbital", np.array([0.0, 1.0, 0.0])), ("meridional", np.array([0.0, 0.0, 1.0]))):
        plane_dict[f"{kind}_origin"] = np.zeros(3, dtype=np.float64)
        plane_dict[f"{kind}_e1"] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        plane_dict[f"{kind}_e2"] = e2.astype(np.float64)
        plane_dict[f"{kind}_a"] = axis_vals
        plane_dict[f"{kind}_b"] = axis_vals
        plane_dict[f"{kind}_Bx"] = np.full(shape2d, 0.1, dtype=np.float32)
        plane_dict[f"{kind}_By"] = np.full(shape2d, 0.1, dtype=np.float32)
        bz = np.full(shape2d, 0.2, dtype=np.float32)
        bz[10:15, 10:15] = 8.0  # strong vertical field patch so tube_reference finds a reference region
        plane_dict[f"{kind}_Bz"] = bz
        plane_dict[f"{kind}_P"] = np.full(shape2d, 0.05, dtype=np.float32)
        plane_dict[f"{kind}_rho_b"] = np.full(shape2d, 1.0, dtype=np.float32)
        plane_dict[f"{kind}_smallb2"] = np.full(shape2d, 2.0, dtype=np.float32)
    np.savez(raw_planes, **plane_dict)

    return dict(
        grid_h5=grid_h5,
        b_h5=b_h5,
        train_h5=train_h5,
        som_pkl=som_pkl,
        model_h5=model_h5,
        raw_planes=raw_planes,
        train_labels=train_labels,
        grid_labels=grid_labels,
    )


def _run_cli(script_path: Path, cli_args: list[str]) -> subprocess.CompletedProcess:
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(script_path.parent)]),
        NUMBA_NUM_THREADS="2",
    )
    res = subprocess.run(
        [sys.executable, str(script_path), *cli_args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, f"script={script_path.name}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


def test_build_model_training_labels_npy_vs_parts(tmp_path, synth_env):
    bm = _load_module("build_model_e8", BBH_DIR / "build_model.py")
    reports = {}
    for layout in ("npy", "parts"):
        sub = tmp_path / layout
        lab_target = sub / "train_labels.npy"
        write_label_layout(lab_target, synth_env["train_labels"], layout)
        out_h5 = sub / "built.h5"
        bm.main(
            [
                "--training-file",
                str(synth_env["train_h5"]),
                "--lattice",
                str(synth_env["som_pkl"]),
                "--scaler",
                "training-float32",
                "--merge-cost",
                "0.1",
                "--training-labels",
                str(lab_target),
                "-o",
                str(out_h5),
            ]
        )
        rep = json.loads(out_h5.with_suffix(".json").read_text())
        for k in ("freeze_seconds", "total_seconds", "model"):
            rep.pop(k, None)
        reports[layout] = rep
    assert "agreement_with_training_labels" in reports["npy"]
    assert reports["npy"] == reports["parts"]


def test_freeze_model_check_labels_npy_vs_parts(tmp_path, synth_env):
    fm = _load_module("freeze_model_e8", BBH_DIR / "freeze_model.py")
    reports = {}
    for layout in ("npy", "parts"):
        sub = tmp_path / layout
        lab_target = sub / "check_labels.npy"
        write_label_layout(lab_target, synth_env["train_labels"], layout)
        out_h5 = sub / "frozen.h5"
        fm.main(
            [
                "--training",
                str(synth_env["train_h5"]),
                "--lattice",
                str(synth_env["som_pkl"]),
                "--scaler-dtype",
                "float32",
                "--merge-cost",
                "0.1",
                "--check-labels",
                str(lab_target),
                "-o",
                str(out_h5),
            ]
        )
        rep = json.loads(out_h5.with_suffix(".json").read_text())
        rep.pop("seconds", None)
        reports[layout] = rep
    assert "agreement_with_check_labels" in reports["npy"]
    assert reports["npy"] == reports["parts"]


def test_check_labels_main_npy_vs_parts(tmp_path, synth_env):
    cl = _load_module("check_labels_e8", BBH_DIR / "check_labels.py")
    outputs = {}
    for layout in ("npy", "parts"):
        sub = tmp_path / layout
        g_lab = sub / "grid_labels.npy"
        t_lab = sub / "train_labels.npy"
        write_label_layout(g_lab, synth_env["grid_labels"], layout)
        write_label_layout(t_lab, synth_env["train_labels"], layout)
        out_json = sub / "check.json"
        cl.main(
            [
                "--grid",
                str(synth_env["grid_h5"]),
                "--labels",
                str(g_lab),
                "--training-file",
                str(synth_env["train_h5"]),
                "--training-labels",
                str(t_lab),
                "--sheet-clusters",
                "1",
                "-o",
                str(out_json),
            ]
        )
        rep = json.loads(out_json.read_text())
        for k in ("labels", "training_labels", "seconds"):
            rep.pop(k, None)
        outputs[layout] = rep
    assert outputs["npy"] == outputs["parts"]
    assert outputs["npy"]["overall"]["label_agreement_at_training_rows"] == 1.0


def test_measure_full_grid_main_npy_vs_parts(tmp_path, synth_env):
    outputs = {}
    masks = {}
    for layout in ("npy", "parts"):
        sub = tmp_path / layout
        g_lab = sub / "grid_labels.npy"
        write_label_layout(g_lab, synth_env["grid_labels"], layout)
        out_dir = sub / "measured"
        _run_cli(
            BBH_DIR / "measure_full_grid.py",
            [
                "--grid",
                str(synth_env["grid_h5"]),
                "--labels",
                str(g_lab),
                "--out",
                str(out_dir),
                "--clusters",
                "1",
            ],
        )
        rep = json.loads((out_dir / "statistics.json").read_text())
        rep.pop("seconds", None)
        rep.pop("peak_GB", None)
        outputs[layout] = rep
        masks[layout] = np.load(out_dir / "sheet_5.npy")
    assert outputs["npy"] == outputs["parts"]
    np.testing.assert_array_equal(masks["npy"], masks["parts"])


def test_proposal_figure_stage_labels_and_stage_sample_npy_vs_parts(tmp_path, synth_env, monkeypatch):
    monkeypatch.syspath_prepend(str(BBH_DIR))
    pf = _load_module("proposal_figure_e8", BBH_DIR / "proposal_figure.py")

    label_infos = {}
    planes_out = {}
    sheet3d_out = {}
    for layout in ("npy", "parts"):
        sub = tmp_path / layout
        out_dir = sub / "out"
        out_dir.mkdir(parents=True)
        ref_lab = sub / "ref_train_labels.npy"
        write_label_layout(ref_lab, synth_env["train_labels"], layout)

        args_labels = types.SimpleNamespace(
            grid_labels=None,
            out=str(out_dir),
            som=str(synth_env["som_pkl"]),
            train_features=str(synth_env["train_h5"]),
            merge_cost=0.1,
            scaler="training-float32",
            ref_labels=str(ref_lab),
            grid=str(synth_env["grid_h5"]),
            tag="t1",
        )
        pf.stage_labels(args_labels)
        info = json.loads((out_dir / "clusters_t1.json").read_text())
        info.pop("predict_seconds", None)
        label_infos[layout] = info

        # Replace out_dir/labels_t1.npy with requested layout (.npy vs .parts) for stage_sample
        pred_labels = np.load(out_dir / "labels_t1.npy")
        (out_dir / "labels_t1.npy").unlink()
        write_label_layout(out_dir / "labels_t1.npy", pred_labels, layout)

        args_sample = types.SimpleNamespace(
            out=str(out_dir),
            raw_planes=str(synth_env["raw_planes"]),
            tag="t1",
            half_width=1.0,
            n=8,
            grid=str(synth_env["grid_h5"]),
            voxel=0.5,
            box_xy=1.0,
            box_z=0.5,
        )
        pf.stage_sample(args_sample)
        planes_out[layout] = dict(np.load(out_dir / "planes_t1.npz"))
        sheet3d_out[layout] = dict(np.load(out_dir / "sheet3d_t1.npz"))

    assert "ref_label_agreement" in label_infos["npy"]
    assert label_infos["npy"] == label_infos["parts"]
    for k in planes_out["npy"]:
        np.testing.assert_array_equal(planes_out["npy"][k], planes_out["parts"][k])
    for k in sheet3d_out["npy"]:
        np.testing.assert_array_equal(sheet3d_out["npy"][k], sheet3d_out["parts"][k])


def test_tube_match_hhr_main_npy_vs_parts(tmp_path, synth_env):
    summaries = {}
    comps = {}
    for layout in ("npy", "parts"):
        sub = tmp_path / layout
        g_lab = sub / "grid_labels.npy"
        write_label_layout(g_lab, synth_env["grid_labels"], layout)
        out_dir = sub / "tube_out"
        _run_cli(
            BBH_DIR / "tube_match_hhr.py",
            [
                "--grid",
                str(synth_env["grid_h5"]),
                "--labels",
                str(g_lab),
                "--raw",
                str(synth_env["raw_planes"]),
                "--model",
                str(synth_env["model_h5"]),
                "--out-dir",
                str(out_dir),
                "--clusters",
                "0",
                "--min-cells",
                "2",
                "--min-elongation",
                "1.1",
            ],
        )
        s = json.loads((out_dir / "summary.json").read_text())
        for k in ("command", "seconds", "peak_rss_GiB"):
            s.pop(k, None)
        summaries[layout] = s
        with h5py.File(out_dir / "components.h5", "r") as f:
            comps[layout] = {k: f[k][()] for k in f}
    assert summaries["npy"] == summaries["parts"]
    for k in comps["npy"]:
        np.testing.assert_array_equal(comps["npy"][k], comps["parts"][k])


def test_flux_tubes_predict_run_and_plot_npy_vs_parts(tmp_path, synth_env):
    pft = _load_module("plot_flux_tubes_e8", FT_DIR / "plot_flux_tubes.py")
    pred_checks = {}
    pred_labels = {}
    run_summaries = {}
    loaded_sets = {}
    slice_pixels = {}

    for layout in ("npy", "parts"):
        sub = tmp_path / layout
        saved_dir = sub / "saved_train"
        # predict_grid formats `{suffix}`: '' for 0.2 and '-mc0.1' for 0.1
        write_label_layout(saved_dir / "labels.train.npy", synth_env["train_labels"], layout)
        write_label_layout(saved_dir / "labels.train-mc0.1.npy", synth_env["train_labels"], layout)

        pred_out = sub / "pred_out"
        pred_out.mkdir(parents=True)
        _run_cli(
            FT_DIR / "predict_grid.py",
            [
                "--som",
                str(synth_env["som_pkl"]),
                "--features",
                str(synth_env["train_h5"]),
                "--grid",
                str(synth_env["grid_h5"]),
                "--merge-costs",
                "0.2",
                "0.1",
                "--train-labels-pattern",
                str(saved_dir / "labels.train{suffix}.npy"),
                "--n-check",
                "40",
                "--chunk",
                "64",
                "--out-dir",
                str(pred_out),
                "--tag",
                "synth",
            ],
        )
        pj = json.loads((pred_out / "predict_grid.json").read_text())
        pred_checks[layout] = pj["training_label_agreement"]
        pred_labels[layout] = np.load(pred_out / "labels_synth_mc0.1.npy")

        # run_flux_tubes.main on .npy vs .parts
        ft_lab = sub / "ft_labels.npy"
        write_label_layout(ft_lab, synth_env["grid_labels"], layout)
        ft_out = sub / "ft_out"
        ft_out.mkdir(parents=True)
        _run_cli(
            FT_DIR / "run_flux_tubes.py",
            [
                "--grid",
                str(synth_env["grid_h5"]),
                "--B",
                str(synth_env["b_h5"]),
                "--labels",
                str(ft_lab),
                "--sets",
                "cand0=0",
                "sheet1=1",
                "--min-cells",
                "2",
                "--min-elongation",
                "1.1",
                "--min-alignment",
                "0.5",
                "--out-dir",
                str(ft_out),
            ],
        )
        fs = json.loads((ft_out / "flux_tubes_summary.json").read_text())
        for k in ("labels", "load_seconds", "total_seconds", "peak_rss_gb"):
            fs.pop(k, None)
        for set_info in fs["sets"].values():
            set_info.pop("seconds", None)
            set_info.pop("peak_rss_gb_so_far", None)
        run_summaries[layout] = fs
        loaded_sets[layout] = pft.load_set(ft_out / "components_cand0.h5")

        # plot_flux_tubes.main reading --labels via open_npy and components via load_set
        _run_cli(
            FT_DIR / "plot_flux_tubes.py",
            [
                "--grid",
                str(synth_env["grid_h5"]),
                "--B",
                str(synth_env["b_h5"]),
                "--labels",
                str(ft_lab),
                "--dir",
                str(ft_out),
                "--candidate",
                "cand0",
                "--candidate-labels",
                "0",
                "--contrast",
                "sheet1",
                "--min-cells",
                "2",
            ],
        )
        for fig_name in ("fig_real_top10.png", "fig_real_slice.png", "fig_real_distributions.png"):
            assert (ft_out / fig_name).stat().st_size > 0
        # fig_real_slice.png is drawn from the loaded labels: compare its pixels across layouts.
        import matplotlib.image as mpimg
        slice_pixels[layout] = mpimg.imread(ft_out / "fig_real_slice.png")

    assert pred_checks["npy"] == pred_checks["parts"]
    np.testing.assert_array_equal(pred_labels["npy"], pred_labels["parts"])
    assert run_summaries["npy"] == run_summaries["parts"]
    for k in loaded_sets["npy"]:
        np.testing.assert_array_equal(loaded_sets["npy"][k], loaded_sets["parts"][k])
    np.testing.assert_array_equal(slice_pixels["npy"], slice_pixels["parts"])


def test_compare_runs_predict_labels_and_main_npy_vs_parts(tmp_path, synth_env):
    cr = _load_module("compare_runs_e8", BBH_DIR / "compare_runs.py")
    model = SOMModel.load(synth_env["model_h5"])

    # 1) predict_labels writes monolithic .npy; sharding its output and running compare_runs.main
    #    against both .npy and .parts produces identical sheet statistics and comparison tables.
    pred_npy = tmp_path / "predicted_labels.npy"
    info = cr.predict_labels(synth_env["grid_h5"], model, pred_npy, chunk_rows=64)
    assert info["n_points"] == len(synth_env["grid_labels"])
    pred_arr = np.load(pred_npy)

    tables = {}
    stats_by_layout = {}
    for layout in ("npy", "parts"):
        sub = tmp_path / f"cmp_{layout}"
        lab_path = write_label_layout(sub / "run_labels.npy", pred_arr, layout)
        out_dir = sub / "out"
        cr.main(
            [
                "--model",
                str(synth_env["model_h5"]),
                "--sheet-clusters",
                "0",
                "--run",
                f"r1={synth_env['grid_h5']}:{lab_path}",
                "--out",
                str(out_dir),
                "--min-cells",
                "2",
                "--chunk-rows",
                "64",
                "--plane-z",
                "0.125",
            ]
        )
        tables[layout] = (out_dir / "comparison.md").read_text()
        st = json.loads((out_dir / "runs/r1/r1_sheet_stats.json").read_text())
        st["run"].pop("labels", None)
        st["run"].pop("label_fractions_seconds", None)
        st["run"].pop("peak_rss_gb", None)
        st.pop("timings_s", None)
        st.pop("peak_rss_gb", None)
        stats_by_layout[layout] = st

    assert tables["npy"] == tables["parts"]
    assert stats_by_layout["npy"] == stats_by_layout["parts"]
