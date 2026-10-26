"""The contingency-table SCE loop must reproduce the original mask-by-mask loop bit for bit."""

import contextlib
import io
import os

import numpy as np
import pytest

from aweSOM import sce


def reference_loop(all_files, number_of_clusters, dimensions, subfolder):
    """The original implementation: every cluster against every other cluster on full masks."""
    runs = all_files
    for i in range(len(runs)):
        run = runs[i]
        clusters = np.load(run, "r").reshape(dimensions)
        with open(subfolder + "/multimap_mappings.txt", "a") as f:
            f.write("{}\n".format(run.removesuffix(".npy")))
        for cid in range(number_of_clusters[i]):
            mask = np.where(clusters == cid, 1, 0)
            total_mask = np.zeros(dimensions, dtype=float)
            total_SQ_scalar = 0.0
            for j in range(len(runs)):
                if j == i:
                    continue
                clustersC = np.load(runs[j], "r").reshape(dimensions)
                for cidC in range(number_of_clusters[j]):
                    maskC = np.where(clustersC == cidC, 1, 0)
                    SQ, SQ_matrix = sce.compute_SQ(mask, maskC)
                    total_mask += SQ_matrix
                    total_SQ_scalar += SQ
            np.save(subfolder + "/mask-{}-id{}.npy".format(run.removesuffix(".npy"), cid), total_mask)
            with open(subfolder + "/multimap_mappings.txt", "a") as f:
                f.write("{} {}\n".format(cid, total_SQ_scalar))


def _write_runs(folder, rng, dims, specs):
    """specs: list of (n_ids, dtype, drop_id) -> files with optionally one id missing (non-contiguous ids).

    Returns bare file names: like the CLI, the loop runs inside the folder and uses the run name in the mask file name.
    """
    files = []
    for k, (n_ids, dtype, drop) in enumerate(specs):
        lab = rng.integers(0, n_ids, size=dims)
        if drop is not None:
            lab[lab == drop] = n_ids - 1
        name = f"labels.run{k}.npy"
        np.save(os.path.join(folder, name), lab.astype(dtype))
        files.append(name)
    return files


def _parse(text):
    """multimap_mappings.txt -> list of (run, cid, Gsum)."""
    rows, run = [], None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 1:
            run = parts[0]
        else:
            rows.append((run, int(parts[0]), float(parts[1])))
    return rows


def _assert_same(out):
    """Bitwise on the NumPy backend. The old loop on a GPU/Metal backend summed in float32 (jax-metal has no
    float64), so there the new exact-count loop is compared at float32 precision instead."""
    new_text, new_masks = out["new"]
    ref_text, ref_masks = out["ref"]
    assert new_masks.keys() == ref_masks.keys()
    if not sce.USE_JAX:
        assert new_text == ref_text  # same text, same values, same order
        for name, ref in ref_masks.items():
            got = new_masks[name]
            assert got.dtype == ref.dtype and got.shape == ref.shape
            assert np.array_equal(got, ref), name
        return
    new_rows, ref_rows = _parse(new_text), _parse(ref_text)
    assert [r[:2] for r in new_rows] == [r[:2] for r in ref_rows]
    np.testing.assert_allclose([r[2] for r in new_rows], [r[2] for r in ref_rows], rtol=1e-5, atol=1e-6)
    for name, ref in ref_masks.items():
        got = new_masks[name]
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, np.asarray(ref, dtype=float), rtol=1e-5, atol=1e-6, err_msg=name)


def _run_both(tmp_path, files, dims):
    nids = sce.find_number_of_clusters(files)
    out = {}
    for name, fn in (("new", sce.loop_over_all_clusters), ("ref", reference_loop)):
        sub = tmp_path / name
        sub.mkdir()
        open(sub / "multimap_mappings.txt", "w").close()
        with contextlib.redirect_stdout(io.StringIO()):
            fn(files, nids, np.array(dims), str(sub))
        masks = {p.name: np.load(p) for p in sub.glob("mask-*.npy")}
        out[name] = (open(sub / "multimap_mappings.txt").read(), masks)
    return out


@pytest.mark.parametrize(
    "dims, specs",
    [
        ((6, 7, 8), [(4, np.int64, None), (5, np.int64, None), (3, np.int64, None)]),
        ((40, 40), [(6, np.int32, None), (6, np.int32, 2), (2, np.int32, None)]),  # one run with a missing id
        ((300,), [(3, np.float64, None), (4, np.float64, None)]),  # float labels as written by old scripts
    ],
)
def test_loop_matches_reference_bitwise(tmp_path, monkeypatch, dims, specs):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    files = _write_runs(str(tmp_path), rng, dims, specs)
    _assert_same(_run_both(tmp_path, files, dims))


def test_loop_with_an_entirely_empty_cluster(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # ids 0..n-1 are assumed; an id that never occurs gives an empty mask, SQ = 0 and an all-zero map
    dims = (5, 5, 5)
    lab = np.full(dims, 2)
    lab[:2] = 0
    np.save(tmp_path / "labels.a.npy", lab)
    np.save(tmp_path / "labels.b.npy", np.where(lab == 0, 1, 0))
    files = ["labels.a.npy", "labels.b.npy"]
    _assert_same(_run_both(tmp_path, files, dims))


def test_pair_quality_from_counts_matches_compute_sq():
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = rng.integers(0, 2, size=200)
        b = rng.integers(0, 2, size=200)
        sq, _ = sce.compute_SQ(a, b)
        got = sce.pair_quality_from_counts(int(a.sum()), int(b.sum()), int((a * b).sum()))
        if sce.USE_JAX:
            assert np.isclose(got, float(sq), rtol=1e-5)
        else:
            assert got == float(sq)
    assert sce.pair_quality_from_counts(0, 5, 0) == 0.0
    assert sce.pair_quality_from_counts(5, 5, 5) == 0.0  # identical masks: Q = 0


def test_load_labels_maps_out_of_range_ids_to_minus_one(tmp_path):
    # ids that would wrap when narrowed to int32 must not become valid cluster ids
    path = tmp_path / "labels.big.npy"
    np.save(path, np.array([0, 2**32, -(2**32), 1, 1.5]))
    assert sce._load_labels(str(path)).tolist() == [0, -1, -1, 1, -1]
