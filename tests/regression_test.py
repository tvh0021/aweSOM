"""Regression tests for bugs fixed on the research/grmhd-scale branch.

Each test reproduces one previously failing input. They are deliberately
small so the whole file runs in a few seconds.
"""

import io
import contextlib

import numpy as np
import pytest

from aweSOM import Lattice
from aweSOM.sce import compute_SQ


def _train(seed_data=0, train=500, **kw):
    data = np.random.default_rng(seed_data).normal(size=(2000, 4))
    som = Lattice(10, 8, 0.3, train, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(data, list("abcd"))
    return som, data


def test_training_is_reproducible_for_a_given_seed():
    # Before the fix, the sample order came from an unseeded Generator and the
    # initial lattice from the global NumPy stream, so two identical calls
    # produced different lattices despite seed=42.
    som_a, _ = _train()
    som_b, _ = _train()
    assert np.array_equal(som_a.lattice, som_b.lattice)


def test_short_training_runs_do_not_divide_by_zero():
    # train < 25 made alpha_freq == 0 and train < 10 made the progress
    # modulus zero, both raising ZeroDivisionError.
    som, _ = _train(train=20)
    assert som.epoch == 20
    som, _ = _train(train=7)
    assert som.epoch == 7


def test_restart_lattice_is_not_modified_in_place():
    som, data = _train(train=200)
    weights = som.lattice.copy()
    with contextlib.redirect_stdout(io.StringIO()):
        som.train_lattice(data, list("abcd"), number_of_steps=100, restart_lattice=weights)
    assert np.array_equal(weights, som.lattice_history[-1]) or not np.shares_memory(
        weights, som.lattice
    )
    assert som.epoch == 300


def test_single_centroid_does_not_crash_merge():
    # A U-matrix with one basin yields one centroid; the merge step used to
    # index into an empty cost list.
    som, _ = _train(train=200)
    som.umat = np.fromfunction(lambda i, j: (i - 5) ** 2 + (j - 4) ** 2, (10, 8), dtype=float)
    with contextlib.redirect_stdout(io.StringIO()):
        clusters = som.assign_cluster_to_lattice(merge_cost=0.2)
    assert clusters.max() == 0


def test_compute_sq_with_empty_masks_is_zero_without_warning():
    with np.errstate(all="raise"):
        sq, sq_matrix = compute_SQ(np.zeros(4), np.zeros(4))
    assert sq == 0.0
    assert np.all(sq_matrix == 0.0)


def test_sce_run_name_strips_only_the_npy_suffix(tmp_path, monkeypatch):
    # str.strip(".npy") removed characters, not the suffix, so a run whose
    # name ended in 'p' or 'n' lost that letter in multimap_mappings.txt.
    from aweSOM import sce

    labels = np.array([0, 0, 1, 1, 2, 2])
    np.save(tmp_path / "labels.x-1p.npy", labels)
    np.save(tmp_path / "labels.x-2n.npy", labels[::-1])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "SCE").mkdir()
    (tmp_path / "SCE" / "multimap_mappings.txt").write_text("")
    with contextlib.redirect_stdout(io.StringIO()):
        sce.loop_over_all_clusters(
            ["labels.x-1p.npy", "labels.x-2n.npy"], np.array([3, 3]), np.array([6]), "SCE"
        )
    text = (tmp_path / "SCE" / "multimap_mappings.txt").read_text()
    assert "labels.x-1p\n" in text and "labels.x-2n\n" in text
    assert (tmp_path / "SCE" / "mask-labels.x-1p-id0.npy").exists()
