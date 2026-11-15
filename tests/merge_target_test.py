"""Target merging preserves the historical threshold path and frozen labels."""
import contextlib
import io

import numpy as np
import pytest
from aweSOM import Lattice, SOMModel


def synthetic():
    lat = Lattice(12, 10)
    lat.lattice = np.random.default_rng(42).normal(size=(120, 3))
    lat.umat = lat.compute_umat()
    return lat


def test_target_and_legacy_path():
    lat = synthetic()
    with contextlib.redirect_stdout(io.StringIO()):
        legacy = lat.assign_cluster_to_lattice(merge_cost=.05).copy()
        explicit = lat.assign_cluster_to_lattice(merge_cost=.05, target_clusters=None)
        assert np.array_equal(legacy, explicit)
        target = lat.assign_cluster_to_lattice(target_clusters=3).copy()
        cost = lat.merge_cost_used
        assert lat.n_clusters == len(np.unique(target)) <= 3
        assert cost >= .05
        assert np.array_equal(target, lat.assign_cluster_to_lattice(merge_cost=cost))
        assert len(np.unique(lat.assign_cluster_to_lattice(merge_cost=np.nextafter(cost, -np.inf)))) > 3
        assert np.array_equal(target, lat.assign_cluster_to_lattice(target_clusters=3))


@pytest.mark.parametrize('target', [0, -1, 1.5, True])
def test_invalid_target(target):
    with pytest.raises(ValueError, match='target_clusters'):
        synthetic().assign_cluster_to_lattice(target_clusters=target)


def test_target_rejects_float32_umat():
    lat = Lattice(12, 10)
    lat.umat = np.random.default_rng(7).random((12, 10)).astype('f4')
    with pytest.raises(ValueError, match='float64 U-matrix'):
        lat.assign_cluster_to_lattice(target_clusters=3)
    with contextlib.redirect_stdout(io.StringIO()):
        lat.assign_cluster_to_lattice(merge_cost=.05)  # legacy path still accepts float32
        lat.umat = lat.umat.astype(np.float64)
        target = lat.assign_cluster_to_lattice(target_clusters=3).copy()
        cost = lat.merge_cost_used
        assert np.array_equal(target, lat.assign_cluster_to_lattice(merge_cost=cost))
        assert len(np.unique(lat.assign_cluster_to_lattice(merge_cost=np.nextafter(cost, -np.inf)))) > 3


def test_target_frozen_roundtrip(tmp_path, monkeypatch):
    lat = synthetic()
    with contextlib.redirect_stdout(io.StringIO()):
        assigned = lat.assign_cluster_to_lattice(target_clusters=3).copy()
    model = SOMModel.from_lattice(lat, scaled_training_data=lat.lattice)
    model.save(tmp_path/'target.h5')
    def forbidden(*args, **kwargs):
        raise AssertionError('inference must never rebuild clusters')
    monkeypatch.setattr(Lattice, 'assign_cluster_to_lattice', forbidden)
    loaded = SOMModel.load(tmp_path/'target.h5')
    assert np.array_equal(loaded.node_cluster, assigned.ravel(order='F'))
    assert float(loaded.metadata['merge_cost_used']) == lat.merge_cost_used
    assert int(loaded.metadata['target_clusters']) == 3
    assert int(loaded.metadata['n_clusters']) <= 3
    assert np.array_equal(loaded.predict(lat.lattice, backend='screen'), model.predict(lat.lattice, backend='screen'))


def test_grid_stream_uses_saved_target_map(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    import h5py

    lat = synthetic()
    with contextlib.redirect_stdout(io.StringIO()):
        lat.assign_cluster_to_lattice(target_clusters=3)
    model = SOMModel.from_lattice(lat, scaled_training_data=lat.lattice)
    path = tmp_path/'model.h5'
    model.save(path)
    grid = tmp_path/'grid.h5'
    with h5py.File(grid, 'w') as h:
        h['features'] = lat.lattice
        h['level'] = np.full(120, 5)
        h['names'] = np.array(model.feature_names, dtype='S')
    script = Path(__file__).parents[1]/'examples/grmhd-bbh/label_grid_stream.py'
    spec = importlib.util.spec_from_file_location('label_grid_stream', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def forbidden(*args, **kwargs):
        raise AssertionError('grid inference must use the saved map')
    monkeypatch.setattr(Lattice, 'assign_cluster_to_lattice', forbidden)
    output = tmp_path/'labels.npy'
    monkeypatch.setattr(sys, 'argv', [str(script), '--grid', str(grid), '--model', str(path), '--output', str(output)])
    module.main()
    assert np.array_equal(np.load(output), model.predict(lat.lattice, backend='patch'))


def test_threshold_mode_records_merge_cost_used_and_n_clusters(tmp_path):
    lat = synthetic()
    with contextlib.redirect_stdout(io.StringIO()):
        assigned = lat.assign_cluster_to_lattice(merge_cost=0.15, target_clusters=None).copy()
    assert lat.merge_cost_used == 0.15
    assert lat.target_clusters is None
    assert lat.n_clusters == len(np.unique(assigned))
    model = SOMModel.from_lattice(lat, scaled_training_data=lat.lattice)
    h5_path = tmp_path / 'threshold.h5'
    txt_path = tmp_path / 'threshold.txt'
    model.save(h5_path)
    model.save_text(txt_path)
    for m in (model, SOMModel.load(h5_path), SOMModel.load(txt_path)):
        assert np.array_equal(m.node_cluster, assigned.ravel(order='F'))
        assert float(m.metadata['merge_cost_used']) == 0.15
        assert int(m.metadata['n_clusters']) == lat.n_clusters
        assert 'target_clusters' not in m.metadata

    with contextlib.redirect_stdout(io.StringIO()):
        target_assigned = lat.assign_cluster_to_lattice(target_clusters=3).copy()
    tmodel = SOMModel.from_lattice(lat, scaled_training_data=lat.lattice)
    target_txt = tmp_path / 'target.txt'
    tmodel.save_text(target_txt)
    loaded_txt = SOMModel.load(target_txt)
    assert np.array_equal(loaded_txt.node_cluster, target_assigned.ravel(order='F'))
    assert float(loaded_txt.metadata['merge_cost_used']) == lat.merge_cost_used
    assert int(loaded_txt.metadata['target_clusters']) == 3
    assert int(loaded_txt.metadata['n_clusters']) == lat.n_clusters <= 3
    assert np.array_equal(
        loaded_txt.predict(lat.lattice, backend='screen'),
        tmodel.predict(lat.lattice, backend='screen'),
    )


def test_flat_umatrix_target_mode():
    for umat in (np.ones((6, 5), dtype=np.float64), np.zeros((6, 5), dtype=np.float64)):
        lat = Lattice(6, 5)
        lat.umat = umat
        with contextlib.redirect_stdout(io.StringIO()):
            assigned = lat.assign_cluster_to_lattice(target_clusters=2)
        assert not np.any(np.isnan(assigned))
        assert 1 <= lat.n_clusters == len(np.unique(assigned)) <= 2
        assert np.isfinite(lat.merge_cost_used) and lat.merge_cost_used >= 0.05


def test_plot_heat_keeps_merge_cost_used():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat = synthetic()
    lat.data_array = lat.lattice
    lat.labels = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            lat.plot_heat(lat.umat, merge=True, merge_cost=0.001)
        assert not hasattr(lat, "merge_cost_used")

        with contextlib.redirect_stdout(io.StringIO()):
            lat.assign_cluster_to_lattice(target_clusters=3)
            cost = lat.merge_cost_used
            assert cost != 0.001
            lat.plot_heat(lat.umat, merge=True, merge_cost=0.001)
        assert lat.merge_cost_used == cost
        model = SOMModel.from_lattice(lat, scaled_training_data=lat.lattice)
        assert float(model.metadata["merge_cost_used"]) == cost
    finally:
        plt.close("all")
