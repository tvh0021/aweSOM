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
