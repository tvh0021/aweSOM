"""Freeze CLI regressions across training feature sets and scaler formats."""
import importlib.util
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest

from aweSOM import Lattice, SOMModel, Scaler
from aweSOM.features_grmhd import SEVEN_FEATURE_NAMES, FeatureThresholds

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'examples/grmhd-bbh' / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def training_case(tmp_path, n=7, kind='median-iqr', weighted=True):
    names = list(SEVEN_FEATURE_NAMES[:n])
    raw = np.random.default_rng(78).normal(size=(80, n)).astype('f4')
    weights = np.array([2. if weighted and name in ('bz_frac', 'pol_frac') else 1. for name in names])
    if kind == 'median-iqr':
        offset = np.median(raw.astype('f8'), axis=0)
        scale = np.subtract(*np.percentile(raw.astype('f8'), [75, 25], axis=0))
    else:
        offset = raw.mean(0).astype('f8')
        scale = (2 * raw.std(0)).astype('f8')
    scaler = Scaler(kind, offset, scale / weights)
    path = tmp_path / 'training.h5'
    with h5py.File(path, 'w') as h:
        h['features'] = raw
        h['names'] = np.array(names, dtype='S')
        h.attrs.update(b2_min=1e-8, s_v=.04, pipeline='v2' if kind == 'median-iqr' else 'v1', scaler_kind=kind)
        g = h.create_group('scaler'); g.attrs['kind'] = kind
        if kind == 'median-iqr':
            g['center'] = offset; g['scale'] = scale / weights
        else:
            g['mean'] = raw.astype('f8').mean(0); g['sigma'] = raw.astype('f8').std(0)
        g['scale_unweighted'] = scale; g['feature_weights'] = weights
    lat = Lattice(3, 2)
    lat.lattice = np.random.default_rng(81).normal(size=(6, n))
    lat.data_array = scaler.transform(raw)
    lat.features_names = names
    lat.umat = lat.compute_umat()
    pkl = tmp_path / 'lattice.pkl'
    with pkl.open('wb') as f: pickle.dump(lat, f)
    return path, pkl, raw, scaler


def freeze_case(tmp_path, n=7, kind='median-iqr', weighted=True):
    path, pkl, raw, scaler = training_case(tmp_path, n, kind, weighted)
    out = tmp_path / 'frozen.h5'
    script('freeze_model').main(['--training', str(path), '--lattice', str(pkl), '--merge-cost', '.1', '-o', str(out)])
    return SOMModel.load(out), raw, scaler


def test_seven_feature_weighted_freeze(tmp_path):
    model, raw, scaler = freeze_case(tmp_path)
    assert model.feature_names == list(SEVEN_FEATURE_NAMES)
    np.testing.assert_array_equal(model.scaler.scale, scaler.scale)
    assert json.loads(model.metadata['feature_weights']) == [1, 1, 1, 1, 1, 2, 2]
    assert FeatureThresholds.from_model(model).feature_names == SEVEN_FEATURE_NAMES
    assert np.isfinite(model.predict(raw)).all()


@pytest.mark.parametrize('n', [5, 6, 7])
@pytest.mark.parametrize('kind', ['manual', 'median-iqr'])
@pytest.mark.parametrize('weighted', [False, True])
def test_feature_scaler_roundtrip(tmp_path, n, kind, weighted):
    model, raw, scaler = freeze_case(tmp_path, n, kind, weighted)
    np.testing.assert_array_equal(model.scaler.offset, scaler.offset)
    np.testing.assert_array_equal(model.scaler.scale, scaler.scale)
    assert len(json.loads((tmp_path / 'frozen.json').read_text())['cluster_means']['0']) == n + 1


def test_legacy_fixture_labels_unchanged(tmp_path):
    old = SOMModel.load(ROOT / 'tests/fixtures/grmhd_five_dev_968e8d6.h5')
    with np.load(ROOT / 'tests/fixtures/grmhd_freeze_dev_968e8d6.npz') as f:
        model, raw, scaler = freeze_case(tmp_path, 5, 'manual', False)
        np.testing.assert_array_equal(old.predict(f['raw']), f['labels'])
        np.testing.assert_array_equal(model.predict(raw), f['labels'])
        for name in ('weights', 'node_cluster', 'node_dist_q99'):
            np.testing.assert_array_equal(getattr(model, name), f[name])
        assert model.reject_distance == f['reject_distance']


def test_build_model_auto_robust_scaler_and_names(tmp_path):
    path, pkl, raw, scaler = training_case(tmp_path)
    with pkl.open('rb') as f: lat = pickle.load(f)
    from aweSOM.inference import best_match_nodes_screen
    lat.features_names = None
    lat.projection_1d = best_match_nodes_screen(lat.lattice, lat.data_array)[0]
    with pkl.open('wb') as f: pickle.dump(lat, f)
    out = tmp_path / 'built.h5'
    script('build_model').main(['--training-file', str(path), '--lattice', str(pkl), '--merge-cost', '.1', '-o', str(out)])
    m = SOMModel.load(out)
    assert m.feature_names == list(SEVEN_FEATURE_NAMES)
    np.testing.assert_array_equal(m.scaler.scale, scaler.scale)
    assert json.loads(m.metadata['feature_weights'])[-2:] == [2., 2.]


def test_ensemble_uses_frozen_robust_scale(tmp_path):
    path, _, raw, scaler = training_case(tmp_path)
    ensemble = script('stack_ensemble')
    cfg = dict(input=str(path), N=80, train=20, ratio=.7, scale=4., alpha_0=.5,
               seed=42, name='tiny', output_dir=str(tmp_path), merge_cost=.1)
    result = ensemble.run_realization(cfg)
    with open(result['som_object_file'], 'rb') as f: lat = pickle.load(f)
    expected = ensemble.train_realization(scaler.transform(raw), list(SEVEN_FEATURE_NAMES),
        result['lattice_xdim'], result['lattice_ydim'], 42, .5, 20, .1)[0]
    np.testing.assert_array_equal(lat.lattice, expected.lattice)


def test_stream_refuses_models_with_different_feature_order(tmp_path, monkeypatch):
    model, raw, _ = freeze_case(tmp_path)
    model.feature_names = list(reversed(model.feature_names))
    other = tmp_path / 'other.h5'; model.save(other)
    grid = tmp_path / 'grid.h5'
    with h5py.File(grid, 'w') as h:
        h['features'] = raw; h['names'] = np.array(SEVEN_FEATURE_NAMES, dtype='S'); h['level'] = np.ones(len(raw))
    monkeypatch.setattr('sys.argv', ['label_grid_stream', '--grid', str(grid),
        '--model', str(tmp_path / 'frozen.h5'), '--model', str(other),
        '--output', str(tmp_path / 'a.npy'), '--output', str(tmp_path / 'b.npy')])
    with pytest.raises(ValueError, match='feature names'):
        script('label_grid_stream').main()


def test_production_freeze_equivalence(tmp_path):
    # Exercise production_pool.fit itself, then freeze exactly its trained weights.
    from production_pool_test import inputs, args
    pool = script('production_pool')
    paths = inputs(tmp_path); a = args(tmp_path, paths)
    a.bz_weight = a.pol_weight = 2.
    records, quota, _, _ = pool.inspect(a)
    a.out.mkdir()
    scaling = pool.pool(a, records, quota, a.out)
    a.anchor_attrs = records[0]['attrs']; a.code_commit = 'synthetic-equivalence'
    result = pool.fit(a, (4, 3), 42, scaling, a.out)
    prod = SOMModel.load(a.out / (result['tag'] + '.h5'))
    assert FeatureThresholds.from_model(prod).feature_names == SEVEN_FEATURE_NAMES
    with h5py.File(a.out / 'pool.h5') as h: raw = h['raw'][:]
    path = tmp_path / 'production-training.h5'
    with h5py.File(path, 'w') as h:
        h['features'] = raw; h['names'] = np.array(prod.feature_names, dtype='S')
        h.attrs.update(a.anchor_attrs)
        g = h.create_group('scaler'); g.attrs['kind'] = 'median-iqr'
        g['center'] = prod.scaler.offset; g['scale'] = prod.scaler.scale
        g['feature_weights'] = scaling['feature_weights']; g['scale_unweighted'] = scaling['scale_unweighted']
    lat = Lattice(prod.xdim, prod.ydim); lat.lattice = prod.weights
    lat.features_names = prod.feature_names; lat.data_array = prod.scaler.transform(raw); lat.umat = lat.compute_umat()
    pkl = tmp_path / 'production.pkl'
    with pkl.open('wb') as f: pickle.dump(lat, f)
    out = tmp_path / 'equivalent.h5'
    script('freeze_model').main(['--training', str(path), '--lattice', str(pkl),
        '--merge-cost', str(a.merge_cost), '-o', str(out)])
    frozen = SOMModel.load(out)
    np.testing.assert_array_equal(frozen.predict(raw), prod.predict(raw))
    np.testing.assert_array_equal(frozen.node_cluster, prod.node_cluster)
    np.testing.assert_array_equal(frozen.scaler.transform(raw), prod.scaler.transform(raw))
    np.testing.assert_array_equal(frozen.node_dist_q99, prod.node_dist_q99)
    assert frozen.reject_distance == prod.reject_distance
    (tmp_path / 'equivalence.json').write_text(json.dumps(dict(rows=len(raw), features=raw.shape[1],
        label_mismatches=int(np.count_nonzero(frozen.predict(raw) != prod.predict(raw))),
        scaler_max_abs_error=float(np.max(np.abs(frozen.scaler.transform(raw)-prod.scaler.transform(raw)))),
        distance_quantiles_equal=True, reject_distance_equal=True)))


def test_build_manual_file_weights(tmp_path):
    path, pkl, raw, _ = training_case(tmp_path, 7, 'manual', True)
    with pkl.open('rb') as f: lat = pickle.load(f)
    from aweSOM.inference import best_match_nodes_screen
    lat.projection_1d = best_match_nodes_screen(lat.lattice, lat.data_array)[0]
    with pkl.open('wb') as f: pickle.dump(lat, f)
    out = tmp_path / 'manual-file.h5'
    script('build_model').main(['--training-file', str(path), '--lattice', str(pkl),
        '--merge-cost', '.1', '--scaler', 'file', '-o', str(out)])
    with h5py.File(path) as h:
        expected = 2 * h['scaler/sigma'][:] / h['scaler/feature_weights'][:]
    np.testing.assert_array_equal(SOMModel.load(out).scaler.scale, expected)


def test_zero_iqr_provenance_is_preserved(tmp_path):
    path, pkl, _, _ = training_case(tmp_path)
    with h5py.File(path, 'r+') as h:
        h['scaler/scale_unweighted'][0] = 0.
    out = tmp_path / 'zero-iqr.h5'
    script('freeze_model').main(['--training', str(path), '--lattice', str(pkl),
        '--merge-cost', '.1', '-o', str(out)])
    assert json.loads(SOMModel.load(out).metadata['scale_unweighted'])[0] == 0.


@pytest.mark.parametrize('consumer', ['freeze_model', 'stack_ensemble'])
@pytest.mark.parametrize('damage', [
    'missing-v2-scaler', 'manual-v2-scaler', 'unapplied-weights',
    'conflicting-weights', 'unknown-weight-name',
])
def test_reject_inconsistent_scaling_before_export_or_training(tmp_path, monkeypatch, consumer, damage):
    path, pkl, raw, _ = training_case(tmp_path)
    with h5py.File(path, 'r+') as h:
        if damage == 'missing-v2-scaler':
            del h['scaler']
        elif damage == 'manual-v2-scaler':
            del h['scaler']
            g = h.create_group('scaler')
            g.attrs['kind'] = 'manual'
            g['mean'] = raw.mean(0)
            g['sigma'] = raw.std(0)
        elif damage == 'unapplied-weights':
            h['scaler/scale'][:] = h['scaler/scale_unweighted'][:]
        elif damage == 'conflicting-weights':
            h.attrs['feature_weights'] = json.dumps([1.] * 7)
        else:
            del h['scaler/feature_weights']
            h['scaler'].attrs['feature_weights'] = json.dumps({'bz_fracc': 2., 'pol_frac': 2.})
    # The training array is optional; provenance must catch these errors itself.
    with pkl.open('rb') as f:
        lat = pickle.load(f)
    lat.data_array = None
    with pkl.open('wb') as f:
        pickle.dump(lat, f)
    mod = script(consumer)
    if consumer == 'stack_ensemble':
        def unexpected_training(*args, **kwargs):
            pytest.fail('invalid scaling reached ensemble training')
        monkeypatch.setattr(mod, 'train_realization', unexpected_training)
    message = ('median-iqr' if damage in ('missing-v2-scaler', 'manual-v2-scaler')
               else 'scale' if damage == 'unapplied-weights' else 'feature_weights')
    with pytest.raises(ValueError, match=message):
        if consumer == 'freeze_model':
            mod.main(['--training', str(path), '--lattice', str(pkl),
                      '--merge-cost', '.1', '-o', str(tmp_path / 'rejected.h5')])
        else:
            mod.run_realization(dict(input=str(path), N=80, train=20, ratio=.7,
                scale=4., alpha_0=.5, seed=42, name='tiny', output_dir=str(tmp_path), merge_cost=.1))
    assert not (tmp_path / 'rejected.h5').exists()
    assert not list(tmp_path.glob('realizations/som_object.*.pkl'))


@pytest.mark.parametrize('declaration', ['pipeline', 'scaler_kind'])
@pytest.mark.parametrize('kind', ['missing', 'manual'])
def test_ensemble_requires_robust_scaler_for_either_declaration(tmp_path, declaration, kind):
    path, _, raw, _ = training_case(tmp_path)
    with h5py.File(path, 'r+') as h:
        del h.attrs['scaler_kind' if declaration == 'pipeline' else 'pipeline']
        del h['scaler']
        if kind == 'manual':
            g = h.create_group('scaler')
            g.attrs['kind'] = 'manual'
            g['mean'] = raw.mean(0)
            g['sigma'] = raw.std(0)
    with pytest.raises(ValueError, match='median-iqr'):
        script('stack_ensemble').run_realization(dict(input=str(path), N=80, train=20,
            ratio=.7, scale=4., alpha_0=.5, seed=42, name='tiny', output_dir=str(tmp_path), merge_cost=.1))
