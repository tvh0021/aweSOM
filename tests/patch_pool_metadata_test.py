"""Patch historical pooled models through the command line."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from aweSOM import SOMModel
from aweSOM.features_grmhd import FeatureThresholds, SEVEN_FEATURE_NAMES
from production_pool_test import inputs, args, pool as dev_pool

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'examples/grmhd-bbh/patch_pool_metadata.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def historical(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('historical-pool')
    source = tmp / 'historical_driver.py'
    source.write_bytes(subprocess.check_output(
        ['git', 'show', '0c65101:examples/grmhd-bbh/production_pool.py'], cwd=ROOT))
    spec = importlib.util.spec_from_file_location('historical_driver', source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    paths = inputs(tmp)
    models = []
    for driver, name in [(old, 'old'), (dev_pool, 'dev')]:
        a = args(tmp, paths)
        a.out = tmp / name
        a.bz_weight = a.pol_weight = 2.
        records, quota, _, _ = driver.inspect(a)
        a.out.mkdir()
        scaling = driver.pool(a, records, quota, a.out)
        a.anchor_attrs = records[0]['attrs']
        a.code_commit = 'historical-driver' if name == 'old' else 'dev-driver'
        result = driver.fit(a, (4, 3), 42, scaling, a.out)
        models.append(a.out / (result['tag'] + '.h5'))
    with h5py.File(tmp / 'old/pool.h5') as h:
        raw = h['raw'][:]
    return models[0], models[1], paths[0], raw


def run_patch(model, anchor, output, *extra):
    return subprocess.run([sys.executable, str(TOOL), '--model', str(model),
                           '--anchor', str(anchor), '-o', str(output), *extra],
                          capture_output=True, text=True)


def test_patch_matches_dev_and_preserves_frozen_model(historical, tmp_path):
    old, dev, anchor, raw = historical
    before = sha(old)
    with pytest.raises(KeyError, match='no feature thresholds'):
        FeatureThresholds.from_model(SOMModel.load(old))
    output = tmp_path / 'patched.h5'
    result = run_patch(old, anchor, output)
    assert result.returncode == 0, result.stderr
    patched = SOMModel.load(output)
    assert FeatureThresholds.from_model(patched).feature_names == SEVEN_FEATURE_NAMES
    expected = SOMModel.load(dev)
    ignored = {'aweSOM_commit', 'metadata_patched_by'}
    assert {k: v for k, v in patched.metadata.items() if k not in ignored} == {
        k: v for k, v in expected.metadata.items() if k not in ignored}
    assert {k for k in patched.metadata.keys() | expected.metadata.keys()
            if patched.metadata.get(k) != expected.metadata.get(k)} == ignored
    record = json.loads(patched.metadata['metadata_patched_by'])
    assert record == dict(tool_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        input_model_sha256=before, anchor_sha256=sha(anchor))
    assert sha(old) == before
    with h5py.File(old) as a, h5py.File(output) as b:
        def unchanged(name, obj):
            if isinstance(obj, h5py.Dataset):
                assert obj.dtype == b[name].dtype
                assert obj.shape == b[name].shape
                np.testing.assert_array_equal(obj[()], b[name][()])
                if obj.dtype.kind != 'O':
                    assert obj[()].tobytes() == b[name][()].tobytes(), name
            for key, value in obj.attrs.items():
                np.testing.assert_array_equal(value, b[name].attrs[key])
        a.visititems(unchanged)
        assert dict(a.attrs) == dict(b.attrs)
    np.testing.assert_array_equal(patched.predict(raw), SOMModel.load(old).predict(raw))
    np.testing.assert_array_equal(patched.predict(raw), expected.predict(raw))


@pytest.mark.parametrize('damage, message', [
    ('b2_min', 'b2_min'), ('s_v', 's_v'), ('names', 'feature names'),
    ('model-names', 'feature names'), ('model-pipeline', 'pipeline'),
    ('anchor-pipeline', 'pipeline'), ('model-scaler', 'median-iqr'),
    ('anchor-scaler', 'scaler'), ('missing-names', 'names'),
    ('missing-threshold', 's_v'), ('winsor', 'winsor'),
    ('conflicting-feature', 'feature_accuracy'),
    ('stale-winsor', 'feature_winsor'),
])
def test_refuse_inconsistent_inputs(historical, tmp_path, damage, message):
    import shutil
    old, _, anchor, _ = historical
    model = tmp_path / 'model.h5'
    ref = tmp_path / 'anchor.h5'
    shutil.copyfile(old, model)
    shutil.copyfile(anchor, ref)
    with h5py.File(model, 'r+') as m, h5py.File(ref, 'r+') as a:
        if damage in ('b2_min', 's_v'):
            a.attrs[damage] = 2.
        elif damage == 'names':
            a['names'][0] = b'wrong'
        elif damage == 'model-names':
            m['feature_names'][0] = 'wrong'
        elif damage == 'model-pipeline':
            m['metadata'].attrs['pipeline'] = 'v1'
        elif damage == 'anchor-pipeline':
            a.attrs['pipeline'] = 'v1'
        elif damage == 'model-scaler':
            m.attrs['scaler_kind'] = 'manual'
        elif damage == 'anchor-scaler':
            a.attrs['scaler_kind'] = 'manual'
        elif damage == 'missing-names':
            del a['names']
        elif damage == 'missing-threshold':
            del m['metadata'].attrs['s_v']
        elif damage == 'winsor':
            a.attrs['winsor_log_chi_B'] = [-1., 1.]
        elif damage == 'conflicting-feature':
            m['metadata'].attrs['feature_accuracy'] = '2'
        else:
            m['metadata'].attrs['feature_winsor_log_chi_B'] = '-1 1'
    before, ref_before = sha(model), sha(ref)
    output = tmp_path / 'rejected.h5'
    result = run_patch(model, ref, output)
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()
    assert sha(model) == before and sha(ref) == ref_before
    assert not list(tmp_path.glob('*.partial'))


@pytest.mark.parametrize('destination', ['model', 'anchor', 'existing', 'symlink', 'dangling'])
def test_never_overwrite(historical, tmp_path, destination):
    old, _, anchor, _ = historical
    output = tmp_path / 'output.h5'
    if destination == 'model':
        output = old
    elif destination == 'anchor':
        output = anchor
    elif destination == 'existing':
        output.write_bytes(b'keep me')
    else:
        output.symlink_to(old if destination == 'symlink' else tmp_path / 'missing')
    before = sha(old), sha(anchor)
    result = run_patch(old, anchor, output)
    assert result.returncode != 0
    assert 'already exists' in result.stderr
    assert (sha(old), sha(anchor)) == before
    if destination == 'existing':
        assert output.read_bytes() == b'keep me'
    if destination in ('symlink', 'dangling'):
        assert output.is_symlink()


def test_non_git_kit(historical, tmp_path):
    import shutil
    old, _, anchor, _ = historical
    tool = tmp_path / TOOL.name
    shutil.copyfile(TOOL, tool)
    output = tmp_path / 'kit-model.h5'
    cmd = [sys.executable, str(tool), '--model', str(old), '--anchor', str(anchor), '-o', str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0 and '--tool-commit' in result.stderr
    assert not output.exists()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    result = subprocess.run(cmd + ['--tool-commit', commit], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(SOMModel.load(output).metadata['metadata_patched_by'])['tool_commit'] == commit


def test_invalid_tool_commit(historical, tmp_path):
    old, _, anchor, _ = historical
    output = tmp_path / 'bad-commit.h5'
    result = run_patch(old, anchor, output, '--tool-commit', 'unknown')
    assert result.returncode != 0 and 'full 40-character' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('key,value', [
    ('accuracy', 2), ('epsilon', .002),
    ('direction_norm', 'other'), ('pipeline', 'v1'),
    ('scaler_kind', 'manual'), ('b2_min', 2.), ('s_v', 2.),
    ('winsor_log_chi_B', [-1., 1.]), ('winsor_quantiles', [.01, .99]),
    ('feature_names', ['wrong']), ('columns', 'wrong'), ('names', ['wrong']),
    ('exponent', 3.), ('gamma_K', 1.4), ('C_B_box', 6.),
])
@pytest.mark.parametrize('override', [False, True])
def test_saved_settings_mismatch_refused(historical, tmp_path, key, value, override):
    import shutil
    old, _, anchor, _ = historical
    model = tmp_path / 'model.h5'
    shutil.copyfile(old, model)
    with h5py.File(model, 'r+') as h:
        settings = json.loads(h['metadata'].attrs['feature_settings'])
        settings[key] = value
        h['metadata'].attrs['feature_settings'] = json.dumps(settings)
    before = sha(model), sha(anchor)
    output = tmp_path / 'rejected.h5'
    result = run_patch(model, anchor, output,
                       *(['--allow-missing-settings'] if override else []))
    assert result.returncode != 0
    assert 'feature_settings' in result.stderr and key in result.stderr
    assert not output.exists()
    assert not list(tmp_path.glob('.*.partial'))
    assert (sha(model), sha(anchor)) == before


@pytest.mark.parametrize('settings', [None, '{garbled', '[]', 'null'])
@pytest.mark.parametrize('override', [False, True])
def test_unavailable_saved_settings_require_explicit_override(historical, tmp_path, settings, override):
    import shutil
    old, _, anchor, _ = historical
    model = tmp_path / 'model.h5'
    shutil.copyfile(old, model)
    with h5py.File(model, 'r+') as h:
        if settings is None:
            del h['metadata'].attrs['feature_settings']
        else:
            h['metadata'].attrs['feature_settings'] = settings
    before = sha(model), sha(anchor)
    output = tmp_path / 'patched.h5'
    result = run_patch(model, anchor, output,
                       *(['--allow-missing-settings'] if override else []))
    if override:
        assert result.returncode == 0, result.stderr
        patched = SOMModel.load(output)
        assert FeatureThresholds.from_model(patched).accuracy == 4
        record = json.loads(patched.metadata['metadata_patched_by'])
        assert record['allow_missing_settings'] is True
    else:
        assert result.returncode != 0
        assert 'feature_settings' in result.stderr
        assert '--allow-missing-settings' in result.stderr
        assert not output.exists()
    assert not list(tmp_path.glob('.*.partial'))
    assert (sha(model), sha(anchor)) == before


def test_saved_settings_accept_equivalent_hdf5_bytes(historical, tmp_path):
    import shutil
    old, _, anchor, _ = historical
    model, ref = tmp_path / 'model.h5', tmp_path / 'anchor.h5'
    shutil.copyfile(old, model)
    shutil.copyfile(anchor, ref)
    with h5py.File(model, 'r+') as m, h5py.File(ref, 'r+') as a:
        a.attrs['inputs'] = np.array(['Bx', 'By'], dtype='S')
        settings = json.loads(m['metadata'].attrs['feature_settings'])
        settings.update(inputs=['Bx', 'By'], feature_names=list(SEVEN_FEATURE_NAMES),
                        columns=' '.join(SEVEN_FEATURE_NAMES), winsor={},
                        scaler_kind='median-iqr')
        m['metadata'].attrs['feature_settings'] = json.dumps(settings)
    output = tmp_path / 'patched.h5'
    result = run_patch(model, ref, output)
    assert result.returncode == 0, result.stderr
    assert FeatureThresholds.from_model(SOMModel.load(output)).accuracy == 4


def test_review_accuracy_anchor_mismatch(historical, tmp_path):
    import shutil
    old, _, anchor, _ = historical
    ref = tmp_path / 'accuracy2.h5'
    shutil.copyfile(anchor, ref)
    with h5py.File(ref, 'r+') as h:
        h.attrs['accuracy'] = 2
    output = tmp_path / 'rejected.h5'
    result = run_patch(old, ref, output)
    assert result.returncode != 0
    assert 'feature_settings' in result.stderr and 'accuracy' in result.stderr
    assert not output.exists()
