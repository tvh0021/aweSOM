"""Exercise the stored robust scaler through training and model freezing."""
import os
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from aweSOM import Scaler, SOMModel


def test_robust_training_round_trip(tmp_path):
    etk = Path('/Volumes/ExternalNVMe/etk-analysis')
    if not etk.exists():
        pytest.skip('local etk-analysis checkout unavailable')
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(42)
    x = rng.normal(size=(300, 6)).astype(np.float32)
    x[-1] = 10
    fitted = Scaler.fit(x, 'median-iqr')
    file = tmp_path/'features.h5'
    from aweSOM.features_grmhd import SIX_FEATURE_NAMES
    with h5py.File(file, 'w') as f:
        f['features'] = x
        f['names'] = np.array(SIX_FEATURE_NAMES, dtype='S')
        f['xyz'] = rng.normal(size=(300,3))
        f['level'] = np.zeros(300, dtype=int)
        f['weight_volume'] = np.ones(300)
        g = f.create_group('scaler')
        g['kind'] = 'median-iqr'
        g['center'] = fitted.offset
        g['scale'] = fitted.scale
        f.attrs.update(pipeline='v2', b2_min=1e-8, s_v=.1)
    env = dict(os.environ, PYTHONPATH=str(root/'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    train = [sys.executable, str(etk/'scripts/train_som.py'), '--input', str(file), '--n-samples', '300', '--train-steps', '300', '--output-dir', str(tmp_path/'train'), '--name', 'v2', '--merge-threshold', '.05']
    refused = subprocess.run(train, env=env, capture_output=True, text=True)
    assert refused.returncode and 'v2 requires --scaler file' in refused.stderr
    subprocess.run(train+['--scaler','file'], env=env, check=True, capture_output=True)
    lattice = next((tmp_path/'train').glob('som_object*.pkl'))
    out = tmp_path/'model.h5'
    build = [sys.executable, str(root/'examples/grmhd-bbh/build_model.py'), '--lattice', str(lattice), '--training-file', str(file), '--merge-cost','.05','-o',str(out)]
    refused = subprocess.run(build, env=env, capture_output=True, text=True)
    assert refused.returncode and 'v2 requires --scaler file' in refused.stderr
    result = subprocess.run(build+['--scaler','file'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    model = SOMModel.load(out)
    assert model.scaler.kind == 'median-iqr'
    assert model.metadata['feature_pipeline'] == 'v2'
    np.testing.assert_array_equal(model.scaler.offset, fitted.offset)
    np.testing.assert_array_equal(model.scaler.scale, fitted.scale)
    import pickle
    with lattice.open('rb') as f:
        lat = pickle.load(f)
    np.testing.assert_array_equal(model.scaler.transform(x), lat.data_array)
