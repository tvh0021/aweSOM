"""Frozen v1 compatibility and seven-column parity with etkplot 38f251f."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from aweSOM import SOMModel, Scaler
from aweSOM.features_grmhd import (
    FEATURE_NAMES, SIX_FEATURE_NAMES, SEVEN_FEATURE_NAMES, INPUTS,
    FeatureThresholds, compute_features, raw_features,
)


@pytest.fixture
def etkf():
    from features_grmhd_test import ETKPLOT
    sys.path.insert(0, str(ETKPLOT))
    try:
        yield pytest.importorskip('etkplot.features')
    finally:
        sys.path.remove(str(ETKPLOT))


@pytest.mark.parametrize('order', [2, 4, 6])
@pytest.mark.parametrize('names', [FEATURE_NAMES, SIX_FEATURE_NAMES])
def test_v1_bytes_from_dev(order, names):
    with np.load(Path(__file__).parent / 'fixtures/grmhd_v1_dev_f9d60d2.npz') as f:
        a = {n: f[f'a_{n}'] for n in INPUTS}
        thr = FeatureThresholds(.5, .7, {'log_chi_B': (-3, .4),
            'log_grad_ln_b2': (-1, 1), 'asinh_div_v': (-1, 1)}, accuracy=order)
        got = compute_features(a, (.13, .21, .35), thr, names=names)
        assert got.tobytes() == f[f'v1_{order}_{len(names)}'].tobytes()


@pytest.mark.parametrize('pipeline', ['v1', 'v2'])
@pytest.mark.parametrize('order', [2, 4, 6])
@pytest.mark.parametrize('floor', [0., .5])
def test_seven_exact_etkplot(etkf, pipeline, order, floor):
    rng = np.random.default_rng(632)
    shape = (12, 11, 10)
    axes = (np.linspace(-3, 3, shape[0]), np.linspace(-2, 2, shape[1]), np.linspace(-1, 1, shape[2]))
    a = {n: rng.uniform(.01, 2, shape) for n in INPUTS}
    for n in ('Bx', 'By', 'Bz'):
        a[n][4, 4, 4] = 0
        a[n][5, 4, 4] = 1e-20
    a['Bx'][6, 5, 4] = np.nan
    centres = [[-2., .3, 0], [2., -.3, 1]]
    opt = SimpleNamespace(pipeline=pipeline, accuracy=order, b2_min=floor, s_v=.7, hole_centres=centres)
    thr = FeatureThresholds(floor, .7, accuracy=order, pipeline=pipeline)
    with np.errstate(invalid='ignore', divide='ignore'):
        want = etkf.compute(SEVEN_FEATURE_NAMES, a, (.13, .21, .35), opt, xyz=axes)
        got = compute_features(a, (.13, .21, .35), thr, names=SEVEN_FEATURE_NAMES, xyz=axes, hole_centres=centres)
        selected = compute_features(a, (.13, .21, .35), thr, names=SEVEN_FEATURE_NAMES,
            xyz=axes, hole_centres=centres, select=(slice(3,-3),)*3)
    expected = np.column_stack([want[n].ravel() for n in SEVEN_FEATURE_NAMES]).astype(np.float32)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(selected, expected.reshape(*shape,7)[3:-3,3:-3,3:-3].reshape(-1,7))


@pytest.mark.parametrize('pipeline', ['v1', 'v2'])
def test_pol_geometry_and_zero(etkf, pipeline):
    thr = FeatureThresholds(0., 1., pipeline=pipeline)
    centres = [[-6., 0., 0.], [6., 0., 10.]]
    for point, expected in [((-6., 0., 0.), np.sqrt(.5)),
                            ((0., 1., 5.), 7/np.sqrt(74)),  # equal 3D distance: first hole
                            ((-1., 1., 9.), 6/np.sqrt(100))]:
        a = {n: np.ones((1,1,1)) for n in ('Bx','By','smallb2')}
        a['Bz'] = np.zeros((1,1,1))
        got = compute_features(a, (1,1,1), thr, names=('pol_frac',), xyz=tuple([v] for v in point), hole_centres=centres)
        assert got[0,0] == pytest.approx(expected, rel=1e-6)
    for n in ('Bx','By'): a[n][:] = 0
    assert compute_features(a,(1,1,1),thr,names=('pol_frac',),xyz=([0],[0],[0]),hole_centres=centres)[0,0] == 0


@pytest.mark.parametrize('bad', [None, [[0,0,0]], [[0,0,0],[np.nan,0,0]]])
def test_explicit_geometry_required(bad):
    a = {n: np.ones((1,1,1)) for n in ('Bx','By','Bz')}
    with pytest.raises(ValueError, match='hole_centres'):
        raw_features(a,(1,1,1),names=('pol_frac',),xyz=([0],[0],[0]),hole_centres=bad)
    with pytest.raises(ValueError, match='coordinates'):
        raw_features(a,(1,1,1),names=('pol_frac',),hole_centres=[[0,0,0],[1,0,0]])


@pytest.mark.parametrize('names', [SIX_FEATURE_NAMES, SEVEN_FEATURE_NAMES])
def test_v2_model_roundtrip_and_mismatches(names, tmp_path):
    thr = FeatureThresholds(1e-8, .04, pipeline='v2', feature_names=names)
    model = SOMModel(xdim=2, ydim=2, weights=np.ones((4,len(names))), node_cluster=np.arange(4),
        feature_names=list(names), scaler=Scaler('median-iqr', np.arange(len(names)), np.ones(len(names))),
        node_dist_q99=np.ones(4), reject_distance=1., metadata=thr.to_metadata())
    for ext in ('h5','txt'):
        p = tmp_path / f'm.{ext}'
        model.save(p) if ext == 'h5' else model.save_text(p)
        loaded = SOMModel.load(p)
        assert FeatureThresholds.from_model(loaded) == thr
        np.testing.assert_array_equal(loaded.scaler.offset, model.scaler.offset)
        np.testing.assert_array_equal(loaded.scaler.scale, model.scaler.scale)
    model.scaler = Scaler.identity(len(names))
    with pytest.raises(ValueError, match='median-iqr'):
        FeatureThresholds.from_model(model)
    for key, val in [('feature_set','grmhd-etkplot-v1'), ('feature_scaler_kind','manual'),
                     ('feature_columns',' '.join(FEATURE_NAMES)), ('feature_direction_norm','comoving')]:
        model.metadata = {**thr.to_metadata(), key: val}
        with pytest.raises(ValueError): FeatureThresholds.from_model(model)


def test_v2_h5_metadata(tmp_path):
    import h5py
    p = tmp_path / 'train.h5'
    with h5py.File(p,'w') as f:
        f.attrs.update(b2_min=1e-8,s_v=.04,pipeline='v2',scaler_kind='median-iqr')
        f['names'] = np.array(SEVEN_FEATURE_NAMES,dtype='S')
    thr = FeatureThresholds.from_h5(p)
    assert thr.feature_names == SEVEN_FEATURE_NAMES
    assert FeatureThresholds.from_metadata(thr.to_metadata()) == thr
    for bad in (-1, np.nan, np.inf):
        with pytest.raises(ValueError): FeatureThresholds(bad,1,pipeline='v2')
    with pytest.raises(ValueError): FeatureThresholds(0,1,pipeline='v2',winsor={'log_chi_B':(-3,1)})
    with pytest.raises(ValueError): raw_features({},(1,1,1),pipeline='v3')
