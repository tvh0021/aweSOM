"""Grid clipping follows file column names for five- and six-column files."""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / 'examples/grmhd-bbh/proposal_figure.py'
spec = importlib.util.spec_from_file_location('proposal_figure', SCRIPT)
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)


@pytest.mark.parametrize('six', [False, True])
def test_clip_columns_follow_names(tmp_path, six):
    names = ['log_beta_inv', 'log_theta', 'log_chi_B', 'log_grad_ln_b2', 'asinh_div_v']
    if six:
        names.append('bz_frac')
    names = names[::-1]  # name lookup must work independently of canonical order
    path = tmp_path / 'features.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('names', data=np.array(names, dtype='S'))
        f.attrs.update(iteration=1, b2_min=0.1, s_v=0.5, winsor_log_chi_B=(-3.0, 1.0))
    clip = pf.grid_clip(path, path)
    assert clip['limits'] == [(names.index('log_chi_B'), -3.0, 1.0)]
