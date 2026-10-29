"""Training-file scaling shared by the GRMHD freezing and ensemble scripts."""
import json

import numpy as np

from .model import Scaler


def scaling_metadata(h, names):
    """Read optional feature-weight provenance without weighting a stored scale twice."""
    result = {}
    group = h.get('scaler')
    for key in ('feature_weights', 'scale_unweighted'):
        value = None
        for source in (group, h):
            if source is None:
                continue
            if key in source:
                value = source[key][()]
                break
            if key in source.attrs:
                value = source.attrs[key]
                break
        if value is None:
            continue
        if isinstance(value, (str, bytes)):
            value = json.loads(value)
        if isinstance(value, dict):
            value = [value.get(n, 1.) for n in names]
        array = np.asarray(value, dtype=np.float64)
        invalid = array <= 0 if key == 'feature_weights' else array < 0
        if array.shape != (len(names),) or not np.isfinite(array).all() or np.any(invalid):
            raise ValueError(f'{key} must contain one valid finite value per feature')
        result[key] = json.dumps(array.tolist())
    return result


def training_scaler(h, raw, names, dtype='float32', mode='auto'):
    """Keep frozen robust scales; reproduce legacy manual reductions by default.

    Stored median-IQR scales already include feature weights. Legacy manual
    training reduced in the input dtype and can carry separate weights.
    ``mode='file'`` selects the stored scaler for either kind.
    """
    meta = scaling_metadata(h, names)
    stored = Scaler.from_feature_file(h) if 'scaler' in h else None
    if mode == 'file' or stored is not None and stored.kind == 'median-iqr':
        if stored is None:
            raise ValueError('file scaling requires a scaler group')
        scaler = stored
        if scaler.kind == 'manual' and 'feature_weights' in meta:
            scaler = Scaler('manual', scaler.offset,
                            scaler.scale / np.array(json.loads(meta['feature_weights'])))
    else:
        x = np.asarray(raw, dtype=dtype)
        scale = (2. * np.std(x, axis=0)).astype(np.float64)
        weights = np.array(json.loads(meta.get('feature_weights', json.dumps([1.] * len(names)))))
        scaler = Scaler('manual', np.mean(x, axis=0).astype(np.float64), scale / weights)
    if len(scaler.scale) != len(names) or raw.ndim != 2 or raw.shape[1] != len(names):
        raise ValueError('training features, names and scaler widths must agree')
    return scaler, meta


def set_lattice_names(lattice, names):
    """Refuse to reinterpret weights in a different feature order."""
    existing = getattr(lattice, 'features_names', None)
    if existing is not None and list(existing) != list(names):
        raise ValueError('lattice and training feature names disagree')
    if np.shape(lattice.lattice)[1] != len(names):
        raise ValueError('lattice and training feature widths disagree')
    lattice.features_names = list(names)
