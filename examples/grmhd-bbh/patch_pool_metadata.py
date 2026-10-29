#!/usr/bin/env python3
"""Add frozen feature thresholds to a copied historical pooled model.

Usage: patch_pool_metadata.py --model MODEL.h5 --anchor FEATURES.h5 -o NEW.h5
For a kit without git, supply --tool-commit with the full commit containing this
script. The output must not exist. Existing model datasets are copied verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import h5py
import numpy as np

from aweSOM.features_grmhd import FeatureThresholds


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def plain(value):
    """Match the historical driver's JSON representation of HDF5 attributes."""
    if isinstance(value, (np.ndarray, np.generic)):
        return plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value.decode() if isinstance(value, bytes) else value


def tool_commit(explicit):
    if explicit is None:
        try:
            explicit = subprocess.check_output(
                ['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'],
                text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError('outside git, supply --tool-commit with the tool commit') from exc
    if not re.fullmatch(r'[0-9a-f]{40}', explicit):
        raise ValueError('tool commit must be a full 40-character lowercase git SHA')
    return explicit


def validate_settings(meta, ref, thresholds, names, allow_missing_settings):
    """Check the historical driver's saved anchor settings before copying."""
    try:
        settings = json.loads(text(meta['feature_settings']))
        if not isinstance(settings, dict):
            raise ValueError('expected a JSON object')
    except (KeyError, ValueError) as exc:
        if allow_missing_settings:
            return
        raise ValueError('missing or invalid feature_settings; '
                         'supply --allow-missing-settings to proceed') from exc

    # Use the same defaults as FeatureThresholds for omitted anchor attributes.
    defaults = dict(accuracy=thresholds.accuracy, epsilon=thresholds.epsilon,
                    pipeline=thresholds.pipeline, scaler_kind=thresholds.scaler_kind,
                    direction_norm='Eulerian B dot B')
    fields = {'accuracy', 'epsilon', 'pipeline', 'scaler_kind', 'direction_norm',
              'b2_min', 's_v', 'exponent', 'gamma_K', 'C_B_box', 'horizon_mask',
              'zslab', 'hole_boxes', 'horizon_centres', 'horizon_radius',
              'horizon_mass', 'horizon_spin', 'bh_positions', 'pol_frac_centres',
              'pol_frac_frame', 'inputs'}
    name_fields = {'names', 'feature_names', 'columns', 'feature_columns'}
    for key, saved in settings.items():
        if key in name_fields:
            expected = names
            saved = saved.split() if isinstance(saved, str) else saved
        elif key == 'winsor':
            expected = {k: list(v) for k, v in thresholds.winsor.items()}
            if saved != expected:
                raise ValueError(f'feature_settings and anchor {key} disagree')
            continue
        elif key in fields or key.startswith('winsor_'):
            if key not in ref.attrs and key not in defaults:
                raise ValueError(f'feature_settings and anchor {key} disagree')
            expected = ref.attrs.get(key, defaults.get(key))
        else:
            continue
        if not np.array_equal(saved, plain(expected)):
            raise ValueError(f'feature_settings and anchor {key} disagree')


def patch(model, anchor, output, commit, allow_missing_settings=False):
    """Validate inputs, copy the model, and publish without overwriting a file."""
    model, anchor, output = map(Path, (model, anchor, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'output already exists: {output}')
    model_hash, anchor_hash = digest(model), digest(anchor)
    with h5py.File(model, 'r') as src, h5py.File(anchor, 'r') as ref:
        meta = src['metadata'].attrs
        thresholds = FeatureThresholds.from_h5_attrs(ref.attrs)
        if text(meta['pipeline']) != 'v2' or thresholds.pipeline != 'v2':
            raise ValueError('model and anchor pipeline must be v2')
        if text(src.attrs['scaler_kind']) != 'median-iqr' or thresholds.scaler_kind != 'median-iqr':
            raise ValueError('model and anchor scaler kind must be median-iqr')
        for key in ('b2_min', 's_v'):
            if float(meta[key]) != getattr(thresholds, key):
                raise ValueError(f'model and anchor {key} disagree')
        names = tuple(text(n) for n in src['feature_names'][:])
        if names != tuple(text(n) for n in ref['names'][:]):
            raise ValueError('model and anchor feature names disagree')
        thresholds.feature_names = names
        thresholds.__post_init__()
        validate_settings(meta, ref, thresholds, names, allow_missing_settings)
        additions = thresholds.to_metadata()
        for key, value in additions.items():
            if key in meta and text(meta[key]) != value:
                raise ValueError(f'existing metadata disagrees: {key}')
        for key in meta:
            if key.startswith('feature_winsor_'):
                raise ValueError(f'v2 forbids existing metadata: {key}')
        if 'metadata_patched_by' in meta:
            raise ValueError('model already has metadata_patched_by')
    provenance = dict(tool_commit=commit, input_model_sha256=model_hash, anchor_sha256=anchor_hash)
    if allow_missing_settings:
        provenance['allow_missing_settings'] = True
    additions['metadata_patched_by'] = json.dumps(provenance, sort_keys=True)
    # Build beside the destination, then use a hard link for atomic no-clobber
    # publication. Copying bytes preserves datasets, including NaN payloads.
    fd, temporary = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.partial', dir=output.parent)
    try:
        with os.fdopen(fd, 'wb') as dst, model.open('rb') as src:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        if digest(temporary) != model_hash or digest(anchor) != anchor_hash:
            raise ValueError('input changed while patching')
        with h5py.File(temporary, 'r+') as dst:
            dst['metadata'].attrs.update(additions)
        os.link(temporary, output)
    finally:
        os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--tool-commit', help='full tool commit SHA; required in a non-git kit')
    parser.add_argument('--allow-missing-settings', action='store_true',
                        help='allow absent or invalid saved feature_settings; recorded in provenance')
    args = parser.parse_args(argv)
    try:
        patch(args.model, args.anchor, args.output, tool_commit(args.tool_commit),
              allow_missing_settings=args.allow_missing_settings)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'patch refused: {exc}\n')
    print(args.output)


if __name__ == '__main__':
    main()
