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

from aweSOM.features_grmhd import FeatureThresholds


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


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


def patch(model, anchor, output, commit):
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
        additions = thresholds.to_metadata()
        for key, value in additions.items():
            if key in meta and text(meta[key]) != value:
                raise ValueError(f'existing metadata disagrees: {key}')
        for key in meta:
            if key.startswith('feature_winsor_'):
                raise ValueError(f'v2 forbids existing metadata: {key}')
        if 'metadata_patched_by' in meta:
            raise ValueError('model already has metadata_patched_by')
    additions['metadata_patched_by'] = json.dumps(dict(
        tool_commit=commit, input_model_sha256=model_hash, anchor_sha256=anchor_hash), sort_keys=True)
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
    args = parser.parse_args(argv)
    try:
        patch(args.model, args.anchor, args.output, tool_commit(args.tool_commit))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'patch refused: {exc}\n')
    print(args.output)


if __name__ == '__main__':
    main()
