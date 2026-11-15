"""compare_runs.py reads frozen sheet selections, including valid empty ones."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare_runs', ROOT/'examples/grmhd-bbh/compare_runs.py')
compare = importlib.util.module_from_spec(spec); spec.loader.exec_module(compare)


class ReachedRun(Exception):
    pass


def run_main(metadata, tmp_path):
    def reached(_):
        raise ReachedRun
    with patch('aweSOM.model.SOMModel.load', return_value=SimpleNamespace(metadata=metadata)), \
            patch.object(compare, 'parse_run', side_effect=reached):
        return compare.main(['--model', 'm.h5', '--run', 'x=g.h5', '--out', str(tmp_path)])


@pytest.mark.parametrize('status', ['none', 'unavailable'])
def test_empty_selection_exits_cleanly(status, tmp_path, capsys):
    run_main({'selected_sheet_clusters': '', 'sheet_clusters': '', 'sheet_selection_status': status}, tmp_path)
    assert f'status {status}' in capsys.readouterr().out


@pytest.mark.parametrize('metadata', [{'selected_sheet_clusters': '4', 'sheet_clusters': '9'}, {'sheet_clusters': '4'}])
def test_selected_ids_reach_run_processing(metadata, tmp_path, capsys):
    with pytest.raises(ReachedRun):
        run_main(metadata, tmp_path)
    assert 'sheet clusters: [4]' in capsys.readouterr().out


def test_missing_selection_metadata_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        run_main({}, tmp_path)


def test_parse_run_accepts_npy_parts_and_manifest():
    assert compare.parse_run('r1=grid.h5') == ('r1', Path('grid.h5'), None)
    assert compare.parse_run('r1=grid.h5:lab.npy') == ('r1', Path('grid.h5'), Path('lab.npy'))
    assert compare.parse_run('r1=grid.h5:lab.parts') == ('r1', Path('grid.h5'), Path('lab.parts'))
    assert compare.parse_run('r1=grid.h5:lab.parts/') == ('r1', Path('grid.h5'), Path('lab.parts/'))
    assert compare.parse_run('r1=grid.h5:lab.parts/manifest.json') == (
        'r1', Path('grid.h5'), Path('lab.parts/manifest.json')
    )


def test_label_fractions_reads_sharded_parts(tmp_path):
    import h5py
    import numpy as np
    from numpy.lib.format import open_memmap
    from aweSOM.npy_parts import part_filename, write_npy_manifest
    from aweSOM.stream import NpyWriter

    grid = tmp_path / 'grid.h5'
    xyz = np.array([[x, y, 0.0] for x in (0.0, 0.5) for y in (0.0, 0.5)], dtype=np.float32)
    with h5py.File(grid, 'w') as h:
        h.create_dataset('xyz', data=xyz)
        h.create_dataset('level', data=np.full(4, 5, dtype=np.int8))
        h.create_dataset('features', data=np.zeros((4, 2), dtype=np.float32))
        h.create_dataset('names', data=np.array([b'f0', b'f1']))

    plain_labels = np.array([0, 1, 1, 2], dtype=np.uint8)
    plain_path = tmp_path / 'labels_plain.npy'
    np.save(plain_path, plain_labels)

    pdir = tmp_path / 'labels_sharded.parts'
    pdir.mkdir()
    parts = []
    for r, (a, b) in enumerate(((0, 2), (2, 4))):
        fname = part_filename(r, 2)
        mm = open_memmap(pdir / fname, mode='w+', dtype=np.uint8, shape=(b - a,))
        del mm
        wr = NpyWriter(pdir / fname)
        wr[0:b - a] = plain_labels[a:b]
        wr.close(sync=False)
        parts.append(dict(rank=r, file=fname, start=a, stop=b, rows=b - a, bytes=b - a, crc32=int(wr.crc32)))
    write_npy_manifest(pdir, dtype=np.uint8, shape=(4,), parts=parts, sync=False)

    assert compare.label_fractions(grid, pdir) == compare.label_fractions(grid, plain_path)
    assert compare.label_fractions(grid, tmp_path / 'labels_sharded.npy') == compare.label_fractions(grid, plain_path)

