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
