"""Smoke test for the README walkthrough (sections 3 and 4, Iris SOM and SCE).

The Python blocks are read from README.md and executed in order in one
namespace, as a user pasting them into a session would. Only three things
differ from the README text:

* the number of training steps is reduced so the test runs in about a minute;
* the manual ``label_map`` line (the README says "change manually") is replaced
  by the majority SOM label of each class, because an unseeded short training
  run does not reproduce the README's cluster numbering;
* matplotlib uses the non-interactive Agg backend.

The shell steps of section 4 (move the label files, run ``sce.py``) are done
with ``shutil`` and ``subprocess``.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
SCE_SCRIPT = REPO / "src" / "aweSOM" / "sce.py"

SECTION3_TRAIN = 2000
SECTION4_TRAIN = [500, 1000, 1500]

MAJORITY_LABEL_MAP = (
    "label_map = {name: int(np.unique(som_labels[iris.target == k], return_counts=True)[0]"
    "[np.argmax(np.unique(som_labels[iris.target == k], return_counts=True)[1])])"
    " for k, name in enumerate(iris.target_names)}"
)


def _python_blocks(text: str, start: str, stop: str) -> list[str]:
    section = text[text.index(start) : text.index(stop)]
    return re.findall(r"```python\n(.*?)```", section, flags=re.S)


def _replace_once(code: str, old: str, new: str) -> str:
    assert code.count(old) == 1, f"README no longer contains {old!r}"
    return code.replace(old, new)


def _run(blocks: list[str], ns: dict, label: str) -> None:
    for i, code in enumerate(blocks):
        exec(compile(code, f"README.md {label} block {i}", "exec"), ns)


def test_readme_iris_som_and_sce(tmp_path, monkeypatch):
    text = README.read_text(encoding="utf-8")
    som_blocks = _python_blocks(text, "## 3. Basic Usage - SOM", "## 4. Basic Usage - SCE")
    sce_blocks = _python_blocks(text, "## 4. Basic Usage - SCE", "## 5. Advanced Usage")
    assert len(som_blocks) == 9 and len(sce_blocks) == 5

    som_code = "\n".join(som_blocks)
    som_code = _replace_once(som_code, "train=100000)", f"train={SECTION3_TRAIN})")
    som_code = _replace_once(
        som_code,
        "label_map = {'setosa' : 2, 'versicolor' : 1, 'virginica' : 0}",
        MAJORITY_LABEL_MAP,
    )
    sce_blocks[0] = _replace_once(
        sce_blocks[0], '"train": [10000, 50000, 100000]', f'"train": {SECTION4_TRAIN}'
    )

    monkeypatch.chdir(tmp_path)
    ns: dict = {"__name__": "__readme__"}
    try:
        # Section 3: single SOM on Iris
        _run([som_code], ns, "section 3")

        data = ns["iris_data_transformed"]
        assert data.shape == (150, 4) and np.isfinite(data).all()
        lattice_map = ns["map"]
        assert lattice_map.lattice.shape == (40 * 15, 4)
        assert np.isfinite(lattice_map.lattice).all()
        assert lattice_map.umat.shape == (40, 15) and np.isfinite(lattice_map.umat).all()
        assert ns["final_clusters"].shape == (40, 15)
        assert ns["som_labels"].shape == (150,)
        for name in ("precision", "recall"):
            values = ns[name]
            assert values.shape == (3,) and np.isfinite(values).all()
            assert ((values >= 0) & (values <= 1)).all()

        # Section 4: 36 SOM realizations, then SCE
        _run(sce_blocks[:1], ns, "section 4")
        label_files = sorted(tmp_path.glob("labels.iris-*.npy"))
        assert len(label_files) == 36

        results = tmp_path / "som_results"  # mkdir som_results; mv labels* som_results/
        results.mkdir()
        for f in label_files:
            shutil.move(str(f), results / f.name)
        subprocess.run(  # cd som_results/; python3 .../sce.py --subfolder SCE --dims 150
            [sys.executable, str(SCE_SCRIPT), "--subfolder", "SCE", "--dims", "150"],
            cwd=results,
            check=True,
            capture_output=True,
        )
        assert (results / "SCE" / "multimap_mappings.txt").is_file()

        _run(sce_blocks[1:], ns, "section 4 after sce.py")

        assert len(ns["ranked_gsum_list"]) == len(ns["map_list"]) > 0
        sce_sum = ns["sce_sum"]
        assert sce_sum.shape == (150,) and np.isfinite(sce_sum).all()
        assert sce_sum.max() > 0
        sce_clusters = ns["sce_clusters"]
        assert sce_clusters.shape == (150,)
        assert set(np.unique(sce_clusters)) <= {0, 1, 2}
    finally:
        plt.close("all")
