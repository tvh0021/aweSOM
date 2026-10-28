"""Shared helpers for the real-data validation benchmarks (planning/benchmarks/real_*.py).

Baseline = the unoptimized public release (commit 1069cff), checked out read-only at
BASELINE_ROOT. Its modules are imported from their files under private names
(``aweSOM_baseline_som`` etc.), so both versions live in one process and the worktree's
``aweSOM`` package stays the one on ``PYTHONPATH``.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

BASELINE_ROOT = os.environ.get("AWESOM_BASELINE", "/Users/tvh0021/git_repos/aweSOM-baseline")
RESULTS = os.environ.get("AWESOM_RESULTS", "/Volumes/ExternalNVMe/awesom-results/2026-10-28/validate")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BBH_DIR = "/Volumes/ExternalNVMe/etk-analysis/som_results"
BBH_H5 = "/Volumes/ExternalNVMe/som-proto-f5/features_bbh_0003-180224-N10000000.h5"
BBH_TAGS = {
    "bbh-29x20": "bbh_0003_180224-29-20-0.5-100000-1u",
    "bbh-51x36": "bbh_0003_180224-51-36-0.5-1000000-1u",
    "bbh-91x64": "bbh_0003_180224-91-64-0.5-10000000-1u",
}
PLASMA_PKL = "/Users/tvh0021/git_repos/aweSOM/examples/plasma-turbulence/som_object.fiducial-5000-64-37-0.1-2097152-1u.pkl"
PLASMA_RUNS = "/Users/tvh0021/git_repos/aweSOM-research/planning/runs/plasma-2026-10-26"


def load_baseline(relpath: str, name: str):
    """Import a baseline source file (relative to BASELINE_ROOT) as module ``name``."""
    path = os.path.join(BASELINE_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        spec.loader.exec_module(mod)
    return mod


def uptime() -> str:
    """Load averages as printed by ``uptime`` (1, 5, 15 min)."""
    out = subprocess.check_output(["uptime"], text=True).strip()
    return out.split("load averages:")[-1].strip() if "load averages:" in out else out


@contextlib.contextmanager
def quiet_fd():
    """Silence stdout at the file-descriptor level (numba prints from compiled loops bypass sys.stdout)."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def timed(fn, reps: int = 3, quiet: bool = True):
    """Run ``fn`` ``reps`` times; return (min seconds, all seconds, uptime before the first run, last result)."""
    load = uptime()
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        if quiet:
            with quiet_fd():
                out = fn()
        else:
            out = fn()
        ts.append(time.perf_counter() - t)
    return min(ts), ts, load, out


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def save_json(name: str, obj) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
    return path


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def log(*a):
    print(*a, file=sys.stderr, flush=True)
