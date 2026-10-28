#!/bin/sh
# Build the column stores (prep.py) for the given iterations, one after another.
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/Users/tvh0021/git_repos/aweSOM-research/.venv/bin/python
export PYTHONPATH=$HERE/../../src
LOG=/Volumes/ExternalNVMe/awesom-results/2026-10-28/physical/logs
cd "$HERE" || exit 1
for it in "$@"; do
  nice -n 10 /usr/bin/time -l $PY prep.py "$it" > "$LOG/prep_$it.log" 2>&1 || exit 1
done
