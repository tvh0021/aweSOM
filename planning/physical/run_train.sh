#!/bin/sh
# Run train.py once per line of a job file (extra arguments per line), P at a time.
#   sh run_train.sh stageA.txt 3 [common args...]
JOBS=$1
P=$2
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
export PY=/Users/tvh0021/git_repos/aweSOM-research/.venv/bin/python
export PYTHONPATH=$HERE/../../src
export LOG=/Volumes/ExternalNVMe/awesom-results/2026-10-28/physical/logs
export COMMON="$*"
cd "$HERE" || exit 1
grep -v '^#' "$JOBS" | grep -v '^$' | tr '\n' '\0' | xargs -0 -P "$P" -n 1 sh -c 'nice -n 10 $PY train.py $0 $COMMON >> $LOG/train.jsonl 2>> $LOG/train.err'
