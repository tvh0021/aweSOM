#!/bin/sh
# Score models on one iteration. Calls are serialised by a lock dir so only one
# scoring process (~7 GB) runs at a time.
#   sh run_score.sh IT COMPONENTS MODELS_FILE|MODEL... [--save-labels ...]
# A single argument ending in .txt is read as a list of model paths (one per line).
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/Users/tvh0021/git_repos/aweSOM-research/.venv/bin/python
export PYTHONPATH=$HERE/../../src
RES=/Volumes/ExternalNVMe/awesom-results/2026-10-28/physical
LOG=$RES/logs
LOCK=$RES/.score.lock
IT=$1; COMP=$2; shift 2
case "$1" in *.txt) F=$1; shift; set -- $(grep -v "^#" "$F") "$@";; esac
cd "$HERE" || exit 1
until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
trap 'rmdir "$LOCK"' EXIT
uptime >> "$LOG/score_$IT.log"
nice -n 10 /usr/bin/time -l $PY score.py --it "$IT" --components "$COMP" "$@" >> "$LOG/score_$IT.log" 2>&1
uptime >> "$LOG/score_$IT.log"
