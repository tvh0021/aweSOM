#!/bin/sh
# Post-processing study (postproc_study.py), serialised with run_score.sh by the same lock.
#   sh run_postproc.sh IT MC TAG [TAG...] [--settings ...] [--save ...]
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/Users/tvh0021/git_repos/aweSOM-research/.venv/bin/python
export PYTHONPATH=$HERE/../../src
RES=/Volumes/ExternalNVMe/awesom-results/2026-10-28/physical
LOG=$RES/logs
LOCK=$RES/.score.lock
IT=$1; MC=$2; shift 2
cd "$HERE" || exit 1
until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
trap 'rmdir "$LOCK"' EXIT
uptime >> "$LOG/postproc_$IT.log"
nice -n 10 /usr/bin/time -l $PY postproc_study.py --it "$IT" --mc "$MC" "$@" >> "$LOG/postproc_$IT.log" 2>&1
uptime >> "$LOG/postproc_$IT.log"
