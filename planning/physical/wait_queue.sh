#!/bin/sh
# Block until no scoring / post-processing job is running or queued.
while pgrep -f "run_score.sh|run_postproc.sh|score.py|postproc_study.py" > /dev/null; do sleep 30; done
uptime
