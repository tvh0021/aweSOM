#!/bin/sh
# Wait for the stage E trainings, then queue their scoring on 135168 (lock-serialised).
HERE=$(cd "$(dirname "$0")" && pwd)
while pgrep -f "train.py" > /dev/null; do sleep 30; done
sh "$HERE/run_score.sh" 135168 0.1 "$HERE/jobs_scoreE.txt"
