#!/bin/sh
# Wait for the running train.py processes to finish, drop the paused stage-D3
# xargs (pid $1), then run stage D4 one job at a time (memory budget).
HERE=$(cd "$(dirname "$0")" && pwd)
while pgrep -f "train.py --variant" > /dev/null; do sleep 15; done
kill -KILL "$1" 2>/dev/null
sh "$HERE/run_train.sh" "$HERE/stageD4.txt" 1
