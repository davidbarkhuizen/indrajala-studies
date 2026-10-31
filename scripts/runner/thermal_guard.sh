#!/bin/bash
# A thermal guard, klip's (the owner, 2026-10-21: "just watch the max temp, it can overheat"; the limit
# set then: ideally <= 80 C, stop at 85 C; the batch runner wraps klip's attempts in it). Runs a
# command in its own process group, samples the CPU's k10temp every 5 s into LOG (time, C), and stops
# the whole group at STOP C or above.
# Usage: scripts/runner/thermal_guard.sh LOG [STOP] -- COMMAND...
set -u
LOG=$1; shift; STOP=85
[ "$1" != "--" ] && { STOP=$1; shift; }
shift  # the --
sensor=$(grep -l '^k10temp$' /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)/temp1_input
setsid "$@" &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
  t=$(( $(cat $sensor) / 1000 ))
  [ $t -gt $peak ] && peak=$t
  echo "$(date +%H:%M:%S) $t" >> $LOG
  if [ $t -ge $STOP ]; then
    echo "$(date +%H:%M:%S) STOP at $t C (limit $STOP)" >> $LOG
    kill -TERM -- -$pid 2>/dev/null
  fi
  sleep 5
done
wait $pid; status=$?
echo "$(date +%H:%M:%S) ended, status $status, peak $peak C" >> $LOG
exit $status
