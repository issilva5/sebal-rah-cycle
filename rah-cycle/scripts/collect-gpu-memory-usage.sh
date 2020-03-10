#!/bin/bash

export LC_NUMERIC="C"
TIME_BETWEEN_COMMANDS=1
echo TIMESTAMP, PID, COMMAND, USED
while [ -e /proc/$1 ]; do
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,nounits | tail -1 | awk -F ',' -v date="$( date +"%s" )" '{if($1 == "pid") print date", none, none, 0"; else print date", "$1", "$2", "$3 }' 2> /dev/null
  sleep $TIME_BETWEEN_COMMANDS
done
