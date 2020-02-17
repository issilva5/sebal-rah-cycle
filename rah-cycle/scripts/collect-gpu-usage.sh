#!/bin/bash

export LC_NUMERIC="C"
TIME_BETWEEN_COMMANDS=1
echo TIMESTAMP, GPU, MEM
while [ -e /proc/$1 ]; do
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,nounits | tail -1 | awk -F ',' -v date="$( date +"%s" )" '{print date", "$1", "$2 }' 2> /dev/null
  sleep $TIME_BETWEEN_COMMANDS
done
