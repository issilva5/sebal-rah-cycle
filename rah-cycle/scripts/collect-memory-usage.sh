#!/bin/bash

export LC_NUMERIC="C"
TIME_BETWEEN_COMMANDS=1
echo TIMESTAMP, USAGE, USAGE_GB, TYPE
while [ -e /proc/$1 ]; do
  ps -o pid,%mem ax | sort -b -k3 -r | grep $1 | awk -v date="$( date +"%s" )" '{ print date, $2, $1, $1 }' 2> /dev/null
  sleep $TIME_BETWEEN_COMMANDS
done
