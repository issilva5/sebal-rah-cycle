#!/bin/bash

INPUT_DIR=$1
OUTPUT_DIR=$2

./Debug/rah-cycle $INPUT_DIR/_B2_converted.tif $INPUT_DIR/_B3_converted.tif $INPUT_DIR/_B4_converted.tif $INPUT_DIR/_B5_converted.tif $INPUT_DIR/_B6_converted.tif $INPUT_DIR/_B7_converted.tif $INPUT_DIR/_B10_converted.tif $INPUT_DIR/MTL.txt $INPUT_DIR/tal_converted.tif $INPUT_DIR/station.csv $OUTPUT_DIR -dist=0.9833 > $OUTPUT_DIR/proctimes.csv &
PID=$(pidof ./Debug/rah-cycle)
echo $PID
sh scripts/collect-cpu-usage.sh $PID > $OUTPUT_DIR/cpu.csv &
sh scripts/collect-memory-usage.sh $PID > $OUTPUT_DIR/mem.csv &
sh scripts/collect-disk-usage.sh $PID > $OUTPUT_DIR/disk.csv
