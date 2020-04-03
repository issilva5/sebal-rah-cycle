#!/bin/bash

SENSOR=$1
INPUT_PATH=$2
OUTPUT_PATH=$3
THREAD=$4
METHOD=$5

if [ $SENSOR == "LC08" ]
then
        ./Debug/rah-cycle $INPUT_PATH/B2.tif $INPUT_PATH/B3.tif $INPUT_PATH/B4.tif $INPUT_PATH/B5.tif $INPUT_PATH/B6.tif $INPUT_PATH/B7.tif $INPUT_PATH/B10.tif $INPUT_PATH/MTL.txt $INPUT_PATH/tal.tif $INPUT_PATH/station.csv $OUTPUT_PATH $INPUT_PATH/land_cover_final.tif -meth=$METHOD -nan=-3.39999995214436425e+38 $THREAD > $OUTPUT_PATH/out.csv &
else
        ./Debug/rah-cycle $INPUT_PATH/B1.tif $INPUT_PATH/B2.tif $INPUT_PATH/B3.tif $INPUT_PATH/B4.tif $INPUT_PATH/B5.tif $INPUT_PATH/B6.tif $INPUT_PATH/B7.tif $INPUT_PATH/MTL.txt $INPUT_PATH/tal.tif $INPUT_PATH/station.csv $OUTPUT_PATH $INPUT_PATH/land_cover_final.tif -meth=$METHOD -nan=-3.39999995214436425e+38 $THREAD > $OUTPUT_PATH/out.csv &
fi

PID=$(pidof ./Debug/rah-cycle)
sh scripts/collect-cpu-usage.sh $PID > $OUTPUT_PATH/cpu.csv &
sh scripts/collect-memory-usage.sh $PID > $OUTPUT_PATH/mem.csv &
sh scripts/collect-disk-usage.sh $PID > $OUTPUT_PATH/disk.csv &
sh scripts/collect-gpu-usage.sh $PID > $OUTPUT_PATH/gpu.csv &
sh scripts/collect-gpu-memory-usage.sh $PID > $OUTPUT_PATH/mem-gpu.csv
