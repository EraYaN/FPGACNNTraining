#!/usr/bin/env bash

if [ $# -eq 0 ]
  then
    EXTRA='default'
else
    EXTRA=$1
fi

DATE=`date +'%Y-%m-%dT%H.%M.%S'`

echo Current Project 'nn_blocking' Date: $DATE

#make -C /home/erwin/SimpleNN clean
FLOW=hw PROJECT=nn_blocking BUILDTIME=$DATE EXTRA=$EXTRA make -C /home/erwin/SimpleNN fpga
EXIT_CODE=$?
BUILD_DIR=`FLOW=hw PROJECT=nn_blocking BUILDTIME=$DATE EXTRA=$EXTRA make -s -C /home/erwin/SimpleNN get_build_dir`
project_name=$(basename $BUILD_DIR)

if [ $EXIT_CODE -eq 0 ]; then       
    if [ -f $BUILD_DIR/acl_quartus_report.txt ]; then
    fmax=`awk -F ":" '/Kernel fmax/ {print $2}' $BUILD_DIR/acl_quartus_report.txt | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'`
    info="fmax: $fmax"
    else
    info=
    fi
    ./send_notification.sh "$project_name" Success "$info"
else    
    ./send_notification.sh "$project_name" Failure "Exit code $EXIT_CODE"
fi