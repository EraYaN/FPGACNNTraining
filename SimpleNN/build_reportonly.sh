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
FLOW=hw PROJECT=nn_blocking BUILDTIME=$DATE EXTRA=$EXTRA make -C /home/erwin/SimpleNN report