#!/bin/bash

if [ $# -eq 0 ]
  then
    EXTRA='default'
else
    EXTRA=$1
fi
LOGDIR=/local/erwin/build_logs

DATE=`date +'%Y-%m-%dT%H.%M.%S'`

LOGNAME=$LOGDIR/kernel_build_hw_${DATE}_${EXTRA}.log
ssh erwin@roma7.m.gsic.titech.ac.jp "cd ~/SimpleNN && bash -login -i -c 'source .bashrc_arria10_rcx2; mkdir -p $LOGDIR; ./build_hw.sh $EXTRA &>> $LOGNAME &'"

echo "Tailing $LOGNAME"

ssh erwin@roma7.m.gsic.titech.ac.jp -t "cd ~/SimpleNN && tail -n 40 -f $LOGNAME"
