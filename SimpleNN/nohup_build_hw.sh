#!/bin/bash

if [ $# -eq 2 ]
  then
    EXTRA=$1
    TARGET=$2
elif [ $# -eq 1 ]
  then
    EXTRA=$1
    TARGET='fpga'
else
    EXTRA='default'
    TARGET='fpga'
fi

LOGDIR=/local/erwin/build_logs

DATE=`date +'%Y-%m-%dT%H.%M.%S'`

LOGNAME=$LOGDIR/kernel_build_hw_${DATE}_${EXTRA}.log
ssh erwin@roma7.m.gsic.titech.ac.jp "cd ~/SimpleNN && bash -login -i -c 'source .bashrc_arria10_rcx2; mkdir -p $LOGDIR; ./build_hw.sh $EXTRA $TARGET &>> $LOGNAME &'"

echo "Tailing $LOGNAME"

ssh erwin@roma7.m.gsic.titech.ac.jp -t "cd ~/SimpleNN && tail -n 40 -f $LOGNAME"
