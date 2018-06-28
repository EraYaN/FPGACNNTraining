#!/usr/bin/env bash
#PYOPENCL_CTX='0' PYOPENCL_COMPILER_OUTPUT=1

if [ $# -eq 0 ]
  then
    EXTRA='default'
else
    EXTRA=$1
fi

KERNEL_NAME=`ls -t kernels/kernel_*${EXTRA}*hw*.aocx 2>/dev/null | head -1`
if [ !  -z  $KERNEL_NAME ] && [ -f $KERNEL_NAME ]; then
    echo "Running script with kernel: $KERNEL_NAME"
    /usr/bin/python36 ./fpga_test.py $KERNEL_NAME
else
    echo "There was no matching kernel found."
fi