#!/usr/bin/env bash
#PYOPENCL_CTX='0' PYOPENCL_COMPILER_OUTPUT=1
KERNEL_NAME=`ls -t kernels/kernel_*_sw_emu*.aocx | head -1`

if [ -f $KERNEL_NAME ]; then
    echo "Running script with kernel: $KERNEL_NAME"
    CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=4 /usr/bin/python3 ./fpga_test.py $KERNEL_NAME
else
    echo "There was no emulator kernel found."
fi