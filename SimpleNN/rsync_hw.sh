#!/bin/bash

rsync . roma7.m.gsic.titech.ac.jp:/home/erwin/SimpleNN -av --exclude .idea --exclude test_hw --exclude test2_hw --exclude "local" --exclude "kernel_*" --exclude archive --exclude .idea --exclude .DS_Store --exclude "*.aocx"
#ssh erwin@rcx0.m.gsic.titech.ac.jp 'cd /home/erwin/SimpleNN && bash -login -i -c "./run_hw.sh"'
# ssh erwin@rcx0.m.gsic.titech.ac.jp bash -c << EOF
#source ~/.bash_profile
#source ~/.bashrc_stratixv
#ll
#pwd
#cd /home/erwin/SimpleNN
#
#echo $PATH
#./build.sh
#./run.sh
#
#EOF
