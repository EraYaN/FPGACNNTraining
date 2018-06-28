#!/bin/bash

#--filter="merge reports_filter.txt"

rsync -zarv -R roma7.m.gsic.titech.ac.jp:/local/erwin/build_cache/kernel_**/reports/ .

rsync -zarv -R roma7.m.gsic.titech.ac.jp:/local/erwin/build_cache/kernel_**/*.txt .