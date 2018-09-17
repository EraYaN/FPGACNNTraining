#!/bin/bash

#--filter="merge reports_filter.txt"

rsync -zarv -R kiev0.m.gsic.titech.ac.jp:/local/erwin/build_cache/kernel_**/reports/ .

rsync -zarv -R kiev0.m.gsic.titech.ac.jp:/local/erwin/build_cache/kernel_**/*.txt .

rsync -zarv -R kiev0.m.gsic.titech.ac.jp:/local/erwin/build_logs/** .