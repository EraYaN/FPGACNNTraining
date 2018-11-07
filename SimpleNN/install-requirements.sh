#!/usr/bin/env bash
source ./.bashrc_arria10_rcx2

CFLAGS=`aocl compile-config` LDFLAGS=`aocl link-config` pip3 install --user -r requirements.txt

plaidml-setup