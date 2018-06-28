#!/bin/bash

fswatch -e .idea -o . | xargs -n1 ./rsync_hw.sh

