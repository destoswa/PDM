#!/bin/bash
set -e

SRC_ROOT_DATA="$1"
SRC_DATA="$2"
echo $SRC_ROOT_DATA
echo $SRC_DATA

python3 ./inference.py --src_root_data $SRC_ROOT_DATA --src_data $SRC_DATA