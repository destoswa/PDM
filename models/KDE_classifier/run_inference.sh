#!/bin/bash
set -e

SRC_ROOT_DATA="$1"
SRC_DATA="$2"
echo $SRC_ROOT_DATA
echo $SRC_DATA
# python3 -c "
# import yaml
# with open('./config/inference.yaml') as f:
#     cfg = yaml.safe_load(f)
# cfg['inference']['src_root_data'] = $SRC_ROOT_DATA
# cfg['inference']['src_data'] = $SRC_DATA
# with open('config.yaml', 'w') as f:
#     yaml.dump(cfg, f)
# "

python3 ./inference.py --src_root_data $SRC_ROOT_DATA --src_data $SRC_DATA