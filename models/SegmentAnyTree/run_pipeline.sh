#!/bin/bash


# python3 oracle_wrapper.py
epochs=$1
batch_size=$2

python3 train.py task=panoptic \
    data=panoptic/treeins_rad8 \
    models=panoptic/area4_ablation_3heads_5 \
    model_name=PointGroup-PAPER \
    training=treeins \
    job_name=treeins_my_first_run \
    epochs=$(($epochs)) \
    batch_size=$(($batch_size)) \

