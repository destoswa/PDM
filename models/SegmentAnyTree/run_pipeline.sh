#!/bin/bash

# python3 oracle_wrapper.py
epochs=$1
batch_size=$2
sample_per_epoch=$3
checkpoint_dir=$4
train_metrics_src=$5
current_loop=$6

python3 train.py task=panoptic \
    data=panoptic/treeins_rad8 \
    models=panoptic/area4_ablation_3heads_5 \
    model_name=PointGroup-PAPER \
    training=treeins \
    job_name=treeins_my_first_run \
    epochs=$(($epochs)) \
    batch_size=$(($batch_size)) \
    data.sample_per_epoch=$(($sample_per_epoch)) \
    checkpoint_dir=$checkpoint_dir \
    +train_metrics_src=$train_metrics_src \
    +current_loop=$current_loop

