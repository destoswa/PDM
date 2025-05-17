#!/bin/bash

# Exit if any command fails
set -e

# Ensure conda is available
eval "$(conda shell.bash hook)"

conda activate pdal_env

src_folder_in="$1"
src_folder_out="$2"
in_type="$3"
out_type="$4"
python src/format_conversions.py $src_folder_in $src_folder_out $in_type $out_type True

conda activate pdm_env