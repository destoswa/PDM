#!/bin/bash

# Exit if any command fails
set -e

# Ensure conda is available
eval "$(conda shell.bash hook)"

echo "Running trimming (on pdm_env)..."
conda activate pdm_env
python src/tiles_loader.py trimming

echo "Running classification (on pdal_env)..."
conda activate pdal_env
python src/tiles_loader.py classification

