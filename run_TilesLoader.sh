#!/bin/bash

# Exit if any command fails
set -e

# Ensure conda is available
eval "$(conda shell.bash hook)"

MODE="$1"
VERBOSE="$2"

if [ "$VERBOSE" == "" ]; then
    VERBOSE=False
fi

# function to activate a certain env and run the script with a certain param
activate_and_run(){
    env="${1}"
    arg="${2}"
    verbose="${3}"
    
    # tests
    if [ "$env" == "" ]; then 
        echo "Missing two argument!"
    elif [ "$arg" == "" ]; then 
        echo "Missing an argument!"
    elif [ "$verbose" == "" ]; then 
        verbose=False
    fi

    # activate and call function
    echo "running $arg (on $env)"
    conda activate $env
    python src/tiles_loader.py $arg $verbose
}

# call the different functions depending on the mode
if [ "$MODE" == "full" ]; then
    activate_and_run "pdal_env" "tiling" "$VERBOSE"
    activate_and_run "pdm_env" "trimming" "$VERBOSE"
    activate_and_run "pdal_env" "classification" "$VERBOSE"
elif [ "$MODE" == "trim_and_class" ]; then
    activate_and_run "pdm_env" "trimming" "$VERBOSE"
    activate_and_run "pdal_env" "classification" "$VERBOSE"
elif [ "$MODE" == "tile_and_trim" ]; then
    activate_and_run "pdal_env" "tiling" "$VERBOSE"
    activate_and_run "pdm_env" "trimming" "$VERBOSE"
elif [ "$MODE" == "tiling" ]; then
    activate_and_run "pdal_env" "tiling" "$VERBOSE"
elif [ "$MODE" == "trimming" ]; then
    activate_and_run "pdm_env" "trimming" "$VERBOSE"
elif [ "$MODE" == "classification" ]; then
    activate_and_run "pdal_env" "classification" "$VERBOSE"
else
    echo "WRONG ARGUMENT!"
    exit 1
fi
#echo "$1"
#activate_and_run "$1" "$2" "$3"
#exit 0
#echo "Running trimming (on pdm_env)..."
#conda activate pdm_env
#python src/tiles_loader.py trimming

#echo "Running classification (on pdal_env)..."
#conda activate pdal_env
#python src/tiles_loader.py classification

