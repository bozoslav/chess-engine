#!/bin/bash
# Quick start script with environment fixes

export KMP_DUPLICATE_LIB_OK=TRUE

cd "$(dirname "$0")"

if [ "$1" == "check" ]; then
    echo "Running setup check..."
    python check_setup.py
elif [ "$1" == "progress" ]; then
    echo "Showing training progress..."
    python view_progress.py
else
    echo "Starting training..."
    python train.py
fi
