#!/bin/bash
# Launcher script for playing against the model
# Handles OpenMP library conflict

export KMP_DUPLICATE_LIB_OK=TRUE

echo "♟️  Starting Chess vs Model..."
python play_vs_model.py "$@"
