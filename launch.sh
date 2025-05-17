#!/bin/zsh
# Launch script for portfolio optimization DRL
# This script must be run with zsh, not sh

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rl

# Run the main application
python main.py