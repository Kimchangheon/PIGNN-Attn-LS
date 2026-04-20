#!/bin/bash

#SBATCH --job-name=gpu_test                 # Job name
#SBATCH --output=./Job_out/gpu_test.out     # Output log file
#SBATCH --error=./Job_out/gpu_test.err      # Error log file
#SBATCH --gres=gpu:a40:1                    # Request 1 A40 GPU
#SBATCH --time=00:10:00                     # Request 10 minutes of runtime
#SBATCH --ntasks=1                          # Run a single task
#SBATCH --cpus-per-task=2                   # Request 2 CPU cores
#SBATCH --partition=a40                     # Use the A40 partition

# Load necessary modules (modify based on environment)
export PYTHONPATH=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:$PYTHONPATH

# Display allocated GPU info with nvidia-smi
# You can also test accessing installed packages, e.g.:
python -c "import numpy; print(f'numpy version: {numpy.__version__}')"

echo "Script finished successfully."
