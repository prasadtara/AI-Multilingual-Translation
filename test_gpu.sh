#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=test_gpu_%j.out
#SBATCH --error=test_gpu_%j.err

# Load Python module
module load python/3.10.4

# Activate virtual environment
source ~/translation_project/AI-Multilingual-Translation/venv/bin/activate

# Show environment info
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Run the test
python test_gpu.py

echo ""
echo "Job completed at: $(date)"
