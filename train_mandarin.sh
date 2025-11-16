#!/bin/bash
#SBATCH --job-name=train_mandarin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=train_mandarin_%j.out
#SBATCH --error=train_mandarin_%j.err

echo "============================================================"
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

# Load Anaconda module
module load anaconda/3

# Initialize conda for bash in the script
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate translation

# Verify environment
echo ""
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Show GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Run training
cd ~/translation_project/AI-Multilingual-Translation
python train_mandarin.py

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "============================================================"
