#!/bin/bash
# ==============================================================================
# Slurm Submission Script for HMER Training on PARAM Kamrupa Supercomputer
# System: PARAM Kamrupa (IIT Guwahati / NSM / C-DAC)
# Hardware: NVIDIA Tesla V100 GPU Nodes (gpu001 - gpu010)
# Project: Handwritten Math Expression Recognition (HMER) to LaTeX Converter
# PI: Prof. K.V. Srikanth | Students: Dinkesh Pal & Rithvik Ponnapalli
# User Account: p.dinkesh
# ==============================================================================

#SBATCH --job-name=HMER_PARAM_KAMRUPA
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/p.dinkesh/hmer_project/logs/hmer_job_%j.out
#SBATCH --error=/scratch/p.dinkesh/hmer_project/logs/hmer_job_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=p.dinkesh@iitg.ac.in

echo "======================================================================"
echo "Starting HMER GPU Training on PARAM Kamrupa (User: p.dinkesh)"
echo "Job ID: $SLURM_JOB_ID | Host: $SLURM_JOB_NODELIST | Date: $(date)"
echo "======================================================================"

# 1. Create scratch directory structure
SCRATCH_DIR="/scratch/p.dinkesh/hmer_project"
mkdir -p "$SCRATCH_DIR/logs"
mkdir -p "$SCRATCH_DIR/checkpoints"
mkdir -p "$SCRATCH_DIR/data"

# 2. Environment Setup & Module Loading (Using Anaconda3-2023.03 with _ctypes & GLIBC 2.17 support)
module purge
module load cuda/11.7
module load Anaconda/Anaconda3-2023.03

# Enable Conda Profile
source $ANACONDA_HOME/etc/profile.d/conda.sh 2>/dev/null || true

export OMP_NUM_THREADS=8

# 3. Launch PyTorch GPU Training
python3 src/training/train.py \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --lambda_arm 1.0 \
    --lambda_count 0.5 \
    --checkpoint_dir "$SCRATCH_DIR/checkpoints"

echo "======================================================================"
echo "PARAM Kamrupa Training Job Finished at: $(date)"
echo "======================================================================"
