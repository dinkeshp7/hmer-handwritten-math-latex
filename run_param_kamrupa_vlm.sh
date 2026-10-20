#!/bin/bash
# ==============================================================================
# Slurm Submission Script for SOTA Vision-Language (Qwen2.5-VL & Nougat) 
# Inference & Full-Page LaTeX Conversion on PARAM Kamrupa Supercomputer (NVIDIA Tesla V100)
# ==============================================================================

#SBATCH --job-name=VLM_PARAM_KAMRUPA
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/p.dinkesh/hmer_project/logs/vlm_job_%j.out
#SBATCH --error=/scratch/p.dinkesh/hmer_project/logs/vlm_job_%j.err

echo "======================================================================"
echo "Starting SOTA Vision-Language Math Engine on PARAM Kamrupa (User: p.dinkesh)"
echo "Job ID: $SLURM_JOB_ID | Host: $SLURM_JOB_NODELIST | Date: $(date)"
echo "======================================================================"

SCRATCH_DIR="/scratch/p.dinkesh/hmer_project"
cd "$SCRATCH_DIR" || exit 1

module purge
module load cuda/11.7
module load Anaconda/Anaconda3-2023.03
source $ANACONDA_HOME/etc/profile.d/conda.sh 2>/dev/null || true
source "$SCRATCH_DIR/venv/bin/activate"
export LD_LIBRARY_PATH=/opt/ohpc/pub/apps/Anaconda3-2023.03/lib:$LD_LIBRARY_PATH
export TORCH_CUDNN_V8_API_DISABLED=1

echo "CUDA Available Device Check:"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

IMAGE_PATH="${1:-page-210103096-01.jpg}"
ENGINE="${2:-qwen_vl}"
MODEL_NAME="${3:-}"

# Generate image-specific and engine-specific output filename
IMAGE_BASENAME=$(basename "$IMAGE_PATH" | sed 's/\.[^.]*$//')
OUTPUT_TEX="output_${ENGINE}_${IMAGE_BASENAME}.tex"

echo "Target Image   : $IMAGE_PATH"
echo "Target Engine  : $ENGINE"
echo "Output Target  : $OUTPUT_TEX"
if [ -n "$MODEL_NAME" ]; then
    echo "Target Model   : $MODEL_NAME"
fi

CMD="python run_full_page_inference.py --image_path $IMAGE_PATH --engine $ENGINE --output_tex $OUTPUT_TEX"
if [ -n "$MODEL_NAME" ]; then
    CMD="$CMD --model_name $MODEL_NAME"
fi

echo "Executing: $CMD"
eval "$CMD"

# Also maintain convenience copy
cp -f "$OUTPUT_TEX" "output_${ENGINE}_solution.tex" 2>/dev/null || true

echo "======================================================================"
echo "Inference Complete. Output LaTeX saved to: $OUTPUT_TEX"
echo "Convenience copy updated at: output_${ENGINE}_solution.tex"
echo "======================================================================"
