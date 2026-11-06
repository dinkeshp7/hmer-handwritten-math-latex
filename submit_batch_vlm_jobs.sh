#!/bin/bash
# ==============================================================================
# Batch SLURM Submission Script for SOTA Qwen2.5-VL-7B Mathematical OCR
# Automatically submits 10-20 real student exam pages across diverse roll numbers
# ==============================================================================

SCRATCH_DIR="/scratch/p.dinkesh/hmer_project"
cd "$SCRATCH_DIR" || exit 1

ENGINE="${1:-qwen_vl}"
LIMIT="${2:-15}"

echo "======================================================================"
echo "  PARAM KAMRUPA BATCH SUBMISSION: SOTA Qwen2.5-VL-7B HMER BENCHMARK   "
echo "======================================================================"
echo "Target Engine : $ENGINE"
echo "Target Limit  : $LIMIT jobs"
echo "Working Dir   : $SCRATCH_DIR"
echo "======================================================================"

# Step 1: Collect candidates
# First check high-value known student scripts, then auto-discover from dataset directories
CANDIDATES=()

# Known benchmark samples across different students & topics
PRIORITY_SAMPLES=(
    "page-210106038-01.jpg"   # 4x4 Decimal Matrix & Row Reductions (Gaussian Elimination)
    "page-210103096-01.jpg"   # Vector Subspace Closure & Null Space Criteria
    "page-240104105-07.jpg"   # Linear Operator & Dimension Proofs
    "dp.jpg"                  # Official Faculty MidSem Model Solution Key
)

for p in "${PRIORITY_SAMPLES[@]}"; do
    if [ -f "$p" ]; then
        CANDIDATES+=("$p")
    elif [ -f "data/$p" ]; then
        CANDIDATES+=("data/$p")
    elif [ -f "data/ma102/$p" ]; then
        CANDIDATES+=("data/ma102/$p")
    fi
done

# Step 2: Auto-discover real student pages across data directory
# Filters for .jpg and .png files that match student answer scripts
if [ -d "data" ]; then
    while IFS= read -r f; do
        # Avoid duplicates
        base=$(basename "$f")
        already_in=0
        for c in "${CANDIDATES[@]}"; do
            if [ "$(basename "$c")" == "$base" ]; then
                already_in=1
                break
            fi
        done
        if [ $already_in -eq 0 ]; then
            CANDIDATES+=("$f")
        fi
    done < <(find data -type f \( -name "page-*.jpg" -o -name "page-*.png" -o -name "*[0-9]*.jpg" \) 2>/dev/null | sort)
fi

TOTAL_FOUND=${#CANDIDATES[@]}
echo "Total candidate student pages discovered: $TOTAL_FOUND"

if [ $TOTAL_FOUND -eq 0 ]; then
    echo "[Error] No student exam page images found in $SCRATCH_DIR or $SCRATCH_DIR/data."
    echo "Please check available files with: find . -maxdepth 3 -name '*.jpg'"
    exit 1
fi

COUNT=0
SUBMITTED_JOBS=()

echo -e "\n--- Submitting SLURM Batch Jobs ---"
printf "%-6s | %-12s | %-35s | %-30s\n" "Index" "Job ID" "Image Filename" "Target LaTeX Output"
echo "-----------------------------------------------------------------------------------------"

for img in "${CANDIDATES[@]}"; do
    if [ $COUNT -ge "$LIMIT" ]; then
        break
    fi

    # Submit job via run_param_kamrupa_vlm.sh
    output=$(sbatch run_param_kamrupa_vlm.sh "$img" "$ENGINE" /scratch/p.dinkesh/hmer_project/models/qwen_vl_7b 2>&1)
    
    # Extract SLURM Job ID (e.g. "Submitted batch job 389105")
    job_id=$(echo "$output" | grep -oE '[0-9]+' | tail -n 1)
    
    if [ -n "$job_id" ]; then
        SUBMITTED_JOBS+=("$job_id")
        base=$(basename "$img" | sed 's/\.[^.]*$//')
        target_tex="output_${ENGINE}_${base}.tex"
        printf "%-6d | %-12s | %-35s | %-30s\n" "$((COUNT + 1))" "$job_id" "$(basename "$img")" "$target_tex"
        COUNT=$((COUNT + 1))
        # Brief 0.5s pause to prevent SLURM scheduler lock contention
        sleep 0.5
    else
        echo "[Warning] Failed to submit job for: $img (Output: $output)"
    fi
done

echo "========================================================================================="
echo "Successfully submitted $COUNT batch jobs for $ENGINE model execution!"
echo "========================================================================================="
echo ""
echo "Monitor all jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Watch batch progress with:"
echo "  watch -n 5 'squeue -u \$USER; echo \"\"; ls -lh output_${ENGINE}_*.tex 2>/dev/null | wc -l; echo \"tex files generated so far\"'"
