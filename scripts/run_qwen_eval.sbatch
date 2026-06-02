#!/bin/bash
#SBATCH -A m2616_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -G 1
#SBATCH -J qicl_qweneval
#SBATCH -o /global/homes/i/ivang/unitary-synthesis/results_qicl/qwen_eval.%j.log

# Evaluate base Qwen2.5-7B-Instruct then SFT-adapter Qwen on the 2x2 ablation.
# Same task seeds across all conditions and across base vs SFT.

set -e
module load python/3.11-24.1.0 >/dev/null 2>&1
source /global/homes/i/ivang/unitary-synthesis/.venv/bin/activate
cd /global/homes/i/ivang/unitary-synthesis

export HF_HOME=/pscratch/sd/i/ivang/hf_cache
ADAPTER="/pscratch/sd/i/ivang/qicl_sft/runs/qwen25_7b_v1/adapter"
MODEL="Qwen/Qwen2.5-7B-Instruct"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== base Qwen ==="
python -u -m quantum_icl.experiment \
    --backend local --model "$MODEL" \
    --tiers B,C_lite,D_lite \
    --conditions zero_shot,feedback_only,structural_retrieval_only,structural_retrieval_plus_feedback \
    --num-tasks 30 --attempts 3 --max-tokens 1024 --seed 42 \
    --workers 1 --tag base_qwen --outdir results_qicl

echo "=== SFT Qwen ==="
python -u -m quantum_icl.experiment \
    --backend local --model "$MODEL" --adapter-path "$ADAPTER" \
    --tiers B,C_lite,D_lite \
    --conditions zero_shot,feedback_only,structural_retrieval_only,structural_retrieval_plus_feedback \
    --num-tasks 30 --attempts 3 --max-tokens 1024 --seed 42 \
    --workers 1 --tag sft_qwen --outdir results_qicl

echo "=== done ==="
