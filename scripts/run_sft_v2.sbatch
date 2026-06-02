#!/bin/bash
#SBATCH -A m2616_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -G 1
#SBATCH -J qicl_sftv2
#SBATCH -o /global/homes/i/ivang/unitary-synthesis/results_qicl/sft_v2.%j.log

# SFT v2: assistant-only loss (so the model can't memorize the boilerplate),
# bigger LoRA (r=32, alpha=64), 3 epochs on the 1800-example dataset.

set -e
module load python/3.11-24.1.0 >/dev/null 2>&1
source /global/homes/i/ivang/unitary-synthesis/.venv/bin/activate
cd /global/homes/i/ivang/unitary-synthesis

export HF_HOME=/pscratch/sd/i/ivang/hf_cache

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUT="/pscratch/sd/i/ivang/qicl_sft/runs/qwen25_7b_v2"
echo "=== launching SFT v2 -> $OUT ==="
python -u -m quantum_icl.sft.train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data /pscratch/sd/i/ivang/qicl_sft/data_v2 \
    --out "$OUT" \
    --epochs 3 --batch-size 2 --grad-accum 8 --lr 1e-4 --max-seq-len 2048 \
    --lora-r 32 --lora-alpha 64
echo "=== SFT v2 finished ==="
