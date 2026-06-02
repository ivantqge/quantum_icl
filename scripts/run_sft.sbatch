#!/bin/bash
#SBATCH -A m2616_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -G 1
#SBATCH -J qicl_sft
#SBATCH -o /global/homes/i/ivang/unitary-synthesis/results_qicl/sft.%j.log

set -e
module load python/3.11-24.1.0 >/dev/null 2>&1
source /global/homes/i/ivang/unitary-synthesis/.venv/bin/activate
cd /global/homes/i/ivang/unitary-synthesis

export HF_HOME=/pscratch/sd/i/ivang/hf_cache
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATA="${DATA:-/pscratch/sd/i/ivang/qicl_sft/data}"
OUT="${OUT:-/pscratch/sd/i/ivang/qicl_sft/runs/qwen25_7b_v1}"

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo "=== launching SFT: model=$MODEL data=$DATA out=$OUT ==="
python -u -m quantum_icl.sft.train \
    --model "$MODEL" --data "$DATA" --out "$OUT" \
    --epochs 2 --batch-size 2 --grad-accum 8 --lr 2e-4 --max-seq-len 2048
echo "=== SFT finished ==="
