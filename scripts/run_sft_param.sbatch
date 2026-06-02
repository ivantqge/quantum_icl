#!/bin/bash
#SBATCH -A m2616_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -G 1
#SBATCH -J qicl_sftp
#SBATCH -o /global/homes/i/ivang/unitary-synthesis/results_qicl/sft_param.%j.log

# Parametrized SFT (uses v2 hyperparams: assistant-only loss, r=32 alpha=64).
# Set DATA and OUT via --export=ALL,DATA=...,OUT=...

set -e
module load python/3.11-24.1.0 >/dev/null 2>&1
source /global/homes/i/ivang/unitary-synthesis/.venv/bin/activate
cd /global/homes/i/ivang/unitary-synthesis

export HF_HOME=/pscratch/sd/i/ivang/hf_cache
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

DATA="${DATA:?set DATA via sbatch --export}"
OUT="${OUT:?set OUT via sbatch --export}"
EPOCHS="${EPOCHS:-3}"

echo "=== SFT: data=$DATA out=$OUT epochs=$EPOCHS ==="
python -u -m quantum_icl.sft.train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data "$DATA" --out "$OUT" \
    --epochs "$EPOCHS" --batch-size 2 --grad-accum 8 --lr 1e-4 \
    --max-seq-len 2048 --lora-r 32 --lora-alpha 64
echo "=== SFT done -> $OUT/adapter ==="
