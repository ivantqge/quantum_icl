#!/bin/bash
#SBATCH -A m2616_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 06:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -G 1
#SBATCH -J qicl_qe
#SBATCH -o /global/homes/i/ivang/unitary-synthesis/results_qicl/qwen_eval_param.%j.log

# Parametrized Qwen eval. Set ADAPTER (or empty for base), TAG, NUM_TASKS,
# TIERS, CONDITIONS via --export=ALL,...

set -e
module load python/3.11-24.1.0 >/dev/null 2>&1
source /global/homes/i/ivang/unitary-synthesis/.venv/bin/activate
cd /global/homes/i/ivang/unitary-synthesis

export HF_HOME=/pscratch/sd/i/ivang/hf_cache
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TAG="${TAG:?set TAG}"
NUM_TASKS="${NUM_TASKS:-100}"
TIERS="${TIERS:-B,C_lite,D_lite}"
CONDITIONS="${CONDITIONS:-zero_shot,feedback_only,structural_retrieval_only,structural_retrieval_plus_feedback}"
ATTEMPTS="${ATTEMPTS:-3}"

ADAPTER_ARG=""
if [ -n "$ADAPTER" ]; then
    ADAPTER_ARG="--adapter-path $ADAPTER"
fi

echo "=== Qwen eval: tag=$TAG adapter=$ADAPTER num=$NUM_TASKS tiers=$TIERS ==="
python -u -m quantum_icl.experiment \
    --backend local --model Qwen/Qwen2.5-7B-Instruct $ADAPTER_ARG \
    --tiers "$TIERS" --conditions "$CONDITIONS" \
    --num-tasks "$NUM_TASKS" --attempts "$ATTEMPTS" --max-tokens 1024 --seed 42 \
    --workers 1 --tag "$TAG" --outdir results_qicl
echo "=== eval done ==="
