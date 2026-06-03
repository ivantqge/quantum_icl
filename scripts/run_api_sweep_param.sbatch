#!/bin/bash
#SBATCH -A m2616
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -J qicl_api
#SBATCH -o /global/homes/i/ivang/unitary-synthesis/results_qicl/api_sweep.%j.log

# Parametrized OpenRouter sweep. Set MODEL, TAG, NUM_TASKS via --export.

set -e
module load python/3.11-24.1.0 >/dev/null 2>&1
source /global/homes/i/ivang/unitary-synthesis/.venv/bin/activate
cd /global/homes/i/ivang/unitary-synthesis

MODEL="${MODEL:?set MODEL}"
TAG="${TAG:?set TAG}"
NUM_TASKS="${NUM_TASKS:-100}"
TIERS="${TIERS:-B,C_lite,D_lite}"
CONDITIONS="${CONDITIONS:-zero_shot,feedback_only,structural_retrieval_only,structural_retrieval_plus_feedback}"
# Workaround for sbatch --export which uses commas as variable separators:
# users pass conditions as 'a|b|c' and we convert to 'a,b,c' here.
CONDITIONS="${CONDITIONS//|/,}"
TIERS="${TIERS//|/,}"
ATTEMPTS="${ATTEMPTS:-3}"
TEMPERATURE="${TEMPERATURE:-0.0}"
PROMPT_VARIANT="${PROMPT_VARIANT:-default}"

python - <<'PY' || echo "NO DIRECT NET"
import urllib.request
print("openrouter reachable:", urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=20).status)
PY

echo "=== sweep: model=$MODEL tag=$TAG num=$NUM_TASKS ==="
python -u -m quantum_icl.experiment \
    --backend openrouter --model "$MODEL" \
    --tiers "$TIERS" --conditions "$CONDITIONS" \
    --num-tasks "$NUM_TASKS" --attempts "$ATTEMPTS" --max-tokens 1024 --seed 42 \
    --workers 12 --tag "$TAG" --outdir results_qicl \
    --temperature "$TEMPERATURE" --prompt-variant "$PROMPT_VARIANT"
echo "=== sweep done ==="
