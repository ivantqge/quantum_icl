#!/bin/bash
#SBATCH -A m2616
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -J qicl_main
#SBATCH -o /global/homes/i/ivang/unitary-synthesis/results_qicl/main_sweep.%j.log

# Main 2x2 ablation: {feedback off/on} x {structural retrieval off/on}.
# Tiers: B (stabilizer states), C_lite (1q Clifford), D_lite (1q Clifford+T).
# Model: gpt-4o-mini (cheap, continuity with prior data; non-reasoning).

set -e
module load python/3.11-24.1.0 >/dev/null 2>&1
source /global/homes/i/ivang/unitary-synthesis/.venv/bin/activate
cd /global/homes/i/ivang/unitary-synthesis

echo "=== connectivity test ==="
python - <<'PY' || echo "NO DIRECT NET"
import urllib.request
print("openrouter reachable:",
      urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=20).status)
PY

MODEL="${MODEL:-openai/gpt-4o-mini}"
TAG="${TAG:-main_$(echo $MODEL | tr '/' '_')}"
echo "=== launching $MODEL ==="
python -u -m quantum_icl.experiment \
    --backend openrouter --model "$MODEL" \
    --tiers B,C_lite,D_lite \
    --conditions zero_shot,feedback_only,structural_retrieval_only,structural_retrieval_plus_feedback \
    --num-tasks 30 --attempts 3 --max-tokens 1024 --seed 42 \
    --workers 12 --tag "$TAG" --outdir results_qicl
echo "=== done $MODEL ==="
