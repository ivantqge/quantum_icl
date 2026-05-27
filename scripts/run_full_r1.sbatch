#!/bin/bash
#SBATCH -A m2616
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -J qicl_r1
#SBATCH -o /global/homes/i/ivang/unitary-synthesis/results_qicl/full_r1n.%j.log

set -e
module load python/3.11-24.1.0 >/dev/null 2>&1
source /global/homes/i/ivang/unitary-synthesis/.venv/bin/activate
cd /global/homes/i/ivang/unitary-synthesis

# OpenRouter key passed via --export on sbatch (OPENROUTER_API_KEY).
echo "=== connectivity test ==="
python - <<'PY' || echo "NO DIRECT NET (may need proxy)"
import urllib.request
print("openrouter reachable:",
      urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=20).status)
PY

echo "=== launching experiment ==="
python -u -m quantum_icl.experiment \
    --backend openrouter --model deepseek/deepseek-r1 \
    --tiers A,B,C,D \
    --conditions zero_shot,fixed_few_shot,random_retrieval,structural_retrieval \
    --num-tasks 20 --attempts 3 --max-tokens 4096 --seed 42 \
    --workers 16 --tag full_r1n --outdir results_qicl
echo "=== experiment finished ==="
