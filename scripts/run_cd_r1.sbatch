#!/bin/bash
#SBATCH -A m2616
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 06:00:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -J qicl_cd
#SBATCH -o /global/homes/i/ivang/unitary-synthesis/results_qicl/cd_r1.%j.log

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

# C/D (hard unitary tiers) only; attempts=2 to bound cost/time. Blocks persist
# incrementally so a timeout can't lose completed work.
echo "=== launching C/D experiment ==="
python -u -m quantum_icl.experiment \
    --backend openrouter --model deepseek/deepseek-r1 \
    --tiers C,D \
    --conditions zero_shot,fixed_few_shot,random_retrieval,structural_retrieval \
    --num-tasks 20 --attempts 2 --max-tokens 4096 --seed 42 \
    --workers 8 --tag cd_r1 --outdir results_qicl
echo "=== experiment finished ==="
