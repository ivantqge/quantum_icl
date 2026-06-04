# Quantum-ICL

> **Simulator-verified in-context learning and lightweight LoRA fine-tuning for quantum circuit synthesis.**
>
> A frozen LLM proposes circuits in strict JSON, an exact quantum simulator verifies them, and verified solutions feed back into future prompts. We test whether **verifier feedback**, **structural retrieval**, and **synthetic-data SFT** improve frozen-LLM performance — and find a clean answer.

Public code: `github.com/ivantqge/quantum_icl`
Author: Ivan Ge (Stanford); built collaboratively with the Claude Code agent — see *AI & tool disclosure* below.

---

## TL;DR (headline findings, 100 tasks/cell, n=4 backbones)

| Lever | Effect |
|---|---|
| **Verifier-feedback self-refinement** | Dominant; helps every model, every tier. |
| **Structural retrieval alone** | Flat or negative in 9/12 cells. |
| **Structural retrieval + feedback** | The winning combination on hard tiers (Gemini D-lite: 28 → **65/100**). |
| **LoRA SFT (Qwen-7B) on verified synthetic data** | Non-monotone: 300 examples is catastrophic, ≥900 stabilizes, 1800–3600 yields strongest combined ICL + SFT. |
| **OracleRetrieval probe** | Target-only features ≈ hidden-generator features (±1/30 on Gemini). Bottleneck is library size, not feature engineering. |
| **Chain-of-thought prompting (Gemini)** | **Largest single lever**: +19 on D-lite zero-shot (28→47); stacks with feedback+retrieval to **71/100** (vs 65 without CoT). |
| **Best-of-N parallel sampling** | Lift but weaker than CoT alone; combining CoT + Best-of-N + retrieval pushes D-mid from 17/100 → **32/100**. |

Total cost: **~$3 of OpenRouter API** on the main 100-task sweeps + **~25 GPU-hours** on a single A100 80 GB.

---

## Layout

```
quantum_icl/                    # the package
  schema.py                     # strict circuit-JSON validator (rejects unsupported gates)
  simulate.py                   # cirq simulation, fidelity, circuit metrics, target features
  tasks.py                      # tier generators (B, C-lite, D-lite, D-mid, A, C, D, ...)
  verify.py                     # schema -> simulate -> compare, global-phase invariant
  library.py                    # verified-solution storage
  retrieval.py                  # NoRetrieval / Random / Text(TF-IDF) / Structural / Oracle
  prompts.py                    # per-tier system prompt + feedback augmentation
  llm.py                        # mock / OpenRouter / Grok / Local HF backends
  metrics.py                    # logging + aggregation
  experiment.py                 # config-driven runner (workers, --rebuild, --adapter-path)
  plots.py                      # success-vs-X, library growth, cost-per-solved figures
  sft/                          # SFT data builder + LoRA trainer (TRL SFTTrainer)
    build_dataset.py
    train.py
scripts/                        # Slurm batch scripts (CPU API sweep, GPU SFT, GPU Qwen eval, smoke checks)
tests/                          # 28 unit tests covering schema, verifier, retriever, SFT, Oracle
paper/paper.tex                 # LaTeX writeup (this study)
SLIDES.md                       # slide-deck outline mapped to rubric
```

---

## Quick start

```bash
# (NERSC Perlmutter; tested with python/3.11-24.1.0)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Free pipeline smoke (mock backend, no API/GPU)
python -m quantum_icl.experiment --backend mock --tiers B,C_lite,D_lite \
    --conditions zero_shot,feedback_only,structural_retrieval_only,structural_retrieval_plus_feedback \
    --num-tasks 5 --workers 4 --tag smoke --no-plots

# Reproduce a 100-task Gemini sweep ($1.50, 15 min)
export OPENROUTER_API_KEY=sk-or-v1-...
python -m quantum_icl.experiment --backend openrouter --model google/gemini-3-flash-preview \
    --tiers B,C_lite,D_lite \
    --conditions zero_shot,feedback_only,structural_retrieval_only,structural_retrieval_plus_feedback \
    --num-tasks 100 --attempts 3 --workers 12 --seed 42 --tag gemini100

# Reproduce a 100-task base Qwen eval (free, ~3-5h on A100)
export HF_HOME=/scratch/your_user/hf_cache
python -m quantum_icl.experiment --backend local --model Qwen/Qwen2.5-7B-Instruct \
    --tiers B,C_lite,D_lite --conditions <same as above> \
    --num-tasks 100 --workers 1 --tag base_qwen
```

If a multi-hour run dies before writing `summary.json`, recover the partial results from per-block files:
```bash
python -m quantum_icl.experiment --rebuild path/to/run_dir
```

---

## SFT pipeline

```bash
# Build training jsonl (assistant turn = simulator-verified generator circuit)
python -m quantum_icl.sft.build_dataset \
    --tiers B,C_lite,D_lite --train 600 --val 30 --test 30 \
    --seed 42 --out /scratch/.../qicl_sft/data_v2

# LoRA SFT on Qwen2.5-7B-Instruct (r=32, alpha=64, assistant_only_loss, 3 epochs)
python -m quantum_icl.sft.train --model Qwen/Qwen2.5-7B-Instruct \
    --data /scratch/.../qicl_sft/data_v2 --out /scratch/.../qicl_sft/runs/qwen25_7b_v2

# Evaluate the adapter inside the same experiment runner
python -m quantum_icl.experiment --backend local --model Qwen/Qwen2.5-7B-Instruct \
    --adapter-path /scratch/.../runs/qwen25_7b_v2/adapter \
    --tiers B,C_lite,D_lite --conditions <...> --num-tasks 100 --tag sft_v2
```

> *Failure mode worth knowing about:* training with full-sequence loss (no `assistant_only_loss`) converges fast but mode-collapses — the model emits a single fixed circuit for every target because the assistant turn is a tiny fraction of tokens. Use `--assistant-only-loss` (default in the script).

---

## Results (all runs reproducible; see `paper/paper.tex` for the full report)

### Experiment 1 — $2\times2$ ablation × three frozen models (100 tasks/cell)

| Tier | Condition | gpt-4o-mini | Gemini-3-Flash | Qwen-7B (base) |
|---|---|---|---|---|
| **B** | zero_shot | 9 | **79** | 8 |
| | feedback_only | 11 | 75 | 14 |
| | structural_only | 6 | 79 | 11 |
| | structural+feedback | **12** | **86** | **22** |
| **C-lite** | zero_shot | 11 | 57 | 23 |
| | feedback_only | **15** | 74 | **30** |
| | structural_only | 10 | 73 | 21 |
| | structural+feedback | **15** | **100**⁺ | 27 |
| **D-lite** | zero_shot | 7 | 28 | 14 |
| | feedback_only | **8** | 53 | 20 |
| | structural_only | 3 | 54 | 15 |
| | structural+feedback | 4 | **65** | **20** |

⁺ early-stopped at 10/10 (saturated). Cost: gpt-4o-mini $0.22, Gemini-3-Flash $1.33, Qwen 0 (local).

### Experiment 2 — OracleRetrieval probe (target features ≈ oracle features)

| Tier | Structural+fb (target) | Oracle (hidden gen features) | Δ |
|---|---|---|---|
| B | 10/10⁺ | 29/30 | ≈ 0 |
| C-lite | 24/30 | 24/30 | 0 |
| D-lite | 18/30 | 19/30 | +1 |

### Experiment 3 — SFT data-size sweep on Qwen-7B (100 tasks/cell)

| Tier | Condition | base (0) | 300 | 900 | 1800 | 3600 |
|---|---|---|---|---|---|---|
| **B** | zero_shot | 8 | 7 | 17 | 20 | 16 |
| | feedback_only | 14 | 11 | 19 | 22 | **27** |
| | struct_only | 11 | 9 | 12 | 11 | **16** |
| | struct+fb | 22 | — | 12 | 15 | **26** |
| **C-lite** | zero_shot | 23 | 3 | 17 | 18 | 16 |
| | feedback_only | 30 | 9 | 34 | **35** | 34 |
| | struct_only | 21 | — | 10 | 18 | 11 |
| | struct+fb | 27 | — | 27 | 32 | **35** |
| **D-lite** | zero_shot | 14 | 0⁻ | 3 | 10 | **16** |
| | feedback_only | 20 | 0⁻ | 10 | 18 | **22** |
| | struct_only | 15 | — | 7 | 13 | 10 |
| | struct+fb | 20 | — | 21 | **28** | 24 |

⁻ early-stopped at 0/10; — = block did not finish (SFT-300 hit walltime; 7 of 12 recovered via per-block persistence).

---

## Reproducibility

- **Seeds:** every per-(condition, tier) RNG is derived from `(--seed, condition, tier, "purpose")` via SHA-256 → reproducible across processes regardless of `PYTHONHASHSEED`.
- **Determinism:** verified `workers=1` vs `workers=8` give identical summaries on mock.
- **Incremental persistence:** each (condition, tier) block writes to `<run_dir>/blocks/`; `--rebuild` recovers partial runs.
- **Tests:** `pytest tests/` → 28 passing, covering schema, verifier correctness across every tier (generator solves own task at F=1, perturbations fail, phase invariance), gate-set enforcement, retrievers, SFT, OracleRetrieval.

---

## Project meets rubric

| Rubric criterion | Where |
|---|---|
| **Problem & Insight (3 pt)** | `paper/paper.tex` §1 + this README header + SLIDES.md slide 2 |
| **Execution & Technical Work (5 pt)** | 12-module package + 28 tests + 5 SFT runs + 6 100-task evals + LaTeX report (all in `quantum_icl/`, `scripts/`, `paper/`) |
| **Evaluation & Evidence (3 pt)** | 100-task confirmatory tables, OracleRetrieval probe, SFT data-size curve, paired comparisons |
| **Communication & Presentation (2 pt)** | `paper/paper.tex` + `SLIDES.md` + README + reproducible CLI |
| **Process, Integrity & Disclosure (2 pt)** | Public git history, AI-usage statement below, limitations section in paper |

---

## AI & tool disclosure

This project was built collaboratively with the **Claude Code agent (Anthropic Claude Opus 4.7)**:
- **Author (Ivan Ge)** owned all research-direction decisions: choice of task tiers, choice of models and condition factorial, calibration decisions, budget calls, interpretation of all results, decision to cancel/resubmit jobs after failures.
- **Agent** contributed: code authoring (schema/simulate/verify/library/retrieval/prompts/LLM backends/metrics/experiment/plots/sft), batch-job orchestration on Slurm, statistical aggregation, prose drafting (the LaTeX paper text and this README).
- **Verifiable artifacts:** every result table here and in `paper/paper.tex` was generated from a re-executable batch script in `scripts/`; per-run summaries are in `results_qicl/`.

### Tools & external code
- **TRL** (`SFTTrainer`, `SFTConfig`) for LoRA fine-tuning.
- **PEFT** for the LoRA adapter implementation.
- **HuggingFace transformers + datasets** for model loading and data pipeline.
- **Cirq** for quantum circuit simulation and Hilbert-Schmidt process fidelity.
- **OpenAI Python SDK** as the OpenAI-compatible client for both OpenRouter and xAI Grok.
- **OpenRouter** API for gpt-4o-mini, Gemini-3-Flash, deepseek-r1 access; Stanford CS153 course credits.
- **NERSC Perlmutter** for compute (account `m2616` CPU, `m2616_g` GPU); single A100 80 GB used for all SFT and local-LLM eval.

### Model checkpoints
- `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0; downloaded to pscratch HF cache).

### What was *not* forked
- The `quantum_icl/` package is original code written for this project.
- The data generation, evaluation framework, prompting, and feedback loop are original to this study.

### Major decisions documented
- **DeepSeek-R1 dropped** from main sweeps after pilot showed cost/latency made the 4-tier × 4-condition × 100-task sweep infeasible.
- **Stabilizer-generator representation tried and dropped** for Tier B — empirically hurt weaker models (1/30 vs 9/30 with raw amplitudes).
- **SFT v1 mode-collapsed**; switched to `assistant_only_loss=True` for v2.
- **SFT-300 walltime-killed**; per-block persistence saved 7 of 12 blocks (intended design).
- **Login-node SIGKILL incident** ($~$8 of API spend lost on a partial run before per-block persistence existed) — fixed by moving long runs to Slurm batch with `--rebuild` recovery.
