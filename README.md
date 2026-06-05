# quantum-icl

A Python framework for studying **simulator-verified in-context learning** and **LoRA fine-tuning** for quantum circuit synthesis. A frozen (or LoRA-tuned) LLM proposes a circuit, then an exact quantum simulator (Cirq) verifies it. Verified solutions accumulate into a retrieval-indexed library that is fed back into future prompts. The framework factorially measures the contribution of three mechanisms — **verifier-feedback self-refinement**, **structural retrieval**, and **synthetic-data SFT** — across four difficulty tiers (graph states, stabilizer states, 1-qubit Clifford / Clifford+T, 2-qubit Clifford+T) and different LLM backends on OpenRouter (Gemini, GPT, Qwen).

## Repository layout

```
quantum_icl/                # the framework (12 modules)
  schema.py                 # strict circuit-JSON validator
  simulate.py               # cirq simulation, fidelity, target features
  tasks.py                  # tier generators (A, B, C, D, C-lite, D-lite, D-mid)
  verify.py                 # global-phase-invariant verification
  library.py                # verified-solution store
  retrieval.py              # none / random / text-TFIDF / structural / oracle retrievers
  prompts.py                # per-tier prompts, feedback augmentation, CoT variant
  llm.py                    # mock / OpenRouter / xAI Grok / local HF backends
  metrics.py                # per-attempt logging + summary aggregation
  experiment.py             # config-driven runner with block-level parallelism
  plots.py                  # 5 default result figures
  config.yaml               # cheap pilot defaults
  sft/                      # supervised fine-tuning subpackage
    build_dataset.py        # chat-format dataset from verified generator circuits
    train.py                # LoRA SFT via TRL SFTTrainer (assistant-only loss)
tests/                      # pytest suite (verifier correctness, retriever, parsing)
requirements.txt
```

## Setup

To run the programs, first install the requirements. Originally ran on Python 3.11.  

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the API backends, export the relevant key:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...     # for --backend openrouter
export XAI_API_KEY=xai-...                 # for --backend grok
```

For the local backend, point the Hugging Face cache somewhere with enough space:

```bash
export HF_HOME=/path/to/large/scratch/hf_cache
```

## Usage

### 1. Free smoke test (mock backend, no API or GPU)

```bash
python -m quantum_icl.experiment \
    --backend mock \
    --tiers B,C_lite,D_lite \
    --conditions zero_shot,feedback_only,structural_retrieval_only,structural_retrieval_plus_feedback \
    --num-tasks 5 --workers 4 --tag smoke --no-plots
```

Writes a timestamped run dir under `results_qicl/smoke_*` containing `summary.json`, per-block JSON files, per-attempt JSONL (full prompts + responses), and CSV exports.

### 2. Real API run (OpenRouter)

```bash
python -m quantum_icl.experiment \
    --backend openrouter --model google/gemini-3-flash-preview \
    --tiers B,C_lite,D_lite,D_mid \
    --conditions zero_shot,feedback_only,structural_retrieval_only,structural_retrieval_plus_feedback \
    --num-tasks 100 --attempts 3 --max-tokens 1024 --workers 12 --seed 42 \
    --temperature 0.0 --prompt-variant default \
    --tag gemini100
```

Optional flags:
- `--prompt-variant cot` — chain-of-thought system prompt.
- `--temperature 0.4 --attempts 5` (with `feedback_only=False`-style condition) — Best-of-N parallel sampling.
- `--rebuild <run_dir>` — recover a run from per-block files after a timeout/crash.

### 3. Local Hugging Face backend for local models (Qwen)

```bash
python -m quantum_icl.experiment \
    --backend local --model Qwen/Qwen2.5-7B-Instruct \
    --tiers B,C_lite,D_lite --conditions ... \
    --num-tasks 100 --workers 1 --tag base_qwen
```

The model is loaded once and shared across (condition, tier) blocks — see `quantum_icl/llm.py::LocalHFLLM`.

### 4. LoRA supervised fine-tuning on simulator-verified synthetic data

Build a dataset (chat-format, assistant turn = the simulator-verified hidden generator circuit):

```bash
python -m quantum_icl.sft.build_dataset \
    --tiers B,C_lite,D_lite \
    --train 600 --val 30 --test 30 \
    --seed 42 --out /path/to/sft_data
```

Train a LoRA adapter (TRL `SFTTrainer`, **`assistant_only_loss=True`** — critical, without it the model mode-collapses):

```bash
python -m quantum_icl.sft.train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data /path/to/sft_data \
    --out  /path/to/run \
    --epochs 3 --batch-size 2 --grad-accum 8 \
    --lr 1e-4 --max-seq-len 2048 --lora-r 32 --lora-alpha 64
```

Evaluate the adapter using the same experiment runner:

```bash
python -m quantum_icl.experiment \
    --backend local --model Qwen/Qwen2.5-7B-Instruct \
    --adapter-path /path/to/run/adapter \
    --tiers B,C_lite,D_lite --conditions ... \
    --num-tasks 100 --workers 1 --tag sft_qwen
```

### 5. Tests

```bash
pytest tests/ -q
```

11 unit tests cover schema validation, verifier correctness on every tier (generator solves own task at fidelity = 1, perturbations fail, global-phase invariance), gate-set enforcement, and retriever ranking.

## CLI reference

`python -m quantum_icl.experiment --help` for the full list. Key flags:

| Flag | Purpose |
|---|---|
| `--backend {mock,openrouter,grok,local}` | LLM backend |
| `--model NAME` | Model id (OpenRouter slug, xAI Grok name, or HF repo) |
| `--adapter-path PATH` | Load a LoRA adapter (local backend only) |
| `--tiers A,B,C,D,C_lite,D_lite,D_mid` | Comma-separated tier list |
| `--conditions ...` | Comma-separated subset of `zero_shot, feedback_only, structural_retrieval_only, structural_retrieval_plus_feedback, fixed_few_shot, random_retrieval, text_retrieval, oracle_retrieval` |
| `--num-tasks N` | Tasks per (condition, tier) cell |
| `--attempts N` | Maximum attempts per task (with feedback or sampling) |
| `--prompt-variant {default,cot}` | Default or chain-of-thought system prompt |
| `--temperature FLOAT` | Sampling temperature (>0 enables Best-of-N when `attempts>1` and feedback off) |
| `--max-tokens N` | Per-call output cap |
| `--workers N` | Concurrent (condition, tier) blocks (deterministic across worker counts) |
| `--seed INT` | Master seed (per-cell seeds are derived via SHA-256, reproducible across processes) |
| `--rebuild RUN_DIR` | Reconstruct `summary.json` from per-block files (recovery) |
| `--tag NAME --outdir DIR` | Output directory and naming |

## AI usage disclosure

This project was developed collaboratively with the Anthropic Claude Code agent (Claude Opus 4.7). The author owned all research-direction decisions — task tiers, ablation factorial, model selection, calibration, budget, interpretation of results, and decisions to cancel/resubmit jobs. The agent contributed to code authoring, batch-job orchestration for experiments, statistical aggregation, and documentation drafting. Every numerical result presented in any derived writeup is generated by a re-executable script in this repository.

## Acknowledgements

External libraries used:

- [Cirq](https://github.com/quantumlib/Cirq) — quantum-circuit simulation and Hilbert–Schmidt process fidelity.
- [TRL](https://github.com/huggingface/trl) — `SFTTrainer` / `SFTConfig` for LoRA supervised fine-tuning with `assistant_only_loss`.
- [PEFT](https://github.com/huggingface/peft) — LoRA adapter implementation.
- [Transformers](https://github.com/huggingface/transformers) and [Datasets](https://github.com/huggingface/datasets) — model loading and data pipeline.
- [OpenAI Python SDK](https://github.com/openai/openai-python) — OpenAI-compatible client used for both OpenRouter and xAI Grok.

Compute / inference credits:

- [OpenRouter](https://openrouter.ai) — Stanford CS153 course credits.

Model checkpoints downloaded from Hugging Face:

- [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Apache-2.0).

## External resources

- [OpenRouter model catalog](https://openrouter.ai/models)
- [Cirq documentation](https://quantumai.google/cirq)
- [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer)
