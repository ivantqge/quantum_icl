# unitary-synthesis

LLM **in-context learning** for quantum circuit synthesis. An LLM is given few-shot
examples and a target, emits a JSON circuit, and is verified by exact Cirq simulation.
Three experiment modes compare a self-improving example library against baselines:

- `independent` — zero-shot, no examples
- `static` — fixed 3-example starter library
- `growing` — library grows with every simulator-verified solution

> **Phase 1** (current): graph-state preparation (H + CZ gates), Grok backend, structured
> logging and plots. **Phase 2** (planned): generalize to true unitary synthesis with a
> universal gate set. See `.claude/plans/` for the full roadmap.

## Setup (NERSC Perlmutter)

```bash
module load python/3.11-24.1.0
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

LLM backends read their API key from the environment — never hardcode it:

```bash
export XAI_API_KEY=<your xai key>            # grok backend; https://console.x.ai
export OPENROUTER_API_KEY=<sk-or-v1-...>      # openrouter backend; https://openrouter.ai
```

**OpenRouter** gives access to many models (GPT, Claude, Gemini, Grok, Llama, …)
through one OpenAI-compatible endpoint, with per-call cost reported automatically.
Pass any OpenRouter model slug via `--model`, e.g. `openai/gpt-4o-mini`,
`anthropic/claude-3.5-sonnet`, `google/gemini-2.0-flash-001`, `x-ai/grok-2-1212`.

## Running

```bash
# Cost-free smoke test with the template MockLLM
python start.py --llm mock --mode all --rounds 2 --tasks-per-round 3 --seed 42

# Real run with xAI Grok (default model: grok-3-mini; --model accepts any string)
python start.py --llm grok --model grok-3-mini --mode all --rounds 5 --tasks-per-round 10 --seed 42

# Real run via OpenRouter (any model slug)
python start.py --llm openrouter --model openai/gpt-4o-mini --mode all --rounds 5 --tasks-per-round 10 --seed 42

# Phase 2: true unitary synthesis over the Clifford+T gate set
python start.py --llm openrouter --model openai/gpt-4o-mini --task-type unitary --gate-set clifford_t --mode all
```

Runs are reproducible: a fixed `--seed` plus `temperature=0.0` gives identical
results across processes (per-mode seeds are derived with a stable SHA-256 hash).

Key flags: `--llm {mock,grok,gemini,anthropic}`, `--model`, `--temperature` (grok, default
0.0), `--mode {independent,static,growing,all}`, `--rounds`, `--tasks-per-round`,
`--retries`, `--retrieval-k`, `--seed`, `--outdir` (default `results/`), `--tag`,
`--no-plots`, `--quiet`.

## Output

Each run writes a timestamped directory `results/<tag>_<UTC timestamp>/`:

| File | Contents |
|---|---|
| `tasks.jsonl` | One record per task, including every LLM attempt's full prompt + response |
| `tasks.csv` | Flat scalar columns (success, fidelity, gate count, depth, tokens, cost, wall time) |
| `rounds.csv` | Per-mode, per-round aggregate stats |
| `summary.json` | Run config + per-mode success rates + total tokens/cost |
| `config.json` | The CLI args used |
| `plots/` | `success_vs_round`, `library_growth`, `success_by_qubit_count`, `cost_and_quality` |

Plots can be regenerated offline from a saved run: `python plots.py results/<run_dir>`.

## Tests

```bash
python -m pytest tests/ -q
```

## Module overview

- `start.py` — CLI entry point
- `graph_task.py` — graph-state task definition + random task generation
- `circuit.py` — circuit definition, JSON parsing, Cirq simulation, verification, `circuit_metrics`
- `library.py` — solution library with similarity-based retrieval
- `llm.py` — `BaseLLM`, `MockLLM`, `GrokLLM`, `GeminiLLM`, `AnthropicLLM`, prompt construction
- `evaluation.py` — experiment runner
- `results_logger.py` — structured run logging (`RunLogger`, `TaskRecord`)
- `plots.py` — result plotting
