# Quantum-ICL — Slide Deck Guidelines

Target: 8–10 slides for a 3–5 minute demo video, with the four required Q1–Q4 video questions and the 15-point rubric mapped to specific slides.

---

## Slide 1 — Title + one-line claim
- **Title:** *Quantum-ICL: Simulator-Verified In-Context Learning and LoRA Fine-Tuning for Quantum Circuit Synthesis*
- **One-line:** *Frozen LLMs can learn quantum circuit synthesis through a closed loop with a simulator — and we measure exactly how.*
- Author / class / date.
- *Rubric: Communication & Presentation (2pt) — clear framing for outsiders.*

---

## Slide 2 — Q1: Why we built this (Problem & Motivation)
- **The bottleneck:** quantum circuit synthesis is hard for LLMs; we want to know *which* lightweight mechanisms actually help frozen LLMs improve on a verified task.
- **Why it matters:** if frozen LLMs can self-improve via simulator feedback and verified retrieval, then *test-time compute* + *small fine-tunes* may substitute for expensive RLHF in scientific domains.
- **Why now:** sub-second simulators give *perfect* binary verification — most domains can't.
- *Rubric: Problem & Insight (3pt).*

---

## Slide 3 — Q2 (research): What we built (Architecture)
- A modular **`quantum_icl/`** Python package (12 modules, 28 unit tests).
- Four **task tiers** at calibrated difficulty: B (stabilizer states), C-lite (1q Clifford), D-lite (1q Clifford+T), D-mid (2q Clifford+T).
- Four **conditions** in a clean factorial ablation: `zero_shot`, `feedback_only`, `structural_retrieval_only`, `structural_retrieval_plus_feedback`.
- Three **retrieval strategies** (random, structural-target-features, oracle-hidden-features diagnostic).
- Four **LLM backends** (gpt-4o-mini, Gemini-3-Flash, Qwen-base, Qwen+LoRA-SFT).
- **Closed feedback loop**: simulator → verifier → state-vector diff → next prompt.
- *Rubric: Execution & Technical Work (5pt) — substantial functional artifact.*

---

## Slide 4 — Q2 (continued): Verifier-feedback + structural-retrieval pipeline
- Diagram (left to right):
  - Target → Prompt → LLM → JSON circuit → Schema check → Simulator → Fidelity
  - On failure: feedback = (prev circuit, fidelity, produced state) → next attempt
  - On success: append to library; future retrieval pulls by target-computable features.
- Key methodological choice: **target-only features** for retrieval (we benchmark vs. *oracle features* as upper bound).
- *Rubric: Execution & Technical Work (5pt) — system architecture.*

---

## Slide 5 — Headline results (Evaluation)
- Big single table: 4 conditions × 3 tiers × 3 frozen models, 100 tasks each.
- Highlight:
  - **Feedback is the dominant lever** (every model, every tier).
  - **Structural retrieval alone is flat/negative** in 9 of 12 cells.
  - **Structural + feedback** is the winning combination on hard tiers — Gemini D-lite: 28 → 65 (+37).
- *Rubric: Evaluation & Evidence (3pt).*

---

## Slide 6 — The SFT data-size finding (most novel)
- Plot: Qwen-7B solve rate vs SFT corpus size {0, 300, 900, 1800, 3600} for each condition × tier.
- Three takeaways:
  - **300 examples is catastrophic** (D-lite zero-shot 14 → 0).
  - **U-shaped recovery** on `struct+fb`: 22 → 15 → 26 as SFT scales.
  - **3600 yields the best Qwen cells** (matches gpt-4o-mini on B feedback).
- *Rubric: Execution + Evaluation (combined).*

---

## Slide 7 — OracleRetrieval probe + limitations
- Comparison: target-only structural features ≈ hidden-generator oracle features (±1 task / 30 on Gemini-3-Flash).
- ⇒ **The retrieval bottleneck is library size + exemplar transfer, not feature engineering.**
- **Limitations:** single seed, n=100 (Wilson 95% CI ≈ ±9 pts); D-mid calibration in progress; library-size and feedback-component ablations queued overnight.
- *Rubric: Evaluation & Evidence (3pt) — limitations honestly stated.*

---

## Slide 8 — Q3: Use cases & impact
- **Direct:** verified ICL pipeline for quantum compiler frontends (LLM → tableau / Solovay-Kitaev fallback).
- **General:** any domain with cheap, perfect verifiers — formal-method-checked code, theorem proving, protein design (with foldability oracle), reversible-circuit synthesis.
- **Educational:** runnable artifact for a graduate quantum / NLP course (we used Stanford CS153 credits to develop this).
- **Open science:** $20 total API + 25 GPU-hours; reproducible end-to-end on a single A100.
- *Rubric: Problem & Insight (3pt) — vision and impact.*

---

## Slide 9 — Q4: Future work
- Library-size sweep (Exp 5): is the retrieval lift just library size?
- Feedback-component ablation: which part of feedback carries the signal?
- Retrieval top-k + diversity (MMR) sweep.
- D-mid (2-qubit Clifford+T) at the same scale; D-full with longer T-sequences.
- Wilson CI + paired McNemar across all cells.
- SFT-3600 with mixed-tier curriculum + held-out generalization tier.
- *Rubric: Problem & Insight (3pt) — meaningful roadmap.*

---

## Slide 10 — Process, integrity, disclosure
- **Public repo + commit history:** `github.com/ivantqge/quantum_icl` — incremental, dated commits (Phase 1 / Phase 2 / SFT v1 / SFT v2 / 2×2 refocus / SFT data-size sweep).
- **AI usage:** built collaboratively with the Claude Code agent (Claude Opus 4.7). Author made all research-direction decisions, picked task tiers, models, and budgets, and interpreted all results. Agent contributed code, batch-job orchestration, and prose drafting. All results re-derivable from `scripts/`.
- **Tooling/credits:** OpenRouter (Stanford CS153 credits), NERSC Perlmutter (account m2616/m2616_g), TRL+PEFT for SFT, Cirq for verification, HuggingFace for Qwen2.5-7B-Instruct.
- **Major decisions discussed:** DeepSeek-R1 dropped (cost/saturation tradeoff), v1 SFT mode-collapse failure documented and remediated, walltime-killed SFT-300 recovered via incremental persistence.
- *Rubric: Process, Integrity & Disclosure (2pt).*

---

## Figures (drop directly into the deck)

All figures are in `paper/figures/` as both PNG (for the deck) and PDF (for LaTeX).

| File | Slide | Purpose |
|---|---|---|
| `fig1_main_2x2.png` | Slide 5 (headline results) | 3-panel grouped bar chart: 4 conditions × 3 frozen models × 3 tiers. The 2×2 ablation at a glance. |
| `fig2_sft_curve.png` | Slide 6 (SFT data-size) | 3-panel line chart: solve rate vs SFT corpus size (0 → 3600) × condition. Shows the **non-monotone** curve and U-shaped struct+fb recovery. |
| `fig3_oracle_vs_struct.png` | Slide 7 (Oracle probe) | Side-by-side bars: structural (target features) vs oracle (hidden gen features) on Gemini-3-Flash. Identical bars ⇒ retrieval bottleneck isn't features. |
| `fig4_base_vs_sft.png` | Slide 6 supplement | Paired base-Qwen vs SFT-3600 across all 12 cells, same seeds. Shows where SFT wins (and the one cell where it interferes). |
| `fig5_attempts_per_solved.png` | Slide 5 efficiency angle | Tier D-lite: mean attempts/solve per condition × model. Shows feedback adds *and* makes solves cheaper. |
| `fig6_capability_ladder.png` | Slide 8 (use cases) | Best-cell solve rate per model × tier. The visual story for the capability-ladder claim. |

## Notes for the demo recording

1. **Open with a 15-second live run** of `python -m quantum_icl.experiment --backend mock --tiers B --conditions structural_retrieval_plus_feedback --num-tasks 5 --no-plots` — shows that the pipeline really runs.
2. Spend ~30 seconds on the architecture diagram (Slide 4).
3. Spend ~60 seconds on the headline results table (Slide 5).
4. Spend ~45 seconds on the SFT curve (Slide 6) — this is the most novel finding.
5. Wrap with Q3 use-cases + Q4 roadmap + Q5 disclosure (Slides 8–10).
6. Total target: **3–5 minutes**. Hard cap at 10 min.
