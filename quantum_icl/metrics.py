"""Logging of raw generations/verifier results and aggregation to summaries."""

import csv
import json
import os
from collections import defaultdict

_ATTEMPT_SCALARS = [
    "condition", "tier", "task_id", "task_index", "attempt", "num_retrieved",
    "valid", "success", "fidelity", "depth", "gate_count",
    "two_qubit_gate_count", "t_count", "prompt_tokens", "completion_tokens",
    "cost_usd", "library_size", "error",
]
_TASK_FIELDS = [
    "condition", "tier", "task_id", "task_index", "solved", "attempts_used",
    "best_fidelity", "depth", "gate_count", "two_qubit_gate_count", "t_count",
    "prompt_tokens", "completion_tokens", "cost_usd", "library_size_at_start",
    "gen_gate_count", "gen_t_count",
]


class MetricsLogger:
    def __init__(self):
        self.attempts = []        # full records (incl. prompts/responses)
        self.tasks = []           # per-task summaries
        self.library_curve = []   # (condition, tier, task_index, size)

    def log_attempt(self, **fields):
        self.attempts.append(fields)

    def log_task(self, **fields):
        self.tasks.append(fields)

    def log_library_size(self, condition, tier, task_index, size):
        self.library_curve.append(
            {"condition": condition, "tier": tier,
             "task_index": task_index, "library_size": size}
        )

    # --- aggregation -------------------------------------------------------

    def aggregate(self) -> dict:
        groups = defaultdict(list)
        for t in self.tasks:
            groups[(t["condition"], t["tier"])].append(t)

        def mean(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs) / len(xs) if xs else 0.0

        out = {}
        for (cond, tier), ts in groups.items():
            solved = [t for t in ts if t["solved"]]
            out[f"{cond}|{tier}"] = {
                "condition": cond, "tier": tier,
                "n_tasks": len(ts),
                "n_solved": len(solved),
                "success_rate": len(solved) / len(ts) if ts else 0.0,
                "mean_best_fidelity": mean([t["best_fidelity"] for t in ts]),
                "mean_attempts_per_solved": mean([t["attempts_used"] for t in solved]),
                "mean_depth_solved": mean([t["depth"] for t in solved]),
                "mean_gate_count_solved": mean([t["gate_count"] for t in solved]),
                "mean_two_qubit_solved": mean([t["two_qubit_gate_count"] for t in solved]),
                "mean_t_count_solved": mean([t["t_count"] for t in solved]),
                "mean_gen_gate_count": mean([t.get("gen_gate_count") for t in ts]),
                "mean_gen_t_count": mean([t.get("gen_t_count") for t in ts]),
                "mean_prompt_tokens": mean([t["prompt_tokens"] for t in ts]),
                "total_cost_usd": sum(t["cost_usd"] for t in ts),
            }
        return out

    # --- output ------------------------------------------------------------

    def write(self, run_dir: str) -> str:
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "attempts.jsonl"), "w") as f:
            for a in self.attempts:
                f.write(json.dumps(a) + "\n")

        with open(os.path.join(run_dir, "attempts.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_ATTEMPT_SCALARS)
            w.writeheader()
            for a in self.attempts:
                w.writerow({k: a.get(k) for k in _ATTEMPT_SCALARS})

        with open(os.path.join(run_dir, "tasks.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_TASK_FIELDS)
            w.writeheader()
            for t in self.tasks:
                w.writerow({k: t.get(k) for k in _TASK_FIELDS})

        with open(os.path.join(run_dir, "library_size.csv"), "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["condition", "tier", "task_index", "library_size"])
            w.writeheader()
            for r in self.library_curve:
                w.writerow(r)

        summary = self.aggregate()
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return run_dir
