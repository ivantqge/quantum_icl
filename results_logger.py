"""Structured result logging for circuit-synthesis experiments.

Writes a self-contained, timestamped run directory containing:
  - tasks.jsonl   one TaskRecord per line, including full prompts/responses
  - tasks.csv     flat scalar columns for quick analysis
  - rounds.csv    per-mode, per-round aggregate stats
  - summary.json  run config + per-mode rates + total tokens/cost
  - config.json   the CLI args used for the run
  - library_<mode>.json  final library snapshot (growing mode)
"""

from dataclasses import dataclass, field, asdict
import csv
import datetime
import json
import os


@dataclass
class TaskRecord:
    """Per-task result, including every LLM attempt."""
    run_tag: str
    mode: str
    round_num: int
    task_id: str
    num_qubits: int
    edge_count: int
    solved: bool
    attempts_used: int
    final_fidelity: float
    gate_count: int = 0
    depth: int = 0
    two_qubit_gate_count: int = 0
    t_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    est_cost_usd: float = 0.0
    wall_time_s: float = 0.0
    error_type: str = ""
    # Per-attempt detail: list of dicts with attempt/system/user/response/
    # fidelity/parse_error keys.
    attempts: list = field(default_factory=list)

    # Scalar columns written to tasks.csv (everything except `attempts`).
    CSV_FIELDS = [
        "run_tag", "mode", "round_num", "task_id", "num_qubits", "edge_count",
        "solved", "attempts_used", "final_fidelity", "gate_count", "depth",
        "two_qubit_gate_count", "t_count", "prompt_tokens", "completion_tokens",
        "est_cost_usd", "wall_time_s", "error_type",
    ]


class RunLogger:
    """Collects TaskRecords and writes a run directory."""

    def __init__(self, outdir: str, run_tag: str):
        self.run_tag = run_tag
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = os.path.join(outdir, f"{run_tag}_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.records: list[TaskRecord] = []

    def log_task(self, record: TaskRecord):
        self.records.append(record)

    def save_run_config(self, config: dict):
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def save_library(self, library, mode: str):
        """Snapshot a SolutionLibrary to library_<mode>.json."""
        path = os.path.join(self.run_dir, f"library_{mode}.json")
        library.save(path)

    def write(self, results: list):
        """Write tasks.jsonl, tasks.csv, rounds.csv, summary.json.

        `results` is a list of ExperimentResult (from evaluation.py).
        """
        self._write_tasks_jsonl()
        self._write_tasks_csv()
        self._write_rounds_csv(results)
        self._write_summary(results)
        return self.run_dir

    def _write_tasks_jsonl(self):
        path = os.path.join(self.run_dir, "tasks.jsonl")
        with open(path, "w") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r)) + "\n")

    def _write_tasks_csv(self):
        path = os.path.join(self.run_dir, "tasks.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TaskRecord.CSV_FIELDS)
            writer.writeheader()
            for r in self.records:
                row = {k: getattr(r, k) for k in TaskRecord.CSV_FIELDS}
                writer.writerow(row)

    def _write_rounds_csv(self, results: list):
        path = os.path.join(self.run_dir, "rounds.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "mode", "round_num", "attempted", "solved",
                "success_rate", "library_size",
            ])
            for res in results:
                for s in res.rounds:
                    writer.writerow([
                        res.mode, s.round_num, s.attempted, s.solved,
                        f"{s.success_rate:.6f}", s.library_size,
                    ])

    def _write_summary(self, results: list):
        # Per-mode aggregates derived from task records.
        per_mode = {}
        for r in self.records:
            m = per_mode.setdefault(r.mode, {
                "tasks": 0, "solved": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "est_cost_usd": 0.0,
                "total_attempts": 0,
            })
            m["tasks"] += 1
            m["solved"] += int(r.solved)
            m["prompt_tokens"] += r.prompt_tokens
            m["completion_tokens"] += r.completion_tokens
            m["est_cost_usd"] += r.est_cost_usd
            m["total_attempts"] += r.attempts_used
        for m in per_mode.values():
            m["success_rate"] = m["solved"] / m["tasks"] if m["tasks"] else 0.0

        summary = {
            "run_tag": self.run_tag,
            "run_dir": self.run_dir,
            "total_tasks": len(self.records),
            "total_prompt_tokens": sum(r.prompt_tokens for r in self.records),
            "total_completion_tokens": sum(r.completion_tokens for r in self.records),
            "total_est_cost_usd": sum(r.est_cost_usd for r in self.records),
            "per_mode": per_mode,
            "modes": {
                res.mode: {
                    "overall_rate": res.overall_rate,
                    "total_solved": res.total_solved,
                    "total_attempted": res.total_attempted,
                    "final_library_size": (
                        res.rounds[-1].library_size if res.rounds else 0
                    ),
                }
                for res in results
            },
        }
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
