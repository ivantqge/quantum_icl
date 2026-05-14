"""Experiment runner and evaluation logic."""

import copy
import random
import time
from dataclasses import dataclass, field

from graph_task import GraphStateTask
from circuit import parse_circuit_json, verify_circuit, circuit_metrics
from library import SolvedExample, SolutionLibrary, create_starter_library
from llm import BaseLLM, build_prompt
from results_logger import TaskRecord


@dataclass
class RoundStats:
    """Statistics for one round of experiments."""
    round_num: int
    attempted: int = 0
    solved: int = 0
    library_size: int = 0

    @property
    def success_rate(self) -> float:
        return self.solved / self.attempted if self.attempted > 0 else 0.0


@dataclass
class ExperimentResult:
    """Full results from an experiment run."""
    mode: str
    rounds: list = field(default_factory=list)  # list of RoundStats

    @property
    def total_attempted(self) -> int:
        return sum(r.attempted for r in self.rounds)

    @property
    def total_solved(self) -> int:
        return sum(r.solved for r in self.rounds)

    @property
    def overall_rate(self) -> float:
        t = self.total_attempted
        return self.total_solved / t if t > 0 else 0.0


def run_experiment(
    mode: str,
    all_tasks: list,
    llm: BaseLLM,
    tasks_per_round: int = 10,
    max_retries: int = 3,
    num_retrieval: int = 3,
    rng: random.Random = None,
    verbose: bool = True,
    logger=None,
    starter_library_factory=create_starter_library,
) -> ExperimentResult:
    """Run a full experiment in the given mode.

    Modes:
    - "independent": no examples shown (empty library)
    - "static": starter library only, never grows
    - "growing": starter library, grows with each verified solution

    If `logger` (a results_logger.RunLogger) is given, a TaskRecord is logged
    for every task, capturing per-attempt prompts/responses, circuit-quality
    metrics, token usage, and wall time. When `logger` is None the function
    behaves exactly as before (backward compatible).
    """
    run_tag = logger.run_tag if logger is not None else ""
    if rng is None:
        rng = random.Random()

    # Initialize library based on mode
    if mode == "independent":
        library = SolutionLibrary()
    else:
        library = starter_library_factory()

    # For static mode, we keep a reference but never add to it
    allow_growth = (mode == "growing")

    result = ExperimentResult(mode=mode)

    num_rounds = (len(all_tasks) + tasks_per_round - 1) // tasks_per_round

    for round_idx in range(num_rounds):
        start = round_idx * tasks_per_round
        end = min(start + tasks_per_round, len(all_tasks))
        round_tasks = all_tasks[start:end]

        stats = RoundStats(round_num=round_idx + 1)

        for task in round_tasks:
            stats.attempted += 1
            solved = False

            # Retrieve examples
            if library.size() > 0:
                examples = library.retrieve(task, k=num_retrieval, rng=rng)
            else:
                examples = []

            # Per-task tracking for logging.
            task_start = time.perf_counter()
            attempts_detail = []
            attempts_used = 0
            last_fidelity = 0.0
            last_error = ""
            solved_circuit = None
            tok_prompt = tok_completion = 0
            cost_usd = 0.0

            # Try up to max_retries times
            for attempt in range(max_retries):
                attempts_used = attempt + 1
                system_msg, user_msg = build_prompt(task, examples)
                response = llm.generate(system_msg, user_msg)

                usage = llm.get_last_usage()
                tok_prompt += usage.get("prompt_tokens", 0)
                tok_completion += usage.get("completion_tokens", 0)
                cost_usd += usage.get("est_cost_usd", 0.0)

                attempt_rec = {
                    "attempt": attempt + 1,
                    "system": system_msg,
                    "user": user_msg,
                    "response": response,
                    "fidelity": 0.0,
                    "parse_error": "",
                }

                try:
                    gate_set = (
                        task.gate_set_name()
                        if hasattr(task, "gate_set_name") else "graph_state"
                    )
                    circuit = parse_circuit_json(response, gate_set=gate_set)
                    passed, fidelity = verify_circuit(task, circuit)
                    attempt_rec["fidelity"] = fidelity
                    last_fidelity = fidelity

                    if passed:
                        if verbose:
                            print(
                                f"  [{mode}] Task {task.task_id}: "
                                f"PASS (fidelity={fidelity:.6f}, "
                                f"attempt={attempt + 1})"
                            )

                        solved_circuit = circuit
                        if allow_growth:
                            # Extract explanation from response
                            explanation = response.split("```")[0].strip()
                            if not explanation:
                                explanation = "Circuit verified correct."
                            library.add(SolvedExample(
                                task=task,
                                explanation=explanation,
                                circuit=circuit,
                                fidelity=fidelity,
                            ))

                        solved = True
                        attempts_detail.append(attempt_rec)
                        break
                    else:
                        if verbose:
                            print(
                                f"  [{mode}] Task {task.task_id}: "
                                f"FAIL fidelity={fidelity:.6f} "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                except (ValueError, Exception) as e:
                    last_error = type(e).__name__
                    attempt_rec["parse_error"] = f"{type(e).__name__}: {e}"
                    if verbose:
                        print(
                            f"  [{mode}] Task {task.task_id}: "
                            f"PARSE ERROR: {e} "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )

                attempts_detail.append(attempt_rec)

            if solved:
                stats.solved += 1

            if logger is not None:
                metrics = circuit_metrics(solved_circuit) if solved_circuit else {}
                logger.log_task(TaskRecord(
                    run_tag=run_tag,
                    mode=mode,
                    round_num=round_idx + 1,
                    task_id=task.task_id,
                    num_qubits=task.num_qubits,
                    edge_count=len(getattr(task, "edges", [])),
                    solved=solved,
                    attempts_used=attempts_used,
                    final_fidelity=last_fidelity,
                    gate_count=metrics.get("gate_count", 0),
                    depth=metrics.get("depth", 0),
                    two_qubit_gate_count=metrics.get("two_qubit_gate_count", 0),
                    t_count=metrics.get("t_count", 0),
                    prompt_tokens=tok_prompt,
                    completion_tokens=tok_completion,
                    est_cost_usd=cost_usd,
                    wall_time_s=time.perf_counter() - task_start,
                    error_type="" if solved else last_error,
                    attempts=attempts_detail,
                ))

        stats.library_size = library.size()
        result.rounds.append(stats)

        if verbose:
            print(
                f"  [{mode}] Round {stats.round_num}: "
                f"{stats.solved}/{stats.attempted} solved, "
                f"library size={stats.library_size}"
            )

    return result


def print_comparison(results: list):
    """Print a comparison table of experiment results."""
    if not results:
        return

    num_rounds = max(len(r.rounds) for r in results)

    # Header
    header = f"{'Mode':<14}"
    for i in range(num_rounds):
        header += f" | {'R' + str(i + 1):>6}"
    header += f" | {'Total':>7} | {'Rate':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        row = f"{r.mode:<14}"
        for i in range(num_rounds):
            if i < len(r.rounds):
                s = r.rounds[i]
                row += f" | {s.solved:>2}/{s.attempted:<2} "
            else:
                row += f" | {'':>6}"
        row += f" | {r.total_solved:>3}/{r.total_attempted:<3}"
        row += f" | {r.overall_rate:>5.1%}"
        print(row)

    print("=" * len(header))

    # Per-qubit breakdown if we have task data
    print()


def print_library_growth(results: list):
    """Print library size over rounds for the growing mode."""
    growing = [r for r in results if r.mode == "growing"]
    if not growing:
        return

    r = growing[0]
    print("\nLibrary growth (growing mode):")
    for s in r.rounds:
        bar = "#" * s.library_size
        print(f"  Round {s.round_num}: {bar} ({s.library_size} examples)")
