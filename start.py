#!/usr/bin/env python3
"""Self-improving quantum circuit synthesis system.

Solves graph state preparation tasks (4-6 qubits, H+CZ gate set) by learning
from verified successes. Compares a growing example library against baselines.

Usage:
    python start.py --mode all --seed 42
    python start.py --mode growing --rounds 5 --tasks-per-round 10
    python start.py --llm grok --model grok-3-mini --mode all
    python start.py --llm anthropic --model claude-sonnet-4-20250514 --mode all
"""

import argparse
import random
import sys

from functools import partial

from graph_task import generate_random_tasks
from unitary_task import generate_unitary_tasks, create_unitary_starter_library
from library import create_starter_library
from llm import MockLLM, MockUnitaryLLM, GeminiLLM, AnthropicLLM, GrokLLM
from evaluation import run_experiment, print_comparison, print_library_growth
from results_logger import RunLogger
from plots import make_all_plots


def main():
    parser = argparse.ArgumentParser(
        description="Self-improving quantum circuit synthesis"
    )
    parser.add_argument(
        "--mode",
        choices=["independent", "static", "growing", "all"],
        default="all",
        help="Experiment mode. 'all' runs all three for comparison (default: all)",
    )
    parser.add_argument(
        "--task-type",
        choices=["graph_state", "unitary"],
        default="graph_state",
        help="Task family: graph-state preparation or unitary synthesis "
             "(default: graph_state)",
    )
    parser.add_argument(
        "--gate-set",
        choices=["clifford_t"],
        default="clifford_t",
        help="Gate set for unitary-synthesis tasks (default: clifford_t)",
    )
    parser.add_argument(
        "--rounds", type=int, default=5,
        help="Number of rounds (default: 5)",
    )
    parser.add_argument(
        "--tasks-per-round", type=int, default=10,
        help="Tasks per round (default: 10)",
    )
    parser.add_argument(
        "--retries", type=int, default=3,
        help="Max LLM retries per task (default: 3)",
    )
    parser.add_argument(
        "--retrieval-k", type=int, default=3,
        help="Number of examples to retrieve (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--llm",
        choices=["mock", "gemini", "anthropic", "grok"],
        default="mock",
        help="LLM backend (default: mock)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name override (default: grok-3-mini / gemini-2.0-flash / "
             "claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature for the grok backend (default: 0.0)",
    )
    parser.add_argument(
        "--outdir", type=str, default="results",
        help="Directory for run output (default: results)",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Run label for the output directory (default: <llm>_<mode>)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-task output",
    )

    args = parser.parse_args()

    # Set up RNG
    rng = random.Random(args.seed)

    # Generate task pool + select the matching starter library
    total_tasks = args.rounds * args.tasks_per_round
    if args.task_type == "unitary":
        print(f"Generating {total_tasks} random unitary-synthesis tasks "
              f"(gate set: {args.gate_set})...")
        all_tasks = generate_unitary_tasks(
            total_tasks,
            qubit_range=(1, 3),
            gate_set=args.gate_set,
            gate_count_range=(2, 6),
            rng=rng,
        )
        starter_library_factory = partial(
            create_unitary_starter_library, args.gate_set
        )
    else:
        print(f"Generating {total_tasks} random graph state tasks...")
        all_tasks = generate_random_tasks(
            total_tasks,
            qubit_range=(4, 6),
            edge_prob=0.5,
            rng=rng,
        )
        starter_library_factory = create_starter_library
    print(f"Generated {len(all_tasks)} tasks "
          f"({args.rounds} rounds x {args.tasks_per_round} tasks)")

    # Verify starter library
    print("Verifying starter library...")
    starter = starter_library_factory()
    print(f"Starter library: {starter.size()} verified examples\n")

    # Set up LLM
    if args.llm == "mock":
        print("Using MockLLM (template-based, ~70% correct)\n")
    elif args.llm == "gemini":
        model = args.model or "gemini-2.0-flash"
        print(f"Using Gemini LLM (model: {model})\n")
    elif args.llm == "anthropic":
        model = args.model or "claude-sonnet-4-20250514"
        print(f"Using Anthropic LLM (model: {model})\n")
    elif args.llm == "grok":
        model = args.model or "grok-3-mini"
        print(f"Using xAI Grok LLM (model: {model})\n")

    # Determine modes to run
    modes = (
        ["independent", "static", "growing"]
        if args.mode == "all"
        else [args.mode]
    )

    # Set up the run logger
    tag = args.tag or f"{args.llm}_{args.mode}"
    logger = RunLogger(outdir=args.outdir, run_tag=tag)
    logger.save_run_config(vars(args))
    print(f"Logging run to {logger.run_dir}\n")

    results = []
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Running experiment: {mode}")
        print(f"{'='*60}")

        # Each mode gets its own LLM instance with a deterministic seed
        # derived from the main seed, so results are reproducible
        mode_seed = hash((args.seed, mode)) % (2**31)
        if args.llm == "mock":
            if args.task_type == "unitary":
                llm = MockUnitaryLLM(all_tasks, rng=random.Random(mode_seed))
            else:
                llm = MockLLM(rng=random.Random(mode_seed))
        elif args.llm == "gemini":
            llm = GeminiLLM(model=args.model or "gemini-2.0-flash")
        elif args.llm == "grok":
            llm = GrokLLM(
                model=args.model or "grok-3-mini",
                temperature=args.temperature,
            )
        else:
            llm = AnthropicLLM(model=args.model or "claude-sonnet-4-20250514")

        exp_rng = random.Random(mode_seed)

        result = run_experiment(
            mode=mode,
            all_tasks=all_tasks,
            llm=llm,
            tasks_per_round=args.tasks_per_round,
            max_retries=args.retries,
            num_retrieval=args.retrieval_k,
            rng=exp_rng,
            verbose=not args.quiet,
            logger=logger,
            starter_library_factory=starter_library_factory,
        )
        results.append(result)

    # Write structured results
    logger.write(results)
    print(f"\nResults written to {logger.run_dir}")

    # Generate plots
    if not args.no_plots:
        plot_dir = make_all_plots(logger.run_dir)
        print(f"Plots written to {plot_dir}")

    # Print comparison
    print_comparison(results)
    print_library_growth(results)


if __name__ == "__main__":
    main()
