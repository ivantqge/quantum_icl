"""Config-driven experiment runner: the self-improving ICL core loop.

For each (condition, tier) it runs every task through up to N attempts, verifies
each generation, optionally grows a verified library, and logs everything.
Conditions differ only in which in-context examples are supplied.
"""

import argparse
import datetime
import hashlib
import json
import os
import random

from . import tasks as tasks_mod
from .library import VerifiedLibrary
from .retrieval import make_retriever
from .prompts import build_messages, fixed_examples
from .verify import verify
from .simulate import circuit_metrics
from .metrics import MetricsLogger
from . import llm as llm_mod

CONDITION_PRESETS = {
    "zero_shot": {"retriever": "none", "use_fixed": False, "grows": False},
    "fixed_few_shot": {"retriever": "none", "use_fixed": True, "grows": False},
    "random_retrieval": {"retriever": "random", "use_fixed": False, "grows": True},
    "text_retrieval": {"retriever": "text", "use_fixed": False, "grows": True},
    "structural_retrieval": {"retriever": "structural", "use_fixed": False, "grows": True},
    "growing_structural_library": {"retriever": "structural", "use_fixed": False, "grows": True},
}

DEFAULT_CONFIG = {
    "seed": 42,
    "llm": {"backend": "mock", "model": "openai/gpt-4o-mini",
            "temperature": 0.0, "max_tokens": 1024},
    "experiment": {
        "tiers": ["A", "B", "C", "D"],
        "conditions": ["zero_shot", "fixed_few_shot",
                       "random_retrieval", "structural_retrieval"],
        "attempts_per_task": 3,
        "num_examples": 3,
        "fidelity_threshold": 0.999,
        "outdir": "results_qicl",
    },
    "tiers": {
        "A": {"num_tasks": 5, "qubit_range": [4, 5], "edge_prob": 0.5},
        "B": {"num_tasks": 5, "qubit_range": [2, 3], "gen_gate_range": [3, 8]},
        "C": {"num_tasks": 5, "qubit_range": [2, 2], "gen_gate_range": [3, 10]},
        "D": {"num_tasks": 5, "qubit_range": [2, 2], "gen_gate_range": [4, 12]},
    },
}


def stable_seed(*parts) -> int:
    s = ":".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2 ** 31)


def _merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base.get(k, {}), v) if isinstance(v, dict) else v
    return out


def load_config(path=None) -> dict:
    cfg = DEFAULT_CONFIG
    if path:
        import yaml
        with open(path) as f:
            cfg = _merge(DEFAULT_CONFIG, yaml.safe_load(f) or {})
    return cfg


def generate_task_pools(cfg) -> dict:
    """One task pool per tier (shared across conditions for fair comparison)."""
    pools = {}
    for tier in cfg["experiment"]["tiers"]:
        tcfg = dict(cfg["tiers"][tier])
        n = tcfg.pop("num_tasks")
        tcfg = {k: (tuple(v) if isinstance(v, list) else v)
                for k, v in tcfg.items()}
        rng = random.Random(stable_seed(cfg["seed"], "tasks", tier))
        pools[tier] = tasks_mod.generate_tasks(tier, n, rng=rng, **tcfg)
    return pools


def run(cfg, run_dir, verbose=True) -> MetricsLogger:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    exp = cfg["experiment"]
    pools = generate_task_pools(cfg)
    all_tasks = [t for tier in pools for t in pools[tier]]
    logger = MetricsLogger()

    backend = cfg["llm"]["backend"]
    shared_llm = None
    if backend != "mock":
        shared_llm = llm_mod.make_llm(
            backend, model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
            max_tokens=cfg["llm"]["max_tokens"],
        )

    k = exp["num_examples"]
    thr = exp["fidelity_threshold"]

    for condition in exp["conditions"]:
        cc = CONDITION_PRESETS[condition]
        for tier in exp["tiers"]:
            tier_tasks = pools[tier]
            retr_rng = random.Random(stable_seed(cfg["seed"], condition, tier, "retr"))
            retriever = make_retriever(cc["retriever"], rng=retr_rng)
            library = VerifiedLibrary()
            fixed = fixed_examples(tier, k) if cc["use_fixed"] else None

            if backend == "mock":
                llm = llm_mod.MockLLM(
                    all_tasks,
                    rng=random.Random(stable_seed(cfg["seed"], condition, tier, "mock")),
                    success_rate=cfg["llm"].get("mock_success_rate", 0.7),
                )
            else:
                llm = shared_llm

            for idx, task in enumerate(tier_tasks):
                lib_at_start = library.size()
                logger.log_library_size(condition, tier, idx, lib_at_start)

                if cc["use_fixed"]:
                    examples = fixed
                elif cc["retriever"] != "none" and library.size() > 0:
                    examples = retriever.select(task, library, k)
                else:
                    examples = []
                retrieved_ids = [e.get("task_id", "fixed") for e in examples]

                gen_m = circuit_metrics(task.generator) if task.generator else {}
                best_fid, solved, used = 0.0, False, 0
                sol_m, pt, ct, cost = {}, 0, 0, 0.0

                for attempt in range(exp["attempts_per_task"]):
                    used = attempt + 1
                    system, user = build_messages(task, examples)
                    resp = llm.generate(system, user)
                    pt += resp.prompt_tokens
                    ct += resp.completion_tokens
                    cost += resp.cost_usd
                    res = verify(task, resp.text, thr)
                    best_fid = max(best_fid, res.fidelity)
                    m = res.metrics or {}
                    logger.log_attempt(
                        condition=condition, tier=tier, task_id=task.task_id,
                        task_index=idx, attempt=attempt + 1,
                        num_retrieved=len(examples), valid=res.valid,
                        success=res.success, fidelity=res.fidelity,
                        depth=m.get("depth"), gate_count=m.get("gate_count"),
                        two_qubit_gate_count=m.get("two_qubit_gate_count"),
                        t_count=m.get("t_count"),
                        prompt_tokens=resp.prompt_tokens,
                        completion_tokens=resp.completion_tokens,
                        cost_usd=resp.cost_usd, library_size=lib_at_start,
                        error=res.error, system=system, user=user,
                        response=resp.text, retrieved_ids=retrieved_ids,
                    )
                    if res.success:
                        solved, sol_m = True, m
                        if cc["grows"]:
                            library.add(task, res.circuit, m)
                        break

                logger.log_task(
                    condition=condition, tier=tier, task_id=task.task_id,
                    task_index=idx, solved=solved, attempts_used=used,
                    best_fidelity=best_fid,
                    depth=sol_m.get("depth"), gate_count=sol_m.get("gate_count"),
                    two_qubit_gate_count=sol_m.get("two_qubit_gate_count"),
                    t_count=sol_m.get("t_count"),
                    prompt_tokens=pt, completion_tokens=ct, cost_usd=cost,
                    library_size_at_start=lib_at_start,
                    gen_gate_count=gen_m.get("gate_count"),
                    gen_t_count=gen_m.get("t_count"),
                )

            if verbose:
                solved = sum(1 for t in logger.tasks
                             if t["condition"] == condition and t["tier"] == tier
                             and t["solved"])
                print(f"  [{condition}|{tier}] {solved}/{len(tier_tasks)} solved, "
                      f"final library={library.size()}")

    logger.write(run_dir)
    if shared_llm is not None:
        print(f"\nLLM totals: calls={shared_llm.total_calls}, "
              f"tokens={shared_llm.total_prompt_tokens}+{shared_llm.total_completion_tokens}, "
              f"cost=${shared_llm.total_cost_usd:.4f}")
    return logger


def main():
    ap = argparse.ArgumentParser(description="Quantum-ICL experiment runner")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--backend", default=None,
                    choices=["mock", "openrouter", "grok", "local"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--tiers", default=None, help="comma list, e.g. A,B")
    ap.add_argument("--conditions", default=None, help="comma list")
    ap.add_argument("--num-tasks", type=int, default=None,
                    help="override num_tasks for every tier")
    ap.add_argument("--attempts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg["llm"]["backend"] = args.backend
    if args.model:
        cfg["llm"]["model"] = args.model
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.tiers:
        cfg["experiment"]["tiers"] = args.tiers.split(",")
    if args.conditions:
        cfg["experiment"]["conditions"] = args.conditions.split(",")
    if args.attempts is not None:
        cfg["experiment"]["attempts_per_task"] = args.attempts
    if args.outdir:
        cfg["experiment"]["outdir"] = args.outdir
    if args.num_tasks is not None:
        for tier in cfg["tiers"]:
            cfg["tiers"][tier]["num_tasks"] = args.num_tasks

    tag = args.tag or cfg["llm"]["backend"]
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(cfg["experiment"]["outdir"], f"{tag}_{ts}")

    print(f"Quantum-ICL run -> {run_dir}")
    print(f"backend={cfg['llm']['backend']} model={cfg['llm']['model']} "
          f"tiers={cfg['experiment']['tiers']} "
          f"conditions={cfg['experiment']['conditions']}")
    run(cfg, run_dir)
    print(f"\nResults written to {run_dir}")

    if not args.no_plots:
        try:
            from .plots import make_all_plots
            pdir = make_all_plots(run_dir)
            print(f"Plots written to {pdir}")
        except Exception as e:
            print(f"(plotting skipped: {e})")


if __name__ == "__main__":
    main()
