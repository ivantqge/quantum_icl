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
from .simulate import circuit_metrics, state_vector


def _fmt_state(vec):
    return "[" + ", ".join(f"{z.real:+.4f}{z.imag:+.4f}j" for z in vec) + "]"
from .metrics import MetricsLogger
from . import llm as llm_mod

CONDITION_PRESETS = {
    # Main 2x2 ablation: {feedback off/on} x {retrieval off/on}.
    "zero_shot":                          {"retriever": "none",       "use_fixed": False, "grows": False, "use_feedback": False},
    "feedback_only":                      {"retriever": "none",       "use_fixed": False, "grows": False, "use_feedback": True},
    "structural_retrieval_only":          {"retriever": "structural", "use_fixed": False, "grows": True,  "use_feedback": False},
    "structural_retrieval_plus_feedback": {"retriever": "structural", "use_fixed": False, "grows": True,  "use_feedback": True},
    # Legacy / ancillary conditions retained for compatibility:
    "fixed_few_shot":                     {"retriever": "none",       "use_fixed": True,  "grows": False, "use_feedback": True},
    "random_retrieval":                   {"retriever": "random",     "use_fixed": False, "grows": True,  "use_feedback": True},
    "text_retrieval":                     {"retriever": "text",       "use_fixed": False, "grows": True,  "use_feedback": True},
    "structural_retrieval":               {"retriever": "structural", "use_fixed": False, "grows": True,  "use_feedback": True},
    "growing_structural_library":         {"retriever": "structural", "use_fixed": False, "grows": True,  "use_feedback": True},
    "oracle_retrieval":                   {"retriever": "oracle",     "use_fixed": False, "grows": True,  "use_feedback": True},
}

DEFAULT_CONFIG = {
    "seed": 42,
    "llm": {"backend": "mock", "model": "openai/gpt-4o-mini",
            "temperature": 0.0, "max_tokens": 1024},
    "experiment": {
        "tiers": ["B", "C_lite", "D_lite"],
        "conditions": [
            "zero_shot", "feedback_only",
            "structural_retrieval_only",
            "structural_retrieval_plus_feedback",
        ],
        "attempts_per_task": 3,
        "num_examples": 3,
        "fidelity_threshold": 0.999,
        "outdir": "results_qicl",
        # Early stopping: after `early_stop_after` tasks, if a block has solved
        # 0 or all of them, skip the rest of that block.
        "early_stopping": True,
        "early_stop_after": 10,
    },
    "tiers": {
        "A": {"num_tasks": 30, "qubit_range": [4, 5], "edge_prob": 0.5},
        "B": {"num_tasks": 30, "qubit_range": [2, 2], "gen_gate_range": [2, 6]},
        "C": {"num_tasks": 30, "qubit_range": [2, 2], "gen_gate_range": [3, 10]},
        "D": {"num_tasks": 30, "qubit_range": [2, 2], "gen_gate_range": [4, 12]},
        "C_lite": {"num_tasks": 20, "qubit_range": [1, 1], "gen_gate_range": [1, 6]},
        "D_lite": {"num_tasks": 30, "qubit_range": [1, 1], "gen_gate_range": [1, 6]},
        "D_mid":  {"num_tasks": 20, "qubit_range": [2, 2], "gen_gate_range": [4, 7]},
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


def _run_block(cfg, condition, tier, tier_tasks, all_tasks, exp, k, thr,
               shared_llm=None):
    """Run one (condition, tier) block sequentially; return its records.

    Each block is self-contained (own library + LLM instance), so blocks can
    execute concurrently without shared state. When `shared_llm` is given
    (e.g. for the local HF backend so the model loads only once) the block
    reuses it and reports per-block cost as the post-pre counter delta.
    """
    cc = CONDITION_PRESETS[condition]
    backend = cfg["llm"]["backend"]
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
        pre_calls = pre_pt = pre_ct = 0
        pre_cost = 0.0
    elif shared_llm is not None:
        llm = shared_llm
        pre_calls = llm.total_calls
        pre_pt = llm.total_prompt_tokens
        pre_ct = llm.total_completion_tokens
        pre_cost = llm.total_cost_usd
    else:
        llm = llm_mod.make_llm(
            backend, model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
            max_tokens=cfg["llm"]["max_tokens"],
            adapter_path=cfg["llm"].get("adapter_path"),
        )
        pre_calls = pre_pt = pre_ct = 0
        pre_cost = 0.0

    attempts_rec, tasks_rec, libcurve_rec = [], [], []

    for idx, task in enumerate(tier_tasks):
        lib_at_start = library.size()
        libcurve_rec.append({"condition": condition, "tier": tier,
                             "task_index": idx, "size": lib_at_start})

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
        feedback = None

        for attempt in range(exp["attempts_per_task"]):
            used = attempt + 1
            system, user = build_messages(
                task, examples, feedback=feedback,
                prompt_variant=cfg.get("experiment", {}).get("prompt_variant", "default"),
            )
            try:
                resp = llm.generate(system, user)
            except Exception as e:
                attempts_rec.append(dict(
                    condition=condition, tier=tier, task_id=task.task_id,
                    task_index=idx, attempt=attempt + 1,
                    num_retrieved=len(examples), valid=False, success=False,
                    fidelity=0.0, depth=None, gate_count=None,
                    two_qubit_gate_count=None, t_count=None, prompt_tokens=0,
                    completion_tokens=0, cost_usd=0.0, library_size=lib_at_start,
                    error=f"generate failed: {type(e).__name__}: {e}",
                    system=system, user=user, response="",
                    retrieved_ids=retrieved_ids,
                ))
                feedback = None
                continue
            pt += resp.prompt_tokens
            ct += resp.completion_tokens
            cost += resp.cost_usd
            res = verify(task, resp.text, thr)
            best_fid = max(best_fid, res.fidelity)
            m = res.metrics or {}
            attempts_rec.append(dict(
                condition=condition, tier=tier, task_id=task.task_id,
                task_index=idx, attempt=attempt + 1,
                num_retrieved=len(examples), valid=res.valid,
                success=res.success, fidelity=res.fidelity,
                depth=m.get("depth"), gate_count=m.get("gate_count"),
                two_qubit_gate_count=m.get("two_qubit_gate_count"),
                t_count=m.get("t_count"), prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens, cost_usd=resp.cost_usd,
                library_size=lib_at_start, error=res.error, system=system,
                user=user, response=resp.text, retrieved_ids=retrieved_ids,
            ))
            if res.success:
                solved, sol_m = True, m
                if cc["grows"]:
                    library.add(task, res.circuit, m)
                break
            # Self-refinement feedback is gated by the condition (2x2 ablation).
            if not cc.get("use_feedback", True):
                feedback = None
                continue
            produced_str = None
            if res.circuit and task.target_kind == "state":
                try:
                    produced_str = _fmt_state(state_vector(res.circuit))
                except Exception:
                    produced_str = None
            feedback = {
                "prev_circuit_json": (json.dumps(res.circuit)
                                      if res.circuit else None),
                "fidelity": res.fidelity, "valid": res.valid,
                "error": res.error, "produced_state_str": produced_str,
            }

        tasks_rec.append(dict(
            condition=condition, tier=tier, task_id=task.task_id,
            task_index=idx, solved=solved, attempts_used=used,
            best_fidelity=best_fid, depth=sol_m.get("depth"),
            gate_count=sol_m.get("gate_count"),
            two_qubit_gate_count=sol_m.get("two_qubit_gate_count"),
            t_count=sol_m.get("t_count"), prompt_tokens=pt,
            completion_tokens=ct, cost_usd=cost,
            library_size_at_start=lib_at_start,
            gen_gate_count=gen_m.get("gate_count"),
            gen_t_count=gen_m.get("t_count"),
        ))

        # Early-stop a saturated/failed block to save budget.
        if exp.get("early_stopping", True):
            n_done = idx + 1
            after = exp.get("early_stop_after", 10)
            if n_done >= after:
                n_solved = sum(1 for t in tasks_rec if t["solved"])
                if n_solved == 0 or n_solved == n_done:
                    break

    return {
        "condition": condition, "tier": tier,
        "attempts": attempts_rec, "tasks": tasks_rec, "libcurve": libcurve_rec,
        "solved": sum(1 for t in tasks_rec if t["solved"]), "n": len(tier_tasks),
        "final_lib": library.size(),
        "llm_calls": llm.total_calls - pre_calls,
        "pt": llm.total_prompt_tokens - pre_pt,
        "ct": llm.total_completion_tokens - pre_ct,
        "cost": llm.total_cost_usd - pre_cost,
    }


def run(cfg, run_dir, verbose=True) -> MetricsLogger:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    exp = cfg["experiment"]
    pools = generate_task_pools(cfg)
    all_tasks = [t for tier in pools for t in pools[tier]]
    logger = MetricsLogger()

    k = exp["num_examples"]
    thr = exp["fidelity_threshold"]
    workers = int(exp.get("workers", 1))

    # Each (condition, tier) block is independent (own library + LLM), so run
    # them concurrently; each block is sequential internally to preserve
    # growing-library order. Records merge deterministically by (cond, tier).
    jobs = [(c, t) for c in exp["conditions"] for t in exp["tiers"]]

    # Load the local HF model once and share it across blocks; API backends
    # are cheap to construct so we keep per-block instances for those.
    shared_llm = None
    if cfg["llm"]["backend"] == "local":
        shared_llm = llm_mod.make_llm(
            "local", model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
            max_tokens=cfg["llm"]["max_tokens"],
            adapter_path=cfg["llm"].get("adapter_path"),
        )

    def work(ct):
        c, t = ct
        return _run_block(cfg, c, t, pools[t], all_tasks, exp, k, thr,
                          shared_llm=shared_llm)

    blocks_dir = os.path.join(run_dir, "blocks")
    os.makedirs(blocks_dir, exist_ok=True)
    results = {}

    def record(blk):
        results[(blk["condition"], blk["tier"])] = blk
        # Persist each block as it finishes so a timeout/kill can never lose
        # completed work; rebuild_from_blocks() can recover a partial run.
        with open(os.path.join(blocks_dir,
                  f"{blk['condition']}__{blk['tier']}.json"), "w") as f:
            json.dump(blk, f)
        if verbose:
            print(f"  [{blk['condition']}|{blk['tier']}] "
                  f"{blk['solved']}/{blk['n']} solved, "
                  f"final library={blk['final_lib']}", flush=True)

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(work, ct): ct for ct in jobs}
            for fut in as_completed(futs):
                record(fut.result())
    else:
        for ct in jobs:
            record(work(ct))

    # Merge records into the logger in deterministic (condition, tier) order.
    for ct in jobs:
        blk = results[ct]
        for r in blk["libcurve"]:
            logger.log_library_size(r["condition"], r["tier"],
                                    r["task_index"], r["size"])
        for r in blk["attempts"]:
            logger.log_attempt(**r)
        for r in blk["tasks"]:
            logger.log_task(**r)

    logger.write(run_dir)
    print(f"\nLLM totals: calls={sum(b['llm_calls'] for b in results.values())}, "
          f"tokens={sum(b['pt'] for b in results.values())}+"
          f"{sum(b['ct'] for b in results.values())}, "
          f"cost=${sum(b['cost'] for b in results.values()):.4f}", flush=True)
    return logger


def rebuild_from_blocks(run_dir):
    """Rebuild merged outputs (csv/jsonl/summary) from per-block JSON files.

    Use to recover a run that was killed/timed out after some blocks finished.
    """
    import glob
    logger = MetricsLogger()
    for fp in sorted(glob.glob(os.path.join(run_dir, "blocks", "*.json"))):
        with open(fp) as f:
            blk = json.load(f)
        for r in blk["libcurve"]:
            logger.log_library_size(r["condition"], r["tier"],
                                    r["task_index"], r["size"])
        for r in blk["attempts"]:
            logger.log_attempt(**r)
        for r in blk["tasks"]:
            logger.log_task(**r)
    logger.write(run_dir)
    return run_dir


def main():
    ap = argparse.ArgumentParser(description="Quantum-ICL experiment runner")
    ap.add_argument("--rebuild", default=None,
                    help="rebuild merged outputs from <run_dir>/blocks and exit")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--backend", default=None,
                    choices=["mock", "openrouter", "grok", "local"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter-path", default=None,
                    help="LoRA adapter directory (local backend only)")
    ap.add_argument("--tiers", default=None, help="comma list, e.g. A,B")
    ap.add_argument("--conditions", default=None, help="comma list")
    ap.add_argument("--num-tasks", type=int, default=None,
                    help="override num_tasks for every tier")
    ap.add_argument("--attempts", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="max output tokens per call (raise for reasoning models)")
    ap.add_argument("--workers", type=int, default=None,
                    help="concurrent (condition,tier) blocks (default 1)")
    ap.add_argument("--prompt-variant", default=None,
                    choices=["default", "cot"],
                    help="prompt style: default (current) or cot (chain-of-thought)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature (default 0)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if args.rebuild:
        out = rebuild_from_blocks(args.rebuild)
        print(f"Rebuilt merged outputs in {out}")
        return

    cfg = load_config(args.config)
    if args.backend:
        cfg["llm"]["backend"] = args.backend
    if args.model:
        cfg["llm"]["model"] = args.model
    if args.adapter_path:
        cfg["llm"]["adapter_path"] = args.adapter_path
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.tiers:
        cfg["experiment"]["tiers"] = args.tiers.split(",")
    if args.conditions:
        cfg["experiment"]["conditions"] = args.conditions.split(",")
    if args.attempts is not None:
        cfg["experiment"]["attempts_per_task"] = args.attempts
    if args.max_tokens is not None:
        cfg["llm"]["max_tokens"] = args.max_tokens
    if args.workers is not None:
        cfg["experiment"]["workers"] = args.workers
    if args.prompt_variant is not None:
        cfg["experiment"]["prompt_variant"] = args.prompt_variant
    if args.temperature is not None:
        cfg["llm"]["temperature"] = args.temperature
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
