"""Build a SFT dataset from simulator-verified circuits.

Each example is a chat-format (system, user, assistant) triple where the
assistant turn is the hidden generator circuit (verified-correct by
construction) for that task. We split a deterministic train/val/test by seed
so the same task never crosses splits.

Usage:
    python -m quantum_icl.sft.build_dataset \
        --tiers B,C_lite,D_lite --train 600 --val 60 --test 60 \
        --out /pscratch/sd/i/ivang/qicl_sft/data
"""

import argparse
import json
import os
import random

from .. import tasks as T
from ..prompts import build_messages


def _example(task) -> dict:
    """Build a single chat example: system + user prompt, assistant = generator JSON."""
    system, user = build_messages(task, examples=[], feedback=None)
    completion = (
        "Strategy: implement the target with the allowed gates.\n"
        f"```json\n{json.dumps(task.generator)}\n```"
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": completion},
        ],
        "tier": task.tier,
        "task_id": task.task_id,
    }


def build_split(tier: str, n: int, rng: random.Random) -> list:
    tasks = T.generate_tasks(tier, n, rng=rng)
    return [_example(t) for t in tasks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="B,C_lite,D_lite")
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tiers = args.tiers.split(",")

    splits = {"train": args.train, "val": args.val, "test": args.test}
    for split_name, n_per_tier in splits.items():
        rows = []
        for tier in tiers:
            rng = random.Random(hash((args.seed, split_name, tier)) & 0x7fffffff)
            rows.extend(build_split(tier, n_per_tier, rng))
        random.Random(args.seed).shuffle(rows)
        path = os.path.join(args.out, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} examples to {path}")


if __name__ == "__main__":
    main()
