"""Smoke test for the local HF backend on a GPU node.

Loads the local model, generates a circuit for one task per tier, verifies it,
and prints the result. Run on a GPU node with HF_HOME pointing at the cache:

    HF_HOME=/pscratch/sd/i/ivang/hf_cache python scripts/smoke_local.py Qwen/Qwen2.5-7B-Instruct
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_icl import tasks as T
from quantum_icl.prompts import build_messages
from quantum_icl.verify import verify
from quantum_icl.llm import LocalHFLLM

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading {model} ...", flush=True)
llm = LocalHFLLM(model=model, max_tokens=512, temperature=0.0)
print("Loaded.\n", flush=True)

for tier in ["A", "B", "C", "D"]:
    task = T.generate_tasks(tier, 1, rng=random.Random(0))[0]
    system, user = build_messages(task, [])
    resp = llm.generate(system, user)
    res = verify(task, resp.text)
    print(f"[{tier}] valid={res.valid} success={res.success} "
          f"fidelity={res.fidelity:.4f} "
          f"tokens={resp.prompt_tokens}+{resp.completion_tokens}", flush=True)
