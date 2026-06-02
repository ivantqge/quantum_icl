"""Quick SFT-v2 collapse check: 3 different prompts, are responses different?"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_icl import tasks as T
from quantum_icl.prompts import build_messages
from quantum_icl.llm import LocalHFLLM

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else \
    "/pscratch/sd/i/ivang/qicl_sft/runs/qwen25_7b_v2/adapter"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

print(f"Loading {MODEL} + adapter {ADAPTER} ...", flush=True)
llm = LocalHFLLM(model=MODEL, max_tokens=512, temperature=0.0, adapter_path=ADAPTER)
print("Loaded.\n", flush=True)

# Three structurally-different tasks (one per tier).
rng = random.Random(123)
tasks = []
for tier in ["B", "C_lite", "D_lite"]:
    tasks.append(T.generate_tasks(tier, 1, rng=rng)[0])

for i, task in enumerate(tasks):
    system, user = build_messages(task, [])
    resp = llm.generate(system, user)
    print(f"=== Task {i + 1} ({task.tier}, task_id {task.task_id[:30]}) ===")
    print(resp.text[:400])
    print()
