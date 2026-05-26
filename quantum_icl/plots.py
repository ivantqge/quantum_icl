"""Result figures for a Quantum-ICL run directory.

Pure consumer of tasks.csv / attempts.csv / library_size.csv, so it can be
re-run offline: python -m quantum_icl.plots <run_dir>
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TRUE = ("True", "true", "1")


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def plot_success_vs_tasks(tasks, out):
    by_cond = defaultdict(list)
    for r in tasks:
        by_cond[r["condition"]].append(r)
    plt.figure(figsize=(7, 4.5))
    for cond, rows in sorted(by_cond.items()):
        rows = sorted(rows, key=lambda r: (r["tier"], int(r["task_index"])))
        cum, ys = 0, []
        for i, r in enumerate(rows, 1):
            cum += 1 if r["solved"] in _TRUE else 0
            ys.append(cum / i)
        plt.plot(range(1, len(ys) + 1), ys, marker=".", label=cond)
    plt.xlabel("Tasks attempted")
    plt.ylabel("Cumulative success rate")
    plt.title("Success rate vs tasks attempted")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_success_vs_library(tasks, out):
    by_cond = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in tasks:
        size = int(_f(r["library_size_at_start"]))
        cell = by_cond[r["condition"]][size]
        cell[0] += 1 if r["solved"] in _TRUE else 0
        cell[1] += 1
    plt.figure(figsize=(7, 4.5))
    plotted = False
    for cond, sizes in sorted(by_cond.items()):
        if len(sizes) <= 1:
            continue  # library never grew (e.g. zero/fixed); skip
        xs = sorted(sizes)
        ys = [sizes[s][0] / sizes[s][1] for s in xs]
        plt.plot(xs, ys, marker="o", label=cond)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("Verified library size (at task start)")
    plt.ylabel("Success rate")
    plt.title("Success rate vs verified library size")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_fidelity_vs_attempt(attempts, out):
    agg = defaultdict(lambda: defaultdict(list))
    for r in attempts:
        agg[r["condition"]][int(r["attempt"])].append(_f(r["fidelity"]))
    plt.figure(figsize=(7, 4.5))
    for cond, byatt in sorted(agg.items()):
        xs = sorted(byatt)
        ys = [sum(byatt[a]) / len(byatt[a]) for a in xs]
        plt.plot(xs, ys, marker="o", label=cond)
    plt.xlabel("Attempt number")
    plt.ylabel("Mean fidelity")
    plt.title("Best fidelity vs attempt number")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_attempts_per_solved(tasks, out):
    agg = defaultdict(list)
    for r in tasks:
        if r["solved"] in _TRUE:
            agg[r["condition"]].append(_f(r["attempts_used"]))
    if not agg:
        return
    conds = sorted(agg)
    means = [sum(agg[c]) / len(agg[c]) for c in conds]
    plt.figure(figsize=(7, 4.5))
    plt.bar(conds, means, color="tab:purple")
    plt.ylabel("Mean attempts per solved task")
    plt.title("Attempts per successful synthesis")
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_circuit_quality(tasks, out):
    metrics = ["depth", "gate_count", "t_count"]
    agg = {m: defaultdict(list) for m in metrics}
    for r in tasks:
        if r["solved"] not in _TRUE:
            continue
        for m in metrics:
            if r.get(m) not in (None, ""):
                agg[m][r["condition"]].append(_f(r[m]))
    conds = sorted({c for m in metrics for c in agg[m]})
    if not conds:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, m in zip(axes, metrics):
        means = [(sum(agg[m][c]) / len(agg[m][c])) if agg[m][c] else 0.0
                 for c in conds]
        ax.bar(conds, means, color="tab:cyan")
        ax.set_title(f"Mean {m} (solved)")
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Circuit quality by condition")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def make_all_plots(run_dir: str) -> str:
    pdir = os.path.join(run_dir, "plots")
    os.makedirs(pdir, exist_ok=True)
    tasks = _read(os.path.join(run_dir, "tasks.csv"))
    attempts = _read(os.path.join(run_dir, "attempts.csv"))
    plot_success_vs_tasks(tasks, os.path.join(pdir, "success_vs_tasks.png"))
    plot_success_vs_library(tasks, os.path.join(pdir, "success_vs_library.png"))
    plot_fidelity_vs_attempt(attempts, os.path.join(pdir, "fidelity_vs_attempt.png"))
    plot_attempts_per_solved(tasks, os.path.join(pdir, "attempts_per_solved.png"))
    plot_circuit_quality(tasks, os.path.join(pdir, "circuit_quality.png"))
    return pdir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m quantum_icl.plots <run_dir>")
        sys.exit(1)
    print("Plots written to", make_all_plots(sys.argv[1]))
