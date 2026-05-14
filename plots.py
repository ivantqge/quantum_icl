"""Plotting for circuit-synthesis experiment results.

Pure consumer of the CSV/JSON files written by results_logger.RunLogger, so it
can be re-run offline on any saved run directory:

    python plots.py results/<run_dir>
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless: NERSC login/compute nodes have no display
import matplotlib.pyplot as plt


def _read_csv(path: str) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_success_vs_round(rounds_csv: str, outpath: str):
    """Per-round success rate, one line per mode."""
    rows = _read_csv(rounds_csv)
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append((int(r["round_num"]), float(r["success_rate"])))

    plt.figure(figsize=(7, 4.5))
    for mode, pts in sorted(by_mode.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=mode)
    plt.xlabel("Round")
    plt.ylabel("Success rate")
    plt.title("Success rate vs round")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()


def plot_library_growth(rounds_csv: str, outpath: str):
    """Library size over rounds for the growing mode."""
    rows = [r for r in _read_csv(rounds_csv) if r["mode"] == "growing"]
    if not rows:
        return
    rows.sort(key=lambda r: int(r["round_num"]))
    xs = [int(r["round_num"]) for r in rows]
    ys = [int(r["library_size"]) for r in rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, marker="s", color="tab:green")
    plt.xlabel("Round")
    plt.ylabel("Library size")
    plt.title("Library growth (growing mode)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()


def plot_success_by_qubit_count(tasks_csv: str, outpath: str):
    """Grouped bar chart: success rate per qubit count, per mode."""
    rows = _read_csv(tasks_csv)
    # (mode, num_qubits) -> [solved_count, total]
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r["mode"], int(r["num_qubits"]))
        agg[key][0] += 1 if r["solved"] in ("True", "true", "1") else 0
        agg[key][1] += 1

    modes = sorted({k[0] for k in agg})
    qubit_counts = sorted({k[1] for k in agg})
    if not modes or not qubit_counts:
        return

    width = 0.8 / max(len(modes), 1)
    plt.figure(figsize=(7, 4.5))
    for i, mode in enumerate(modes):
        rates = [
            (agg[(mode, q)][0] / agg[(mode, q)][1]) if agg[(mode, q)][1] else 0.0
            for q in qubit_counts
        ]
        xs = [q + i * width for q in range(len(qubit_counts))]
        plt.bar(xs, rates, width=width, label=mode)
    plt.xticks(
        [q + width * (len(modes) - 1) / 2 for q in range(len(qubit_counts))],
        qubit_counts,
    )
    plt.xlabel("Number of qubits")
    plt.ylabel("Success rate")
    plt.title("Success rate by qubit count")
    plt.ylim(0, 1.02)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()


def plot_cost_and_quality(tasks_csv: str, outpath: str):
    """Mean circuit quality and cumulative token cost per mode."""
    rows = _read_csv(tasks_csv)
    modes = sorted({r["mode"] for r in rows})
    if not modes:
        return

    def _mean(mode, field, solved_only=False):
        vals = [
            float(r[field]) for r in rows
            if r["mode"] == mode
            and (not solved_only or r["solved"] in ("True", "true", "1"))
        ]
        return sum(vals) / len(vals) if vals else 0.0

    def _sum(mode, field):
        return sum(float(r[field]) for r in rows if r["mode"] == mode)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = [
        ("Mean gate count (solved)", lambda m: _mean(m, "gate_count", True)),
        ("Mean depth (solved)", lambda m: _mean(m, "depth", True)),
        ("Mean attempts/task", lambda m: _mean(m, "attempts_used")),
        ("Total est. cost (USD)", lambda m: _sum(m, "est_cost_usd")),
    ]
    for ax, (title, fn) in zip(axes, metrics):
        ax.bar(modes, [fn(m) for m in modes], color="tab:blue")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Cost and circuit quality by mode")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def make_all_plots(run_dir: str):
    """Generate all plots into <run_dir>/plots/."""
    rounds_csv = os.path.join(run_dir, "rounds.csv")
    tasks_csv = os.path.join(run_dir, "tasks.csv")
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if os.path.exists(rounds_csv):
        plot_success_vs_round(rounds_csv, os.path.join(plot_dir, "success_vs_round.png"))
        plot_library_growth(rounds_csv, os.path.join(plot_dir, "library_growth.png"))
    if os.path.exists(tasks_csv):
        plot_success_by_qubit_count(
            tasks_csv, os.path.join(plot_dir, "success_by_qubit_count.png")
        )
        plot_cost_and_quality(
            tasks_csv, os.path.join(plot_dir, "cost_and_quality.png")
        )
    return plot_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plots.py <run_dir>")
        sys.exit(1)
    out = make_all_plots(sys.argv[1])
    print(f"Plots written to {out}")
