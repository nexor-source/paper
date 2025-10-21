"""
Extended challenge scenario to demonstrate the advantage of the Hungarian
matching (Ours) over a simple greedy selector (Greedy) in large,
conflicting environments.

Main ideas:
    - Construct tasks and workers belonging to several “speciality” classes
      (e.g. Compute-heavy, Bandwidth-heavy, Mobility-heavy).
    - Each worker is strong in exactly one speciality and mediocre/weak in
      others.
    - Each step generates tasks whose preferred specialities follow a pattern
      that changes over time (staggered and rotating demand).
    - Greedy tends to consume the same strong workers for the first few tasks
      and later leaves some tasks matched with mediocre workers.
    - Hungarian (Ours) considers all pairings jointly and avoids clashes.

The script runs multiple independent steps, prints detailed assignments, and
saves heatmaps & reward plots under output/challenge/.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from matching_utils import run_hungarian_matching
from task_replicator import Assignment


OUTPUT_DIR = Path("output") / "challenge"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SPECIALITY_NAMES = ["Compute", "Bandwidth", "Mobility"]


@dataclass
class Worker:
    wid: int
    speciality: int  # 0: compute, 1: bandwidth, 2: mobility
    base_skill: float


@dataclass
class Task:
    tid: int
    speciality: int
    difficulty: float


def build_workers(num_groups: int = 3, workers_per_group: int = 7) -> List[Worker]:
    """Construct workers strongly specialised in one speciality."""
    workers: List[Worker] = []
    wid = 0
    rng = np.random.default_rng(1234)
    for spec in range(num_groups):
        for _ in range(workers_per_group):
            base_skill = rng.uniform(0.75, 0.95)
            workers.append(Worker(wid=wid, speciality=spec, base_skill=base_skill))
            wid += 1
    return workers


def build_tasks(step_idx: int,
                num_tasks: int = 30,
                difficulty_span: Tuple[float, float] = (0.6, 1.2)) -> List[Task]:
    """
    Generate tasks for a given step with rotating demand.

    We rotate which speciality is most demanded across steps, e.g.
    step 0: compute-heavy, step 1: bandwidth-heavy, step 2: mobility-heavy, etc.
    """
    rng = np.random.default_rng(1000 + step_idx)
    tasks: List[Task] = []

    # Determine speciality distribution: rotate peaks every step.
    base_counts = np.array([1.0, 1.0, 1.0])
    peak_spec = step_idx % len(SPECIALITY_NAMES)
    base_counts[peak_spec] += 6.0   # emphasise one speciality
    other_spec = (step_idx + 1) % len(SPECIALITY_NAMES)
    base_counts[other_spec] += 2.0  # secondary emphasis
    probs = base_counts / base_counts.sum()

    speciality_choices = rng.choice(len(SPECIALITY_NAMES), size=num_tasks, p=probs)
    diffs = rng.uniform(difficulty_span[0], difficulty_span[1], size=num_tasks)

    for tid, (spec, diff) in enumerate(zip(speciality_choices, diffs)):
        tasks.append(Task(tid=tid, speciality=int(spec), difficulty=float(diff)))
    return tasks


def compute_profit(worker: Worker, task: Task,
                   same_spec_bonus: float = 0.35,
                   cross_penalty: float = 0.4,
                   noise: float = 0.05) -> float:
    """
    Return the expected net reward contribution for a worker-task pair.
    Larger when speciality matches; penalised otherwise.
    """
    rng = np.random.default_rng(worker.wid * 8191 + task.tid * 4093)
    base = worker.base_skill * (1.1 - 0.05 * abs(task.speciality - worker.speciality))

    if worker.speciality == task.speciality:
        val = base + same_spec_bonus * task.difficulty
    else:
        val = base - cross_penalty * task.difficulty

    val += rng.normal(0.0, noise)
    val = max(0.01, min(0.99, val))  # clamp
    return val


def build_profit_matrix(workers: List[Worker],
                        tasks: List[Task]) -> np.ndarray:
    matrix = np.zeros((len(tasks), len(workers)), dtype=float)
    for i, task in enumerate(tasks):
        for j, worker in enumerate(workers):
            matrix[i, j] = compute_profit(worker, task)
    return matrix


def hungarian_select(profits: np.ndarray) -> Tuple[float, List[Tuple[int, int, float]]]:
    m, n = profits.shape
    task_ids = list(range(m))
    worker_ids = list(range(n))
    lookup: Dict[Tuple[int, int], Assignment] = {}
    for i in task_ids:
        for j in worker_ids:
            lookup[(i, j)] = Assignment(worker_id=j, task_id=i, context=np.zeros(1, dtype=float))

    selected, _, _ = run_hungarian_matching(
        task_ids,
        worker_ids,
        profits,
        lookup,
        allow_unmatch=False,
    )

    total = 0.0
    picks: List[Tuple[int, int, float]] = []
    for a in selected:
        val = float(profits[a.task_id, a.worker_id])
        picks.append((a.task_id, a.worker_id, val))
        total += val
    return total, picks


def greedy_select(profits: np.ndarray) -> Tuple[float, List[Tuple[int, int, float]]]:
    m, n = profits.shape
    combos: List[Tuple[float, int, int]] = []
    for i in range(m):
        for j in range(n):
            combos.append((float(profits[i, j]), i, j))
    combos.sort(reverse=True)

    used_tasks = set()
    used_workers = set()
    picks: List[Tuple[int, int, float]] = []
    total = 0.0

    for val, i, j in combos:
        if i in used_tasks or j in used_workers:
            continue
        picks.append((i, j, val))
        total += val
        used_tasks.add(i)
        used_workers.add(j)
        if len(used_tasks) == m:
            break
    return total, picks


def plot_heatmap(step_idx: int,
                 profits: np.ndarray,
                 picks_o: List[Tuple[int, int, float]],
                 picks_g: List[Tuple[int, int, float]],
                 workers: List[Worker],
                 tasks: List[Task]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    vmax = 1.0
    worker_labels = [
        f"W{w.wid}\n{SPECIALITY_NAMES[w.speciality][0]}"
        for w in workers
    ]
    task_labels = [
        f"T{t.tid}\n{SPECIALITY_NAMES[t.speciality][0]}"
        for t in tasks
    ]

    for ax, picks, title in zip(
        axes,
        [picks_o, picks_g],
        ["Ours (Hungarian)", "Greedy"],
    ):
        im = ax.imshow(profits, cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(workers)))
        ax.set_xticklabels(worker_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(task_labels, fontsize=8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Worker")
        ax.set_ylabel("Task")

        for t, w, val in picks:
            ax.scatter(w, t, marker="o", s=90, facecolors="none", edgecolors="white", linewidths=1.8)
            ax.text(w, t, f"{val:.2f}", color="white", ha="center", va="center", fontsize=8, weight="bold")

    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="Expected net reward")
    fig.suptitle(f"Step {step_idx}: Ours vs Greedy selections", fontsize=13, fontweight="bold")
    fig.savefig(OUTPUT_DIR / f"step_{step_idx:03d}.png", dpi=150)
    plt.close(fig)


def plot_totals(step_totals: Dict[str, List[float]],
                cumulative: Dict[str, List[float]]) -> None:
    steps = np.arange(len(next(iter(step_totals.values()))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    colors = {
        "Ours": "C0",
        "Greedy": "C2",
    }

    # determine marker spacing to avoid “sugar-coated hawthorn” clutter
    def _markevery(series: List[float]) -> int:
        n = max(1, len(series))
        # show at most about 12 markers per line
        return max(1, n // 12)

    # Step rewards (use line + light markers; highlight gap)
    bar_width = 0.35
    offset = {"Ours": -bar_width / 2, "Greedy": bar_width / 2}
    for label, series in step_totals.items():
        axes[0].bar(
            steps + offset[label],
            series,
            width=bar_width,
            label=label,
            color=colors.get(label, None),
            alpha=0.65,
        )
    axes[0].set_title("Per-Step Expected Reward")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Reward")
    tick_positions = np.linspace(0, len(steps) - 1, num=6)
    tick_positions = np.unique(np.round(tick_positions).astype(int))
    axes[0].set_xticks(tick_positions)
    # zoom y-range around data
    step_min = min(min(series) for series in step_totals.values())
    step_max = max(max(series) for series in step_totals.values())
    margin = max(0.5, 0.05 * (step_max - step_min))
    axes[0].set_ylim(step_min - margin, step_max + margin)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    # Cumulative rewards
    for label, series in cumulative.items():
        axes[1].plot(
            steps,
            series,
            label=label,
            color=colors.get(label, None),
            linewidth=1.8,
            marker="o",
            markersize=4,
            markevery=_markevery(series),
        )
    axes[1].set_title("Cumulative Expected Reward")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Cumulative reward")
    cum_min = min(min(series) for series in cumulative.values())
    cum_max = max(max(series) for series in cumulative.values())
    cum_margin = max(5.0, 0.03 * (cum_max - cum_min))
    axes[1].set_ylim(cum_min - cum_margin, cum_max + cum_margin)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Challenge Scenario – Ours vs Greedy", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "challenge_totals.png", dpi=150)
    plt.close(fig)


def run_challenge(
    steps: int = 60,
    num_tasks: int = 30,
    workers_per_group: int = 8,
) -> None:
    workers = build_workers(workers_per_group=workers_per_group)
    print(f"[info] workers: {len(workers)} (per speciality {workers_per_group})")

    step_totals = {"Ours": [], "Greedy": []}
    cumulative = {"Ours": [], "Greedy": []}
    cum_vals = {"Ours": 0.0, "Greedy": 0.0}

    for step_idx in range(steps):
        tasks = build_tasks(step_idx, num_tasks=num_tasks)
        profits = build_profit_matrix(workers, tasks)

        total_o, picks_o = hungarian_select(profits)
        total_g, picks_g = greedy_select(profits)

        step_totals["Ours"].append(total_o)
        step_totals["Greedy"].append(total_g)

        cum_vals["Ours"] += total_o
        cum_vals["Greedy"] += total_g

        cumulative["Ours"].append(cum_vals["Ours"])
        cumulative["Greedy"].append(cum_vals["Greedy"])

        print(f"Step {step_idx:03d} | Ours={total_o:.3f}, Greedy={total_g:.3f}")

        if step_idx < 10 or step_idx % max(1, steps // 10) == 0:
            plot_heatmap(step_idx, profits, picks_o, picks_g, workers, tasks)

    plot_totals(step_totals, cumulative)

    print("\n=== Challenge Summary ===")
    for label in ["Ours", "Greedy"]:
        print(f"{label:>8s} cumulative reward: {cum_vals[label]:.3f}")

    diff = cum_vals["Ours"] - cum_vals["Greedy"]
    print(f"Difference (Ours - Greedy): {diff:.3f}")
    print(f"Figures saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_challenge()
