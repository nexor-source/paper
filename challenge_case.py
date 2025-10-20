import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from matching_utils import run_hungarian_matching
from task_replicator import Assignment


OUTPUT_DIR = Path("output") / "challenge"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ScenarioStep:
    profits: np.ndarray  # shape=(n_tasks, n_workers)
    label: str


# 手工构造的动态挑战场景：每一步的收益矩阵都不同，
# 设计成 “局部最优≠全局最优” 的结构，匈牙利算法能明显胜出。
SCENARIO: List[ScenarioStep] = [
    ScenarioStep(
        profits=np.array([
            [0.95, 0.94, 0.25],
            [0.93, 0.50, 0.20],
            [0.25, 0.60, 0.92],
        ]),
        label="Step 0: data-heavy + compute-heavy + mobility mix",
    ),
    ScenarioStep(
        profits=np.array([
            [0.90, 0.40, 0.95],
            [0.85, 0.92, 0.30],
            [0.70, 0.91, 0.35],
        ]),
        label="Step 1: task preferences rotate (A/B swap)",
    ),
    ScenarioStep(
        profits=np.array([
            [0.88, 0.96, 0.30],
            [0.86, 0.35, 0.95],
            [0.60, 0.92, 0.40],
        ]),
        label="Step 2: compute-heavy reappears",
    ),
    ScenarioStep(
        profits=np.array([
            [0.92, 0.33, 0.94],
            [0.81, 0.93, 0.32],
            [0.68, 0.90, 0.34],
        ]),
        label="Step 3: bandwidth-critical burst",
    ),
    ScenarioStep(
        profits=np.array([
            [0.91, 0.95, 0.36],
            [0.87, 0.38, 0.93],
            [0.63, 0.89, 0.37],
        ]),
        label="Step 4: mobility-critical burst",
    ),
]


def run_original_step(step: ScenarioStep) -> Tuple[float, List[Tuple[int, int, float]]]:
    n_tasks, n_workers = step.profits.shape
    task_ids = list(range(n_tasks))
    worker_ids = list(range(n_workers))

    pair_lookup = {}
    for i in task_ids:
        for j in worker_ids:
            # 用零向量充当上下文即可，只需要 task_id / worker_id。
            pair_lookup[(i, j)] = Assignment(worker_id=j, task_id=i, context=np.zeros(7, dtype=float))

    selected, _, _ = run_hungarian_matching(
        task_ids,
        worker_ids,
        step.profits,
        pair_lookup,
        allow_unmatch=False,
    )

    picks = []
    total = 0.0
    for a in selected:
        net = float(step.profits[a.task_id, a.worker_id])
        if net <= 0.0:
            continue
        picks.append((a.task_id, a.worker_id, net))
        total += net
    return total, picks


def run_greedy_step(step: ScenarioStep) -> Tuple[float, List[Tuple[int, int, float]]]:
    n_tasks, n_workers = step.profits.shape
    combos: List[Tuple[float, int, int]] = []
    for i in range(n_tasks):
        for j in range(n_workers):
            net = float(step.profits[i, j])
            if net <= 0.0:
                continue
            combos.append((net, i, j))
    combos.sort(reverse=True)

    used_tasks = set()
    used_workers = set()
    picks = []
    total = 0.0
    for net, i, j in combos:
        if i in used_tasks or j in used_workers:
            continue
        used_tasks.add(i)
        used_workers.add(j)
        picks.append((i, j, net))
        total += net
    return total, picks


def _plot_step(step_idx: int, step: ScenarioStep,
               picks_o: List[Tuple[int, int, float]],
               picks_g: List[Tuple[int, int, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    vmax = 1.0

    for ax, label, picks in zip(
        axes,
        ["Original (Hungarian)", "Greedy"],
        [picks_o, picks_g],
    ):
        im = ax.imshow(step.profits, cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Worker")
        ax.set_ylabel("Task")
        ax.set_xticks(range(step.profits.shape[1]))
        ax.set_yticks(range(step.profits.shape[0]))
        for t, w, val in picks:
            ax.scatter(w, t, marker="o", s=120, facecolors="none", edgecolors="white", linewidths=2.0)
            ax.text(w, t, f"{val:.2f}", color="white", ha="center", va="center", fontsize=9, weight="bold")

    fig.colorbar(im, ax=axes, fraction=0.045, pad=0.04, label="Expected net reward")
    fig.suptitle(step.label, fontsize=12, fontweight="bold")
    save_path = OUTPUT_DIR / f"step_{step_idx:02d}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_totals(step_totals_o: List[float],
                 step_totals_g: List[float],
                 cum_o: List[float],
                 cum_g: List[float]) -> None:
    steps = np.arange(len(step_totals_o))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

    axes[0].plot(steps, step_totals_o, marker="o", label="Original", color="C0")
    axes[0].plot(steps, step_totals_g, marker="s", label="Greedy", color="C2")
    axes[0].set_title("Step Reward")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Expected net reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, cum_o, marker="o", label="Original", color="C0")
    axes[1].plot(steps, cum_g, marker="s", label="Greedy", color="C2")
    axes[1].set_title("Cumulative Reward")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Cumulative expected reward")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Original vs Greedy on Challenge Scenario", fontsize=13, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "challenge_totals.png", dpi=150)
    plt.close(fig)


def run_challenge():
    print("=== Challenge Scenario: Global vs Greedy Matching ===")
    cum_original = 0.0
    cum_greedy = 0.0
    step_totals_o: List[float] = []
    step_totals_g: List[float] = []
    cum_series_o: List[float] = []
    cum_series_g: List[float] = []

    for idx, step in enumerate(SCENARIO):
        total_o, picks_o = run_original_step(step)
        total_g, picks_g = run_greedy_step(step)
        cum_original += total_o
        cum_greedy += total_g

        step_totals_o.append(total_o)
        step_totals_g.append(total_g)
        cum_series_o.append(cum_original)
        cum_series_g.append(cum_greedy)

        print(f"\n{step.label}")
        print("  Original (Hungarian):")
        for t, w, v in picks_o:
            print(f"    task {t} <- worker {w}, net={v:.3f}")
        print(f"    step total = {total_o:.3f} | cumulative = {cum_original:.3f}")

        print("  Greedy:")
        for t, w, v in picks_g:
            print(f"    task {t} <- worker {w}, net={v:.3f}")
        print(f"    step total = {total_g:.3f} | cumulative = {cum_greedy:.3f}")

        _plot_step(idx, step, picks_o, picks_g)

    _plot_totals(step_totals_o, step_totals_g, cum_series_o, cum_series_g)

    gap = cum_original - cum_greedy
    print("\n=== Summary ===")
    print(f"Original cumulative expected reward : {cum_original:.3f}")
    print(f"Greedy cumulative expected reward   : {cum_greedy:.3f}")
    print(f"Difference (Original - Greedy)      : {gap:.3f}")
    print(f"Saved visualizations to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_challenge()
