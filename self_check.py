import numpy as np
from typing import List, Dict

from config import (
    RANDOM_SEED,
    PARTITION_SPLIT_THRESHOLD,
    REPLICATION_COST,
    MAX_PARTITION_DEPTH,
)
from normalizer import ContextNormalizer
from task_replicator import TaskReplicator, Assignment
from scheduler import Scheduler, Task, Worker, spawn_new_worker
from baselines import RandomBaseline, GreedyBaseline


def build_env(num_workers: int = 8):
    """构造一个固定环境（无工人动态），返回 (workers, normalizer, replicator, scheduler, rng)。"""
    rng = np.random.default_rng(RANDOM_SEED)
    workers: List[Worker] = [spawn_new_worker(i, rng) for i in range(num_workers)]
    normalizer = ContextNormalizer()
    replicator = TaskReplicator(
        context_dim=7,
        partition_split_threshold=PARTITION_SPLIT_THRESHOLD,
        budget=1,
        replication_cost=REPLICATION_COST,
        max_partition_depth=MAX_PARTITION_DEPTH,
    )
    scheduler = Scheduler(workers, normalizer, replicator, enable_worker_dynamics=False)
    return workers, normalizer, replicator, scheduler, rng


def sample_tasks(rng: np.random.Generator, n: int, start_id: int = 0) -> List[Task]:
    """构造 n 个任务，带唯一 task_id。"""
    tasks: List[Task] = []
    for k in range(n):
        task_type = int(rng.integers(0, 10))
        data_size = float(rng.uniform(100, 3000))
        deadline = float(rng.uniform(1, 3))
        tasks.append(Task(start_id + k, task_type, data_size, deadline))
    return tasks


def pretrain_replicator(
    scheduler: Scheduler,
    replicator: TaskReplicator,
    rng: np.random.Generator,
    rounds: int = 300,
    tasks_per_round: int = 10,
) -> None:
    """快速“预热”分区估计，避免所有 posterior_mean 初始都相同。

    做法：随机生成任务，随机挑选若干 worker-task 对，按真实概率 p(context)
    做一次伯努利采样并更新 replicator。这样会形成一定的非均匀后验。"""
    for r in range(rounds):
        tasks = sample_tasks(rng, tasks_per_round, start_id=r * tasks_per_round)
        candidates = scheduler.generate_candidate_assignments(tasks)
        if not candidates:
            continue
        sz = min(12, len(candidates))
        idx = rng.choice(len(candidates), size=sz, replace=False)
        selected = [candidates[i] for i in idx]
        rewards: Dict[Assignment, float] = {}
        for a in selected:
            p = scheduler.evaluate_reward_complex(a.context)
            rwd = float(rng.binomial(1, p))
            rewards[a] = rwd
        replicator.update_assignments_reward(selected, rewards)


def deterministic_realized_reward(
    scheduler: Scheduler,
    a: Assignment,
    base_seed: int = 20240923,
) -> float:
    """给定 assignment，基于 (task_id, worker_id, base_seed) 生成可重复的伯努利样本。

    说明：避免 NumPy 标量与位运算的兼容性问题，全部使用 Python int 进行混合。
    """
    t = int(getattr(a, "task_id", 0))
    w = int(getattr(a, "worker_id", 0))
    mask = (1 << 63) - 1
    # 简单可复现的混合哈希（无需安全性）
    s = (int(base_seed) ^ (t * 1315423911) ^ (w * 2654435761)) & mask
    rng = np.random.default_rng(int(s))
    p = scheduler.evaluate_reward_complex(a.context)
    return float(rng.binomial(1, p))


def net_expected_true(scheduler: Scheduler, ass: List[Assignment]) -> float:
    rc = float(scheduler.replicator.replication_cost)
    return float(sum(scheduler.evaluate_reward_complex(a.context) - rc for a in ass))


def net_realized_deterministic(scheduler: Scheduler, ass: List[Assignment]) -> float:
    rc = float(scheduler.replicator.replication_cost)
    return float(sum(deterministic_realized_reward(scheduler, a) - rc for a in ass))


def run_one_check(num_workers: int = 8, num_tasks: int = 12) -> None:
    workers, normalizer, replicator, scheduler, rng = build_env(num_workers)

    # 可选：预热学习，让 Original/Greedy 的后验有区分度
    pretrain_replicator(scheduler, replicator, rng, rounds=300, tasks_per_round=10)

    # 构造一批“相同的候选 assignment”，四种策略都在此集合上决策
    tasks = sample_tasks(rng, num_tasks, start_id=10_000)
    candidates = scheduler.generate_candidate_assignments(tasks)
    if not candidates:
        print("[self-check] 候选为空，检查输入规模")
        return

    rc = float(replicator.replication_cost)

    # 策略：Original（posterior + 匈牙利）
    sel_original = replicator.select_assignments(candidates, allow_unmatch=True)

    # 策略：Greedy（posterior 排序逐个挑）
    sel_greedy = GreedyBaseline(replicator).select(candidates, lambda a: 0.0)

    # 策略：Random（不使用 eval_net，仅随机一对一）
    sel_random = RandomBaseline(np.random.default_rng(RANDOM_SEED)).select(
        candidates,
        lambda a: 0.0,
    )

    # 策略：Oracle（真实 p(context) + 匈牙利）
    sel_oracle = scheduler._oracle_select_assignments(candidates)

    # 计算期望净收益（真实 p）与确定性实现净收益
    results = []
    for name, sel in [
        ("Original", sel_original),
        ("Random", sel_random),
        ("Greedy", sel_greedy),
        ("Oracle", sel_oracle),
    ]:
        exp_true = net_expected_true(scheduler, sel)
        real_fix = net_realized_deterministic(scheduler, sel)
        results.append((name, len(sel), exp_true, real_fix))

    print("\n=== Self Check on Same Candidates ===")
    print(f"workers={len(workers)}, tasks={len(tasks)}, candidates={len(candidates)}, rc={rc}")
    print("name      count   expected(True p)   realized(deterministic)")
    for name, cnt, exp_t, real_f in results:
        print(f"{name:<9} {cnt:5d}   {exp_t:16.4f}   {real_f:20.4f}")


if __name__ == "__main__":
    # 可根据需要调整工人数/任务数
    run_one_check(num_workers=10, num_tasks=14)
