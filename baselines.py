import numpy as np
from typing import List, Callable, Set, Optional

from task_replicator import Assignment


class RandomBaseline:
    """
    Randomly assign tasks to workers (one-to-one),
    filtering out pairs with non-positive expected net reward.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def select(self, candidates: List[Assignment], eval_net: Callable[[Assignment], float]) -> List[Assignment]:
        if not candidates:
            return []
        # 随机选择不需要对 candidates 做任何过滤，不使用 eval_net
        valid = candidates

        # Shuffle and greedily pick unique task/worker pairs
        idx = np.arange(len(valid))
        self.rng.shuffle(idx)

        used_tasks: Set[int] = set()
        used_workers: Set[int] = set()
        selected: List[Assignment] = []
        for k in idx:
            a = valid[k]
            if a.task_id in used_tasks or a.worker_id in used_workers:
                continue
            selected.append(a)
            used_tasks.add(a.task_id)
            used_workers.add(a.worker_id)
        return selected


class GreedyBaseline:
    """
    Greedy by true expected net reward: sort all candidate pairs by
    evaluate_reward_complex(context) - cost and pick non-conflicting pairs.
    """

    def select(self, candidates: List[Assignment], eval_net: Callable[[Assignment], float]) -> List[Assignment]:
        if not candidates:
            return []
        # Sort by net reward desc
        ranked = sorted(candidates, key=lambda a: eval_net(a), reverse=True)
        used_tasks: Set[int] = set()
        used_workers: Set[int] = set()
        selected: List[Assignment] = []
        for a in ranked:
            if eval_net(a) <= 0.0:
                break
            if a.task_id in used_tasks or a.worker_id in used_workers:
                continue
            selected.append(a)
            used_tasks.add(a.task_id)
            used_workers.add(a.worker_id)
        return selected

