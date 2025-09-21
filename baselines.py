import numpy as np
from typing import List, Callable, Set, Optional

from task_replicator import Assignment, TaskReplicator


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
        # Random selection keeps all candidates without using eval_net
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
    """Greedy selector backed by a TaskReplicator without Hungarian matching."""

    def __init__(self, replicator: TaskReplicator):
        self.replicator = replicator

    def _estimated_net(self, assignment: Assignment) -> float:
        partition = self.replicator.root_partition.find_partition(assignment.context)
        return float(partition.posterior_mean() - self.replicator.replication_cost)

    def select(self, candidates: List[Assignment], _eval_net: Callable[[Assignment], float]) -> List[Assignment]:
        if not candidates:
            return []
        ranked = sorted(candidates, key=self._estimated_net, reverse=True)
        used_tasks: Set[int] = set()
        used_workers: Set[int] = set()
        selected: List[Assignment] = []
        for assignment in ranked:
            net = self._estimated_net(assignment)
            if net <= 0.0:
                break
            if assignment.task_id in used_tasks or assignment.worker_id in used_workers:
                continue
            selected.append(assignment)
            used_tasks.add(assignment.task_id)
            used_workers.add(assignment.worker_id)
        return selected

