import numpy as np
from typing import Dict, List, Sequence, Tuple, TYPE_CHECKING
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from task_replicator import Assignment

DEFAULT_INVALID_COST = 1e12


def run_hungarian_matching(
    task_ids: Sequence[int],
    worker_ids: Sequence[int],
    profits: np.ndarray,
    pair_lookup: Dict[Tuple[int, int], "Assignment"],
    *,
    allow_unmatch: bool = True,
    eps: float = 1e-12,
    invalid_cost: float = DEFAULT_INVALID_COST,
):
    """匈牙利算法求解获取 worker-task pair

    Args:
        task_ids: Row order task identifiers corresponding to ``profits``.
        worker_ids: Column order worker identifiers corresponding to ``profits``.
        profits: Net reward matrix shaped (len(task_ids), len(worker_ids)).
        pair_lookup: Map back to original ``Assignment`` objects.
        allow_unmatch: Whether dummy rows/cols can absorb unmatched items with zero cost.
        eps: Minimum net reward required to keep a match in the output.
        invalid_cost: Cost assigned to invalid pairs when building the square matrix.

    Returns:
        A tuple ``(selected, row_ind, col_ind)`` where ``selected`` is the list of
        chosen assignments (net > eps) and the index arrays mirror SciPy's
        ``linear_sum_assignment`` output for debugging/analysis.
    """
    # 匈牙利算法获取 assignments
    m = len(task_ids)
    n = len(worker_ids)

    if m == 0 or n == 0:
        return [], np.array([], dtype=int), np.array([], dtype=int)

    tasks = list(task_ids)
    workers = list(worker_ids)

    size = m + n
    cost_square = np.zeros((size, size), dtype=float)

    cost_square[:m, :n] = invalid_cost
    valid_mask = np.isfinite(profits)
    cost_square[:m, :n][valid_mask] = -profits[valid_mask]

    if not allow_unmatch:
        if m > 0:
            cost_square[:m, n:n + m] = invalid_cost
        if n > 0:
            cost_square[m:m + n, :n] = invalid_cost

    row_ind, col_ind = linear_sum_assignment(cost_square)

    # 过滤掉负受益的匹配（按道理来说，这是保险起见）
    selected: List["Assignment"] = []
    for i, j in zip(row_ind, col_ind):
        if i < m and j < n:
            net = profits[i, j]
            if net > eps:
                assignment = pair_lookup.get((tasks[i], workers[j]))
                if assignment is not None:
                    selected.append(assignment)

    return selected, row_ind, col_ind
