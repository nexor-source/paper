import math
import numpy as np
from typing import List, Tuple, Optional
from matching_utils import run_hungarian_matching
from config import *

class ContextSpacePartition:
    """
    上下文空间的一个划分单元（超立方体），维护该区域内样本计数和奖励估计，
    支持 [动态二分细分] 以 [细化学习] 。
    """
    def __init__(self, bounds: List[Tuple[float, float]], depth: int = 0):
        """初始化上下文划分区域

        Args:
            bounds (List[Tuple[float, float]]): 每个维度的区间，如[(min, max), ...]，shape=(d, 2)。
            depth (int, optional): 当前划分层级，根节点为0。默认为0。
        """
        self.bounds = bounds    # 每个维度的区间列表
        self.depth = depth      # 当前划分层级
        self.sample_count = 0   # 该区域被采样的次数
        self.estimated_quality = 0.0  # 该区域内任务副本的平均完成质量估计
        self.children = None    # 子划分列表，未细分时为None

        self.default_mean = 0.5               # no-data fallback
        self.sum_reward = 0.0                 # cumulative reward stored in this partition
        self.data_points = []                 # cached (context, reward) samples for adaptive splits
    
    def posterior_mean(self) -> float:
        """融合先验与真实样本的估计质量（后验均值）

        Returns:
            float: 当前分区的估计质量（先验均值与观测数据的加权平均）。
        """
        if self.sample_count <= 0:
            return self.default_mean  # 没有任何信息时，退回默认均值
        return self.sum_reward / float(self.sample_count)

    def contains(self, context: np.ndarray) -> bool:
        """判断上下文是否在该划分区域内

        Args:
            context (np.ndarray, shape=(d,)): 归一化上下文向量。

        Returns:
            bool: 上下文是否在当前划分区域内。
        """
        EPS = 1e-8  # 容忍边界浮点误差
        return all(self.bounds[d][0] <= context[d] < self.bounds[d][1] or
                abs(context[d] - self.bounds[d][1]) < EPS  # 允许等于上界
                for d in range(len(context)))

    def update_reward(self, reward: float, debug: bool = False, context: Optional[np.ndarray] = None):
        """根据观察到的副本完成奖励更新该区域的样本计数和平均质量估计

        Args:
            reward (float): 当前副本任务完成奖励，通常取0或1。
            debug (bool): 用于代码内部调用，如果为true则不会增加样本数量，只是更新估计质量
        """
        if not debug:
            self.sample_count += 1
            if context is not None:
                try:
                    ctx_tuple = tuple(float(x) for x in context)
                except Exception:
                    ctx_tuple = tuple(context)
                self.data_points.append((ctx_tuple, float(reward)))
        self.sum_reward += reward
        # 仅作为一个“纯样本平均”的快速回显；真正用于决策请调用 posterior_mean()
        # self.estimated_quality = self.sum_reward / self.sample_count
        self.estimated_quality = self.posterior_mean()
    
    def _binary_entropy(self, success: float, total: float) -> float:
        if total <= 0.0:
            return 0.0
        p = success / total
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))

    def _select_split_dimension(self) -> Tuple[Optional[int], Optional[float]]:
        """选择信息增益最大的维度进行二分。"""
        if not self.data_points:
            return None, None

        total_samples = float(len(self.data_points))
        parent_entropy = self._binary_entropy(self.sum_reward, total_samples)
        best_gain = float("-inf")
        best_dim = None
        best_mid = None

        for dim, (lo, hi) in enumerate(self.bounds):
            mid = (lo + hi) / 2.0
            left_total = 0.0
            right_total = 0.0
            left_success = 0.0
            right_success = 0.0

            for ctx, reward in self.data_points:
                value = ctx[dim]
                if value < mid:
                    left_total += 1.0
                    left_success += reward
                else:
                    right_total += 1.0
                    right_success += reward

            if left_total == 0.0 or right_total == 0.0:
                continue

            total = left_total + right_total
            if total <= 0.0:
                continue

            left_entropy = self._binary_entropy(left_success, left_total)
            right_entropy = self._binary_entropy(right_success, right_total)
            weighted_entropy = (left_total / total) * left_entropy + (right_total / total) * right_entropy
            gain = parent_entropy - weighted_entropy
            if gain > best_gain:
                best_gain = gain
                best_dim = dim
                best_mid = mid

        if best_dim is None:
            widths = [b[1] - b[0] for b in self.bounds]
            if not widths:
                return None, None
            best_dim = int(np.argmax(widths))
            lo, hi = self.bounds[best_dim]
            best_mid = (lo + hi) / 2.0

        return best_dim, best_mid

    def subdivide(self):
        """基于信息增益选择维度，将该划分区域二分生成两个子区域

        Notes:
            - 若该区域已被细分，则不再重复细分。
            - 每次细分会更新 `children` 列表。
        """
        if self.children is not None:
            # 如果这个partition已细分则不再细分
            print("Partition already subdivided.")
            return

        split_dim, split_value = self._select_split_dimension()
        if split_dim is None or split_value is None:
            # 无法找到合适的划分维度，放弃细分
            return

        left_bounds = list(self.bounds)
        right_bounds = list(self.bounds)
        left_bounds[split_dim] = (self.bounds[split_dim][0], split_value)
        right_bounds[split_dim] = (split_value, self.bounds[split_dim][1])

        left_child = ContextSpacePartition(left_bounds, self.depth + 1)
        right_child = ContextSpacePartition(right_bounds, self.depth + 1)
        self.children = [left_child, right_child]

        left_child.data_points = []
        right_child.data_points = []
        left_child.sample_count = 0
        right_child.sample_count = 0
        left_child.sum_reward = 0.0
        right_child.sum_reward = 0.0

        for ctx, reward in self.data_points:
            target = left_child if ctx[split_dim] < split_value else right_child
            target.data_points.append((ctx, reward))
            target.sample_count += 1
            target.sum_reward += reward

        for child in self.children:
            child.sum_reward = float(child.sum_reward)
            child.estimated_quality = child.posterior_mean()

        # 分裂后父节点不再向子节点传递额外先验
        self.data_points = []
    
    def find_partition(self, context: np.ndarray):
        """递归寻找包含指定上下文的最底层划分区域（叶节点）

        Args:
            context (np.ndarray, shape=(d,)): 归一化上下文向量。

        Returns:
            ContextSpacePartition: 对应的叶节点对象。

        Raises:
            ValueError: 当上下文不在任何子划分内时抛出。
        """
        if self.children is None:
            return self
        for child in self.children:
            if child.contains(context):
                return child.find_partition(context)
        raise ValueError(f"Context {context} not contained in any child partition")

class Assignment:
    """
    表示一个工人-任务对，封装对应的上下文信息，
    是调度系统中任务分配的基本单位。
    """
    def __init__(self, worker_id: int, task_id: int, context: np.ndarray):
        """初始化 Assignment

        Args:
            worker_id (int): 工人 ID。
            task_id (int): 任务 ID。
            context (np.ndarray, shape=(d,)): 归一化后的上下文特征向量。
        """
        self.worker_id = worker_id
        self.task_id = task_id
        self.context = context

class TaskReplicator:
    """
    任务分配器，基于上下文划分估计副本质量，
    利用匈牙利算法实现任务-工人最大收益匹配，
    动态细分上下文划分空间保证长期学习精度。
    """
    def __init__(self, context_dim: int, partition_split_threshold: int, budget: int, replication_cost: float, max_partition_depth: Optional[int] = None):
        """初始化任务分配器

        Args:
            context_dim (int): 上下文向量维度 d。
            partition_split_threshold (int): 划分细分的阈值（样本数达到时细分）。
            budget (int): 每个任务允许的最大副本数。
            replication_cost (float): 每次分配的成本。
            max_partition_depth (Optional[int], optional): 最大细分层级，防止过度细分。默认为 None（不限制）。
        """
        self.context_dim = context_dim
        self.budget = budget
        self.replication_cost = replication_cost
        self.max_partition_depth = max_partition_depth
        
        # 初始化根划分，单位超立方体[0,1]^d
        self.root_partition = ContextSpacePartition(bounds=[(0,1)]*context_dim)
        self.partitions = [self.root_partition]
        self.partition_split_threshold = partition_split_threshold
        try:
            self.use_ucb = bool(globals().get('REPLICATOR_USE_UCB', False))
        except Exception:
            self.use_ucb = False
        try:
            self.ucb_coef = float(globals().get('REPLICATOR_UCB_COEF', 0.0))
        except Exception:
            self.ucb_coef = 0.0
        try:
            self.ucb_min_pulls = max(1.0, float(globals().get('REPLICATOR_UCB_MIN_PULLS', 1)))
        except Exception:
            self.ucb_min_pulls = 1.0
        self.use_ucb = bool(self.use_ucb and self.ucb_coef > 0.0)
        try:
            self.min_samples_before_split = int(globals().get('PARTITION_MIN_SAMPLES', self.partition_split_threshold))
        except Exception:
            self.min_samples_before_split = self.partition_split_threshold
        self.min_samples_before_split = max(1, self.min_samples_before_split)
        try:
            self.variance_split_threshold = float(globals().get('PARTITION_VARIANCE_THRESHOLD', 0.02))
        except Exception:
            self.variance_split_threshold = 0.02
        self.variance_split_threshold = max(0.0, self.variance_split_threshold)
        self.total_updates = 0
        self.split_events = 0  # number of times partitions were subdivided

        # 调试用的计数器
        self._run_counter = 0
    
    def _debug_print_assignment_table(self, task_ids, worker_ids, net_matrix, selected_set,
                                     max_rows=30, max_cols=30, digits=3):
        """打印任务×工人净收益表：选中=[v]，未选中=(v)，无候选=--

        说明：
        - net_matrix 期望为 shape=(len(task_ids), len(worker_ids)) 的二维数组/ndarray，
          非候选位置可用 NaN 或 +/-Inf 标记；函数将把非有限值视作“无候选”。
        - selected_set 为 {(task_id, worker_id), ...} 的集合，用于高亮被选中的配对。
        """
        # 兼容 Assignment 集合：若传入为 Assignment 对象集合，则转为 (task_id, worker_id)
        try:
            _it = iter(selected_set)
            _sample = next(_it) if selected_set else None
        except Exception:
            _sample = None
        if _sample is not None and not (isinstance(_sample, tuple) and len(_sample) == 2):
            try:
                selected_set = {(getattr(x, 'task_id'), getattr(x, 'worker_id')) for x in selected_set}
            except Exception:
                selected_set = set()

        # 可选截断，避免过大输出
        t_ids = task_ids[:max_rows]
        w_ids = worker_ids[:max_cols]
        # 头部
        col_head = ["task\\worker"] + [str(w) for w in w_ids]
        col_widths = [max(len(col_head[0]), 12)] + [max(len(str(w)), 8) for w in w_ids]
        def fmt_cell(s, w): 
            s = str(s)
            return s + " " * max(0, w - len(s))
        print("\n=== Assignment Check (every 100 runs) ===")
        print(" | ".join(fmt_cell(h, col_widths[i]) for i, h in enumerate(col_head)))
        print("-" * (sum(col_widths) + 3*len(col_widths)))

        for ti, t in enumerate(t_ids):
            row_cells = [fmt_cell(str(t), col_widths[0])]
            for wj, w in enumerate(w_ids):
                # 支持 ndarray 与 list-of-lists 两种索引方式
                try:
                    v = net_matrix[ti, wj]
                except Exception:
                    v = net_matrix[ti][wj]
                # 允许用 NaN / +/-Inf / None 表示“无候选”
                try:
                    is_valid = np.isfinite(v)
                except Exception:
                    # 对非常规类型做兜底
                    try:
                        is_valid = np.isfinite(float(v))
                    except Exception:
                        is_valid = False
                if (v is None) or (not is_valid):  # 非候选
                    cell = "--"
                else:
                    mark_left, mark_right = ("[", "]") if (t, w) in selected_set else ("(", ")")
                    try:
                        cell = f"{mark_left}{float(v):.{digits}f}{mark_right}"
                    except Exception:
                        cell = f"{mark_left}{v}{mark_right}"
                row_cells.append(fmt_cell(cell, col_widths[wj+1]))
            print(" | ".join(row_cells))
        print("=== End ===\n")
        return

    def estimated_net(self, partition: ContextSpacePartition, include_ucb: bool = True) -> float:
        mean = float(partition.posterior_mean())
        if include_ucb and self.use_ucb:
            pulls = float(max(partition.sample_count, 1.0))
            pulls = max(pulls, float(self.ucb_min_pulls))
            total = max(self.total_updates, 1)
            bonus = 0.0
            try:
                bonus = self.ucb_coef * math.sqrt(max(0.0, math.log(total + 1.0) / pulls))
            except Exception:
                bonus = 0.0
            mean = min(1.0, mean + bonus)
        return float(mean - self.replication_cost)

    def assignment_net(self, assignment: 'Assignment', include_ucb: bool = True) -> float:
        partition = self.root_partition.find_partition(assignment.context)
        return self.estimated_net(partition, include_ucb=include_ucb)

    def _posterior_variance(self, partition: ContextSpacePartition) -> float:
        """Compute Beta posterior variance using observed rewards (no extra priors)."""
        total = float(partition.sample_count)
        if total <= 0.0:
            return 1.0
        alpha = float(partition.sum_reward)
        beta_param = total - alpha
        eps = 1e-9
        alpha = max(alpha, eps)
        beta_param = max(beta_param, eps)
        denom = (alpha + beta_param) ** 2 * (alpha + beta_param + 1.0)
        if denom <= 0.0:
            return 0.0
        variance = (alpha * beta_param) / denom
        return float(max(variance, eps))

    def _should_split(self, partition: ContextSpacePartition) -> bool:
        """Decide whether to subdivide a partition based on variance & sample count."""
        if partition.children is not None:
            return False
        if self.max_partition_depth is not None and partition.depth >= self.max_partition_depth:
            return False

        min_required = max(self.min_samples_before_split, self.partition_split_threshold + partition.depth)
        if partition.sample_count < min_required:
            return False

        variance = self._posterior_variance(partition)
        if variance <= self.variance_split_threshold:
            return False

        return True

    # 覆盖式统一接口：允许通过参数控制是否可不匹配
    def select_assignments(self, candidate_assignments: List[Assignment], allow_unmatch: bool = True):
        """选择工-任务匹配（允许通过参数控制是否可不匹配）。

        - allow_unmatch=True：允许任务或工人不匹配（虚拟节点成本0）；
        - allow_unmatch=False：尽量匹配（不匹配成本为极大常数）。最终仅返回净收益>0的匹配。
        """
        if not candidate_assignments:
            return []

        task_ids = sorted({a.task_id for a in candidate_assignments})
        worker_ids = sorted({a.worker_id for a in candidate_assignments})
        task_idx = {t: i for i, t in enumerate(task_ids)}
        worker_idx = {w: j for j, w in enumerate(worker_ids)}
        m, n = len(task_ids), len(worker_ids)

        # 净收益矩阵（非候选 = -inf）
        profits = np.full((m, n), -np.inf, dtype=float)
        pair2a = {}
        for a in candidate_assignments:
            i, j = task_idx[a.task_id], worker_idx[a.worker_id]
            partition = self.root_partition.find_partition(a.context)
            net = self.estimated_net(partition, include_ucb=True)
            profits[i, j] = net
            pair2a[(a.task_id, a.worker_id)] = a

        EPS = 1e-12
        selected, row_ind, col_ind = run_hungarian_matching(
            task_ids,
            worker_ids,
            profits,
            pair2a,
            allow_unmatch=allow_unmatch,
            eps=EPS,
        )
        # Debug: 每100轮打印一次候选净收益与被选配对
        try:
            self._run_counter += 1
        except Exception:
            self._run_counter = 1
        if (self._run_counter % 10000) == 0:
            try:
                # 构造包含虚拟行/列的方阵视图，直观展示不匹配情况
                t_ids_sq = list(task_ids) + [f"__DUMMY_T_{k}" for k in range(n)]
                w_ids_sq = list(worker_ids) + [f"__DUMMY_W_{k}" for k in range(m)]
                size_sq = m + n
                net_square = np.full((size_sq, size_sq), np.nan, dtype=float)
                # 真实候选净收益
                net_square[:m, :n] = profits
                # 不匹配成本对应的“净收益”显示：允许不匹配时显示为 0；否则显示为空
                if allow_unmatch:
                    if m > 0:
                        net_square[:m, n:n + m] = 0.0  # 任务 -> 虚拟工人
                    if n > 0:
                        net_square[m:m + n, :n] = 0.0  # 虚拟任务 -> 工人

                # 选中配对（包括落在虚拟行/列的“未匹配”）
                selected_pairs_sq = set()
                for i, j in zip(row_ind, col_ind):
                    ri = t_ids_sq[i]
                    cj = w_ids_sq[j]
                    if i < m and j < n:
                        if profits[i, j] > EPS:
                            selected_pairs_sq.add((ri, cj))
                    elif allow_unmatch:
                        selected_pairs_sq.add((ri, cj))

                self._debug_print_assignment_table(t_ids_sq, w_ids_sq, net_square, selected_pairs_sq)
            except Exception as _e:
                print(f"[debug] print assignment table failed: {_e}")
        return selected

    
    def update_assignments_reward(self, selected_assignments: List[Assignment], rewards: dict):
        """更新选中分配的奖励并判断是否细分

        Args:
            selected_assignments (List[Assignment]): 被选中的工人-任务对。
            rewards (dict): 奖励字典，键为 Assignment，值为奖励 (0 或 1)。

        Notes:
            - 当样本数达到阈值 `partition_split_threshold` 时会进行二分细分。
        """
        for a in selected_assignments:
            p = self.root_partition.find_partition(a.context)
            reward = rewards.get(a, 0)
            p.update_reward(reward, context=a.context)
            self.total_updates += 1
            # 样本数达到阈值后进行二分细分（受最大层级限制）
            if self._should_split(p):
                p.subdivide()
                self.split_events += 1
                if p in self.partitions:
                    self.partitions.remove(p)
                    self.partitions.extend(p.children)
                else:
                    print("Warning: subdividing a partition not in the main list.")

if __name__ == "__main__":
    # 参数定义
    CONTEXT_DIM = 7
    REPLICATION_COST = 0.1
    BUDGET = 1
    
    replicator = TaskReplicator(CONTEXT_DIM, PARTITION_SPLIT_THRESHOLD, BUDGET, REPLICATION_COST)
    
    # 生成模拟候选工人-任务对，随机上下文
    candidates = []
    for w in range(10):
        for task in range(10):
            # 这里我们直接模拟生成 assignment 的上下文向量，也就是task和worker的组合特征
            ctx = np.random.rand(CONTEXT_DIM)
            candidates.append(Assignment(w, task, ctx))
    
    # 任务选择
    selected = replicator.select_assignments(candidates)
    print(f"选中工人-任务对数量: {len(selected)}")
    
    # 模拟奖励观测，0或1随机
    rewards = {a: np.random.binomial(1, 0.5) for a in selected}
    # 更新统计
    replicator.update_assignments_reward(selected, rewards)
