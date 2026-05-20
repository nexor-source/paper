import math
import numpy as np
from typing import List, Tuple, Optional
from matching_utils import run_hungarian_matching
from config import *

class ContextSpacePartition:
    """
    上下文空间的一个划分单元（超立方体）。
    兼容 Ch3的精确信息熵分裂 与 Ch4的DP无状态几何分裂。
    """
    def __init__(self, bounds: List[Tuple[float, float]], depth: int = 0, enable_dp: bool = False, epsilon: float = 1.0):
        self.bounds = bounds    
        self.depth = depth      
        self.enable_dp = enable_dp
        self.epsilon = epsilon
        
        self.sample_count = 0   
        self.estimated_quality = 0.0  
        self.children = None    

        self.default_mean = 0.5               
        self.sum_reward = 0.0                 
        self.data_points = []                 
    
    def _get_noisy_stats(self) -> Tuple[float, float]:
        """【第四章新增】获取加噪后的统计量"""
        if not self.enable_dp:
            return self.sum_reward, float(self.sample_count)
        
        # 注入 Laplace 噪声，敏感度为 1，预算分配给 sum 和 count 所以是 2/epsilon
        scale = 2.0 / self.epsilon
        noisy_sum = self.sum_reward + np.random.laplace(0, scale)
        noisy_count = self.sample_count + np.random.laplace(0, scale)
        return noisy_sum, max(1.0, noisy_count)

    def posterior_mean(self) -> float:
        """获取后验均值（DP模式下使用加噪数据）"""
        if self.enable_dp:
            n_sum, n_count = self._get_noisy_stats()
            # 截断到 [0, 1] 保证合法性
            return float(min(max(n_sum / n_count, 0.0), 1.0))
        else:
            if self.sample_count <= 0:
                return self.default_mean  
            return self.sum_reward / float(self.sample_count)

    def contains(self, context: np.ndarray) -> bool:
        EPS = 1e-8  
        return all(self.bounds[d][0] <= context[d] < self.bounds[d][1] or
                abs(context[d] - self.bounds[d][1]) < EPS  
                for d in range(len(context)))

    def update_reward(self, reward: float, debug: bool = False, context: Optional[np.ndarray] = None):
        if not debug:
            self.sample_count += 1
            # 【核心修改】如果是DP模式，绝对不保存原始context以节省内存并保护隐私
            if not self.enable_dp and context is not None:
                try:
                    ctx_tuple = tuple(float(x) for x in context)
                except Exception:
                    ctx_tuple = tuple(context)
                self.data_points.append((ctx_tuple, float(reward)))
        self.sum_reward += reward
        self.estimated_quality = self.posterior_mean()
    
    def _binary_entropy(self, success: float, total: float) -> float:
        if total <= 0.0: return 0.0
        p = success / total
        if p <= 0.0 or p >= 1.0: return 0.0
        return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))

    def _select_split_dimension(self) -> Tuple[Optional[int], Optional[float]]:
        # 【第四章新增】DP模式下，直接寻找跨度最大的维度进行物理中心点切割（无状态分割）
        if self.enable_dp:
            widths = [b[1] - b[0] for b in self.bounds]
            if not widths:
                return None, None
            best_dim = int(np.argmax(widths))
            lo, hi = self.bounds[best_dim]
            best_mid = (lo + hi) / 2.0
            return best_dim, best_mid

        # 第三章逻辑：信息增益
        if not self.data_points: return None, None
        total_samples = float(len(self.data_points))
        parent_entropy = self._binary_entropy(self.sum_reward, total_samples)
        best_gain, best_dim, best_mid = float("-inf"), None, None

        for dim, (lo, hi) in enumerate(self.bounds):
            mid = (lo + hi) / 2.0
            left_total, right_total, left_success, right_success = 0.0, 0.0, 0.0, 0.0
            for ctx, reward in self.data_points:
                if ctx[dim] < mid:
                    left_total += 1.0
                    left_success += reward
                else:
                    right_total += 1.0
                    right_success += reward

            if left_total == 0.0 or right_total == 0.0: continue
            total = left_total + right_total
            left_entropy = self._binary_entropy(left_success, left_total)
            right_entropy = self._binary_entropy(right_success, right_total)
            weighted_entropy = (left_total / total) * left_entropy + (right_total / total) * right_entropy
            gain = parent_entropy - weighted_entropy
            if gain > best_gain:
                best_gain, best_dim, best_mid = gain, dim, mid

        if best_dim is None:
            widths = [b[1] - b[0] for b in self.bounds]
            if not widths: return None, None
            best_dim = int(np.argmax(widths))
            lo, hi = self.bounds[best_dim]
            best_mid = (lo + hi) / 2.0

        return best_dim, best_mid

    def subdivide(self):
        if self.children is not None: return

        split_dim, split_value = self._select_split_dimension()
        if split_dim is None or split_value is None: return

        left_bounds = list(self.bounds)
        right_bounds = list(self.bounds)
        left_bounds[split_dim] = (self.bounds[split_dim][0], split_value)
        right_bounds[split_dim] = (split_value, self.bounds[split_dim][1])

        left_child = ContextSpacePartition(left_bounds, self.depth + 1, self.enable_dp, self.epsilon)
        right_child = ContextSpacePartition(right_bounds, self.depth + 1, self.enable_dp, self.epsilon)
        self.children = [left_child, right_child]

        # 【核心修改】DP模式下，子节点重新从0开始积累，不继承任何历史（无状态）
        if not self.enable_dp:
            for ctx, reward in self.data_points:
                target = left_child if ctx[split_dim] < split_value else right_child
                target.data_points.append((ctx, reward))
                target.sample_count += 1
                target.sum_reward += reward

        for child in self.children:
            child.sum_reward = float(child.sum_reward)
            child.estimated_quality = child.posterior_mean()

        self.data_points = []
    
    def find_partition(self, context: np.ndarray):
        if self.children is None: return self
        for child in self.children:
            if child.contains(context):
                return child.find_partition(context)
        raise ValueError(f"Context {context} not contained in any child partition")

class Assignment:
    def __init__(self, worker_id: int, task_id: int, context: np.ndarray):
        self.worker_id = worker_id
        self.task_id = task_id
        self.context = context

class TaskReplicator:
    def __init__(self, context_dim: int, partition_split_threshold: int, budget: int, replication_cost: float, max_partition_depth: Optional[int] = None, enable_dp: bool = False, dp_epsilon: float = 1.0):
        self.context_dim = context_dim
        self.budget = budget
        self.replication_cost = replication_cost
        self.max_partition_depth = max_partition_depth
        
        # DP 相关配置
        self.enable_dp = enable_dp
        self.dp_epsilon = dp_epsilon
        self.dp_beta = float(globals().get('DP_UCB_BETA', 0.5))
        
        self.root_partition = ContextSpacePartition(bounds=[(0,1)]*context_dim, enable_dp=self.enable_dp, epsilon=self.dp_epsilon)
        self.partitions = [self.root_partition]
        self.partition_split_threshold = partition_split_threshold
        
        try: self.use_ucb = bool(globals().get('REPLICATOR_USE_UCB', False))
        except: self.use_ucb = False
        try: self.ucb_coef = float(globals().get('REPLICATOR_UCB_COEF', 0.0))
        except: self.ucb_coef = 0.0
        try: self.ucb_min_pulls = max(1.0, float(globals().get('REPLICATOR_UCB_MIN_PULLS', 1)))
        except: self.ucb_min_pulls = 1.0
        
        self.use_ucb = bool(self.use_ucb and self.ucb_coef > 0.0)
        try: self.min_samples_before_split = max(1, int(globals().get('PARTITION_MIN_SAMPLES', self.partition_split_threshold)))
        except: self.min_samples_before_split = self.partition_split_threshold
        try: self.variance_split_threshold = max(0.0, float(globals().get('PARTITION_VARIANCE_THRESHOLD', 0.02)))
        except: self.variance_split_threshold = 0.02
        
        self.total_updates = 0
        self.split_events = 0  
        self._run_counter = 0

    def estimated_net(self, partition: ContextSpacePartition, include_ucb: bool = True) -> float:
        mean = float(partition.posterior_mean())
        if include_ucb and self.use_ucb:
            # 【第四章新增】获取计算 UCB 的 pulls
            if self.enable_dp:
                _, pulls = partition._get_noisy_stats()
            else:
                pulls = float(max(partition.sample_count, 1.0))
                
            pulls = max(pulls, float(self.ucb_min_pulls))
            total = max(self.total_updates, 1)
            
            # 系统探索补偿项 c_sys
            sys_bonus = self.ucb_coef * math.sqrt(max(0.0, math.log(total + 1.0) / pulls))
            
            # 隐私探索补偿项 c_dp
            dp_bonus = 0.0
            if self.enable_dp:
                dp_bonus = self.dp_beta * (2.0 / (self.dp_epsilon * pulls))
                
            bonus = sys_bonus + dp_bonus
            mean = min(1.0, mean + bonus)
            
        return float(mean - self.replication_cost)

    def assignment_net(self, assignment: 'Assignment', include_ucb: bool = True) -> float:
        partition = self.root_partition.find_partition(assignment.context)
        return self.estimated_net(partition, include_ucb=include_ucb)

    def _posterior_variance(self, partition: ContextSpacePartition) -> float:
        total = float(partition.sample_count)
        if total <= 0.0: return 1.0
        alpha = float(partition.sum_reward)
        beta_param = total - alpha
        eps = 1e-9
        alpha, beta_param = max(alpha, eps), max(beta_param, eps)
        denom = (alpha + beta_param) ** 2 * (alpha + beta_param + 1.0)
        return float(max((alpha * beta_param) / denom, eps)) if denom > 0 else 0.0

    def _should_split(self, partition: ContextSpacePartition) -> bool:
        if partition.children is not None: return False
        if self.max_partition_depth is not None and partition.depth >= self.max_partition_depth: return False

        # 【核心修改】DP模式下，不计算方差，直接依靠加噪样本量强行门控
        if self.enable_dp:
            _, n_count = partition._get_noisy_stats()
            min_required = max(self.min_samples_before_split, self.partition_split_threshold + partition.depth)
            # 为了抵抗噪声，DP模式需要稍微更多的样本才分裂
            return n_count > (min_required + (2.0 / self.dp_epsilon))
        
        # 第三章逻辑
        min_required = max(self.min_samples_before_split, self.partition_split_threshold + partition.depth)
        if partition.sample_count < min_required: return False
        if self._posterior_variance(partition) <= self.variance_split_threshold: return False
        return True

    def select_assignments(self, candidate_assignments: List[Assignment], allow_unmatch: bool = True, use_ucb: bool = True):
        if not candidate_assignments: return []

        task_ids = sorted({a.task_id for a in candidate_assignments})
        worker_ids = sorted({a.worker_id for a in candidate_assignments})
        task_idx, worker_idx = {t: i for i, t in enumerate(task_ids)}, {w: j for j, w in enumerate(worker_ids)}
        m, n = len(task_ids), len(worker_ids)

        profits = np.full((m, n), -np.inf, dtype=float)
        pair2a = {}
        for a in candidate_assignments:
            i, j = task_idx[a.task_id], worker_idx[a.worker_id]
            partition = self.root_partition.find_partition(a.context)
            net = self.estimated_net(partition, include_ucb=use_ucb)
            profits[i, j] = net
            pair2a[(a.task_id, a.worker_id)] = a

        EPS = 1e-12
        selected, row_ind, col_ind = run_hungarian_matching(task_ids, worker_ids, profits, pair2a, allow_unmatch=allow_unmatch, eps=EPS)
        return selected
    
    def update_assignments_reward(self, selected_assignments: List[Assignment], rewards: dict):
        for a in selected_assignments:
            p = self.root_partition.find_partition(a.context)
            reward = rewards.get(a, 0)
            p.update_reward(reward, context=a.context)
            self.total_updates += 1
            if self._should_split(p):
                p.subdivide()
                self.split_events += 1
                if p in self.partitions:
                    self.partitions.remove(p)
                    self.partitions.extend(p.children)