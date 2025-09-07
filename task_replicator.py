import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
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

        # 新增：层级先验（“父经验”）。注意：prior_weight 不计入细分阈值
        self.prior_mean = 0.5                 # 缺省先验均值（对二元奖励可设 0.5）
        self.prior_weight = 0.0               # 先验“伪样本权重”
        self.sum_reward = 0.0                 # 该 partition 所有样本带来的reward总和
    
    def posterior_mean(self) -> float:
        """融合先验与真实样本的估计质量（后验均值）

        Returns:
            float: 当前分区的估计质量（先验均值与观测数据的加权平均）。
        """
        total_w = self.prior_weight + self.sample_count
        if total_w <= 0:
            return self.prior_mean  # 没有任何信息时，退回先验均值
        return (self.prior_mean * self.prior_weight + self.sum_reward) / total_w

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

    def update_reward(self, reward: float, debug: bool = False):
        """根据观察到的副本完成奖励更新该区域的样本计数和平均质量估计

        Args:
            reward (float): 当前副本任务完成奖励，通常取0或1。
            debug (bool): 用于代码内部调用，如果为true则不会增加样本数量，只是更新估计质量
        """
        if not debug:
            self.sample_count += 1
        self.sum_reward += reward
        # 仅作为一个“纯样本平均”的快速回显；真正用于决策请调用 posterior_mean()
        # self.estimated_quality = self.sum_reward / self.sample_count
        self.estimated_quality = self.posterior_mean()
    
    def subdivide(self):
        """将该划分区域沿每个维度中点二分，产生 2^d 个子区域

        Notes:
            - 若该区域已被细分，则不再重复细分。
            - 每次细分会更新 `children` 列表。
        """
        if self.children is not None:
            # 如果这个partition已细分则不再细分
            print("Partition already subdivided.")
            return
        
        # 计算每个维度的中点
        midpoints = [(b[0] + b[1]) / 2 for b in self.bounds]
        new_bounds_list = []
        
        # 遍历生成2^d个子划分的边界
        for i in range(2 ** len(self.bounds)):
            new_bounds = []
            # 每个新生成的划分的每个维度
            for d in range(len(self.bounds)):
                # 根据二进制位选择下界或上界（巧妙）
                if (i >> d) & 1 == 0:
                    new_bounds.append((self.bounds[d][0], midpoints[d]))
                else:
                    new_bounds.append((midpoints[d], self.bounds[d][1]))
            new_bounds_list.append(new_bounds)
        # 创建子划分对象列表
        self.children = [ContextSpacePartition(b, self.depth + 1) for b in new_bounds_list]

        # —— 关键：将父分区“后验均值”作为子分区“先验均值”，并给予弱化先验权重 ——
        parent_post = self.posterior_mean()
        total_prior = LAMBDA_PRIOR * min(self.sample_count, PRIOR_CAP)  # 控制强度与上限
        d = len(self.bounds)
        per_child_prior = total_prior / (2 ** d) if total_prior > 0 else 0.0

        for child in self.children:
            child.prior_mean = parent_post
            child.prior_weight = per_child_prior
            child.update_reward(0, True)
            # 重要：不继承 sample_count / sum_reward，避免“过度自信”
            # child.sample_count = 0
            # child.sum_reward  = 0.0
    
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
    
    def select_assignments(self, candidate_assignments: List[Assignment]):
        """对候选工人-任务对进行最优匹配选择

        Args:
            candidate_assignments (List[Assignment]): 候选工人-任务对列表，长度 n。

        Returns:
            List[Assignment]: 选中的工人-任务分配列表，长度 m。
        """
        # 建立映射 map{Assignment: ContextSpacePartition}
        partition_map = {a: self.root_partition.find_partition(a.context) for a in candidate_assignments}
        
        # 获得所有任务与工人的id集合
        task_ids = sorted(set(a.task_id for a in candidate_assignments))
        worker_ids = sorted(set(a.worker_id for a in candidate_assignments))
        # 将任务和工人的id重新映射为矩阵索引(0~n-1)
        task_idx = {task: i for i, task in enumerate(task_ids)}  
        worker_idx = {worker: j for j, worker in enumerate(worker_ids)}
        
        # 初始化收益矩阵（任务×工人），匈牙利算法求最小成本匹配，
        # 因此收益取负值，未探索划分质量赋大正数（此处为极大负值对应正收益）
        LARGE_VALUE = 1e6
        cost_matrix = np.full((len(task_ids), len(worker_ids)), -LARGE_VALUE)
        
        # 填充收益矩阵
        for a in candidate_assignments:
            i = task_idx[a.task_id]
            j = worker_idx[a.worker_id]
            p = partition_map[a]
            # 未被探索的划分赋极大估计质量，保证算法探索
            if p.sample_count == 0:
                estimated_quality = LARGE_VALUE
            else:
                estimated_quality = p.estimated_quality
            net_quality = estimated_quality - self.replication_cost
            cost_matrix[i, j] = -net_quality  # 转为成本矩阵，匈牙利算法求最小值
        
        # 使用匈牙利算法求最大匹配（最小成本匹配）
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        selected = []
        for i, j in zip(row_ind, col_ind):
            # 只选净收益大于0的匹配，避免选负收益方案
            if -cost_matrix[i, j] > 0:
                task = task_ids[i]
                worker = worker_ids[j]
                # 找到对应assignment对象并加入结果
                for a in candidate_assignments:
                    if a.task_id == task and a.worker_id == worker:
                        selected.append(a)
                        break
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
            p.update_reward(reward)
            # 样本数达到阈值后进行二分细分（受最大层级限制）
            if p.sample_count >= self.partition_split_threshold:
                # 若设置了最大层级限制，且当前层级已达到或超过上限，则不再细分
                if self.max_partition_depth is not None and p.depth >= self.max_partition_depth:
                    continue
                p.subdivide()
                if p in self.partitions:
                    self.partitions.remove(p)
                    self.partitions.extend(p.children)

if __name__ == "__main__":
    # 参数定义
    CONTEXT_DIM = 7
    PARTITION_SPLIT_THRESHOLD = 10
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
