import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment

class ContextSpacePartition:
    """
    上下文空间的一个划分单元（超立方体），维护该区域内样本计数和奖励估计，
    支持 [动态二分细分] 以 [细化学习] 。
    """
    def __init__(self, bounds: List[Tuple[float, float]], depth: int = 0):
        """
        初始化上下文划分区域
        :param bounds: 每个维度的区间，如[(min, max), (min, max), ...],shape=(d,2)
        :param depth: 当前划分层级，根节点depth=0
        """
        self.bounds = bounds    # 每个维度的区间列表
        self.depth = depth         # 当前划分层级
        self.sample_count = 0              # 该区域被采样的次数（副本被选中次数）
        self.estimated_quality = 0.0      # 该区域内任务副本的平均完成质量估计
        self.children = None              # 子划分列表，未细分时为None
    
    def contains(self, context:np.ndarray) -> bool:
        """ 判断上下文是否在该划分区域内
        Args:
            context (np.ndarray,shape=(d,)): 归一化上下文向量

        Returns:
            bool: 上下文是否在当前划分区域内
        """
        EPS = 1e-8  # 容忍边界浮点误差
        return all(self.bounds[d][0] <= context[d] < self.bounds[d][1] or
                abs(context[d] - self.bounds[d][1]) < EPS  # 允许等于上界
                for d in range(len(context)))

    def update_reward(self, reward: float):
        """
        根据观察到的副本完成奖励更新该区域的样本计数和平均质量估计
        :param reward: 当前副本任务完成奖励，通常0或1
        """
        self.sample_count += 1
        # 使用滑动平均更新估计质量，保证数值稳定
        self.estimated_quality = ((self.estimated_quality * (self.sample_count - 1)) + reward) / self.sample_count
    
    def subdivide(self):
        """
        将该划分区域沿每个维度中点二分，产生2^d个子区域，
        细化划分以提升估计的精度和灵活性
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
    
    def find_partition(self, context: np.ndarray):
        """
        递归寻找包含指定上下文的最底层划分区域（叶节点）
        :param context: 归一化上下文向量,shape=(d,)
        :return: 对应的叶节点 ContextSpacePartition 对象
        """
        # print("Current partition bounds:")
        # if self.children: 
        #     for child in self.children:
        #         print(" ", child.bounds)
        #         print("Incoming context:", context)
        

        if self.children is None:
            return self
        for child in self.children:
            if child.contains(context):
                return child.find_partition(context)
        raise ValueError(f"Context {context} not contained in any child partition")

class Assignment:
    """
    表示一个工人-任务对，封装对应的上下文信息，
    是调度系统中任务分配的基本单位
    """
    def __init__(self, worker_id: int, task_id: int, context: np.ndarray):
        self.worker_id = worker_id
        self.task_id = task_id
        self.context = context  # 归一化后的上下文特征向量

class TaskReplicator:
    """
    任务分配器，基于上下文划分估计副本质量，
    利用匈牙利算法实现任务-工人最大收益匹配，
    动态细分上下文划分空间保证长期学习精度
    """
    def __init__(self, context_dim: int, initial_partition_size: int, budget: int, replication_cost: float):
        self.context_dim = context_dim
        self.budget = budget  # 每个任务允许的最大副本数，代码中假设每个任务只给一个worker分配
        self.replication_cost = replication_cost
        
        # 初始化根划分，单位超立方体[0,1]^d
        self.root_partition = ContextSpacePartition(bounds=[(0,1)]*context_dim)
        self.partitions = [self.root_partition]  # 当前所有叶子划分
        self.initial_partition_size = initial_partition_size  # 细分阈值
    
    def select_assignments(self, candidate_assignments: List[Assignment]):
        """
        对给定候选工人-任务对，选择一组最优匹配副本
        :param candidate_assignments: 所有候选分配方案列表,shape=(n,)
        :return: 选中的 [工人-任务] 分配列表,shape=(m,)
        """
        # 建立映射 map{Assignment: ContextSpacePartition}
        partition_map = {a: self.root_partition.find_partition(a.context) for a in candidate_assignments}
        
        # 获得所有任务与工人的id集合
        task_ids = sorted(set(a.task_id for a in candidate_assignments))    # 任务id set
        worker_ids = sorted(set(a.worker_id for a in candidate_assignments))    # 工人id set
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
        """
        更新选择的assignments对应的reward并判断细分
        :param selected_assignments: 被选中的工人-任务对
        :param rewards: dict{Assignment: reward}，任务完成奖励，0或1
        """
        for a in selected_assignments:
            p = self.root_partition.find_partition(a.context)
            reward = rewards.get(a, 0)
            p.update_reward(reward)
            # 样本数达到阈值后进行二分细分
            if p.sample_count >= self.initial_partition_size:
                p.subdivide()
                if p in self.partitions:
                    self.partitions.remove(p)
                    self.partitions.extend(p.children)

if __name__ == "__main__":
    # 参数定义
    context_dim = 7
    budget = 1  # 每任务唯一worker分配
    initial_partition_size = 10
    replication_cost = 0.1
    
    # 初始化调度器
    replicator = TaskReplicator(context_dim, initial_partition_size, budget, replication_cost)
    
    # 生成模拟候选工人-任务对，随机上下文
    candidates = []
    for w in range(10):
        for task in range(10):
            ctx = np.random.rand(context_dim)
            candidates.append(Assignment(w, task, ctx))
    
    # 任务选择
    selected = replicator.select_assignments(candidates)
    print(f"选中工人-任务对数量: {len(selected)}")
    
    # 模拟奖励观测，0或1随机
    rewards = {a: np.random.binomial(1, 0.5) for a in selected}
    
    # 更新统计
    replicator.update_assignments_reward(selected, rewards)
