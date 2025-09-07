import numpy as np
import os
from collections import deque
from typing import List, Dict
from scipy.optimize import linear_sum_assignment

# 你之前定义的 ContextNormalizer, Assignment, TaskReplicator 类这里省略，假设已经实现并导入
from normalizer import ContextNormalizer
from task_replicator import Assignment, TaskReplicator
from visualizer import PartitionVisualizer
from config import *

# 全局记录每一步的 loss（oracle 与算法期望净收益之差的非负部分）
LOSS_HISTORY: List[float] = []

class Task:
    """
    任务实体，包含任务特征与任务 ID。
    """

    def __init__(self, task_id: int, task_type: int, data_size: float, deadline: float):
        """初始化任务对象

        Args:
            task_id (int): 任务 ID。可传入 -1 表示由队列自动分配。
            task_type (int): 任务类型编码（例如 0-9）。
            data_size (float): 数据大小，单位 MB。
            deadline (float): 截止时间（相对当前时间的秒数或任意时间度量）。

        Notes:
            - 任务 ID 若为 -1，将在入队时由 TaskQueue 进行自增分配。
        """
        self.task_id = task_id
        self.task_type = task_type
        self.data_size = data_size
        self.deadline = deadline


class Worker:
    """
    工人实体，包含工人 ID 和能力特征。
    """

    def __init__(
        self,
        worker_id: int,
        driving_speed: float,
        bandwidth: float,
        processor_perf: float,
        physical_distance: float,
        weather: int,
    ):
        """初始化工人对象

        Args:
            worker_id (int): 工人 ID。
            driving_speed (float): 行驶速度（m/s）。
            bandwidth (float): 网络带宽（Mbps）。
            processor_perf (float): 处理器性能（GHz）。
            physical_distance (float): 距离任务点的物理距离（m）。
            weather (int): 天气类别编码（如 0-4）。

        Notes:
            - 这些特征会被归一化到 [0, 1] 后用于调度模型。
        """
        self.worker_id = worker_id
        self.driving_speed = driving_speed
        self.bandwidth = bandwidth
        self.processor_perf = processor_perf
        self.physical_distance = physical_distance
        self.weather = weather


class TaskQueue:
    """
    任务队列，支持任务动态加入和批量调度。
    """

    def __init__(self):
        """初始化任务队列

        Notes:
            - 使用双端队列（deque）保存任务。
            - `next_task_id` 用于为未指定 ID 的任务自动分配递增 ID。
        """
        self.queue = deque()
        self.next_task_id = 0

    def add_task(self, task_type: int, data_size: float, deadline: float) -> None:
        """新增任务到队列末尾

        Args:
            task_type (int): 任务类型编码。
            data_size (float): 数据大小（MB）。
            deadline (float): 截止时间。

        Returns:
            None

        Notes:
            - 自动为任务分配自增的 `task_id`。
        """
        task = Task(self.next_task_id, task_type, data_size, deadline)
        self.queue.append(task)
        self.next_task_id += 1

    def get_tasks_batch(self, batch_size: int) -> List[Task]:
        """从队列头部弹出一个批次的任务

        Args:
            batch_size (int): 批大小（最多返回该数量的任务）。

        Returns:
            List[Task]: 取出的任务列表，长度不超过 `batch_size`。
        """
        tasks: List[Task] = []
        for _ in range(min(batch_size, len(self.queue))):
            tasks.append(self.queue.popleft())
        return tasks


class Scheduler:
    """
    任务调度器，管理任务流、工人资源与调用分配算法。
    """

    def __init__(
        self,
        workers: List[Worker],
        context_normalizer: ContextNormalizer,
        replicator: TaskReplicator,
    ):
        """初始化调度器

        Args:
            workers (List[Worker]): 工人资源列表。
            context_normalizer (ContextNormalizer): 上下文归一化器。
            replicator (TaskReplicator): 任务-工人分配器（基于上下文划分 + 匈牙利算法）。

        Notes:
            - 内部维护一个 `TaskQueue` 和一个简单的时间步 `time`。
        """
        self.workers = workers
        self.normalizer = context_normalizer
        self.replicator = replicator
        self.task_queue = TaskQueue()
        self.time = 0  # 模拟时间步

    def generate_candidate_assignments(self, tasks: List[Task]) -> List[Assignment]:
        """根据任务与工人生成候选的工人-任务对

        Args:
            tasks (List[Task]): 待调度的任务列表。

        Returns:
            List[Assignment]: 候选分配对列表，每个元素包含 (worker_id, task_id, normalized_context)。

        Notes:
            - 原始特征通过 `ContextNormalizer.normalize_context` 归一化为 [0,1]^d。
            - 该方法不进行筛选，返回所有工人 x 任务的组合。
        """
        candidates: List[Assignment] = []
        for task in tasks:
            for worker in self.workers:
                raw_context = {
                    "driving_speed": worker.driving_speed,
                    "bandwidth": worker.bandwidth,
                    # "processor_performance": worker.processor_perf,
                    # "physical_distance": worker.physical_distance,
                    # "task_type": task.task_type,
                    # "data_size": task.data_size,
                    # "weather": worker.weather,
                }
                norm_context = self.normalizer.normalize_context(raw_context)
                assignment = Assignment(worker.worker_id, task.task_id, norm_context)
                candidates.append(assignment)
        return candidates

    def _oracle_select_assignments(self, candidate_assignments: List[Assignment]) -> List[Assignment]:
        """全知全能（oracle）下基于真实成功概率的最优匹配。

        使用与 TaskReplicator 相同的约束与代价形式，但将收益替换为真实的期望净收益
        p(context) - replication_cost。

        Args:
            candidate_assignments: 候选工人-任务对

        Returns:
            List[Assignment]: 基于真实概率的最优匹配集合（仅保留净收益>0的对）。
        """
        if not candidate_assignments:
            return []

        # 建立任务/工人索引
        task_ids = sorted(set(a.task_id for a in candidate_assignments))
        worker_ids = sorted(set(a.worker_id for a in candidate_assignments))
        task_idx = {t: i for i, t in enumerate(task_ids)}
        worker_idx = {w: j for j, w in enumerate(worker_ids)}

        # 真实期望净收益矩阵（注意：linear_sum_assignment 求最小成本，因此取负）
        LARGE_VALUE = 1e6
        cost_matrix = np.full((len(task_ids), len(worker_ids)), -LARGE_VALUE, dtype=float)
        for a in candidate_assignments:
            i = task_idx[a.task_id]
            j = worker_idx[a.worker_id]
            p = self.evaluate_reward(a.context)
            net = p - self.replicator.replication_cost
            cost_matrix[i, j] = -net

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        selected: List[Assignment] = []
        for i, j in zip(row_ind, col_ind):
            # 过滤净收益<=0的匹配
            if -cost_matrix[i, j] <= 0:
                continue
            t_id = task_ids[i]
            w_id = worker_ids[j]
            for a in candidate_assignments:
                if a.task_id == t_id and a.worker_id == w_id:
                    selected.append(a)
                    break
        return selected

    def _expected_total_reward(self, assignments: List[Assignment]) -> float:
        """基于真实成功概率的期望总净收益: sum(p(context) - replication_cost)."""
        if not assignments:
            return 0.0
        rc = self.replicator.replication_cost
        return float(sum(self.evaluate_reward(a.context) - rc for a in assignments))

    def evaluate_reward(self, context: np.ndarray) -> float:
        """根据 context 向量模拟得到【成功概率】
        
        Args:
            context (np.ndarray): 归一化上下文向量，shape=(d,)
        
        Returns:
            float: 成功概率 p ∈ [0,1]
        """
        # 示例规则：速度、带宽、处理器性能高 → 概率高
        #           距离、数据量大 → 概率低
        # 权重可以按需调整
        driving_speed = context[0] if len(context) > 0 else 0
        bandwidth = context[1] if len(context) > 1 else 0
        processor_perf = context[2] if len(context) > 2 else 0
        distance = context[3] if len(context) > 3 else 0
        task_type = context[4] if len(context) > 4 else 0
        data_size = context[5] if len(context) > 5 else 0
        weather = context[6] if len(context) > 6 else 0
        
        # 一个线性组合示例（权重可调）
        score = (
            0.3 * driving_speed
            + 0.25 * bandwidth
            + 0.2 * processor_perf
            - 0.15 * distance
            - 0.1 * data_size
        )
        # 天气的简单影响（越差越扣分）
        score -= 0.05 * weather  

        # Sigmoid 压缩到 (0,1)
        p = 1 / (1 + np.exp(-score * 5))  # 乘系数调整斜率
        return float(np.clip(p, 0.01, 0.99))  # 保证不为0或1


    def step(self, new_tasks: List[Task], batch_size: int) -> None:
        """执行一次调度时间步：入队新任务、批量选择、匹配与反馈更新

        Args:
            new_tasks (List[Task]): 新到达的任务列表（其 task_id 可为占位，最终入队时将重分配）。
            batch_size (int): 本次调度最多出队参与匹配的任务数。

        Returns:
            None

        Notes:
            - 奖励在此处使用 `np.random.binomial(1, 0.6)` 进行模拟；实际系统应由执行反馈提供。
            - 最终通过 `TaskReplicator.update_assignments_reward` 更新统计并触发可能的细分。
        """
        # 1. 新任务入队
        for task in new_tasks:
            self.task_queue.add_task(task.task_type, task.data_size, task.deadline)

        # 2. 取批任务准备调度
        tasks_to_schedule = self.task_queue.get_tasks_batch(batch_size)
        if not tasks_to_schedule:
            print("No tasks to schedule at time", self.time)
            self.time += 1
            return

        # 3. 生成所有候选工人-任务对
        candidates = self.generate_candidate_assignments(tasks_to_schedule)

        # 4. 调用分配算法
        selected_assignments = self.replicator.select_assignments(candidates)

        # 4.1 计算基于真实概率的最优匹配（oracle）与一步 regret/loss
        oracle_assignments = self._oracle_select_assignments(candidates)
        alg_expected = self._expected_total_reward(selected_assignments)
        oracle_expected = self._expected_total_reward(oracle_assignments)
        step_loss = max(0.0, oracle_expected - alg_expected)
        # 记录本步 loss 到全局列表
        LOSS_HISTORY.append(float(step_loss))

        # 5. 模拟执行和奖励（这里用随机模拟，真实场景由系统反馈）
        rewards: Dict[Assignment, float] = {}
        for a in selected_assignments:
            # 自定义线性成功率
            print(a.context)
            p = self.evaluate_reward(a.context)
            # 模拟成功与否
            rewards[a] = np.random.binomial(1, p)

        # 6. 更新统计
        self.replicator.update_assignments_reward(selected_assignments, rewards)

        print(
            f"Time {self.time}: Scheduled {len(selected_assignments)} assignments from {len(tasks_to_schedule)} tasks | "
            f"expected={alg_expected:.4f}, oracle={oracle_expected:.4f}, loss={step_loss:.4f}"
        )
        self.time += 1


# 示例运行
if __name__ == "__main__":
    # 初始化工人资源
    workers = [
        Worker(0, 25, 800, 3.0, 250, 1),
        Worker(1, 40, 400, 3.5, 300, 2),
        Worker(2, 5, 150, 2.5, 100, 0),
        # ...更多工人
    ]
    # 补充到 10 个工人示例
    for i in range(3, 10):
        workers.append(
            Worker(
                i,
                np.random.uniform(0, WORKER_FEATURE_VALUES_RANGE["driving_speed"][1]),
                np.random.uniform(0, WORKER_FEATURE_VALUES_RANGE["bandwidth"][1]),
                np.random.uniform(2, WORKER_FEATURE_VALUES_RANGE["processor_performance"][1]),
                np.random.uniform(100, WORKER_FEATURE_VALUES_RANGE["data_size"][1]),
                np.random.randint(0, WORKER_FEATURE_VALUES_RANGE["task_type"][1] + 1),
            )
        )

    normalizer = ContextNormalizer()
    # 注意：此处的构造参数名与实现保持一致（例如 partition_split_threshold）
    replicator = TaskReplicator(
        context_dim=2,
        partition_split_threshold=10,
        budget=1,
        replication_cost=0.1,
        max_partition_depth=MAX_PARTITION_DEPTH,
    )
    scheduler = Scheduler(workers, normalizer, replicator)
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)

    # 模拟任务流，多轮调度
    for step_i in range(300):
        # 模拟每步新任务到达：任务类型 0~9 随机，数据大小 100~3000MB，deadline 1~3 秒
        new_tasks: List[Task] = []
        for _ in range(np.random.randint(6, 16)):
            task_type = np.random.randint(0, 10)
            data_size = np.random.uniform(100, 3000)
            deadline = np.random.uniform(1, 3)
            new_tasks.append(Task(-1, task_type, data_size, deadline))  # task_id 自动分配

        scheduler.step(new_tasks, batch_size=10)  # 每轮调度最多 5 个任务

        if step_i % 10 == 0:
            print(f"--- After {step_i} steps ---")
            visualizer = PartitionVisualizer(replicator.partitions)
            # 保存分区可视化到文件，避免阻塞显示窗口
            os.makedirs("output", exist_ok=True)
            visualizer.plot_2d_partitions(
                dim_x=0,
                dim_y=1,
                iteration=step_i,
                save_path=f"output/partition_{step_i}.png",
            )

    # 所有步骤完成后，保存 loss 曲线到文件 output/loss.png
    try:
        import matplotlib.pyplot as plt

        os.makedirs("output", exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(LOSS_HISTORY)), LOSS_HISTORY, label="loss", color="tab:red")
        plt.title("Loss over Steps")
        plt.xlabel("Step")
        plt.ylabel("Loss")

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/loss.png", dpi=150)
        plt.close()
        print("Saved loss curve to output/loss.png")
    except Exception as e:
        print(f"Failed to save loss curve: {e}")
