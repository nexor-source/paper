import numpy as np
import os
from collections import deque
from typing import List, Dict, Callable, Tuple
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
        enable_worker_dynamics: bool = True,
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
        self.enable_worker_dynamics = enable_worker_dynamics
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
                # Keep order aligned with evaluate_reward2 indices
                raw_context = {
                    "driving_speed": float(worker.driving_speed),
                    "bandwidth": float(worker.bandwidth),
                    "processor_performance": float(worker.processor_perf),
                    "physical_distance": float(worker.physical_distance),
                    "task_type": int(task.task_type),
                    "data_size": float(task.data_size),
                    "weather": int(worker.weather),
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
            p = self.evaluate_reward2(a.context)
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

    def _apply_worker_dynamics(self) -> None:
        """模拟工人来去与能力漂移（使用全局 np.random，配合固定种子可复现）。"""
        # 保障 next_worker_id 存在
        if not hasattr(self, "next_worker_id"):
            self.next_worker_id = (max((w.worker_id for w in self.workers), default=-1) + 1)

        # 读取配置（若缺省则采用兜底值）
        try:
            dynamics = WORKER_DYNAMICS
        except NameError:
            dynamics = {
                "leave_prob": 0.05,
                "join_prob": 0.10,
                "join_count_range": (0, 2),
                "drift_frac": {
                    "driving_speed": 0.03,
                    "bandwidth": 0.05,
                    "processor_performance": 0.02,
                    "physical_distance": 0.05,
                },
                "weather_change_prob": 0.03,
            }

        leave_prob = float(dynamics.get("leave_prob", 0.05))
        join_prob = float(dynamics.get("join_prob", 0.10))
        join_lo, join_hi = dynamics.get("join_count_range", (0, 2))
        drift_frac = dynamics.get("drift_frac", {})
        weather_change_prob = float(dynamics.get("weather_change_prob", 0.03))

        # 1) 工人离开（逐个伯努利），至少保留 1 个
        keep_flags = []
        for _w in self.workers:
            keep_flags.append(np.random.random() >= leave_prob)
        if any(keep_flags) is False and len(self.workers) > 0:
            keep_flags[np.random.randint(0, len(self.workers))] = True
        self.workers = [w for w, keep in zip(self.workers, keep_flags) if keep]

        # 2) 工人新增（按概率触发，数量在区间内均匀采样）
        n_join = 0
        if np.random.random() < join_prob:
            if join_hi >= join_lo and join_lo >= 0:
                n_join = int(np.random.randint(join_lo, join_hi + 1))
        for _ in range(n_join):
            ds_min, ds_max = WORKER_FEATURE_VALUES_RANGE.get("driving_speed", (0.0, 40.0))
            bw_min, bw_max = WORKER_FEATURE_VALUES_RANGE.get("bandwidth", (0.0, 1000.0))
            pp_min, pp_max = WORKER_FEATURE_VALUES_RANGE.get("processor_performance", (1.0, 5.0))
            pd_min, pd_max = WORKER_FEATURE_VALUES_RANGE.get("physical_distance", (0.0, 1000.0))
            weather_max = int(WORKER_FEATURE_VALUES_RANGE.get("weather", (0, 4))[1])

            new_w = Worker(
                self.next_worker_id,
                float(np.random.uniform(ds_min, ds_max)),
                float(np.random.uniform(bw_min, bw_max)),
                float(np.random.uniform(pp_min, pp_max)),
                float(np.random.uniform(pd_min, pd_max)),
                int(np.random.randint(0, weather_max + 1)),
            )
            self.workers.append(new_w)
            self.next_worker_id += 1

        # 3) 属性缓慢漂移（高斯噪声按范围比例），裁剪到范围内
        def clip(v: float, lo: float, hi: float) -> float:
            return float(min(max(v, lo), hi))

        ranges = WORKER_FEATURE_VALUES_RANGE
        for w in self.workers:
            if "driving_speed" in drift_frac and "driving_speed" in ranges:
                lo, hi = ranges["driving_speed"]
                std = float(drift_frac["driving_speed"]) * (hi - lo)
                w.driving_speed = clip(w.driving_speed + float(np.random.normal(0.0, std)), lo, hi)
            if "bandwidth" in drift_frac and "bandwidth" in ranges:
                lo, hi = ranges["bandwidth"]
                std = float(drift_frac["bandwidth"]) * (hi - lo)
                w.bandwidth = clip(w.bandwidth + float(np.random.normal(0.0, std)), lo, hi)
            if "processor_performance" in drift_frac and "processor_performance" in ranges:
                lo, hi = ranges["processor_performance"]
                std = float(drift_frac["processor_performance"]) * (hi - lo)
                w.processor_perf = clip(w.processor_perf + float(np.random.normal(0.0, std)), lo, hi)
            if "physical_distance" in drift_frac and "physical_distance" in ranges:
                lo, hi = ranges["physical_distance"]
                std = float(drift_frac["physical_distance"]) * (hi - lo)
                w.physical_distance = clip(w.physical_distance + float(np.random.normal(0.0, std)), lo, hi)
            # weather：小概率变化为任意类别
            if np.random.random() < weather_change_prob:
                max_w = int(ranges.get("weather", (0, 4))[1])
                w.weather = int(np.random.randint(0, max_w + 1))

    def _expected_total_reward(self, assignments: List[Assignment]) -> float:
        """基于真实成功概率的期望总净收益: sum(p(context) - replication_cost)."""
        if not assignments:
            return 0.0
        rc = self.replicator.replication_cost
        return float(sum(self.evaluate_reward2(a.context) - rc for a in assignments))

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


    def evaluate_reward2(self, context: np.ndarray) -> float:
        """根据 context 向量模拟得到【成功概率】，但是更复杂且连续的函数

        Args:
            context (np.ndarray): 归一化上下文向量，shape=(d,)
        Returns:
            float: 成功概率 p ∈ [0,1]
        Notes:
            设计细节：
            特征将分为正向和负向两类，正向特征考虑平方根的边际效用递减，
            负向特征考虑凸函数 (^1.5) 的惩罚，同时正向特征之间考虑交互作用（几何平均）。
            最终通过平滑门控（sigmoid）反映短板效应。

        """
        # Extract features (normalized to [0,1]); missing dims treated as 0
        driving_speed = float(context[0]) if len(context) > 0 else 0.0
        bandwidth = float(context[1]) if len(context) > 1 else 0.0
        processor_perf = float(context[2]) if len(context) > 2 else 0.0
        distance = float(context[3]) if len(context) > 3 else 0.0
        # task_type = float(context[4]) if len(context) > 4 else 0.0
        data_size = float(context[5]) if len(context) > 5 else 0.0
        weather = float(context[6]) if len(context) > 6 else 0.0

        eps = 1e-6

        # Positive part: diminishing returns + interactions
        pos_basic = (
            0.35 * np.sqrt(driving_speed + eps)
            + 0.35 * np.sqrt(bandwidth + eps)
            + 0.20 * np.sqrt(processor_perf + eps)
        )
        pos_synergy = (
            0.18 * np.sqrt((driving_speed * bandwidth) + eps)
            + 0.10 * np.sqrt((bandwidth * processor_perf) + eps)
        )

        # Negative part: convex penalties
        neg = (
            0.25 * (distance ** 1.5)
            + 0.20 * (data_size ** 1.2)
            + 0.10 * (weather ** 1.2)
        )

        # Smooth gating (soft-AND). Do not gate on missing dims.
        def gate_for(x: float, thr: float = 0.2, k: float = 6.0) -> float:
            return 1.0 / (1.0 + np.exp(-k * (x - thr)))

        gates = [gate_for(driving_speed), gate_for(bandwidth)]
        if len(context) > 2:
            gates.append(gate_for(processor_perf))
        gate = float(np.prod(gates)) if gates else 1.0

        raw = (pos_basic + pos_synergy) * gate - neg
        p = 1.0 / (1.0 + np.exp(-3.0 * raw))
        return float(np.clip(p, 0.01, 0.99))

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

        # 2.5 工人动态（离开/新增/属性漂移）
        self._apply_worker_dynamics()
        candidates = self.generate_candidate_assignments(tasks_to_schedule)
        # 3. 生成所有候选工人-任务对
        candidates = self.generate_candidate_assignments(tasks_to_schedule)
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
            p = self.evaluate_reward2(a.context)
            # 模拟成功与否
            rewards[a] = np.random.binomial(1, p)

        # 6. 更新统计
        self.replicator.update_assignments_reward(selected_assignments, rewards)

        print(
            f"Time {self.time}: Scheduled {len(selected_assignments)} assignments from {len(tasks_to_schedule)} tasks | "
            f"expected={alg_expected:.4f}, oracle={oracle_expected:.4f}, loss={step_loss:.4f}"
        )
        self.time += 1

    def step_with_selector(
        self,
        new_tasks: List[Task],
        batch_size: int,
        selector: Callable[[List[Assignment], Callable[[Assignment], float]], List[Assignment]],
        update_model: bool = False,
    ) -> Dict[str, float]:
        """通用一步：自定义 selector 进行分配，可选更新模型，返回指标。"""
        # 1. 新任务入队
        for task in new_tasks:
            self.task_queue.add_task(task.task_type, task.data_size, task.deadline)

        # 2. 取批任务准备调度
        tasks_to_schedule = self.task_queue.get_tasks_batch(batch_size)
        if not tasks_to_schedule:
            self.time += 1
            return {"loss": 0.0, "expected": 0.0, "oracle": 0.0, "realized_net": 0.0}

        # 2.5 工人动态（可开关）
        if getattr(self, "enable_worker_dynamics", True):
            self._apply_worker_dynamics()

        # 3. 候选
        candidates = self.generate_candidate_assignments(tasks_to_schedule)

        rc = float(self.replicator.replication_cost)

        def eval_net(a: Assignment) -> float:
            return float(self.evaluate_reward2(a.context) - rc)

        # 4. 选择
        selected_assignments = selector(candidates, eval_net)

        # 4.1 Oracle 与 loss
        oracle_assignments = self._oracle_select_assignments(candidates)
        alg_expected = self._expected_total_reward(selected_assignments)
        oracle_expected = self._expected_total_reward(oracle_assignments)
        step_loss = max(0.0, oracle_expected - alg_expected)

        # 5. 模拟执行与奖励
        realized_net = 0.0
        rewards: Dict[Assignment, float] = {}
        for a in selected_assignments:
            p = self.evaluate_reward2(a.context)
            r = float(np.random.binomial(1, p))
            rewards[a] = r
            realized_net += (r - rc)

        # 6. 可选模型更新
        if update_model and selected_assignments:
            self.replicator.update_assignments_reward(selected_assignments, rewards)

        self.time += 1
        return {
            "loss": float(step_loss),
            "expected": float(alg_expected),
            "oracle": float(oracle_expected),
            "realized_net": float(realized_net),
            "sel_workers": sorted({int(a.worker_id) for a in selected_assignments}),
            "sel_tasks": sorted({int(a.task_id) for a in selected_assignments}),
        }

def _clone_workers(workers: List["Worker"]) -> List["Worker"]:
    return [
        Worker(
            w.worker_id,
            w.driving_speed,
            w.bandwidth,
            w.processor_perf,
            w.physical_distance,
            w.weather,
        )
        for w in workers
    ]


# 若启用对比实验，则优先运行并提前退出，避免执行下方旧版主流程
if __name__ == "__main__" and bool(globals().get("RUN_COMPARISON", False)):
    np.random.seed(RANDOM_SEED)
    os.makedirs("output", exist_ok=True)

    # 基础工人集
    base_workers: List[Worker] = [
        # 固定的三个样例工人
        # id, speed, bw, cpu, distance, weather
        Worker(0, 25, 800, 3.0, 250, 1),
        Worker(1, 40, 400, 3.5, 300, 2),
        Worker(2, 5, 150, 2.5, 100, 0),
    ]
    for i in range(3, 10):
        base_workers.append(
            Worker(
                i,
                float(np.random.uniform(0, WORKER_FEATURE_VALUES_RANGE["driving_speed"][1])),
                float(np.random.uniform(0, WORKER_FEATURE_VALUES_RANGE["bandwidth"][1])),
                float(np.random.uniform(WORKER_FEATURE_VALUES_RANGE["processor_performance"][0], WORKER_FEATURE_VALUES_RANGE["processor_performance"][1])),
                float(np.random.uniform(0, WORKER_FEATURE_VALUES_RANGE["physical_distance"][1])),
                int(np.random.randint(0, WORKER_FEATURE_VALUES_RANGE["weather"][1] + 1)),
            )
        )

    normalizer = ContextNormalizer()

    steps = int(globals().get("COMPARISON_STEPS", 300))
    batch_size = int(globals().get("COMPARISON_BATCH_SIZE", 10))
    arrivals_min, arrivals_max = globals().get("ARRIVALS_PER_STEP", (6, 16))

    # 预生成任务流，三种算法共享
    task_stream: List[List[Task]] = []
    for _ in range(steps):
        new_tasks: List[Task] = []
        n_new = int(np.random.randint(arrivals_min, arrivals_max))
        for _k in range(n_new):
            task_type = int(np.random.randint(0, 10))
            data_size = float(np.random.uniform(100, 3000))
            deadline = float(np.random.uniform(1, 3))
            new_tasks.append(Task(-1, task_type, data_size, deadline))
        task_stream.append(new_tasks)

    from baselines import RandomBaseline, GreedyBaseline
    import matplotlib.pyplot as plt

    def run_original() -> Tuple[List[float], List[float]]:
        workers = _clone_workers(base_workers)
        replicator = TaskReplicator(
            context_dim=7,
            partition_split_threshold=10,
            budget=1,
            replication_cost=0.1,
            max_partition_depth=MAX_PARTITION_DEPTH,
        )
        scheduler = Scheduler(
            workers,
            normalizer,
            replicator,
            enable_worker_dynamics=bool(globals().get("ENABLE_WORKER_DYNAMICS_COMPARISON", False)),
        )
        loss_c, cum_c = [], []
        cum = 0.0
        np.random.seed(RANDOM_SEED)
        for s in range(steps):
            res = scheduler.step_with_selector(
                task_stream[s],
                batch_size,
                lambda cands, _e: scheduler.replicator.select_assignments(cands),
                update_model=True,
            )
            loss_c.append(res["loss"])
            cum += res["realized_net"]
            cum_c.append(cum)
        return loss_c, cum_c

    def run_with_selector(sel) -> Tuple[List[float], List[float]]:
        workers = _clone_workers(base_workers)
        replicator = TaskReplicator(
            context_dim=7,
            partition_split_threshold=10,
            budget=1,
            replication_cost=0.1,
            max_partition_depth=MAX_PARTITION_DEPTH,
        )
        scheduler = Scheduler(
            workers,
            normalizer,
            replicator,
            enable_worker_dynamics=bool(globals().get("ENABLE_WORKER_DYNAMICS_COMPARISON", False)),
        )
        loss_c, cum_c = [], []
        cum = 0.0
        np.random.seed(RANDOM_SEED)
        for s in range(steps):
            res = scheduler.step_with_selector(task_stream[s], batch_size, sel, update_model=False)
            loss_c.append(res["loss"])
            cum += res["realized_net"]
            cum_c.append(cum)
        return loss_c, cum_c

    # Baselines
    rand_selector = RandomBaseline()
    rand_fn = lambda c, e: rand_selector.select(c, e)
    greedy_selector = GreedyBaseline()
    greedy_fn = lambda c, e: greedy_selector.select(c, e)

    # Oracle policy (for cumulative reward plot)
    def run_oracle() -> Tuple[List[float], List[float]]:
        workers = _clone_workers(base_workers)
        replicator = TaskReplicator(
            context_dim=7,
            partition_split_threshold=10,
            budget=1,
            replication_cost=0.1,
            max_partition_depth=MAX_PARTITION_DEPTH,
        )
        scheduler = Scheduler(
            workers,
            normalizer,
            replicator,
            enable_worker_dynamics=bool(globals().get("ENABLE_WORKER_DYNAMICS_COMPARISON", False)),
        )
        loss_c, cum_c = [], []
        cum = 0.0
        np.random.seed(RANDOM_SEED)
        for s in range(steps):
            res = scheduler.step_with_selector(
                task_stream[s],
                batch_size,
                lambda cands, _e: scheduler._oracle_select_assignments(cands),
                update_model=False,
            )
            loss_c.append(res["loss"])  # should be ~0 for oracle
            cum += res["realized_net"]
            cum_c.append(cum)
        return loss_c, cum_c

    loss_o, cum_o = run_original()
    loss_r, cum_r = run_with_selector(rand_fn)
    loss_g, cum_g = run_with_selector(greedy_fn)
    loss_orc, cum_orc = run_oracle()

    plt.figure(figsize=(9, 4))
    plt.plot(loss_o, label="Original")
    plt.plot(loss_r, label="Random")
    plt.plot(loss_g, label="Greedy")
    plt.title("Loss Comparison")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/compare_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(cum_o, label="Original")
    plt.plot(cum_r, label="Random")
    plt.plot(cum_g, label="Greedy")
    plt.plot(cum_orc, label="Oracle")
    plt.title("Cumulative Net Reward")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward (sum(reward - cost))")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/compare_cum_reward.png", dpi=150)
    plt.close()
    print("Saved loss (Original/Random) and cumulative reward (Original/Random/Oracle) plots.")

    import sys as _sys
    _sys.exit(0)



# 示例运行
if __name__ == "__main__":
    # 固定随机种子，保证模拟可复现
    np.random.seed(RANDOM_SEED)
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
