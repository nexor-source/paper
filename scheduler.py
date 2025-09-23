import numpy as np
import os
from collections import deque
from typing import List, Dict, Callable, Tuple, Optional
from matching_utils import run_hungarian_matching

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




def spawn_new_worker(worker_id: int, rng: Optional[np.random.Generator] = None) -> Worker:
    """Sample a new Worker using WORKER_FEATURE_VALUES_RANGE."""
    ranges = WORKER_FEATURE_VALUES_RANGE

    if rng is None:
        uniform = np.random.uniform
        randint_fn = np.random.randint
    else:
        uniform = rng.uniform
        if hasattr(rng, 'integers'):
            randint_fn = rng.integers
        else:
            randint_fn = rng.randint

    ds_min, ds_max = ranges.get('driving_speed', (0.0, 0.0))
    bw_min, bw_max = ranges.get('bandwidth', (0.0, 0.0))
    pp_min, pp_max = ranges.get('processor_performance', (1.0, 1.0))
    pd_min, pd_max = ranges.get('physical_distance', (0.0, 0.0))
    weather_min, weather_max = ranges.get('weather', (0, 0))

    return Worker(
        worker_id,
        float(uniform(ds_min, ds_max)),
        float(uniform(bw_min, bw_max)),
        float(uniform(pp_min, pp_max)),
        float(uniform(pd_min, pd_max)),
        int(randint_fn(int(weather_min), int(weather_max) + 1)),
    )


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
                # 需要保证 raw_context 必!须!是!这!个!顺!序! 以匹配 evaluate_reward_complex 函数使用 index 来读取特征
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
        """Oracle 版本（允许不匹配）——最小化版：
        - 最大化真实净收益 p(context) - replication_cost  等价为  最小化成本 = -net
        - 一对一约束 + 可空配对（虚拟节点，净值=0，任务/工人两边都可不配）
        """
        if not candidate_assignments:
            return []

        # 索引
        task_ids = sorted({a.task_id for a in candidate_assignments})
        worker_ids = sorted({a.worker_id for a in candidate_assignments})
        t_idx = {t: i for i, t in enumerate(task_ids)}
        w_idx = {w: j for j, w in enumerate(worker_ids)}
        m, n = len(task_ids), len(worker_ids)
        EPS = 1e-12

        # 仅给候选边赋值；非候选设为 -inf（最大化语义下不可行，稍后转为 +inf 成本）
        profits = np.full((m, n), -np.inf, dtype=float)

        # O(1) 回找
        pair2a = {}

        for a in candidate_assignments:
            i, j = t_idx[a.task_id], w_idx[a.worker_id]
            p = self.evaluate_reward_complex(a.context)
            net = float(p - self.replicator.replication_cost)
            profits[i, j] = net
            pair2a[(a.task_id, a.worker_id)] = a

        selected, _row_ind, _col_ind = run_hungarian_matching(
            task_ids,
            worker_ids,
            profits,
            pair2a,
            allow_unmatch=True,
            eps=EPS,
        )
        return selected



    def _apply_worker_dynamics(self) -> None:
        """模拟实现工人的离开，新增和属性漂移（使用全局 np.random）。"""
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
            new_w = spawn_new_worker(self.next_worker_id)
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
        return float(sum(self.evaluate_reward_complex(a.context) - rc for a in assignments))

    def evaluate_reward_simple(self, context: np.ndarray) -> float:
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


    def evaluate_reward_complex(self, context: np.ndarray) -> float:
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

        # 特征积极部分：边际效益递减(平方根) + 相互作用
        pos_basic = (
            0.35 * np.sqrt(driving_speed + eps)
            + 0.35 * np.sqrt(bandwidth + eps)
            + 0.20 * np.sqrt(processor_perf + eps)
        )
        pos_synergy = (
            0.18 * np.sqrt((driving_speed * bandwidth) + eps)
            + 0.10 * np.sqrt((bandwidth * processor_perf) + eps)
        )

        # 负面部分：斜率递增惩罚项
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

    def step_with_selector(
        self,
        new_tasks: List[Task],
        batch_size: int,
        selector: Callable[[List[Assignment], Callable[[Assignment], float]], List[Assignment]],
        update_model: bool = False,
        eval_net_fn: Optional[Callable[[np.ndarray], float]] = None,
    ) -> Dict[str, float]:
        """通用一步：自定义 selector 进行分配，可选更新模型，返回指标。"""
        # 1. 新任务 new_tasks 入队
        for task in new_tasks:
            self.task_queue.add_task(task.task_type, task.data_size, task.deadline)

        # 2. 取 batch_size 大小的批任务准备调度
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
        reward_fn = eval_net_fn if eval_net_fn is not None else self.evaluate_reward_complex

        def eval_net(a: Assignment) -> float:
            return float(reward_fn(a.context) - rc)

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
            p = self.evaluate_reward_complex(a.context)
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
            "assignment_count": len(selected_assignments),
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


# 预生成工人动态时间线：给定初始工人与步数，离线模拟每一步的工人集合
def _generate_worker_timeline(base_workers: List["Worker"], steps: int) -> List[List["Worker"]]:
    rng = np.random.default_rng(int(globals().get("RANDOM_SEED", 42)))
    # 读取配置
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

    def clip(v: float, lo: float, hi: float) -> float:
        return float(min(max(v, lo), hi))

    ranges = WORKER_FEATURE_VALUES_RANGE

    workers: List[Worker] = _clone_workers(base_workers)
    next_worker_id = (max((w.worker_id for w in workers), default=-1) + 1)

    timeline: List[List[Worker]] = []
    for _s in range(steps):
        # 1) 离开
        keep_flags = [rng.random() >= leave_prob for _ in workers]
        if any(keep_flags) is False and len(workers) > 0:
            keep_flags[rng.integers(0, len(workers))] = True
        workers = [w for w, keep in zip(workers, keep_flags) if keep]

        # 2) 加入
        n_join = 0
        if rng.random() < join_prob:
            if join_hi >= join_lo and join_lo >= 0:
                n_join = int(rng.integers(int(join_lo), int(join_hi) + 1))
        for _ in range(n_join):
            workers.append(spawn_new_worker(next_worker_id, rng))
            next_worker_id += 1

        # 3) 漂移
        for w in workers:
            if "driving_speed" in drift_frac and "driving_speed" in ranges:
                lo, hi = ranges["driving_speed"]
                std = float(drift_frac["driving_speed"]) * (hi - lo)
                w.driving_speed = clip(w.driving_speed + float(rng.normal(0.0, std)), lo, hi)
            if "bandwidth" in drift_frac and "bandwidth" in ranges:
                lo, hi = ranges["bandwidth"]
                std = float(drift_frac["bandwidth"]) * (hi - lo)
                w.bandwidth = clip(w.bandwidth + float(rng.normal(0.0, std)), lo, hi)
            if "processor_performance" in drift_frac and "processor_performance" in ranges:
                lo, hi = ranges["processor_performance"]
                std = float(drift_frac["processor_performance"]) * (hi - lo)
                w.processor_perf = clip(w.processor_perf + float(rng.normal(0.0, std)), lo, hi)
            if "physical_distance" in drift_frac and "physical_distance" in ranges:
                lo, hi = ranges["physical_distance"]
                std = float(drift_frac["physical_distance"]) * (hi - lo)
                w.physical_distance = clip(w.physical_distance + float(rng.normal(0.0, std)), lo, hi)
            if rng.random() < weather_change_prob:
                max_w = int(ranges.get("weather", (0, 4))[1])
                w.weather = int(rng.integers(0, max_w + 1))

        # 存储本步快照
        timeline.append(_clone_workers(workers))

    return timeline

# 若启用对比实验，则优先运行并提前退出，避免执行下方旧版主流程
def run_experiment() -> None:
    """统一的主实验入口。

    - 初始化随机种子与输出目录；
    - 构建基础工人集与 ContextNormalizer；
    - 预生成共享任务流（Original/Random/Greedy/Oracle 共用，同一分布、同一顺序）；
    - 分别运行 Original、Random、Greedy、Oracle 四种策略并对比：
      Original 在循环内每 10 步输出一次上下文划分可视化 `output/partition_{step}.png`；
    - 最后保存对比图到 `output/compare_loss.png` 与 `output/compare_cum_reward.png`。

    相关配置见 config.py：RANDOM_SEED、COMPARISON_STEPS、COMPARISON_BATCH_SIZE、
    ARRIVALS_PER_STEP、ENABLE_WORKER_DYNAMICS_COMPARISON、MAX_PARTITION_DEPTH。
    """
    # 固定随机种子，并确保输出目录存在
    np.random.seed(RANDOM_SEED)
    os.makedirs("output", exist_ok=True)

    debug_counts = bool(globals().get("DEBUG_ASSIGNMENT_COUNTS", True))

    # 基础工人集
    base_workers: List[Worker] = []
    # base_workers: List[Worker] = [
    #     # 固定的三个样例工人
    #     # id, speed, bw, cpu, distance, weather
    #     Worker(0, 25, 800, 3.0, 250, 1),
    #     Worker(1, 40, 400, 3.5, 300, 2),
    #     Worker(2, 5, 150, 2.5, 100, 0),
    # ]

    for i in range(3, 50):
        base_workers.append(spawn_new_worker(i))

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

    # 预生成工人时间线（如启用），确保各策略面向同一“世界线”
    worker_timeline = None
    if bool(globals().get("ENABLE_WORKER_DYNAMICS_COMPARISON", False)) and bool(globals().get("USE_PREGENERATED_WORKER_TIMELINE", True)):
        worker_timeline = _generate_worker_timeline(base_workers, steps)

    from baselines import RandomBaseline, GreedyBaseline
    import matplotlib.pyplot as plt

    def run_original() -> Tuple[List[float], List[float], List[int]]:
        workers = _clone_workers(base_workers)
        replicator = TaskReplicator(
            context_dim=7,
            partition_split_threshold=PARTITION_SPLIT_THRESHOLD,
            budget=1,
            replication_cost=REPLICATION_COST,
            max_partition_depth=MAX_PARTITION_DEPTH,
        )
        scheduler = Scheduler(
            workers,
            normalizer,
            replicator,
            enable_worker_dynamics=bool(globals().get("ENABLE_WORKER_DYNAMICS_COMPARISON", False)),
        )
        if worker_timeline is not None:
            scheduler.enable_worker_dynamics = False
        loss_c, cum_c, assign_counts = [], [], []
        cum = 0.0
        np.random.seed(RANDOM_SEED)
        for s in range(steps):
            if worker_timeline is not None:
                scheduler.workers = _clone_workers(worker_timeline[s])
            res = scheduler.step_with_selector(
                task_stream[s],
                batch_size,
                lambda cands, _e: scheduler.replicator.select_assignments(cands, allow_unmatch=True),
                update_model=True,
            )
            loss_c.append(res["loss"])
            cum += res["realized_net"]
            cum_c.append(cum)
            assign_counts.append(int(res.get("assignment_count", len(res.get("sel_tasks", [])))))
            if s % 50 == 0:
                try:
                    visualizer = PartitionVisualizer(replicator.partitions)
                    os.makedirs("output", exist_ok=True)
                    visualizer.plot_2d_partitions(
                        dim_x=0,
                        dim_y=1,
                        iteration=s,
                        save_path=f"output/partition_{s}.png",
                    )
                except Exception as _e:
                    print(f"[viz] failed to render partition at step {s}: {_e}")
        return loss_c, cum_c, assign_counts

    def run_with_selector(
        selector_factory,
        *,
        update_model: bool = False,
        use_oracle_eval: bool = True,
    ) -> Tuple[List[float], List[float], List[int]]:
        """Run a baseline selector (e.g. RandomBaseline/GreedyBaseline)."""
        workers = _clone_workers(base_workers)
        replicator = TaskReplicator(
            context_dim=7,
            partition_split_threshold=PARTITION_SPLIT_THRESHOLD,
            budget=1,
            replication_cost=REPLICATION_COST,
            max_partition_depth=MAX_PARTITION_DEPTH,
        )
        scheduler = Scheduler(
            workers,
            normalizer,
            replicator,
            enable_worker_dynamics=bool(globals().get("ENABLE_WORKER_DYNAMICS_COMPARISON", False)),
        )
        if worker_timeline is not None:
            scheduler.enable_worker_dynamics = False
        selector = selector_factory(replicator)
        loss_c, cum_c, assign_counts = [], [], []
        cum = 0.0
        np.random.seed(RANDOM_SEED)
        eval_fn = scheduler.evaluate_reward_complex if use_oracle_eval else None
        for s in range(steps):
            if worker_timeline is not None:
                scheduler.workers = _clone_workers(worker_timeline[s])
            res = scheduler.step_with_selector(
                task_stream[s],
                batch_size,
                selector,
                update_model=update_model,
                eval_net_fn=eval_fn,
            )
            loss_c.append(res["loss"])
            cum += res["realized_net"]
            cum_c.append(cum)
            assign_counts.append(int(res.get("assignment_count", len(res.get("sel_tasks", [])))))
        return loss_c, cum_c, assign_counts

    # Baselines
    loss_r, cum_r, assign_r = run_with_selector(
        lambda _rep: RandomBaseline().select,
        update_model=False,
        use_oracle_eval=False,
    )
    loss_g, cum_g, assign_g = run_with_selector(
        lambda rep: GreedyBaseline(rep).select,
        update_model=True,
        use_oracle_eval=False,
    )
    # Oracle policy (for cumulative reward plot)
    def run_oracle() -> Tuple[List[float], List[float], List[int]]:
        """Oracle（带虚拟节点，允许不匹配）用于提供上界参考。

        使用真实成功概率评估净收益，不进行学习更新，理论上本策略的 loss 约为 0。
        返回每步 loss 以及累计净收益曲线。
        """
        workers = _clone_workers(base_workers)
        replicator = TaskReplicator(
            context_dim=7,
            partition_split_threshold=PARTITION_SPLIT_THRESHOLD,
            budget=1,
            replication_cost=REPLICATION_COST,
            max_partition_depth=MAX_PARTITION_DEPTH,
        )
        scheduler = Scheduler(
            workers,
            normalizer,
            replicator,
            enable_worker_dynamics=bool(globals().get("ENABLE_WORKER_DYNAMICS_COMPARISON", False)),
        )
        if worker_timeline is not None:
            scheduler.enable_worker_dynamics = False
        loss_c, cum_c, assign_counts = [], [], []
        cum = 0.0
        np.random.seed(RANDOM_SEED)
        for s in range(steps):
            if worker_timeline is not None:
                scheduler.workers = _clone_workers(worker_timeline[s])
            res = scheduler.step_with_selector(
                task_stream[s],
                batch_size,
                lambda cands, _e: scheduler._oracle_select_assignments(cands),
                update_model=False,
            )
            loss_c.append(res["loss"])  # should be ~0 for oracle
            cum += res["realized_net"]
            cum_c.append(cum)
            assign_counts.append(int(res.get("assignment_count", len(res.get("sel_tasks", [])))))
        return loss_c, cum_c, assign_counts

    loss_o, cum_o, assign_o = run_original()
    loss_orc, cum_orc, assign_orc = run_oracle()

    if debug_counts:
        assignment_counts = {
            "Original": assign_o,
            "Random": assign_r,
            "Greedy": assign_g,
            "Oracle": assign_orc,
        }
        base_counts = assignment_counts["Original"]
        print("[assign-count] preview (first 10 steps):")
        for name, seq in assignment_counts.items():
            preview = seq[:10]
            print(f"  {name:<8}: {preview}")
        for name, seq in assignment_counts.items():
            if name == "Original":
                continue
            if len(seq) != len(base_counts):
                print(f"[assign-count] {name} length mismatch: {len(seq)} vs {len(base_counts)}")
                continue
            mismatches = [i for i, (a, b) in enumerate(zip(base_counts, seq)) if a != b]
            if mismatches:
                first = mismatches[0]
                print(f"[assign-count] {name} differs at {len(mismatches)} steps; first mismatch step {first}: Original={base_counts[first]}, {name}={seq[first]}")
            else:
                print(f"[assign-count] {name} matches Original for all {len(base_counts)} steps")

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

    return


if __name__ == "__main__":
    if bool(globals().get("RUN_COMPARISON", False)):
        run_experiment()
    else:
        print("RUN_COMPARISON 为 False。请修改 config.py 中的 RUN_COMPARISON = True 后再运行本脚本。")
