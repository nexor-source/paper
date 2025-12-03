import numpy as np
import os
from collections import deque
from typing import List, Dict, Callable, Tuple, Optional
from matching_utils import run_hungarian_matching

from normalizer import ContextNormalizer
from task_replicator import Assignment, TaskReplicator
from visualizer import PartitionVisualizer, render_assignment_matrix
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
        enable_delayed_feedback: Optional[bool] = None,
        feedback_delay_range: Optional[Tuple[int, int]] = None,
        feedback_drop_prob: Optional[float] = None,
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
        cfg_enable_delay = bool(globals().get("ENABLE_DELAYED_FEEDBACK", False))
        self.enable_delayed_feedback = cfg_enable_delay if enable_delayed_feedback is None else bool(enable_delayed_feedback)
        cfg_delay_range = globals().get("FEEDBACK_DELAY_RANGE", (0, 0))
        if feedback_delay_range is None:
            if isinstance(cfg_delay_range, (list, tuple)) and len(cfg_delay_range) >= 2:
                self.feedback_delay_range = (int(cfg_delay_range[0]), int(cfg_delay_range[1]))
            else:
                self.feedback_delay_range = (0, 0)
        else:
            self.feedback_delay_range = (int(feedback_delay_range[0]), int(feedback_delay_range[1]))
        cfg_drop = float(globals().get("FEEDBACK_DROP_PROB", 0.0))
        self.feedback_drop_prob = float(cfg_drop if feedback_drop_prob is None else feedback_drop_prob)
        self.pending_feedback: List[Tuple[int, List[Assignment], Dict[Assignment, float]]] = []

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

    def _sample_feedback_delay(self) -> int:
        """Sample a feedback delay (in steps) when delayed feedback is enabled."""
        if not self.enable_delayed_feedback:
            return 0
        lo, hi = self.feedback_delay_range
        try:
            return int(np.random.randint(int(lo), int(hi) + 1))
        except Exception:
            return 0

    def _flush_pending_feedback(self) -> Tuple[int, int]:
        """Apply any feedback items whose ready_time <= current time."""
        if not self.pending_feedback:
            return 0, 0
        ready: List[Tuple[int, List[Assignment], Dict[Assignment, float]]] = []
        still_pending: List[Tuple[int, List[Assignment], Dict[Assignment, float]]] = []
        for item in self.pending_feedback:
            ready_time, assignments, rewards = item
            if ready_time <= self.time:
                ready.append(item)
            else:
                still_pending.append(item)
        self.pending_feedback = still_pending
        if not ready:
            return 0, 0

        merged_rewards: Dict[Assignment, float] = {}
        merged_assignments: List[Assignment] = []
        for _ready_time, assignments, rewards in ready:
            merged_assignments.extend(assignments)
            for a, r in rewards.items():
                merged_rewards[a] = r
        if merged_assignments:
            self.replicator.update_assignments_reward(merged_assignments, merged_rewards)
        return len(merged_assignments), len(ready)

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

        """模拟真实工人的离开、加入与属性漂移；使用全局 np.random。"""

        if not hasattr(self, "next_worker_id"):

            self.next_worker_id = (max((w.worker_id for w in self.workers), default=-1) + 1)



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



        keep_flags = [np.random.random() >= leave_prob for _ in self.workers]

        if any(keep_flags) is False and len(self.workers) > 0:

            keep_flags[np.random.randint(0, len(self.workers))] = True

        self.workers = [w for w, keep in zip(self.workers, keep_flags) if keep]



        n_join = 0
        if np.random.random() < join_prob:
            if join_hi >= join_lo and join_lo >= 0:
                n_join = int(np.random.randint(int(join_lo), int(join_hi) + 1))
        for _ in range(n_join):
            new_w = spawn_new_worker(self.next_worker_id)
            self.workers.append(new_w)
            self.next_worker_id += 1


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
        *,
        predict_net_fn: Optional[Callable[[Assignment], float]] = None,
        collect_details: bool = False,
    ) -> Dict[str, float]:
        """通用一步：自定义 selector 进行分配，可选更新模型，返回指标。"""
        # 1. 新任务 new_tasks 入队
        flushed_assignments, flushed_batches = self._flush_pending_feedback()
        for task in new_tasks:
            self.task_queue.add_task(task.task_type, task.data_size, task.deadline)

        # 2. 取 batch_size 大小的批任务准备调度
        tasks_to_schedule = self.task_queue.get_tasks_batch(batch_size)
        if not tasks_to_schedule:
            self.time += 1
            return {
                "loss": 0.0,
                "expected": 0.0,
                "oracle": 0.0,
                "realized_net": 0.0,
                "pending_feedback": len(self.pending_feedback),
                "flushed_feedback": int(flushed_assignments),
                "flushed_batches": int(flushed_batches),
            }

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
        alg_count = len(selected_assignments)
        oracle_count = len(oracle_assignments)
        alg_avg_expected = float(alg_expected / alg_count) if alg_count > 0 else 0.0
        oracle_avg_expected = float(oracle_expected / oracle_count) if oracle_count > 0 else 0.0

        pred_sel_sum = 0.0
        pred_sel_abs_sum = 0.0
        pred_sel_count = 0
        pred_all_sum = 0.0
        pred_all_abs_sum = 0.0
        pred_all_count = 0
        if predict_net_fn is not None and candidates:
            diff_map: Dict[Assignment, float] = {}
            for assignment in candidates:
                try:
                    pred_val = float(predict_net_fn(assignment))
                except Exception:
                    pred_val = float("nan")
                true_val = float(reward_fn(assignment.context) - rc)
                if np.isfinite(pred_val):
                    diff = pred_val - true_val
                    diff_map[assignment] = diff
                    pred_all_sum += diff
                    pred_all_abs_sum += abs(diff)
                    pred_all_count += 1
            if selected_assignments:
                for assignment in selected_assignments:
                    diff = diff_map.get(assignment)
                    if diff is None:
                        try:
                            pred_val = float(predict_net_fn(assignment))
                        except Exception:
                            continue
                        if not np.isfinite(pred_val):
                            continue
                        diff = pred_val - float(reward_fn(assignment.context) - rc)
                    pred_sel_sum += diff
                    pred_sel_abs_sum += abs(diff)
                    pred_sel_count += 1

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
            assignments_for_update = list(selected_assignments)
            rewards_for_update = dict(rewards)
            if self.enable_delayed_feedback:
                if self.feedback_drop_prob > 0.0:
                    kept_assignments: List[Assignment] = []
                    for a in assignments_for_update:
                        if np.random.random() < self.feedback_drop_prob:
                            continue
                        kept_assignments.append(a)
                    assignments_for_update = kept_assignments
                    rewards_for_update = {a: rewards_for_update[a] for a in assignments_for_update}
                if assignments_for_update:
                    delay_steps = self._sample_feedback_delay()
                    if delay_steps <= 0:
                        self.replicator.update_assignments_reward(assignments_for_update, rewards_for_update)
                    else:
                        self.pending_feedback.append(
                            (self.time + int(delay_steps), assignments_for_update, rewards_for_update)
                        )
            else:
                self.replicator.update_assignments_reward(assignments_for_update, rewards_for_update)

        self.time += 1
        inspection_payload = None
        if collect_details:
            inspection_payload = {
                "candidates": list(candidates),
                "selected": list(selected_assignments),
            }
        result = {
            "loss": float(step_loss),
            "expected": float(alg_expected),
            "oracle": float(oracle_expected),
            "realized_net": float(realized_net),
            "sel_workers": sorted({int(a.worker_id) for a in selected_assignments}),
            "sel_tasks": sorted({int(a.task_id) for a in selected_assignments}),
            "assignment_count": int(alg_count),
            "oracle_assignment_count": int(oracle_count),
            "expected_avg": float(alg_avg_expected),
            "oracle_avg": float(oracle_avg_expected),
        }
        result["pending_feedback"] = len(self.pending_feedback)
        result["flushed_feedback"] = int(flushed_assignments)
        result["flushed_batches"] = int(flushed_batches)
        if pred_sel_count > 0:
            result["pred_error_sum"] = float(pred_sel_sum)
            result["pred_error_abs_sum"] = float(pred_sel_abs_sum)
            result["pred_error_count"] = float(pred_sel_count)
            result["pred_error_mean"] = float(pred_sel_sum / pred_sel_count)
            result["pred_error_abs_mean"] = float(pred_sel_abs_sum / pred_sel_count)
        if pred_all_count > 0:
            result["pred_error_all_sum"] = float(pred_all_sum)
            result["pred_error_all_abs_sum"] = float(pred_all_abs_sum)
            result["pred_error_all_count"] = float(pred_all_count)
            result["pred_error_all_mean"] = float(pred_all_sum / pred_all_count)
            result["pred_error_all_abs_mean"] = float(pred_all_abs_sum / pred_all_count)
        if inspection_payload is not None:
            result["inspection"] = inspection_payload
        return result

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
    - 预生成共享任务流（Ours/Random/Greedy/Oracle 共用，同一分布、同一顺序）；
    - 分别运行 Ours、Random、Greedy、Oracle 四种策略并对比：
      Ours 在循环内每 10 步输出一次上下文划分可视化 `output/partition_{step}.png`；
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

    for i in range(3, 50):
        base_workers.append(spawn_new_worker(i))

    normalizer = ContextNormalizer()

    steps = int(globals().get("COMPARISON_STEPS", 300))
    batch_size = int(globals().get("COMPARISON_BATCH_SIZE", 10))
    arrivals_min, arrivals_max = globals().get("ARRIVALS_PER_STEP", (6, 16))

    inspection_steps_cfg = globals().get("ASSIGNMENT_INSPECTION_STEPS", None)
    inspection_steps: List[int] = []
    if inspection_steps_cfg is None:
        inspection_count = int(globals().get("ASSIGNMENT_INSPECTION_COUNT", 0))
        if inspection_count > 0 and steps > 0:
            seed = int(globals().get("ASSIGNMENT_INSPECTION_SEED", RANDOM_SEED + 7))
            rng_inspection = np.random.default_rng(seed)
            sample_size = min(int(inspection_count), steps)
            sampled = rng_inspection.choice(steps, size=sample_size, replace=False)
            sampled_array = np.atleast_1d(sampled)
            inspection_steps = sorted(int(x) for x in sampled_array.tolist())
    else:
        if isinstance(inspection_steps_cfg, (list, tuple, set)):
            candidates = [int(v) for v in inspection_steps_cfg]
        else:
            candidates = [int(inspection_steps_cfg)]
        inspection_steps = sorted({int(s) for s in candidates if 0 <= int(s) < steps})

    inspection_dir_name = str(globals().get("ASSIGNMENT_INSPECTION_DIR", "assignment_inspections"))
    inspection_dir = os.path.join("output", inspection_dir_name)
    if inspection_steps:
        print(f"[inspection] Capturing assignment grids at steps: {inspection_steps}")

    def _maybe_render_inspection(
        method_label: str,
        step_idx: int,
        scheduler_inst: "Scheduler",
        predict_fn: Callable[[Assignment], float],
        payload: Optional[Dict[str, List[Assignment]]],
    ) -> None:
        if not inspection_steps or payload is None:
            return
        candidates = payload.get("candidates", [])
        if not candidates:
            return
        selected_assignments = payload.get("selected", [])
        task_ids = sorted({int(a.task_id) for a in candidates})
        worker_ids = sorted({int(a.worker_id) for a in candidates})

        predicted_map: Dict[Tuple[int, int], float] = {}
        for a in candidates:
            key = (int(a.worker_id), int(a.task_id))
            try:
                predicted_map[key] = float(predict_fn(a))
            except Exception as exc:
                print(f"[inspection][{method_label}] failed to compute predicted net for {key}: {exc}")
                predicted_map[key] = float("nan")

        rc_val = float(scheduler_inst.replicator.replication_cost)
        true_map: Dict[Tuple[int, int], float] = {}
        for a in candidates:
            key = (int(a.worker_id), int(a.task_id))
            try:
                expected = float(scheduler_inst.evaluate_reward_complex(a.context) - rc_val)
            except Exception as exc:
                print(f"[inspection][{method_label}] failed to compute true net for {key}: {exc}")
                expected = float("nan")
            true_map[key] = expected

        selected_pairs = {(int(a.worker_id), int(a.task_id)) for a in selected_assignments}
        safe_name = method_label.replace(" ", "_").lower()
        save_path = os.path.join(inspection_dir, f"{safe_name}_step_{step_idx:04d}.png")
        try:
            render_assignment_matrix(
                method_name=method_label,
                step_index=step_idx,
                task_ids=task_ids,
                worker_ids=worker_ids,
                predicted_net=predicted_map,
                true_net=true_map,
                selected_pairs=selected_pairs,
                save_path=save_path,
            )
        except Exception as exc:
            print(f"[inspection][{method_label}] failed to render matrix at step {step_idx}: {exc}")

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
    COLOR_MAP = {
        "Ours-Immediate": "#1f77b4",
        "Ours-Delayed": "#ff7f0e",
        "Random": "#2ca02c",
        "Greedy": "#d62728",
        "Oracle": "#9467bd",
    }
    def _c(label: str) -> str:
        return COLOR_MAP.get(label, "#444444")

    def run_original(method_label: str = "Ours", enable_delay: Optional[bool] = None) -> Tuple[List[float], List[float], List[float], List[int], List[float], List[float], List[float], int, List[int], List[int]]:
        """Run the main method; enable_delay overrides config when set."""
        workers = _clone_workers(base_workers)
        replicator = TaskReplicator(
            context_dim=7,
            partition_split_threshold=PARTITION_SPLIT_THRESHOLD,
            budget=1,
            replication_cost=REPLICATION_COST,
            max_partition_depth=MAX_PARTITION_DEPTH,
        )
        enable_delay_flag = bool(globals().get("ENABLE_DELAYED_FEEDBACK", False)) if enable_delay is None else bool(enable_delay)
        scheduler = Scheduler(
            workers,
            normalizer,
            replicator,
            enable_worker_dynamics=bool(globals().get("ENABLE_WORKER_DYNAMICS_COMPARISON", False)),
            enable_delayed_feedback=enable_delay_flag,
        )
        if worker_timeline is not None:
            scheduler.enable_worker_dynamics = False
        loss_c, cum_c, cum_exp_c, assign_counts = [], [], [], []
        avg_loss_series: List[float] = []
        pred_error_mean_series: List[float] = []
        pred_error_abs_series: List[float] = []
        pending_series: List[int] = []
        flushed_series: List[int] = []
        cum = 0.0
        cum_exp = 0.0
        pred_error_sel_sum = 0.0
        pred_error_sel_abs_sum = 0.0
        pred_error_sel_count = 0.0
        pred_error_all_sum = 0.0
        pred_error_all_abs_sum = 0.0
        pred_error_all_count = 0.0
        np.random.seed(RANDOM_SEED)

        def predict_net(a: Assignment) -> float:
            try:
                return float(replicator.assignment_net(a, include_ucb=False))
            except AttributeError:
                partition = replicator.root_partition.find_partition(a.context)
                return float(partition.posterior_mean() - replicator.replication_cost)

        for s in range(steps):
            if worker_timeline is not None:
                scheduler.workers = _clone_workers(worker_timeline[s])
            collect_details = s in inspection_steps
            res = scheduler.step_with_selector(
                task_stream[s],
                batch_size,
                lambda cands, _e: scheduler.replicator.select_assignments(cands, allow_unmatch=True, use_ucb=True),
                update_model=True,
                predict_net_fn=predict_net,
                collect_details=collect_details,
            )
            loss_c.append(res["loss"])
            cum += res["realized_net"]
            cum_c.append(cum)
            cum_exp += float(res.get("expected", 0.0))
            cum_exp_c.append(cum_exp)
            assign_counts.append(int(res.get("assignment_count", len(res.get("sel_tasks", [])))))
            avg_alg = float(res.get("expected_avg", 0.0))
            avg_orc = float(res.get("oracle_avg", 0.0))
            avg_loss_series.append(float(avg_alg - avg_orc))
            pred_error_mean_series.append(float(res.get("pred_error_all_mean", res.get("pred_error_mean", np.nan))))
            pred_error_abs_series.append(float(res.get("pred_error_all_abs_mean", res.get("pred_error_abs_mean", np.nan))))
            pending_series.append(int(res.get("pending_feedback", 0)))
            flushed_series.append(int(res.get("flushed_feedback", 0)))
            if "pred_error_count" in res:
                pred_error_sel_sum += float(res.get("pred_error_sum", 0.0))
                pred_error_sel_abs_sum += float(res.get("pred_error_abs_sum", 0.0))
                pred_error_sel_count += float(res.get("pred_error_count", 0.0))
            if "pred_error_all_count" in res:
                pred_error_all_sum += float(res.get("pred_error_all_sum", 0.0))
                pred_error_all_abs_sum += float(res.get("pred_error_all_abs_sum", 0.0))
                pred_error_all_count += float(res.get("pred_error_all_count", 0.0))
            if collect_details:
                payload = res.get("inspection")
                _maybe_render_inspection(method_label, s, scheduler, predict_net, payload)
            if s % 50 == 0:
                try:
                    visualizer = PartitionVisualizer(replicator.partitions)
                    os.makedirs("output", exist_ok=True)
                    part_dir = os.path.join("output", "partitions")
                    os.makedirs(part_dir, exist_ok=True)
                    visualizer.plot_2d_partitions(
                        dim_x=0,
                        dim_y=1,
                        iteration=s,
                        save_path=os.path.join(part_dir, f"partition_{s}.png"),
                    )
                except Exception as _e:
                    print(f"[viz] failed to render partition at step {s}: {_e}")
        split_events = replicator.split_events
        if pred_error_all_count > 0:
            avg_err_all = float(pred_error_all_sum / pred_error_all_count)
            avg_abs_err_all = float(pred_error_all_abs_sum / pred_error_all_count)
            print("[prediction-bias][{}][all] mean={:.4f}, mean_abs={:.4f}, samples={}".format(
                method_label,
                avg_err_all,
                avg_abs_err_all,
                int(pred_error_all_count),
            ))
        if pred_error_sel_count > 0:
            avg_err_sel = float(pred_error_sel_sum / pred_error_sel_count)
            avg_abs_err_sel = float(pred_error_sel_abs_sum / pred_error_sel_count)
            print("[prediction-bias][{}][selected] mean={:.4f}, mean_abs={:.4f}, samples={}".format(
                method_label,
                avg_err_sel,
                avg_abs_err_sel,
                int(pred_error_sel_count),
            ))
        return (
            loss_c,
            cum_c,
            cum_exp_c,
            assign_counts,
            pred_error_mean_series,
            pred_error_abs_series,
            avg_loss_series,
            split_events,
            pending_series,
            flushed_series,
        )

    def run_with_selector(
        selector_factory,
        *,
        method_label: str,
        update_model: bool = False,
        use_oracle_eval: bool = True,
        predict_net_builder: Optional[Callable[["Scheduler", TaskReplicator], Callable[[Assignment], float]]] = None,
    ) -> Tuple[List[float], List[float], List[float], List[int], List[float], List[float], List[float], int]:
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
        loss_c, cum_c, cum_exp_c, assign_counts = [], [], [], []
        avg_loss_series: List[float] = []
        pred_error_mean_series: List[float] = []
        pred_error_abs_series: List[float] = []
        cum = 0.0
        cum_exp = 0.0
        pred_error_sel_sum = 0.0
        pred_error_sel_abs_sum = 0.0
        pred_error_sel_count = 0.0
        pred_error_all_sum = 0.0
        pred_error_all_abs_sum = 0.0
        pred_error_all_count = 0.0
        np.random.seed(RANDOM_SEED)
        eval_fn = scheduler.evaluate_reward_complex if use_oracle_eval else None

        track_pred_error = predict_net_builder is not None
        if predict_net_builder is not None:
            predict_fn = predict_net_builder(scheduler, replicator)
        else:
            def predict_fn(a: Assignment) -> float:
                return float(scheduler.evaluate_reward_complex(a.context) - scheduler.replicator.replication_cost)

        for s in range(steps):
            if worker_timeline is not None:
                scheduler.workers = _clone_workers(worker_timeline[s])
            collect_details = s in inspection_steps
            res = scheduler.step_with_selector(
                task_stream[s],
                batch_size,
                selector,
                update_model=update_model,
                eval_net_fn=eval_fn,
                predict_net_fn=predict_fn if track_pred_error else None,
                collect_details=collect_details,
            )
            loss_c.append(res["loss"])
            cum += res["realized_net"]
            cum_c.append(cum)
            cum_exp += float(res.get("expected", 0.0))
            cum_exp_c.append(cum_exp)
            assign_counts.append(int(res.get("assignment_count", len(res.get("sel_tasks", [])))))
            avg_alg = float(res.get("expected_avg", 0.0))
            avg_orc = float(res.get("oracle_avg", 0.0))
            avg_loss_series.append(float(avg_alg - avg_orc))
            pred_error_mean_series.append(float(res.get("pred_error_all_mean", res.get("pred_error_mean", np.nan))))
            pred_error_abs_series.append(float(res.get("pred_error_all_abs_mean", res.get("pred_error_abs_mean", np.nan))))
            if track_pred_error and "pred_error_count" in res:
                pred_error_sel_sum += float(res.get("pred_error_sum", 0.0))
                pred_error_sel_abs_sum += float(res.get("pred_error_abs_sum", 0.0))
                pred_error_sel_count += float(res.get("pred_error_count", 0.0))
            if track_pred_error and "pred_error_all_count" in res:
                pred_error_all_sum += float(res.get("pred_error_all_sum", 0.0))
                pred_error_all_abs_sum += float(res.get("pred_error_all_abs_sum", 0.0))
                pred_error_all_count += float(res.get("pred_error_all_count", 0.0))
            if collect_details:
                payload = res.get("inspection")
                _maybe_render_inspection(method_label, s, scheduler, predict_fn, payload)
        split_events = replicator.split_events
        if track_pred_error and pred_error_all_count > 0:
            avg_err_all = float(pred_error_all_sum / pred_error_all_count)
            avg_abs_err_all = float(pred_error_all_abs_sum / pred_error_all_count)
            print("[prediction-bias][{}][all] mean={:.4f}, mean_abs={:.4f}, samples={}".format(
                method_label,
                avg_err_all,
                avg_abs_err_all,
                int(pred_error_all_count),
            ))
        if track_pred_error and pred_error_sel_count > 0:
            avg_err_sel = float(pred_error_sel_sum / pred_error_sel_count)
            avg_abs_err_sel = float(pred_error_sel_abs_sum / pred_error_sel_count)
            print("[prediction-bias][{}][selected] mean={:.4f}, mean_abs={:.4f}, samples={}".format(
                method_label,
            avg_err_sel,
            avg_abs_err_sel,
            int(pred_error_sel_count),
        ))
        return loss_c, cum_c, cum_exp_c, assign_counts, pred_error_mean_series, pred_error_abs_series, avg_loss_series, split_events

    # Baselines
    loss_r, cum_r, cum_er, assign_r, _pred_err_r, _pred_err_abs_r, avg_loss_r, _split_r = run_with_selector(
        lambda _rep: RandomBaseline().select,
        method_label="Random",
        update_model=False,
        use_oracle_eval=False,
    )
    loss_g, cum_g, cum_eg, assign_g, pred_err_g, pred_err_abs_g, avg_loss_g, split_g = run_with_selector(
        # 传入一个 lambda，调用 rep.select_assignments 并设置 use_ucb=False
        lambda rep: lambda cands, _e: rep.select_assignments(cands, allow_unmatch=True, use_ucb=False),
        method_label="Greedy",
        update_model=True,
        use_oracle_eval=False,
        # 确保预测值也不带 UCB
        predict_net_builder=lambda sched, rep: (
            lambda a: float(rep.assignment_net(a, include_ucb=False)) # 这里的预测已经是不带UCB的了
        ),
    )
    # Oracle policy (for cumulative reward plot)
    def run_oracle() -> Tuple[List[float], List[float], List[float], List[int], List[float], int]:
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
        loss_c, cum_c, cum_exp_c, assign_counts = [], [], [], []
        avg_loss_series: List[float] = []
        cum = 0.0
        cum_exp = 0.0
        np.random.seed(RANDOM_SEED)

        def predict_net(a: Assignment) -> float:
            return float(scheduler.evaluate_reward_complex(a.context) - scheduler.replicator.replication_cost)

        for s in range(steps):
            if worker_timeline is not None:
                scheduler.workers = _clone_workers(worker_timeline[s])
            collect_details = s in inspection_steps
            res = scheduler.step_with_selector(
                task_stream[s],
                batch_size,
                lambda cands, _e: scheduler._oracle_select_assignments(cands),
                update_model=False,
                collect_details=collect_details,
            )
            loss_c.append(res["loss"])  # should be ~0 for oracle
            cum += res["realized_net"]
            cum_c.append(cum)
            cum_exp += float(res.get("oracle", 0.0))
            cum_exp_c.append(cum_exp)
            assign_counts.append(int(res.get("assignment_count", len(res.get("sel_tasks", [])))))
            avg_alg = float(res.get("expected_avg", 0.0))
            avg_orc = float(res.get("oracle_avg", 0.0))
            avg_loss_series.append(float(avg_alg - avg_orc))
            if collect_details:
                payload = res.get("inspection")
                _maybe_render_inspection("Oracle", s, scheduler, predict_net, payload)
        split_events = replicator.split_events
        return loss_c, cum_c, cum_exp_c, assign_counts, avg_loss_series, split_events

    run_delayed_variant = bool(globals().get("RUN_DELAYED_FEEDBACK_VARIANT", True))
    loss_o, cum_o, cum_eo, assign_o, pred_err_o, pred_err_abs_o, avg_loss_o, split_o, pending_o, flushed_o = run_original(
        method_label="Ours-Immediate",
        enable_delay=False,
    )
    loss_od: List[float] = []
    cum_od: List[float] = []
    cum_eod: List[float] = []
    assign_od: List[int] = []
    pred_err_od: List[float] = []
    pred_err_abs_od: List[float] = []
    avg_loss_od: List[float] = []
    split_od = 0
    pending_od: List[int] = []
    flushed_od: List[int] = []
    if run_delayed_variant:
        (
            loss_od,
            cum_od,
            cum_eod,
            assign_od,
            pred_err_od,
            pred_err_abs_od,
            avg_loss_od,
            split_od,
            pending_od,
            flushed_od,
        ) = run_original(method_label="Ours-Delayed", enable_delay=True)
    loss_orc, cum_orc, cum_eorc, assign_orc, avg_loss_orc, split_orc = run_oracle()

    print("[partition-splits] Ours-Immediate:", int(split_o))
    if run_delayed_variant:
        print("[partition-splits] Ours-Delayed :", int(split_od))
    print("[partition-splits] Greedy :", int(split_g))
    print("[partition-splits] Oracle :", int(split_orc))

    # (debug prints removed)

    plt.figure(figsize=(9, 4))
    plt.plot(loss_o, label="Ours-Immediate", linewidth=1.0, alpha=0.7)
    if run_delayed_variant and loss_od:
        plt.plot(loss_od, label="Ours-Delayed", linewidth=1.0, alpha=0.7)
    plt.plot(loss_r, label="Random", linewidth=1.0, alpha=0.7)
    plt.plot(loss_g, label="Greedy", linewidth=1.0, alpha=0.7)
    plt.title("Loss Comparison (raw)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/compare_loss.png", dpi=150)
    plt.close()

    # Additionally provide a smoothed view with rolling mean and quantile band
    def _rolling_stats(arr, window=50, qlo=0.1, qhi=0.9):
        arr = np.asarray(arr, dtype=float)
        n = len(arr)
        window = max(1, min(window, n))
        if window == 1:
            return arr, None, None
        # rolling mean via convolution
        kernel = np.ones(window, dtype=float) / float(window)
        mean = np.convolve(arr, kernel, mode='valid')
        # rolling quantiles (simple loop keeps it clear and small n)
        lo, hi = [], []
        for i in range(0, n - window + 1):
            w = arr[i:i+window]
            lo.append(float(np.quantile(w, qlo)))
            hi.append(float(np.quantile(w, qhi)))
        return mean, np.asarray(lo), np.asarray(hi)

    smooth_win = int(globals().get("LOSS_SMOOTH_WINDOW", 50))
    qlo = float(globals().get("LOSS_SMOOTH_QLO", 0.1))
    qhi = float(globals().get("LOSS_SMOOTH_QHI", 0.9))
    stride = int(globals().get("LOSS_DOWNSAMPLE", 1))
    stride = max(1, stride)

    def _prep(series):
        m, l, h = _rolling_stats(series, window=smooth_win, qlo=qlo, qhi=qhi)
        x = np.arange(len(m))
        if stride > 1:
            x = x[::stride]
            m = m[::stride]
            if l is not None and h is not None:
                l = l[::stride]
                h = h[::stride]
        return x, m, l, h

    xo, mo, lo_b, hi_b = _prep(loss_o)
    xr, mr, lr_b, hr_b = _prep(loss_r)
    xg, mg, lg_b, hg_b = _prep(loss_g)
    xod = mod = lod_b = hid_b = None
    if run_delayed_variant and loss_od:
        xod, mod, lod_b, hid_b = _prep(loss_od)

    plt.figure(figsize=(9, 4))
    if lo_b is not None and hi_b is not None:
        plt.fill_between(xo, lo_b, hi_b, color=_c("Ours-Immediate"), alpha=0.12)
    plt.plot(xo, mo, label="Ours-Immediate (mean)", color=_c("Ours-Immediate"), linewidth=2.0)
    if run_delayed_variant and mod is not None and lod_b is not None and hid_b is not None:
        plt.fill_between(xod, lod_b, hid_b, color=_c("Ours-Delayed"), alpha=0.12)
    if run_delayed_variant and mod is not None:
        plt.plot(xod, mod, label="Ours-Delayed (mean)", color=_c("Ours-Delayed"), linewidth=2.0)

    if lr_b is not None and hr_b is not None:
        plt.fill_between(xr, lr_b, hr_b, color=_c("Random"), alpha=0.12)
    plt.plot(xr, mr, label="Random (mean)", color=_c("Random"), linewidth=2.0)

    if lg_b is not None and hg_b is not None:
        plt.fill_between(xg, lg_b, hg_b, color=_c("Greedy"), alpha=0.12)
    plt.plot(xg, mg, label="Greedy (mean)", color=_c("Greedy"), linewidth=2.0)

    plt.title(f"Loss (rolling mean, window={smooth_win})")
    plt.xlabel("Step (offset by window)")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/compare_loss_smooth.png", dpi=150)
    plt.close()

    # Average loss per assignment (raw)
    plt.figure(figsize=(9, 4))
    plt.plot(avg_loss_o, label="Ours-Immediate", linewidth=1.0, alpha=0.7)
    if run_delayed_variant and avg_loss_od:
        plt.plot(avg_loss_od, label="Ours-Delayed", linewidth=1.0, alpha=0.7)
    plt.plot(avg_loss_r, label="Random", linewidth=1.0, alpha=0.7)
    plt.plot(avg_loss_g, label="Greedy", linewidth=1.0, alpha=0.7)
    plt.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.4)
    plt.title("Average Loss per Assignment (raw)")
    plt.xlabel("Step")
    plt.ylabel("Avg loss (method - oracle)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/compare_avg_loss.png", dpi=150)
    plt.close()

    # Smoothed average loss
    xo_avg, mo_avg, lo_avg_b, hi_avg_b = _prep(avg_loss_o)
    xr_avg, mr_avg, lr_avg_b, hr_avg_b = _prep(avg_loss_r)
    xg_avg, mg_avg, lg_avg_b, hg_avg_b = _prep(avg_loss_g)
    xod_avg = mod_avg = lod_avg_b = hid_avg_b = None
    if run_delayed_variant and avg_loss_od:
        xod_avg, mod_avg, lod_avg_b, hid_avg_b = _prep(avg_loss_od)

    plt.figure(figsize=(9, 4))
    if lo_avg_b is not None and hi_avg_b is not None:
        plt.fill_between(xo_avg, lo_avg_b, hi_avg_b, color=_c("Ours-Immediate"), alpha=0.12)
    plt.plot(xo_avg, mo_avg, label="Ours-Immediate (mean)", color=_c("Ours-Immediate"), linewidth=2.0)
    if run_delayed_variant and mod_avg is not None and lod_avg_b is not None and hid_avg_b is not None:
        plt.fill_between(xod_avg, lod_avg_b, hid_avg_b, color=_c("Ours-Delayed"), alpha=0.12)
    if run_delayed_variant and mod_avg is not None:
        plt.plot(xod_avg, mod_avg, label="Ours-Delayed (mean)", color=_c("Ours-Delayed"), linewidth=2.0)

    if lr_avg_b is not None and hr_avg_b is not None:
        plt.fill_between(xr_avg, lr_avg_b, hr_avg_b, color=_c("Random"), alpha=0.12)
    plt.plot(xr_avg, mr_avg, label="Random (mean)", color=_c("Random"), linewidth=2.0)

    if lg_avg_b is not None and hg_avg_b is not None:
        plt.fill_between(xg_avg, lg_avg_b, hg_avg_b, color=_c("Greedy"), alpha=0.12)
    plt.plot(xg_avg, mg_avg, label="Greedy (mean)", color=_c("Greedy"), linewidth=2.0)

    plt.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.4)
    plt.title(f"Average Loss per Assignment (rolling mean, window={smooth_win})")
    plt.xlabel("Step (offset by window)")
    plt.ylabel("Avg loss (method - oracle)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/compare_avg_loss_smooth.png", dpi=150)
    plt.close()

    err_o = np.asarray(pred_err_o, dtype=float)
    err_g = np.asarray(pred_err_g, dtype=float)
    err_abs_o = np.asarray(pred_err_abs_o, dtype=float)
    err_abs_g = np.asarray(pred_err_abs_g, dtype=float)
    err_od = np.asarray(pred_err_od, dtype=float) if run_delayed_variant else np.asarray([])
    err_abs_od = np.asarray(pred_err_abs_od, dtype=float) if run_delayed_variant else np.asarray([])
    steps_axis = np.arange(len(err_o))
    steps_axis_g = np.arange(len(err_g))
    steps_axis_od = np.arange(len(err_od))

    plt.figure(figsize=(9, 4))
    plt.plot(steps_axis, err_o, label="Ours-Immediate", color=_c("Ours-Immediate"), linewidth=1.0, alpha=0.85)
    if run_delayed_variant and err_od.size > 0:
        plt.plot(steps_axis_od, err_od, label="Ours-Delayed", color=_c("Ours-Delayed"), linewidth=1.0, alpha=0.85)
    plt.plot(steps_axis_g, err_g, label="Greedy", color=_c("Greedy"), linewidth=1.0, alpha=0.85)
    plt.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.4)
    plt.title("Prediction Bias (Signed Error on Selected Assignments)")
    plt.xlabel("Step")
    plt.ylabel("Predicted net - true net")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/pred_error_mean.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(steps_axis, err_abs_o, label="Ours-Immediate", color=_c("Ours-Immediate"), linewidth=1.0, alpha=0.85)
    if run_delayed_variant and err_abs_od.size > 0:
        plt.plot(steps_axis_od, err_abs_od, label="Ours-Delayed", color=_c("Ours-Delayed"), linewidth=1.0, alpha=0.85)
    plt.plot(steps_axis_g, err_abs_g, label="Greedy", color=_c("Greedy"), linewidth=1.0, alpha=0.85)
    plt.title("Prediction Error Magnitude on Selected Assignments")
    plt.xlabel("Step")
    plt.ylabel("|Predicted net - true net|")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/pred_error_abs.png", dpi=150)
    plt.close()

    # Smoothed absolute prediction error with the same rolling window stats as loss
    x_abs_o, mean_abs_o, lo_abs_o, hi_abs_o = _prep(err_abs_o)
    x_abs_g, mean_abs_g, lo_abs_g, hi_abs_g = _prep(err_abs_g)
    x_abs_od = mean_abs_od = lo_abs_od = hi_abs_od = None
    if run_delayed_variant and err_abs_od.size > 0:
        x_abs_od, mean_abs_od, lo_abs_od, hi_abs_od = _prep(err_abs_od)

    plt.figure(figsize=(9, 4))
    if lo_abs_o is not None and hi_abs_o is not None and len(lo_abs_o) > 0:
        plt.fill_between(x_abs_o, lo_abs_o, hi_abs_o, color=_c("Ours-Immediate"), alpha=0.12)
    plt.plot(x_abs_o, mean_abs_o, label="Ours-Immediate (mean)", color=_c("Ours-Immediate"), linewidth=2.0)
    if run_delayed_variant and lo_abs_od is not None and hi_abs_od is not None and len(lo_abs_od) > 0:
        plt.fill_between(x_abs_od, lo_abs_od, hi_abs_od, color=_c("Ours-Delayed"), alpha=0.12)
    if run_delayed_variant and mean_abs_od is not None:
        plt.plot(x_abs_od, mean_abs_od, label="Ours-Delayed (mean)", color=_c("Ours-Delayed"), linewidth=2.0)

    if lo_abs_g is not None and hi_abs_g is not None and len(lo_abs_g) > 0:
        plt.fill_between(x_abs_g, lo_abs_g, hi_abs_g, color=_c("Greedy"), alpha=0.12)
    plt.plot(x_abs_g, mean_abs_g, label="Greedy (mean)", color=_c("Greedy"), linewidth=2.0)

    plt.title(f"|Prediction Error| (rolling mean, window={smooth_win})")
    plt.xlabel("Step (offset by window)")
    plt.ylabel("|Predicted net - true net|")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/pred_error_abs_smooth.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(cum_o, label="Ours-Immediate")
    if run_delayed_variant and cum_od:
        plt.plot(cum_od, label="Ours-Delayed")
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

    def _cumulative_regret(method_cum, oracle_cum):
        method_arr = np.asarray(method_cum, dtype=float)
        oracle_arr = np.asarray(oracle_cum, dtype=float)
        if method_arr.size == 0 or oracle_arr.size == 0:
            return np.asarray([])
        n = min(method_arr.size, oracle_arr.size)
        return oracle_arr[:n] - method_arr[:n]

    regret_o = _cumulative_regret(cum_o, cum_orc)
    regret_r = _cumulative_regret(cum_r, cum_orc)
    regret_g = _cumulative_regret(cum_g, cum_orc)
    regret_od = _cumulative_regret(cum_od, cum_orc) if run_delayed_variant and cum_od else np.asarray([])

    if regret_o.size > 0:
        plt.figure(figsize=(9, 4))
        plt.plot(np.arange(len(regret_o)), regret_o, label="Ours-Immediate", linewidth=1.6)
        if regret_od.size > 0:
            plt.plot(np.arange(len(regret_od)), regret_od, label="Ours-Delayed", linewidth=1.6, linestyle=':')
        if regret_r.size > 0:
            plt.plot(np.arange(len(regret_r)), regret_r, label="Random", linewidth=1.0, alpha=0.85)
        if regret_g.size > 0:
            plt.plot(np.arange(len(regret_g)), regret_g, label="Greedy", linewidth=1.0, alpha=0.85)
        plt.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.4)
        plt.title("Cumulative Regret (Oracle - Realized Method Reward)")
        plt.xlabel("Step")
        plt.ylabel("Cumulative Regret")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/compare_cum_regret.png", dpi=150)
        plt.close()

    # New: Expected cumulative net reward (sum of expected net per step)
    plt.figure(figsize=(9, 4))
    # 使用不同的线型，避免重叠时“看成三条线”的错觉
    plt.plot(cum_eo, label="Ours-Immediate", color=_c("Ours-Immediate"), linestyle='-', linewidth=2.0, alpha=0.95, zorder=3)
    if run_delayed_variant and cum_eod:
        plt.plot(cum_eod, label="Ours-Delayed", color=_c("Ours-Delayed"), linestyle=':', linewidth=2.0, alpha=0.95, zorder=2.5)
    plt.plot(cum_er, label="Random",   color=_c("Random"), linestyle='--', linewidth=2.0, alpha=0.95, zorder=2)
    plt.plot(cum_eg, label="Greedy",    color=_c("Greedy"), linestyle='-.', linewidth=2.0, alpha=0.95, zorder=4)
    plt.plot(cum_eorc, label="Oracle",  color=_c("Oracle"), linestyle='-', linewidth=2.5, alpha=0.95, zorder=5)
    plt.title("Expected Cumulative Net Reward")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Expected Reward (sum(E[r]-cost))")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/compare_cum_expected.png", dpi=150)
    plt.close()

    regret_eo = _cumulative_regret(cum_eo, cum_eorc)
    regret_er = _cumulative_regret(cum_er, cum_eorc)
    regret_eg = _cumulative_regret(cum_eg, cum_eorc)
    regret_eod = _cumulative_regret(cum_eod, cum_eorc) if run_delayed_variant and cum_eod else np.asarray([])

    if regret_eo.size > 0:
        plt.figure(figsize=(9, 4))
        plt.plot(np.arange(len(regret_eo)), regret_eo, label="Ours-Immediate", color=_c("Ours-Immediate"), linewidth=2.0, alpha=0.9)
        if regret_eod.size > 0:
            plt.plot(np.arange(len(regret_eod)), regret_eod, label="Ours-Delayed", color=_c("Ours-Delayed"), linestyle=':', linewidth=1.8, alpha=0.9)
        if regret_er.size > 0:
            plt.plot(np.arange(len(regret_er)), regret_er, label="Random", color=_c("Random"), linestyle='--', linewidth=1.6, alpha=0.9)
        if regret_eg.size > 0:
            plt.plot(np.arange(len(regret_eg)), regret_eg, label="Greedy", color=_c("Greedy"), linestyle='-.', linewidth=1.6, alpha=0.9)
        plt.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.4)
        plt.title("Cumulative Expected Regret")
        plt.xlabel("Step")
        plt.ylabel("Cumulative Expected Regret")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/compare_cum_expected_regret.png", dpi=150)
        plt.close()

    if run_delayed_variant and pending_od:
        plt.figure(figsize=(9, 4))
        plt.plot(pending_od, label="Pending feedback count (delayed)", color=_c("Ours-Delayed"), linewidth=1.6)
        if flushed_od:
            plt.plot(flushed_od, label="Applied feedback per step", color=_c("Ours-Immediate"), linestyle='--', linewidth=1.2, alpha=0.8)
        plt.title("Delayed Feedback Queue Dynamics")
        plt.xlabel("Step")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/delayed_feedback_queue.png", dpi=150)
        plt.close()

    # Print average expected net per selected assignment to help tune REPLICATION_COST
    def _avg_expected(cum_e_seq, assign_seq):
        total = float(cum_e_seq[-1]) if cum_e_seq else 0.0
        nsel = int(sum(assign_seq)) if assign_seq else 0
        avg = (total / nsel) if nsel > 0 else 0.0
        return avg, total, nsel

    avg_o, tot_o, n_o = _avg_expected(cum_eo, assign_o)
    avg_od, tot_od, n_od = _avg_expected(cum_eod, assign_od) if run_delayed_variant else (0.0, 0.0, 0)
    avg_r, tot_r, n_r = _avg_expected(cum_er, assign_r)
    avg_g, tot_g, n_g = _avg_expected(cum_eg, assign_g)
    avg_orc, tot_orc, n_orc = _avg_expected(cum_eorc, assign_orc)

    print("[avg-expected-net] per selected assignment:")
    print(f"  Ours-Immediate: {avg_o:.4f} (total={tot_o:.2f}, selected={n_o})")
    if run_delayed_variant:
        print(f"  Ours-Delayed   : {avg_od:.4f} (total={tot_od:.2f}, selected={n_od})")
    print(f"  Random  : {avg_r:.4f} (total={tot_r:.2f}, selected={n_r})")
    print(f"  Greedy  : {avg_g:.4f} (total={tot_g:.2f}, selected={n_g})")
    print(f"  Oracle  : {avg_orc:.4f} (total={tot_orc:.2f}, selected={n_orc})")
    print("Saved comparison plots including delayed-feedback variant and queue dynamics.")

    return


if __name__ == "__main__":
    if bool(globals().get("RUN_COMPARISON", False)):
        run_experiment()
    else:
        print("RUN_COMPARISON 为 False。请修改 config.py 中的 RUN_COMPARISON = True 后再运行本脚本。")
