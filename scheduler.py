import numpy as np
import os
from collections import deque
from typing import List, Dict, Callable, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matching_utils import run_hungarian_matching

from normalizer import ContextNormalizer
from task_replicator import Assignment, TaskReplicator
from visualizer import PartitionVisualizer, render_assignment_matrix
from config import *

USE_CN = bool(globals().get("PLOT_USE_CHINESE", True))

def _t(cn: str, en: str) -> str:
    return cn if USE_CN else en

def _tf(cn: str, en: str, **kwargs) -> str:
    template = cn if USE_CN else en
    return template.format(**kwargs)

def _configure_chinese_font() -> None:
    candidates = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Source Han Sans SC", "PingFang SC", "WenQuanYi Zen Hei",
    ]
    if USE_CN:
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                break
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["figure.titlesize"] = 13

_configure_chinese_font()

class Task:
    def __init__(self, task_id: int, task_type: int, data_size: float, deadline: float):
        self.task_id = task_id
        self.task_type = task_type
        self.data_size = data_size
        self.deadline = deadline

class Worker:
    def __init__(self, worker_id: int, driving_speed: float, bandwidth: float, processor_perf: float, physical_distance: float, weather: int):
        self.worker_id = worker_id
        self.driving_speed = driving_speed
        self.bandwidth = bandwidth
        self.processor_perf = processor_perf
        self.physical_distance = physical_distance
        self.weather = weather

def spawn_new_worker(worker_id: int, rng: Optional[np.random.Generator] = None) -> Worker:
    ranges = WORKER_FEATURE_VALUES_RANGE
    if rng is None:
        uniform = np.random.uniform
        randint_fn = np.random.randint
    else:
        uniform = rng.uniform
        randint_fn = rng.integers if hasattr(rng, 'integers') else rng.randint

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
    def __init__(self):
        self.queue = deque()
        self.next_task_id = 0

    def add_task(self, task_type: int, data_size: float, deadline: float) -> None:
        task = Task(self.next_task_id, task_type, data_size, deadline)
        self.queue.append(task)
        self.next_task_id += 1

    def get_tasks_batch(self, batch_size: int) -> List[Task]:
        tasks: List[Task] = []
        for _ in range(min(batch_size, len(self.queue))):
            tasks.append(self.queue.popleft())
        return tasks

class Scheduler:
    def __init__(self, workers: List[Worker], context_normalizer: ContextNormalizer, replicator: TaskReplicator, enable_worker_dynamics: bool = True, enable_delayed_feedback: Optional[bool] = None, feedback_delay_range: Optional[Tuple[int, int]] = None, feedback_drop_prob: Optional[float] = None):
        self.workers = workers
        self.normalizer = context_normalizer
        self.replicator = replicator
        self.enable_worker_dynamics = enable_worker_dynamics
        self.task_queue = TaskQueue()
        self.time = 0  
        cfg_enable_delay = bool(globals().get("ENABLE_DELAYED_FEEDBACK", False))
        self.enable_delayed_feedback = cfg_enable_delay if enable_delayed_feedback is None else bool(enable_delayed_feedback)
        cfg_delay_range = globals().get("FEEDBACK_DELAY_RANGE", (0, 0))
        if feedback_delay_range is None:
            self.feedback_delay_range = (int(cfg_delay_range[0]), int(cfg_delay_range[1])) if isinstance(cfg_delay_range, (list, tuple)) and len(cfg_delay_range) >= 2 else (0, 0)
        else:
            self.feedback_delay_range = (int(feedback_delay_range[0]), int(feedback_delay_range[1]))
        cfg_drop = float(globals().get("FEEDBACK_DROP_PROB", 0.0))
        self.feedback_drop_prob = float(cfg_drop if feedback_drop_prob is None else feedback_drop_prob)
        self.pending_feedback: List[Tuple[int, List[Assignment], Dict[Assignment, float]]] = []

    def generate_candidate_assignments(self, tasks: List[Task]) -> List[Assignment]:
        candidates: List[Assignment] = []
        for task in tasks:
            for worker in self.workers:
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
                candidates.append(Assignment(worker.worker_id, task.task_id, norm_context))
        return candidates

    def _sample_feedback_delay(self) -> int:
        if not self.enable_delayed_feedback: return 0
        lo, hi = self.feedback_delay_range
        return int(np.random.randint(int(lo), int(hi) + 1))

    def _flush_pending_feedback(self) -> Tuple[int, int]:
        if not self.pending_feedback: return 0, 0
        ready, still_pending = [], []
        for item in self.pending_feedback:
            if item[0] <= self.time: ready.append(item)
            else: still_pending.append(item)
        self.pending_feedback = still_pending
        if not ready: return 0, 0

        merged_rewards, merged_assignments = {}, []
        for _, assignments, rewards in ready:
            merged_assignments.extend(assignments)
            merged_rewards.update(rewards)
        if merged_assignments:
            self.replicator.update_assignments_reward(merged_assignments, merged_rewards)
        return len(merged_assignments), len(ready)

    def _oracle_select_assignments(self, candidate_assignments: List[Assignment]) -> List[Assignment]:
        if not candidate_assignments: return []
        task_ids = sorted({a.task_id for a in candidate_assignments})
        worker_ids = sorted({a.worker_id for a in candidate_assignments})
        t_idx = {t: i for i, t in enumerate(task_ids)}
        w_idx = {w: j for j, w in enumerate(worker_ids)}
        m, n = len(task_ids), len(worker_ids)
        profits = np.full((m, n), -np.inf, dtype=float)
        pair2a = {}
        for a in candidate_assignments:
            net = float(self.evaluate_reward_complex(a.context) - self.replicator.replication_cost)
            profits[t_idx[a.task_id], w_idx[a.worker_id]] = net
            pair2a[(a.task_id, a.worker_id)] = a

        selected, _, _ = run_hungarian_matching(task_ids, worker_ids, profits, pair2a, allow_unmatch=True, eps=1e-12)
        return selected

    def _apply_worker_dynamics(self) -> None:
        if not hasattr(self, "next_worker_id"):
            self.next_worker_id = (max((w.worker_id for w in self.workers), default=-1) + 1)
        dynamics = globals().get("WORKER_DYNAMICS", {"leave_prob": 0.05, "join_prob": 0.10, "join_count_range": (0, 2), "drift_frac": {"driving_speed": 0.03, "bandwidth": 0.05, "processor_performance": 0.02, "physical_distance": 0.05}, "weather_change_prob": 0.03})
        
        keep_flags = [np.random.random() >= dynamics["leave_prob"] for _ in self.workers]
        if not any(keep_flags) and self.workers: keep_flags[np.random.randint(0, len(self.workers))] = True
        self.workers = [w for w, keep in zip(self.workers, keep_flags) if keep]

        if np.random.random() < dynamics["join_prob"]:
            n_join = int(np.random.randint(dynamics["join_count_range"][0], dynamics["join_count_range"][1] + 1))
            for _ in range(n_join):
                self.workers.append(spawn_new_worker(self.next_worker_id))
                self.next_worker_id += 1

        def clip(v, lo, hi): return float(min(max(v, lo), hi))
        for w in self.workers:
            for feat in ["driving_speed", "bandwidth", "processor_performance", "physical_distance"]:
                if feat in dynamics["drift_frac"] and feat in WORKER_FEATURE_VALUES_RANGE:
                    lo, hi = WORKER_FEATURE_VALUES_RANGE[feat]
                    std = float(dynamics["drift_frac"][feat]) * (hi - lo)
                    setattr(w, ("processor_perf" if feat=="processor_performance" else feat), clip(getattr(w, "processor_perf" if feat=="processor_performance" else feat) + float(np.random.normal(0.0, std)), lo, hi))
            if np.random.random() < dynamics["weather_change_prob"]:
                w.weather = int(np.random.randint(0, WORKER_FEATURE_VALUES_RANGE.get("weather", (0, 4))[1] + 1))

    def _expected_total_reward(self, assignments: List[Assignment]) -> float:
        rc = self.replicator.replication_cost
        return sum(float(self.evaluate_reward_complex(a.context) - rc) for a in assignments) if assignments else 0.0

    def evaluate_reward_complex(self, context: np.ndarray) -> float:
        ds, bw, pp, pd, ds_size, weather = (float(context[i]) if len(context)>i else 0.0 for i in [0,1,2,3,5,6])
        eps = 1e-6
        pos_basic = 0.35 * np.sqrt(ds + eps) + 0.35 * np.sqrt(bw + eps) + 0.20 * np.sqrt(pp + eps)
        pos_synergy = 0.18 * np.sqrt((ds * bw) + eps) + 0.10 * np.sqrt((bw * pp) + eps)
        neg = 0.25 * (pd ** 1.5) + 0.20 * (ds_size ** 1.2) + 0.10 * (weather ** 1.2)
        def gate_for(x, thr=0.2, k=6.0): return 1.0 / (1.0 + np.exp(-k * (x - thr)))
        gates = [gate_for(ds), gate_for(bw)]
        if len(context) > 2: gates.append(gate_for(pp))
        gate = float(np.prod(gates)) if gates else 1.0
        p = 1.0 / (1.0 + np.exp(-3.0 * ((pos_basic + pos_synergy) * gate - neg)))
        return float(np.clip(p, 0.01, 0.99))

    def step_with_selector(self, new_tasks, batch_size, selector, update_model=False):
        self._flush_pending_feedback()
        for task in new_tasks: self.task_queue.add_task(task.task_type, task.data_size, task.deadline)
        tasks_to_schedule = self.task_queue.get_tasks_batch(batch_size)
        if not tasks_to_schedule:
            self.time += 1
            return {"loss": 0.0, "expected": 0.0, "oracle": 0.0, "realized_net": 0.0}

        if self.enable_worker_dynamics: self._apply_worker_dynamics()
        candidates = self.generate_candidate_assignments(tasks_to_schedule)
        rc = float(self.replicator.replication_cost)
        
        selected_assignments = selector(candidates, lambda a: float(self.evaluate_reward_complex(a.context) - rc))
        oracle_assignments = self._oracle_select_assignments(candidates)
        
        alg_expected = self._expected_total_reward(selected_assignments)
        oracle_expected = self._expected_total_reward(oracle_assignments)
        
        realized_net = 0.0
        rewards = {}
        for a in selected_assignments:
            r = float(np.random.binomial(1, self.evaluate_reward_complex(a.context)))
            rewards[a] = r
            realized_net += (r - rc)

        if update_model and selected_assignments:
            self.replicator.update_assignments_reward(list(selected_assignments), dict(rewards))

        self.time += 1
        return {
            "loss": max(0.0, oracle_expected - alg_expected),
            "expected": float(alg_expected),
            "oracle": float(oracle_expected),
            "realized_net": float(realized_net)
        }

def _clone_workers(workers: List["Worker"]) -> List["Worker"]:
    return [Worker(w.worker_id, w.driving_speed, w.bandwidth, w.processor_perf, w.physical_distance, w.weather) for w in workers]

def _generate_worker_timeline(base_workers, steps):
    rng = np.random.default_rng(RANDOM_SEED)
    scheduler = Scheduler(_clone_workers(base_workers), ContextNormalizer(), None)
    timeline = []
    for _ in range(steps):
        scheduler._apply_worker_dynamics()
        timeline.append(_clone_workers(scheduler.workers))
    return timeline

def run_experiment() -> None:
    np.random.seed(RANDOM_SEED)
    os.makedirs("output", exist_ok=True)
    
    base_workers = [spawn_new_worker(i) for i in range(3, 50)]
    normalizer = ContextNormalizer()
    steps = int(globals().get("COMPARISON_STEPS", 1000)) # 建议稍微调小一点，比如1000或2000，加速运行
    batch_size = int(globals().get("COMPARISON_BATCH_SIZE", 10))
    arrivals_min, arrivals_max = globals().get("ARRIVALS_PER_STEP", (6, 16))

    task_stream = [[Task(-1, int(np.random.randint(0, 10)), float(np.random.uniform(100, 3000)), float(np.random.uniform(1, 3))) for _ in range(int(np.random.randint(arrivals_min, arrivals_max)))] for _ in range(steps)]
    worker_timeline = _generate_worker_timeline(base_workers, steps)

    COLOR_MAP = {
        "Ours-Immediate": "#1f77b4", # 蓝色 (第三章原始)
        "Ours-DP (eps=1.0)": "#2ca02c", # 绿色 (差分隐私, 中等)
        "Ours-DP (eps=0.1)": "#ff7f0e", # 橙色 (差分隐私, 强)
        "Greedy": "#d62728", # 红色
        "Random": "#9467bd"  # 紫色
    }

    # ============== 核心运行器 ==============
    def run_method(method_label: str, enable_dp: bool = False, dp_epsilon: float = 1.0, is_greedy: bool = False, is_random: bool = False, is_oracle: bool = False):
        print(f"[{method_label}] 实验开始运行...")
        np.random.seed(RANDOM_SEED)
        workers = _clone_workers(base_workers)
        
        replicator = TaskReplicator(
            context_dim=7, 
            partition_split_threshold=PARTITION_SPLIT_THRESHOLD, 
            budget=1, 
            replication_cost=REPLICATION_COST, 
            max_partition_depth=MAX_PARTITION_DEPTH,
            enable_dp=enable_dp,       # 传入DP参数
            dp_epsilon=dp_epsilon      # 传入epsilon
        )
        scheduler = Scheduler(workers, normalizer, replicator, enable_worker_dynamics=False)
        
        from baselines import RandomBaseline, GreedyBaseline
        if is_random: selector = lambda cands, eval_fn: RandomBaseline(np.random.default_rng(RANDOM_SEED)).select(cands, eval_fn)
        elif is_greedy: selector = lambda cands, eval_fn: GreedyBaseline(replicator).select(cands, eval_fn)
        elif is_oracle: selector = lambda cands, eval_fn: scheduler._oracle_select_assignments(cands)
        else: selector = lambda cands, eval_fn: replicator.select_assignments(cands, allow_unmatch=True, use_ucb=True)

        loss_c, cum_c, cum_exp_c, mem_c = [], [], [], []
        cum = 0.0
        cum_exp = 0.0
        
        for s in range(steps):
            scheduler.workers = _clone_workers(worker_timeline[s])
            # Greedy / Random / Oracle 不需要更新模型 (Greedy虽然更新，但在 baseline 内部已经写死或不影响DP)
            update_model = not (is_random or is_oracle)
            res = scheduler.step_with_selector(task_stream[s], batch_size, selector, update_model=update_model)
            
            loss_c.append(res["loss"])
            cum += res["realized_net"]
            cum_c.append(cum)
            cum_exp += res["oracle"] if is_oracle else res["expected"]
            cum_exp_c.append(cum_exp)
            
            mem_c.append(get_replicator_memory_kb(replicator))
            
        print(f"[{method_label}] 运行完毕. 最终累计收益: {cum:.2f}, 分裂次数: {replicator.split_events}")
        return loss_c, cum_c, cum_exp_c, mem_c

    # ============== 执行各类算法 ==============
    # 1. Oracle (上界)
    loss_orc, cum_orc, cum_eorc, mem_orc = run_method("Oracle", is_oracle=True)
    # 2. 第三章最优算法 (Ours-Immediate)
    loss_o, cum_o, cum_eo, mem_o = run_method("Ours-Immediate", enable_dp=False)
    # 3. 第四章 DP 算法 (eps=1.0)
    loss_dp1, cum_dp1, cum_edp1, mem_dp1 = run_method("Ours-DP (eps=1.0)", enable_dp=True, dp_epsilon=1.0)
    # 4. 第四章 DP 算法 (eps=0.1) - 强隐私
    loss_dp01, cum_dp01, cum_edp01, mem_dp01 = run_method("Ours-DP (eps=0.1)", enable_dp=True, dp_epsilon=0.1)
    # 5. Greedy
    loss_g, cum_g, cum_eg, mem_g = run_method("Greedy", is_greedy=True)
    # 6. Random
    loss_r, cum_r, cum_er, mem_r = run_method("Random", is_random=True)

    # ============== 画图部分 ==============
    print("\n正在生成实验图表...")

    def _cumulative_regret(method_cum, oracle_cum):
        method_arr = np.asarray(method_cum, dtype=float)
        oracle_arr = np.asarray(oracle_cum, dtype=float)
        n = min(method_arr.size, oracle_arr.size)
        return oracle_arr[:n] - method_arr[:n]

    reg_o = _cumulative_regret(cum_o, cum_orc)
    reg_dp1 = _cumulative_regret(cum_dp1, cum_orc)
    reg_dp01 = _cumulative_regret(cum_dp01, cum_orc)
    reg_g = _cumulative_regret(cum_g, cum_orc)
    reg_r = _cumulative_regret(cum_r, cum_orc)
    
    steps_arr = np.arange(len(reg_o))

    # 1. 累积遗憾 (Cumulative Regret) —— 重点展示 Trade-off
    plt.figure(figsize=(9, 5))
    plt.plot(steps_arr, reg_dp01, label="Ours-DP ($\epsilon=0.1$)", color=COLOR_MAP["Ours-DP (eps=0.1)"], linewidth=2.0)
    plt.plot(steps_arr, reg_dp1, label="Ours-DP ($\epsilon=1.0$)", color=COLOR_MAP["Ours-DP (eps=1.0)"], linewidth=2.0)
    plt.plot(steps_arr, reg_o, label="Ours-Immediate (No DP)", color=COLOR_MAP["Ours-Immediate"], linewidth=2.0)
    plt.plot(steps_arr, reg_g, label="Greedy", color=COLOR_MAP["Greedy"], linestyle='--', linewidth=1.5)
    plt.plot(steps_arr, reg_r, label="Random", color=COLOR_MAP["Random"], linestyle='-.', linewidth=1.5)
    plt.title(_t("累计遗憾对比：隐私与效用的权衡 (Trade-off)", "Cumulative Regret: Privacy-Utility Trade-off"))
    plt.xlabel(_t("时间步 (Step)", "Step"))
    plt.ylabel(_t("累计遗憾 (Cumulative Regret)", "Cumulative Regret"))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/chapter4_dp_regret.png", dpi=150)
    plt.close()

    # 2. 累计真实收益 (Cumulative Reward)
    plt.figure(figsize=(9, 5))
    plt.plot(steps_arr, cum_orc[:len(steps_arr)], label="Oracle (UpperBound)", color="black", linestyle="--", linewidth=2.0)
    plt.plot(steps_arr, cum_o, label="Ours-Immediate (No DP)", color=COLOR_MAP["Ours-Immediate"], linewidth=2.0)
    plt.plot(steps_arr, cum_dp1, label="Ours-DP ($\epsilon=1.0$)", color=COLOR_MAP["Ours-DP (eps=1.0)"], linewidth=2.0)
    plt.plot(steps_arr, cum_dp01, label="Ours-DP ($\epsilon=0.1$)", color=COLOR_MAP["Ours-DP (eps=0.1)"], linewidth=2.0)
    plt.plot(steps_arr, cum_g, label="Greedy", color=COLOR_MAP["Greedy"], linestyle='--', linewidth=1.5)
    plt.title(_t("累计净收益对比", "Cumulative Net Reward"))
    plt.xlabel(_t("时间步 (Step)", "Step"))
    plt.ylabel(_t("累计收益", "Cumulative Reward"))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/chapter4_dp_reward.png", dpi=150)
    plt.close()

    # 3. 内存开销对比 (Memory Usage) —— 本章核心亮点
    plt.figure(figsize=(9, 5))
    # 第 3 章的重状态算法
    plt.plot(steps_arr, mem_o, label="Ours-Immediate (重状态, No DP)", color=COLOR_MAP["Ours-Immediate"], linewidth=2.5)
    # 第 4 章的无状态算法
    plt.plot(steps_arr, mem_dp1, label="Ours-DP (无状态, $\epsilon=1.0$)", color=COLOR_MAP["Ours-DP (eps=1.0)"], linewidth=2.5)
    
    plt.title(_t("不同算法随时间步增长的内存开销对比", "Memory Usage Comparison Over Steps"))
    plt.xlabel(_t("时间步 (Step)", "Step"))
    plt.ylabel(_t("内存开销 (Memory Usage) / KB", "Memory Usage (KB)"))
    
    # 填充颜色，使得对比更直观
    plt.fill_between(steps_arr, 0, mem_o, color=COLOR_MAP["Ours-Immediate"], alpha=0.1)
    plt.fill_between(steps_arr, 0, mem_dp1, color=COLOR_MAP["Ours-DP (eps=1.0)"], alpha=0.2)
    
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/chapter4_memory_usage.png", dpi=150)
    plt.close()

    print("生成完毕！请查看 output/ 目录下的 chapter4_dp_regret.png、chapter4_dp_reward.png 以及 chapter4_memory_usage.png")
    print("生成完毕！请查看 output/ 目录下的 chapter4_dp_regret.png 和 chapter4_dp_reward.png")

def get_replicator_memory_kb(replicator: TaskReplicator) -> float:
    """
    计算 TaskReplicator 内部维护上下文划分树的内存开销 (KB)。
    主要包含：树节点的常数开销 + 缓存历史样本的开销。
    """
    if replicator is None:
        return 0.0
        
    NODE_OVERHEAD_BYTES = 128  # 假设每个树节点(边界、统计量等)占用约128字节
    # 每个样本存储 context_tuple (dim个float) 和 reward (1个float)
    SAMPLE_BYTES = (replicator.context_dim + 1) * 8  

    total_nodes = 0
    total_samples = 0

    def traverse(node):
        nonlocal total_nodes, total_samples
        total_nodes += 1
        total_samples += len(node.data_points)
        if node.children is not None:
            for child in node.children:
                traverse(child)

    traverse(replicator.root_partition)
    
    # 计算总字节数并转换为 KB
    total_bytes = (total_nodes * NODE_OVERHEAD_BYTES) + (total_samples * SAMPLE_BYTES)
    return total_bytes / 1024.0

if __name__ == "__main__":
    run_experiment()