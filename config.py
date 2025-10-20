# config.py
WORKER_FEATURE_VALUES_RANGE = {
    "driving_speed": (0.0, 40.0),       # 0-40 m/s (~0-144 km/h) 车辆行驶速度
    "bandwidth": (0.0, 1000.0),         # 0-1000 Mbps 当前可用带宽
    "processor_performance": (1.0, 5.0),# 1-5 GHz 计算能力
    "physical_distance": (0.0, 1000.0), # 0-1000 m 到附近基站的平均距离
    "task_type": (0, 9),                # 10 任务类型
    "data_size": (0.0, 5000.0),         # 0-5000 MB 任务数据量
    "weather": (0, 4)                   # 5 类天气，编码 0-4？必须做到天气越大越恶劣
}

# 随机性控制：使用固定随机种子保证可复现
RANDOM_SEED = 42

# 工人动态变化配置（来来往往 + 属性缓慢漂移）
# - leave_prob: 现有工人每一步离开的概率（独立）
# - join_prob: 每一步触发新增工人的概率
# - join_count_range: 触发新增时新增工人数的闭区间范围（含端点）
# - drift_frac: 各属性相对其取值范围宽度的高斯噪声标准差*比例*（边界处会裁剪）
# - weather_change_prob: 每步天气状态变动概率（类别 0..4 中随机）
WORKER_DYNAMICS = {
    "leave_prob": 0.025,
    "join_prob": 0.10,
    "join_count_range": (0, 3),
    "drift_frac": {
        "driving_speed": 0.03,
        "bandwidth": 0.05,
        "processor_performance": 0.02,
        "physical_distance": 0.05,
    },
    "weather_change_prob": 0.03,
}

# 从父分区的真实样本里“借”百分之多少给子分区。
LAMBDA_PRIOR = 0.5

# 从父分区最多继承样本数
PRIOR_CAP = 10

# 最大允许的上下文划分层级（根为0）。达到该层级后不再细分。
# 可按需调整，若不希望限制，可在创建 TaskReplicator 时传入 None 覆盖。
MAX_PARTITION_DEPTH = 64

# 分区细分策略：'all' | 'longest' | 'topk'
# - 'all': 沿所有维度二分（原始行为），产生 2^d 子区
# - 'longest': 仅沿当前区间长度最大的维度二分，产生 2 个子区
# - 'topk': 沿区间长度排名前 K 的维度二分，产生 2^K 子区（由 PARTITION_SPLIT_TOP_K 控制）
PARTITION_SPLIT_STRATEGY = 'longest'
PARTITION_SPLIT_TOP_K = 1
PARTITION_SPLIT_THRESHOLD = 10

# Baseline/Comparison experiment settings
RUN_COMPARISON = True
COMPARISON_STEPS = 1000
COMPARISON_BATCH_SIZE = 10
ARRIVALS_PER_STEP = (6, 16)  # 每个时刻到达的任务数量的上下限
ENABLE_WORKER_DYNAMICS_COMPARISON = True  # 由于所有baseline采用了相同的randomseed，所以随机结果是相同的

# Assignment inspection configuration (visual spot checks)
ASSIGNMENT_INSPECTION_COUNT = 5          # 0 ~ COMPARISON_STEPS-1 中随机抽查的 step 数量
ASSIGNMENT_INSPECTION_SEED = 1234        # 抽查 step 使用的随机种子
ASSIGNMENT_INSPECTION_DIR = "assignment_inspections"  # 输出目录（位于 output/ 下）
# Optional: manually specify steps to inspect; set to None to fall back to random sampling
ASSIGNMENT_INSPECTION_STEPS = None

# 算法模拟worker的特征的时候混入了task的特征

# 注释统一度量

# evaluate_reward_complex 函数通过固定的函数来计算工人和任务的特征，最终给出一个 0 ~ 1 之间的值，作为一个 worker-task 是否能够完成任务的一个概率
# 并且通过这样的概率，来作为 worker-task 的奖励。但这样的奖励过于具体化，我们还需要在其之上添加一些随机性，以便更好地模拟真实场景中的不确定性。
# 具体来说，我们可以在奖励上添加一个高斯噪声项，因为我们算法的本质相当于是动态划分上下文空间的多臂老虎机模拟，在原版的多臂老虎机中，据说奖励是随机的，
# 这样添加随机噪声也符合算法的基本设定，你觉得呢，如果你也是这么觉得的那就直接开始做出更改吧

# scheduler.py 的 step 方法已经废弃了

REPLICATION_COST = 0.2  # 默认的复制成本

# TaskReplicator exploration / UCB configuration
REPLICATOR_USE_UCB = True            # 是否对 Original/Greedy 的估计加入 UCB 探索项
REPLICATOR_UCB_COEF = 1.5            # UCB 系数 c：越大越激进探索
REPLICATOR_UCB_MIN_PULLS = 1         # 分母里加的平滑项，防止除 0
REPLICATOR_PRIOR_MEAN = 0.6          # 初始化/划分后子区域的乐观先验均值
REPLICATOR_PRIOR_WEIGHT = 2.0        # 乐观先验的“伪样本”权重

