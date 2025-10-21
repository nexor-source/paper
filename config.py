# config.py
WORKER_FEATURE_VALUES_RANGE = {
    "driving_speed": (0.0, 40.0),       # 0-40 m/s (~0-144 km/h) 最大行驶速度
    "bandwidth": (0.0, 1000.0),         # 0-1000 Mbps 当前链路带宽
    "processor_performance": (1.0, 5.0),# 1-5 GHz CPU 性能
    "physical_distance": (0.0, 1000.0), # 0-1000 m 与任务站点的距离
    "task_type": (0, 9),                # 任务类型编号范围
    "data_size": (0.0, 5000.0),         # 0-5000 MB 数据量
    "weather": (0, 4)                   # 天气等级 0-4，越大越恶劣
}

# 全局随机种子
RANDOM_SEED = 43

# 工人动态：默认关闭（静态环境）
WORKER_DYNAMICS = {
    "leave_prob": 0.03,
    "join_prob": 0.15,
    "join_count_range": (1, 3),
    "drift_frac": {
        "driving_speed": 0.03,
        "bandwidth": 0.05,
        "processor_performance": 0.02,
        "physical_distance": 0.05,
    },
    "weather_change_prob": 0.03
}

# 目标保持工人数量的大致区间（仅在启用动态时使用）
WORKER_COUNT_MIN = 8
WORKER_COUNT_MAX = 12

# 根节点到子节点的先验复制强度
LAMBDA_PRIOR = 0.5
PRIOR_CAP = 10

MAX_PARTITION_DEPTH = 64
PARTITION_SPLIT_STRATEGY = 'longest'
PARTITION_SPLIT_TOP_K = 1
PARTITION_SPLIT_THRESHOLD = 10

# Adaptive partition split control
PARTITION_MIN_SAMPLES = 6             # 每个叶子至少采样 6 次后才允许细分
PARTITION_VARIANCE_THRESHOLD = 0.005   # Beta 后验方差大于该值时才考虑继续细分

# Baseline/Comparison experiment settings
RUN_COMPARISON = True
COMPARISON_STEPS = 4000
COMPARISON_BATCH_SIZE = 10
ARRIVALS_PER_STEP = (6, 16)
ENABLE_WORKER_DYNAMICS_COMPARISON = True

# Plot smoothing parameters
LOSS_SMOOTH_WINDOW = 100

ASSIGNMENT_INSPECTION_COUNT = 5
ASSIGNMENT_INSPECTION_SEED = 1234
ASSIGNMENT_INSPECTION_DIR = "assignment_inspections"
ASSIGNMENT_INSPECTION_STEPS = None

REPLICATION_COST = 0.2

REPLICATOR_USE_UCB = True
REPLICATOR_UCB_COEF = 0.3
REPLICATOR_UCB_MIN_PULLS = 1
REPLICATOR_PRIOR_MEAN = 0.6
REPLICATOR_PRIOR_WEIGHT = 2.0
