# config.py

MAX_PARTITION_DEPTH = 4

WORKER_FEATURE_VALUES_RANGE = {
    "driving_speed": (0.0, 40.0),       # 0-40 m/s (~0-144 km/h)
    "bandwidth": (0.0, 1000.0),         # 0-1000 Mbps
    "processor_performance": (1.0, 5.0),# 1-5 GHz
    "physical_distance": (0.0, 1000.0), # 0-1000 m
    "task_type": (0, 9),                # 10 种任务，编码 0-9
    "data_size": (0.0, 5000.0),         # 0-5000 MB
    "weather": (0, 4)                   # 5 类天气，编码 0-4？必须做到天气越大越恶劣
}

# 随机性控制：使用固定随机种子保证可复现
RANDOM_SEED = 42

# 工人动态变化配置（来来往往 + 属性缓慢漂移）
# - leave_prob: 现有工人每一步离开的概率（独立）
# - join_prob: 每一步触发新增工人的概率
# - join_count_range: 触发新增时新增工人数的闭区间范围（含端点）
# - drift_frac: 各属性相对其取值范围宽度的高斯噪声标准差比例（边界处会裁剪）
# - weather_change_prob: 每步天气状态变动概率（类别 0..4 中随机）
WORKER_DYNAMICS = {
    "leave_prob": 0.025,
    "join_prob": 0.10,
    "join_count_range": (0, 2),
    "drift_frac": {
        "driving_speed": 0.03,
        "bandwidth": 0.05,
        # "processor_performance": 0.02,
        # "physical_distance": 0.05,
    },
    # "weather_change_prob": 0.03,
}

# 从父分区的真实样本里“借”百分之多少给子分区。
LAMBDA_PRIOR = 0.5  

# 从父分区最多继承样本数
PRIOR_CAP = 10

# 最大允许的上下文划分层级（根为0）。达到该层级后不再细分。
# 可按需调整，若不希望限制，可在创建 TaskReplicator 时传入 None 覆盖。
MAX_PARTITION_DEPTH = 8


# scheduler.py 的 evaluate_reward 函数将reward定义为几个指标的线性组合，这使得上下文空间对应的reward图过于简单。
# 考虑将 reward 定义为更复杂的函数，但保持连续性以及各个指标的组合是合理的。

# task_replicator.py 的 subdivide 函数，一旦确定要划分，就会对 partition 的所有维度进行二分生成 2^d 个子分区
# （d 为维度数）。这种做法可能过于激进，考虑改为只划分某些维度，或允许多种划分方式。（比如只划分目前长度最长的维度，或者根据样本分布选择划分维度）

# 分区细分策略：'all' | 'longest' | 'topk'
# - 'all': 沿所有维度二分（原始行为），产生 2^d 子区
# - 'longest': 仅沿当前区间长度最大的维度二分，产生 2 个子区
# - 'topk': 沿区间长度排名前 K 的维度二分，产生 2^K 子区（由 PARTITION_SPLIT_TOP_K 控制）
PARTITION_SPLIT_STRATEGY = 'longest'
PARTITION_SPLIT_TOP_K = 1

# 给worker在steps递增的过程中添加变动性，模拟真实场景下worker不断发生变化来来往往的场景
# 使用随机种子控制，保证在相同的种子输入下程序可以模拟得到完全相同的结果
