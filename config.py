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

# 从父分区的真实样本里“借”百分之多少给子分区。
LAMBDA_PRIOR = 0.5  

# 从父分区最多继承样本数
PRIOR_CAP = 10

# 最大允许的上下文划分层级（根为0）。达到该层级后不再细分。
# 可按需调整，若不希望限制，可在创建 TaskReplicator 时传入 None 覆盖。
MAX_PARTITION_DEPTH = 4


# scheduler.py 的 evaluate_reward 函数将reward定义为几个指标的线性组合，这使得上下文空间对应的reward图过于简单。
# 考虑将 reward 定义为更复杂的函数，但保持连续性以及各个指标的组合是合理的。

# task_replicator.py 的 subdivide 函数，一旦确定要划分，就会对 partition 的所有维度进行二分生成 2^d 个子分区
# （d 为维度数）。这种做法可能过于激进，考虑改为只划分某些维度，或允许多种划分方式。（比如只划分目前长度最长的维度，或者根据样本分布选择划分维度）

# 给worker添加一些变动的感觉，模拟真实场景下worker不断发生变化的场景
