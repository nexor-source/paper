# config.py

WORKER_FEATURE_VALUES_RANGE = {
    "driving_speed": (0.0, 40.0),       # 0-40 m/s (~0-144 km/h)
    "bandwidth": (0.0, 1000.0),         # 0-1000 Mbps
    "processor_performance": (1.0, 5.0),# 1-5 GHz
    "physical_distance": (0.0, 1000.0), # 0-1000 m
    "task_type": (0, 9),                # 10 种任务，编码 0-9
    "data_size": (0.0, 5000.0),         # 0-5000 MB
    "weather": (0, 4)                   # 5 类天气，编码 0-4
}

# 从父分区的真实样本里“借”百分之多少给子分区。
LAMBDA_PRIOR = 0.5  

# 从父分区最多继承样本数
PRIOR_CAP = 10

