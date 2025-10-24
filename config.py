"""Global configuration for the task replication experiments."""

# Ranges for worker features (min, max). All values are normalised to [0, 1]
# before entering the learning model.
WORKER_FEATURE_VALUES_RANGE = {
    "driving_speed": (0.0, 40.0),        # m/s
    "bandwidth": (0.0, 1000.0),          # Mbps
    "processor_performance": (1.0, 5.0), # GHz
    "physical_distance": (0.0, 1000.0),  # metres
    "task_type": (0, 9),
    "data_size": (0.0, 5000.0),          # MB
    "weather": (0, 4),
}

# Reproducibility
RANDOM_SEED = 43

# Worker arrival/departure dynamics
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
    "weather_change_prob": 0.03,
}

# Target worker pool size (used when dynamics are enabled)
WORKER_COUNT_MIN = 8
WORKER_COUNT_MAX = 12

# Partition control
MAX_PARTITION_DEPTH = 64
PARTITION_SPLIT_THRESHOLD = 10
PARTITION_MIN_SAMPLES = 6            # minimum samples before split is considered
PARTITION_VARIANCE_THRESHOLD = 0.005 # variance gate for splitting

# Experiment settings
RUN_COMPARISON = True
COMPARISON_STEPS = 8000
COMPARISON_BATCH_SIZE = 10
ARRIVALS_PER_STEP = (6, 16)
ENABLE_WORKER_DYNAMICS_COMPARISON = True

# Plot smoothing parameters
LOSS_SMOOTH_WINDOW = 100

# Assignment inspection snapshots
ASSIGNMENT_INSPECTION_COUNT = 5
ASSIGNMENT_INSPECTION_SEED = 1234
ASSIGNMENT_INSPECTION_DIR = "assignment_inspections"
ASSIGNMENT_INSPECTION_STEPS = None

# Economic parameters
REPLICATION_COST = 0.2

# UCB behaviour for the replicator
REPLICATOR_USE_UCB = True
REPLICATOR_UCB_COEF = 0.3
REPLICATOR_UCB_MIN_PULLS = 1
