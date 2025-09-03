import numpy as np

class ContextNormalizer:
    """
    上下文特征归一化器，将真实世界特征映射到[0,1]区间，
    支持数值型和类别型特征
    """
    def __init__(self):
        # 预设每个特征的最小最大范围，用于归一化
        # 可根据实际数据分布调整
        self.ranges = {
            "driving_speed": (0.0, 40.0),     # 0-40 m/s (~0-144 km/h)
            "bandwidth": (0.0, 1000.0),       # 0-1000 Mbps
            "processor_performance": (1.0, 5.0),  # 1-5 GHz
            "physical_distance": (0.0, 1000.0),    # 0-1000 m
            "task_type": (0, 9),              # 10种任务，编码0-9
            "data_size": (0.0, 5000.0),       # 0-5000 MB
            "weather": (0, 4)                 # 5类天气，编码0-4
        }

    def normalize_value(self, key, value):
        """
        数值型归一化到[0,1]
        """
        v_min, v_max = self.ranges[key]
        if v_max == v_min:
            return 0.0
        norm = (value - v_min) / (v_max - v_min)
        # 保证范围在0-1内
        return min(max(norm, 0.0), 1.0)

    def normalize_context(self, raw_features: dict) -> np.ndarray:
        """
        输入原始特征字典，输出归一化上下文向量(np.array)
        """
        norm_vec = []
        # 车速
        norm_vec.append(self.normalize_value("driving_speed", raw_features.get("driving_speed", 0.0)))
        # 带宽
        norm_vec.append(self.normalize_value("bandwidth", raw_features.get("bandwidth", 0.0)))
        # 处理器性能
        norm_vec.append(self.normalize_value("processor_performance", raw_features.get("processor_performance", 1.0)))
        # 物理距离
        norm_vec.append(self.normalize_value("physical_distance", raw_features.get("physical_distance", 0.0)))
        # 任务类型（类别编码归一化）
        norm_vec.append(self.normalize_value("task_type", raw_features.get("task_type", 0)))
        # 数据大小
        norm_vec.append(self.normalize_value("data_size", raw_features.get("data_size", 0.0)))
        # 天气（类别编码归一化）
        norm_vec.append(self.normalize_value("weather", raw_features.get("weather", 0)))

        return np.array(norm_vec, dtype=np.float32)

# 测试示例
if __name__ == "__main__":
    normalizer = ContextNormalizer()

    raw_sample = {
        "driving_speed": 15.5,         # m/s
        "bandwidth": 150,              # Mbps
        "processor_performance": 3.2,  # GHz
        "physical_distance": 230,      # m
        "task_type": 3,                # 第4类任务，编码0开始
        "data_size": 1200,             # MB
        "weather": 2                  # 天气类别编码
    }

    normalized_context = normalizer.normalize_context(raw_sample)
    print("归一化上下文向量:", normalized_context)
