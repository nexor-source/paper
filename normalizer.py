import numpy as np
from typing import Dict

class ContextNormalizer:
    """
    上下文特征归一化器，将真实世界特征映射到 [0,1] 区间，
    支持数值型和类别型特征。
    """

    def __init__(self):
        """初始化特征取值范围字典

        Notes:
            - 数值范围用于线性归一化，类别特征则用编码范围。
        """
        self.ranges = {
            "driving_speed": (0.0, 40.0),       # 0-40 m/s (~0-144 km/h)
            "bandwidth": (0.0, 1000.0),         # 0-1000 Mbps
            "processor_performance": (1.0, 5.0),# 1-5 GHz
            "physical_distance": (0.0, 1000.0), # 0-1000 m
            "task_type": (0, 9),                # 10 种任务，编码 0-9
            "data_size": (0.0, 5000.0),         # 0-5000 MB
            "weather": (0, 4)                   # 5 类天气，编码 0-4
        }

    def normalize_value(self, key: str, value: float) -> float:
        """单值归一化（裁剪到 [0,1]），并对异常情况给出提示"""
        if key not in self.ranges:
            raise KeyError(f"[ContextNormalizer] 特征 '{key}' 未在 ranges 中定义！")

        v_min, v_max = self.ranges[key]
        if v_max == v_min:
            raise ValueError(f"[ContextNormalizer] 特征 '{key}' 的范围无效：min==max")

        norm = (value - v_min) / (v_max - v_min)

        # 提示超出范围的情况
        if norm < 0 or norm > 1:
            print(f"[Warning] 特征 '{key}' 的值 {value} 超出范围 {v_min}~{v_max}，结果已裁剪到 [0,1]")

        return min(max(norm, 0.0), 1.0)

    def normalize_context(self, raw_features: Dict[str, any]) -> np.ndarray:
        """将原始特征字典映射为归一化向量

        Args:
            raw_features (dict): 输入特征字典，键必须在 self.ranges 中定义。

        Returns:
            np.ndarray: 归一化后的向量，shape=(len(raw_features),)，dtype=float32。
        """
        norm_vec = []
        for key, value in raw_features.items():
            norm_vec.append(self.normalize_value(key, value))
        return np.array(norm_vec, dtype=np.float32)


# 测试示例
if __name__ == "__main__":
    normalizer = ContextNormalizer()

    raw_sample = {
        "driving_speed": 15.5,         # m/s
        "bandwidth": 150,              # Mbps
        "processor_performance": 3.2,  # GHz
        "physical_distance": 230,      # m
        "task_type": 3,                # 第 4 类任务
        "data_size": 1200,             # MB
        "weather": 2                   # 天气类别编码
    }

    normalized_context = normalizer.normalize_context(raw_sample)
    print("归一化上下文向量:", normalized_context)
