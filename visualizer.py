import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from typing import List


class PartitionVisualizer:
    """
    上下文空间划分可视化器：以二维形式展示当前所有叶子划分的分布和采样状态。
    """

    def __init__(self, partitions: List):
        """初始化可视化器

        Args:
            partitions (List): 当前所有叶子节点的列表，每个元素为 ContextSpacePartition。
        """
        self.partitions = partitions

    def plot_2d_partitions(self, dim_x: int, dim_y: int, iteration: int, save_path: str = None) -> None:
        """可视化上下文空间在指定两个维度上的划分结果，使用颜色表示期望奖励

        Args:
            dim_x (int): 第一个特征维度索引（如 0 代表车速）。
            dim_y (int): 第二个特征维度索引（如 1 代表带宽）。
            iteration (int): 当前迭代轮次，用于图表标题或保存文件命名。
            save_path (str, optional): 图片保存路径。若为 None，则直接显示图表。

        Returns:
            None

        Notes:
            - 每个矩形框代表一个划分单元（partition）。
            - 矩形颜色表示期望奖励（estimated_quality），颜色越深表示奖励越高。
            - 坐标范围固定在 [0,1] × [0,1]，对应归一化的上下文空间。
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Context Partition Visualization (Iteration {iteration})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Feature {dim_x}")
        ax.set_ylabel(f"Feature {dim_y}")

        # 获取所有分区的期望奖励，用于归一化颜色
        qualities = [p.estimated_quality for p in self.partitions if p.sample_count > 0]
        if qualities:
            min_quality, max_quality = min(qualities), max(qualities)
        else:
            min_quality, max_quality = 0, 1  # 默认范围

        for p in self.partitions:
            # 取当前划分在指定两个维度的上下界
            x_min, x_max = p.bounds[dim_x]
            y_min, y_max = p.bounds[dim_y]

            # 根据期望奖励计算颜色
            if p.sample_count > 0:
                normalized_quality = p.estimated_quality
                # normalized_quality = (p.estimated_quality - min_quality) / (max_quality - min_quality + 1e-6)
                color = cm.viridis(normalized_quality)  # 使用 Viridis 颜色映射
            else:
                color = (1, 1, 1, 0)  # 空白区域为透明

            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1,
                edgecolor="blue",
                facecolor=color,
            )
            ax.add_patch(rect)

            # 标记样本数量
            if p.sample_count > 0:
                cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
                ax.text(
                    cx,
                    cy,
                    f"{p.sample_count}",
                    color="white",
                    fontsize=10,
                    ha="center",
                    va="center",
                )

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=min_quality, vmax=max_quality))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Estimated Quality")

        ax.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved partition plot to {save_path}")
            plt.close()
        else:
            plt.show()