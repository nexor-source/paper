import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List

class PartitionVisualizer:
    """
    上下文空间划分可视化器：以2D形式展示当前所有叶子划分的分布和采样状态
    """
    def __init__(self, partitions: List):
        """
        :param partitions: List of ContextSpacePartition（即当前所有叶子节点）
        """
        self.partitions = partitions

    def plot_2d_partitions(self, dim_x: int, dim_y: int, iteration: int, save_path: str = None):
        """
        可视化上下文空间中在第 dim_x 与 dim_y 两维上的划分结果。
        :param dim_x: 第一个特征维度索引（如0代表车速）
        :param dim_y: 第二个特征维度索引（如1代表带宽）
        :param iteration: 当前迭代轮次，用于标题或保存文件命名
        :param save_path: 若提供路径则保存为图片，否则显示
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Context Partition Visualization (Iteration {iteration})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Feature {dim_x}")
        ax.set_ylabel(f"Feature {dim_y}")
        
        for p in self.partitions:
            # 取当前划分在指定两个维度的上下界
            x_min, x_max = p.bounds[dim_x]
            y_min, y_max = p.bounds[dim_y]
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=1, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

            # 标记样本数量
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            ax.text(cx, cy, str(p.sample_count), color='red', fontsize=30, ha='center', va='center')

        ax.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved partition plot to {save_path}")
            plt.close()
        else:
            plt.show()
