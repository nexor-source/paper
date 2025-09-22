import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from typing import List, Optional, Dict


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

    def plot_2d_partitions(
        self,
        dim_x: int,
        dim_y: int,
        iteration: int,
        save_path: Optional[str] = None,
        slice_point: Optional[Dict[int, float]] = None,
        default_slice_value: float = 0.52,
    ) -> None:
        """Visualize the partitioning on a 2D slice defined by two dimensions."""

        if not self.partitions:
            return

        total_dims = len(self.partitions[0].bounds)
        slice_point = slice_point or {}
        eps = 1e-9

        def matches_slice(partition) -> bool:
            if total_dims <= 0:
                return True
            for dim in range(total_dims):
                if dim in (dim_x, dim_y):
                    continue
                value = slice_point.get(dim, default_slice_value)
                lo, hi = partition.bounds[dim]
                if value < lo - eps or value > hi + eps:
                    return False
            return True

        filtered = [p for p in self.partitions if matches_slice(p)]
        if not filtered:
            filtered = self.partitions

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Context Partition Visualization (Iteration {iteration})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Feature {dim_x}")
        ax.set_ylabel(f"Feature {dim_y}")

        qualities = [p.estimated_quality for p in filtered]
        if qualities:
            min_quality, max_quality = min(qualities), max(qualities)
        else:
            min_quality, max_quality = 0.0, 1.0
        if max_quality - min_quality < 1e-9:
            max_quality = min_quality + 1e-9
        norm = plt.Normalize(vmin=min_quality, vmax=max_quality)

        for partition in filtered:
            x_min, x_max = partition.bounds[dim_x]
            y_min, y_max = partition.bounds[dim_y]
            color = cm.viridis(norm(partition.estimated_quality))

            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1,
                edgecolor="blue",
                facecolor=color,
            )
            ax.add_patch(rect)

            if partition.sample_count > 0:
                cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
                ax.text(
                    cx,
                    cy,
                    f"{partition.sample_count}",
                    color="white",
                    fontsize=10,
                    ha="center",
                    va="center",
                )

        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
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