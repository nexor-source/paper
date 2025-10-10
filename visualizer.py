import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from typing import Dict, List, Optional, Sequence, Set, Tuple


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


def render_assignment_matrix(
    *,
    method_name: str,
    step_index: int,
    task_ids: Sequence[int],
    worker_ids: Sequence[int],
    predicted_net: Dict[Tuple[int, int], float],
    true_net: Dict[Tuple[int, int], float],
    selected_pairs: Set[Tuple[int, int]],
    save_path: str,
) -> None:
    """Render a worker-task grid with predicted/true net reward values."""
    if not task_ids or not worker_ids:
        return

    # Determine figure size adaptively: wider for more tasks, taller for more workers
    width = max(8.0, 1.5 + 0.85 * len(task_ids))
    height = max(5.5, 1.8 + 0.55 * len(worker_ids))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")

    # Prepare table content
    cell_text: List[List[str]] = []
    cell_colors: List[List[str]] = []
    selected_set = set(selected_pairs)

    for worker_id in worker_ids:
        row_text: List[str] = []
        row_colors: List[str] = []
        for task_id in task_ids:
            key = (int(worker_id), int(task_id))
            if key in predicted_net:
                pred_val = predicted_net.get(key, float("nan"))
                true_val = true_net.get(key, float("nan"))
                pred_str = f"{pred_val:.3f}" if pred_val == pred_val else "nan"
                true_str = f"{true_val:.3f}" if true_val == true_val else "nan"
                row_text.append(f"{pred_str}\n({true_str})")
                if key in selected_set:
                    row_colors.append("#ffe8a5")  # highlight
                else:
                    row_colors.append("#f5f5f5")
            else:
                row_text.append("")
                row_colors.append("#e0e0e0")
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=[str(w) for w in worker_ids],
        colLabels=[str(t) for t in task_ids],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.45, 2.2)

    title = f"{method_name} — Step {step_index}"
    subtitle = "cell: predicted net\n(parentheses: oracle net)"
    ax.set_title(title, fontsize=14, pad=16)
    fig.text(0.02, 0.93, subtitle, fontsize=9, ha="left", va="top")

    selected_pred_sum = sum(predicted_net.get(key, 0.0) for key in selected_set)
    selected_true_sum = sum(true_net.get(key, 0.0) for key in selected_set)
    summary = (
        f"Selected assignments: {len(selected_set)} | "
        f"Predicted net sum = {selected_pred_sum:.3f} | "
        f"True net sum = {selected_true_sum:.3f}"
    )
    fig.text(0.5, 0.04, summary, fontsize=10, ha="center", va="center")

    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.92))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
