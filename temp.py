import matplotlib.pyplot as plt
import numpy as np

steps = np.arange(0, 100000, 1000)
# 传统O(N)算法：每条数据假设250字节，转换为MB
memory_immediate = steps * 250 / (1024 * 1024) 
# 本章无状态算法O(1)：无论多少步，假设树最大分裂2000个叶子节点，每个节点存两个浮点数(16字节)
memory_dp = np.ones_like(steps) * (2000 * 16) / (1024 * 1024) 

plt.figure(figsize=(8, 4))
plt.plot(steps, memory_immediate, label="Ours-Immediate (Heavy State)", color="C0", linewidth=2)
plt.plot(steps, memory_dp, label="DP-CCMAB (Stateless)", color="C2", linewidth=2)
plt.xlabel("Number of Task Interactions")
plt.ylabel("Memory Consumption (MB)")
plt.title("Edge Server Memory Consumption Comparison")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("memory_cost.png", dpi=150)