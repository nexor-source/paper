import numpy as np
from collections import deque
from typing import List, Dict

# 你之前定义的ContextNormalizer, Assignment, TaskReplicator类这里省略，假设已经实现并导入
from normalizer import ContextNormalizer
from main import Assignment, TaskReplicator
from visualizer import PartitionVisualizer

class Task:
    """
    任务实体，包含任务特征与任务ID
    """
    def __init__(self, task_id: int, task_type: int, data_size: float, deadline: float):
        self.task_id = task_id
        self.task_type = task_type
        self.data_size = data_size
        self.deadline = deadline

class Worker:
    """
    工人实体，包含工人ID和能力特征
    """
    def __init__(self, worker_id: int, driving_speed: float, bandwidth: float, processor_perf: float, physical_distance: float, weather: int):
        self.worker_id = worker_id
        self.driving_speed = driving_speed
        self.bandwidth = bandwidth
        self.processor_perf = processor_perf
        self.physical_distance = physical_distance
        self.weather = weather

class TaskQueue:
    """
    任务队列，支持任务动态加入和批量调度
    """
    def __init__(self):
        self.queue = deque()
        self.next_task_id = 0

    def add_task(self, task_type: int, data_size: float, deadline: float):
        task = Task(self.next_task_id, task_type, data_size, deadline)
        self.queue.append(task)
        self.next_task_id += 1

    def get_tasks_batch(self, batch_size: int) -> List[Task]:
        tasks = []
        for _ in range(min(batch_size, len(self.queue))):
            tasks.append(self.queue.popleft())
        return tasks

class Scheduler:
    """
    任务调度器，管理任务流、工人资源与调用分配算法
    """
    def __init__(self, workers: List[Worker], context_normalizer: ContextNormalizer, replicator: TaskReplicator):
        self.workers = workers
        self.normalizer = context_normalizer
        self.replicator = replicator
        self.task_queue = TaskQueue()
        self.time = 0  # 模拟时间步

    def generate_candidate_assignments(self, tasks: List[Task]) -> List[Assignment]:
        candidates = []
        for task in tasks:
            for worker in self.workers:
                raw_context = {
                    "driving_speed": worker.driving_speed,
                    "bandwidth": worker.bandwidth,
                    "processor_performance": worker.processor_perf,
                    "physical_distance": worker.physical_distance,
                    "task_type": task.task_type,
                    "data_size": task.data_size,
                    "weather": worker.weather
                }
                norm_context = self.normalizer.normalize_context(raw_context)
                assignment = Assignment(worker.worker_id, task.task_id, norm_context)
                candidates.append(assignment)
        return candidates

    def step(self, new_tasks: List[Task], batch_size: int):
        # 1. 新任务入队
        for task in new_tasks:
            self.task_queue.add_task(task.task_type, task.data_size, task.deadline)
        
        # 2. 取批任务准备调度
        tasks_to_schedule = self.task_queue.get_tasks_batch(batch_size)
        if not tasks_to_schedule:
            print("No tasks to schedule at time", self.time)
            return
        
        # 3. 生成所有候选工人-任务对
        candidates = self.generate_candidate_assignments(tasks_to_schedule)
        
        # 4. 调用分配算法
        selected_assignments = self.replicator.select_assignments(candidates)
        
        # 5. 模拟执行和奖励（这里用随机模拟，真实场景由系统反馈）
        rewards = {}
        for a in selected_assignments:
            rewards[a] = np.random.binomial(1, 0.6)  # 60%成功率示例
        
        # 6. 更新统计
        self.replicator.update_statistics(selected_assignments, rewards)
        
        print(f"Time {self.time}: Scheduled {len(selected_assignments)} assignments from {len(tasks_to_schedule)} tasks")
        self.time += 1

# 示例运行
if __name__ == "__main__":
    # 初始化工人资源
    workers = [
        Worker(0, 12, 100, 3.0, 250, 1),
        Worker(1, 15, 120, 3.5, 300, 2),
        Worker(2, 10, 90, 2.5, 100, 0),
        # ...更多工人
    ]
    # 补充到10个工人示例
    for i in range(3,10):
        workers.append(Worker(i, np.random.uniform(5,20), np.random.uniform(50,150), np.random.uniform(2,4), np.random.uniform(100,400), np.random.randint(0,5)))

    normalizer = ContextNormalizer()
    replicator = TaskReplicator(context_dim=7, initial_partition_size=10, budget=1, replication_cost=0.1)
    scheduler = Scheduler(workers, normalizer, replicator)

    # 模拟任务流，多轮调度
    for step_i in range(10):
        # 模拟每步新任务到达，任务类型0~9随机，数据大小100~3000MB，deadline随机1~3秒
        new_tasks = []
        for _ in range(np.random.randint(2,5)):
            task_type = np.random.randint(0,10)
            data_size = np.random.uniform(100,3000)
            deadline = np.random.uniform(1,3)
            new_tasks.append(Task(-1, task_type, data_size, deadline))  # task_id自动分配
        scheduler.step(new_tasks, batch_size=3)  # 每轮调度最多3个任务

        # 展示 replicator.partitions 里的每个 partition 的 sample_count
        print(f"Step {step_i}: Current Partitions and Sample Counts:")
        for partition in replicator.partitions:
            print(f"Partition {partition.bounds}: Sample Count = {partition.sample_count}")

        visualizer = PartitionVisualizer(replicator.partitions)
        visualizer.plot_2d_partitions(dim_x=0, dim_y=1, iteration=step_i)

