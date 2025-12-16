# ray_check_gpu.py

import os
import ray

@ray.remote
def get_gpu_id():
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "No GPU")
    print(f"Task {ray.get_runtime_context().get_worker_id()} running on GPU(s): {gpu_ids}")
    return gpu_ids

# 启动 Ray 客户端连接到集群
ray.init(address='auto')  # 自动连接已启动的集群

# 获取集群总 GPU 数量（可选）
resources = ray.cluster_resources()
total_gpus = resources.get("GPU", 0)
print(f"Detected total GPUs in cluster: {total_gpus}")

# 启动 32 个任务，每个请求 1 个 GPU
num_tasks = int(total_gpus)  # 或者硬编码 32
print(f"Launching {num_tasks} tasks to occupy all GPUs...")
executor = get_gpu_id.options(num_gpus=2) 

# 并行提交任务
futures = [executor.remote() for _ in range(num_tasks)]

# 等待结果
results = ray.get(futures)

print("All tasks completed.")
print("GPU assignments:", results)

# 统计每个 GPU 被分配了多少次
from collections import Counter
flat_gpus = []
for r in results:
    if r != "No GPU":
        flat_gpus.extend(r.split(','))

gpu_count = Counter(flat_gpus)
print("GPU usage count:", dict(gpu_count))