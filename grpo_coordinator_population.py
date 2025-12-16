import os, sys
import subprocess
import json
import time
import threading
from queue import Queue
from datetime import datetime
import ray
import shutil

# ... (前面不变的代码)
VERL_ROOT = "."  # 替换为你的实际绝对路径

if VERL_ROOT not in sys.path:
    sys.path.insert(0, VERL_ROOT)

# --- 1. 全局配置 (请根据您的环境修改) ---

# --- 脚本路径配置 ---
CONSTRAINED_ROLLOUT_PY = "./constrained_rollout.py"
EXPRESSION_CHECK_PY = "./expression_check.py"
REWARD_CALCULATION_SH = "./examples/grpo_trainer/run_qwen2_5_3b_llm_math_meta_llm_rl_population.sh"
MERGE_ROLLOUT_SH = "./merger_rollout.sh"
EVAL_SH = "./eval.sh"
MODEL_UPDATE_SH = "./examples/grpo_trainer/run_qwen2_5_05b_instr_math_meta_rl.sh"
MERGE_OUTER_SH = "./merger_outer.sh"

# --- 训练参数 ---
TOTAL_ITERATIONS = 50  # 总共迭代多少轮
NUM_ROLLOUTS_PER_ITER = 8 # 每轮生成多少个rollout (4或8)
INITIAL_MODEL_PATH = "path_to_initial_meta_optimizer"

# --- GPU 配置 ---
# 可用的GPU设备ID列表，例如 [0, 1, 2, 3, 4, 5, 6, 7]
AVAILABLE_GPUS = list(range(32))
print(f"Available GPUs: {AVAILABLE_GPUS}")
GPUS_PER_REWARD_TASK = 4
GPUS_PER_EVAL_TASK = 4
GPUS_PER_UPDATE_TASK = 8 # Rollout

# --- 文件和目录结构 ---
BASE_OUTPUT_DIR = "./outputs/meta_grpo_dlc"  # 输出目录
ROLLOUTS_DIR = os.path.join(BASE_OUTPUT_DIR, "llm_rollouts")
# 注意：你的描述中提到了一个新的路径，这里采用原来的定义，但使用时会构造目标路径
CHECKED_ROLLOUTS_DIR = os.path.join(BASE_OUTPUT_DIR, "checked_rollouts") 
REWARDS_DIR = os.path.join(BASE_OUTPUT_DIR, "rewards")
MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "merged_model/meta_llm_rl_from_sft")


# --- 2. GPU资源管理器 ---
# ... (GPUManager 类不变)
class GPUManager:
    """一个线程安全的GPU资源管理器"""
    def __init__(self, gpu_list):
        self.gpu_queue = Queue()
        for gpu in gpu_list:
            self.gpu_queue.put(gpu)
        self.lock = threading.Lock()

    def acquire(self, num_gpus):
        """请求并获取指定数量的GPU，返回GPU ID列表"""
        with self.lock:
            if self.gpu_queue.qsize() < num_gpus:
                print(f"[{datetime.now()}] Insufficient GPU resources. Required: {num_gpus}, Available: {self.gpu_queue.qsize()}")
                return None # 资源不足
            
            gpus = [self.gpu_queue.get() for _ in range(num_gpus)]
            print(f"[{datetime.now()}] Acquired GPUs: {gpus}")
            return gpus

    def release(self, gpus):
        """释放GPU资源"""
        with self.lock:
            if gpus:
                for gpu in gpus:
                    self.gpu_queue.put(gpu)
                print(f"[{datetime.now()}] Released GPUs: {gpus}")

# --- 3. 辅助函数 ---
# ... (run_command, stream_output, parallel_task_runner, execute_task_on_ray, run_parallel_ray_tasks 不变)
def run_command(command, env=None, capture_output=True, text=True):
    """运行一个shell命令并打印日志"""
    print(f"[{datetime.now()}] Running command: {' '.join(command)}")
    try:
        process = subprocess.run(
            command, 
            check=True, 
            env=env, 
            capture_output=capture_output, 
            text=text
        )
        print(f"[{datetime.now()}] Command finished successfully.")
        print(f"Output:\n{process.stdout}")
        return process
    except subprocess.CalledProcessError as e:
        print(f"!!!!!! ERROR in command: {' '.join(command)} !!!!")
        print(f"Return code: {e.returncode}")
        print(f"Output:\n{e.stdout}")
        print(f"Error Output:\n{e.stderr}")
        raise

def stream_output(stream, prefix):
    """一个辅助函数，用于从流中逐行读取并添加前缀后打印"""
    try:
        # 使用 iter(stream.readline, '') 确保在流关闭时循环能正确终止
        for line in iter(stream.readline, ''):
            if line:
                print(f"[{prefix}] {line.strip()}", flush=True)
    finally:
        stream.close()


def parallel_task_runner(tasks, gpus_per_task, gpu_manager, max_retries=2): # <--- MODIFICATION: 增加 max_retries 参数
    """
    并行运行一组任务，管理GPU资源。(最终版：带重试机制 + 实时流式输出)
    'tasks' 是一个元组列表: [(command, env), (command, env), ...]
    """
    task_queue = Queue()
    # <--- MODIFICATION: 任务队列中增加'重试次数'
    for i, task in enumerate(tasks):
        task_queue.put((i, task, 0)) # (task_index, task_data, retries)

    active_processes = {}  # {pid: (gpus, p, command, task_index, out_thread, err_thread, retries)}
    
    # <--- MODIFICATION: 区分两种失败类型
    failed_to_start_tasks = []
    terminally_failed_tasks = [] 

    try:
        while not task_queue.empty() or active_processes:
            # --- 阶段1: 尽可能多地启动新任务 ---
            while not task_queue.empty() and gpu_manager.gpu_queue.qsize() >= gpus_per_task:
                gpus_to_acquire = gpu_manager.acquire(gpus_per_task)
                if not gpus_to_acquire:
                    break

                # <--- MODIFICATION: 从队列中获取重试次数
                task_index, (command, env_update), retries = task_queue.get()
                
                try:
                    current_env = os.environ.copy()
                    if env_update:
                        current_env.update(env_update)
                    
                    # <--- MODIFICATION: 在日志中显示尝试次数
                    attempt_num = retries + 1
                    print(f"[{datetime.now()}] >>> Starting task {task_index} (Attempt {attempt_num}/{max_retries + 1}) on GPUs {gpus_to_acquire}: {' '.join(command)}", flush=True)
                    
                    p = subprocess.Popen(
                        command, env=current_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                        text=True, encoding='utf-8', errors='replace'
                    )
                    
                    time.sleep(5)

                    stdout_prefix = f"Task-{task_index}|A:{attempt_num}|PID:{p.pid}|OUT"
                    stderr_prefix = f"Task-{task_index}|A:{attempt_num}|PID:{p.pid}|ERR"
                    
                    out_thread = threading.Thread(target=stream_output, args=(p.stdout, stdout_prefix))
                    err_thread = threading.Thread(target=stream_output, args=(p.stderr, stderr_prefix))
                    out_thread.start()
                    err_thread.start()
                    
                    # <--- MODIFICATION: 将重试次数存入 active_processes
                    active_processes[p.pid] = (gpus_to_acquire, p, (command, env_update), task_index, out_thread, err_thread, retries)

                except Exception as e:
                    # <--- MODIFICATION: 实现启动失败的重试逻辑 ---
                    print(f"!!!!!! FATAL ERROR trying to start task {task_index} (Attempt {attempt_num}/{max_retries + 1}) !!!!!!", flush=True)
                    print(f"    Exception: {e}", flush=True)
                    gpu_manager.release(gpus_to_acquire)
                    
                    if retries < max_retries:
                        new_retries = retries + 1
                        print(f"    Retrying task {task_index} in 5 seconds... (Next attempt: {new_retries + 1})", flush=True)
                        time.sleep(5) # 增加一个短暂的延迟避免快速连续失败
                        task_queue.put((task_index, (command, env_update), new_retries))
                    else:
                        print(f"    Task {task_index} has failed to start after {max_retries + 1} attempts. Giving up.", flush=True)
                        failed_to_start_tasks.append((task_index, command))

            # --- 阶段2: 检查所有正在运行的进程 ---
            finished_pids = []
            for pid, (gpus, p, task_data, task_index, out_thread, err_thread, retries) in list(active_processes.items()):
                if p.poll() is not None:
                    out_thread.join()
                    err_thread.join()
                    
                    if p.returncode != 0:
                        # <--- MODIFICATION: 实现运行失败的重试逻辑 ---
                        attempt_num = retries + 1
                        print(f"!!!!!! ERROR: Task finished with non-zero exit code (PID: {pid}, Index: {task_index}, Attempt {attempt_num}/{max_retries + 1}) !!!!!!", flush=True)
                        print(f"    Return code: {p.returncode}", flush=True)

                        if retries < max_retries:
                            new_retries = retries + 1
                            print(f"    Retrying task {task_index} in 5 seconds... (Next attempt: {new_retries + 1})", flush=True)
                            time.sleep(5)
                            task_queue.put((task_index, task_data, new_retries))
                        else:
                            print(f"    Task {task_index} has failed after {max_retries + 1} attempts. Giving up.", flush=True)
                            terminally_failed_tasks.append((task_index, task_data[0], p.returncode))
                    else:
                        print(f"[{datetime.now()}] <<< Task (PID: {pid}, Index: {task_index}) finished successfully.", flush=True)

                    gpu_manager.release(gpus)
                    finished_pids.append(pid)
            
            for pid in finished_pids:
                del active_processes[pid]
            
            # --- 阶段3: 等待 ---
            if active_processes or not task_queue.empty():
                time.sleep(5)

    except Exception as e:
        print(f"An unexpected error occurred in the task runner: {e}", flush=True)
    finally:
        # 清理逻辑保持不变...
        print("Cleaning up active processes...", flush=True)
        # ... (和上一版一样的清理代码)

        # <--- MODIFICATION: 最终的失败报告 ---
        if failed_to_start_tasks or terminally_failed_tasks:
            print("\n" + "="*50, flush=True)
            print("!!! SOME TASKS FAILED PERMANENTLY !!!", flush=True)
            if failed_to_start_tasks:
                print("\n--- Tasks That FAILED TO START ---", flush=True)
                for idx, cmd in failed_to_start_tasks:
                    print(f"  Task {idx}: {' '.join(cmd)}", flush=True)
            
            if terminally_failed_tasks:
                print("\n--- Tasks That FAILED AFTER RUNNING ---", flush=True)
                for idx, cmd, code in terminally_failed_tasks:
                    print(f"  Task {idx} (Exit Code: {code}): {' '.join(cmd)}", flush=True)
            
            print("="*50 + "\n", flush=True)
            raise RuntimeError("Some tasks failed permanently after all retries. Please check the logs.")


@ray.remote(num_gpus=8)
def execute_task_on_ray(command, env_update=None, task_name="unnamed_task"):
    """
    一个通用的Ray远程任务，用于在集群中的一个worker上执行shell命令。
    (最终版：主动查询并设置CUDA_VISIBLE_DEVICES)
    """
    hostname = os.uname()[1]
    start_time = datetime.now()
    
    # 准备将要传递给子进程的环境变量
    # current_env = os.environ.copy()

    print(f"[{start_time}] >> RAY TASK '{task_name}' STARTING <<")
    # print(f"    Node: {hostname}, Explicitly Set CUDA_VISIBLE_DEVICES = {assigned_gpus_msg}")
    print(f"    Command: {' '.join(command)}")

    try:
        process = subprocess.run(
            command,
            check=True,
            env=os.environ.copy(), # 使用我们手动构建好的环境
            capture_output=True,
            text=True
        )
        
        end_time = datetime.now()
        print(f"[{end_time}] >> RAY TASK '{task_name}' SUCCESS << (Duration: {end_time - start_time})")
        return {"status": "success", "node": hostname, "stdout": process.stdout, "stderr": process.stderr}

    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        error_message = (
            f"!!!!!! ERROR in RAY TASK '{task_name}' on node {hostname} !!!!!!\n"
            f"    Return code: {e.returncode}\n"
            f"    Command: {e.args}\n"
            f"    --- STDOUT ---\n{e.stdout}\n"
            f"    --- STDERR ---\n{e.stderr}\n"
        )
        print(error_message)
        raise RuntimeError(error_message) from e


def run_parallel_ray_tasks(tasks, gpus_per_task, task_group_name):
    """
    使用Ray并行运行一组任务。
    'tasks' 是一个元组列表: [(command, env_update), ...]
    """
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=None,
        )
    print(f"\n[{datetime.now()}] Submitting {len(tasks)} parallel Ray tasks for '{task_group_name}'...")

    resources = ray.cluster_resources()
    total_gpus = resources.get("GPU", 0)
    print(f"Detected total GPUs in cluster: {total_gpus}")
    print(f"Each task requests {gpus_per_task} GPU(s).")
    print("==========================================================")

    # 异步提交所有任务
    task_refs = []
    for i, (command, env_update) in enumerate(tasks):
        task_name = f"{task_group_name}_{i+1}"
        # 为这组任务配置所需的资源
        task_executor = execute_task_on_ray.options(num_gpus=gpus_per_task)
        task_refs.append(task_executor.remote(command, env_update, task_name))

    # 等待所有任务完成
    # ray.get() 会阻塞直到所有任务结束，如果任何一个任务抛出异常，它会在这里重新抛出
    try:
        results = ray.get(task_refs)
        print(f"[{datetime.now()}] All tasks for '{task_group_name}' completed successfully.")
        return results
    except Exception as e:
        print(f"Error while executing tasks for '{task_group_name}': {e}")
        print(f"!!!!!! A task in group '{task_group_name}' failed. Halting execution. !!!!!!")
        # 异常会由 ray.get() 自动传播，我们可以在这里记录一下然后重新抛出
        return 0.0

# --- 新增的辅助函数 ---
def find_best_model_path(loop_num, checked_rollouts_dir, models_dir):
    """
    查找上一轮（loop_num - 1）中 Reward 最高的 Rollout 对应的 Merged Model 路径。
    """
    prev_loop_num = loop_num - 1
    if prev_loop_num < 0:
        print("Loop 0: Using initial model path.")
        return "Qwen/Qwen2.5-3B" 

    # 构造上一轮的 checked_rollouts 文件路径
    # 注意：这里使用你描述中的路径结构来读取reward结果
    rollout_file_path = os.path.join(checked_rollouts_dir, f"loop_{prev_loop_num}_checked_with_reward.json")
    
    if not os.path.exists(rollout_file_path):
        print(f"WARNING: Cannot find previous loop's reward file: {rollout_file_path}. Using default model.")
        return "Qwen/Qwen2.5-3B"

    try:
        with open(rollout_file_path, 'r') as f:
            checked_data = json.load(f)
    except Exception as e:
        print(f"ERROR reading reward file {rollout_file_path}: {e}. Using default model.")
        return "Qwen/Qwen2.5-3B"

    best_reward = -float('inf')
    best_rollout_index = None

    # 查找最高的 reward
    for item in checked_data:
        # 确保reward存在且可比较
        reward = item.get('reward', -1.0) 
        if reward > best_reward:
            best_reward = reward
            best_rollout_index = item['rollout_index']

    if best_rollout_index is not None:
        # 构造最佳 Rollout 对应的 Merged Model 路径
        best_model_path = os.path.join(
            models_dir, 
            f"meta_outerloop{prev_loop_num}_rollout{best_rollout_index}"
        )
        if os.path.exists(best_model_path):
            print(f"Found best model from loop {prev_loop_num} (Rollout {best_rollout_index}) with reward {best_reward:.4f}.")
            return best_model_path
        else:
            print(f"WARNING: Best merged model path not found: {best_model_path}. Using default model.")
            return "Qwen/Qwen2.5-3B"
    else:
        print(f"WARNING: Could not determine best rollout for loop {prev_loop_num}. Using default model.")
        return "Qwen/Qwen2.5-3B"

# --- 4. 主流程 (修改 Step 3 部分) ---
def main():
    """GRPO算法主协调流程"""
    # 初始化目录
    for dir_path in [ROLLOUTS_DIR, CHECKED_ROLLOUTS_DIR, REWARDS_DIR, MODELS_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    gpu_manager = GPUManager(AVAILABLE_GPUS)
    # current_model_path 保持为 Outer-Loop Model Path，我们新增一个变量来跟踪 Inner-Loop Best Model
    current_model_path = INITIAL_MODEL_PATH 
    
    # 新增：用于 warm-start 内环训练的 Rollout Best Model 路径
    # 在第一次迭代（loop_num=0）中，它将是 None
    best_inner_loop_model_path = None 

    for loop_num in range(TOTAL_ITERATIONS):
        print(f"\n{'='*30} STARTING GRPO ITERATION {loop_num} {'='*30}")

        # 在开始内环训练之前，首先确定 Warm-Start 模型路径
        # 第一次循环 (loop_num=0) 时， best_inner_loop_model_path 将为 None，使用默认路径
        # 之后的循环 (loop_num > 0) 将使用上一轮的最佳 Rollout 模型路径
        best_inner_loop_model_path = find_best_model_path(
            loop_num, 
            CHECKED_ROLLOUTS_DIR, 
            MODELS_DIR
        )

        # 确定 Rollout 训练的起点模型
        # 如果找到上一轮最佳 Rollout 模型，就用它。否则使用 Outer-Loop 的最新模型 (current_model_path)
        # 注意：这里的逻辑根据你对 Rollout 训练的起点期望而定。如果你希望 Rollout 训练总是从 Outer-Loop Model 开始，则不需要这个逻辑。
        # 我先保持 Rollout 训练使用 current_model_path (即 Outer-Loop 的最新模型)，因为 Rollout 是生成数据的步骤。
        
        
        # --- Step 1: 生成Rollouts ---
        print(f"\n--- [Step 1/7] Generating {NUM_ROLLOUTS_PER_ITER} rollouts for iteration {loop_num} ---")
        print(f"\n--- [Step 2/7] Checking expression format for iteration {loop_num} ---")

        rollouts_path = os.path.join(ROLLOUTS_DIR, f"loop_{loop_num}_rollouts.json")
        checked_rollouts_path = os.path.join(CHECKED_ROLLOUTS_DIR, f"loop_{loop_num}_checked_with_reward.json")

        if not os.path.exists(rollouts_path) or not os.path.exists(checked_rollouts_path):
            env = os.environ.copy()
            env["VLLM_USE_V1"]="0"
            command = [
                "python", CONSTRAINED_ROLLOUT_PY,
                "--model_path", current_model_path,
                "--output_file_path", rollouts_path,
                "--num_samples", str(NUM_ROLLOUTS_PER_ITER)
            ]
            run_command(command=command, env=env)

            # --- Step 2: 格式检查和初步打分 ---
            run_command([
                "python", EXPRESSION_CHECK_PY,
                "--input_file", rollouts_path,
                "--output_file", checked_rollouts_path
            ])
        else:
            print(f"Checked rollouts already exist at {checked_rollouts_path}, skipping generation and checking.")

        # --- Step 3: 并行计算Reward (Inner-Loop 训练) ---
        print(f"\n--- [Step 3/7] Calculating rewards for iteration {loop_num} (Parallel) ---")
        
        # 构造 Warm-Start 参数（如果找到了最佳模型）
        warm_start_param = []
        if best_inner_loop_model_path:
            # 假设 REWARD_CALCULATION_SH 使用 model.load_from 参数
            # 注意：如果你的脚本配置不同，这里需要调整！
            warm_start_param = f"actor_rollout_ref.model.path={best_inner_loop_model_path}"
            print(f"Inner-Loop Warm-Start Model: {best_inner_loop_model_path}")
        else:
             print("Inner-Loop starting from scratch (or initial model).")

        with open(checked_rollouts_path, 'r') as f:
            checked_data = json.load(f)

        reward_tasks = []
        rollouts_to_process = []
        for item in checked_data:
            if item.get('reward') == 0.0:
                rollout_index = str(item['rollout_index'])
                rollout_str = item['rollout'].replace(" ", "")  # 确保字符串格式正确

                print(f"Processing rollout {rollout_index} with expression: {rollout_str}")

                custom_fn = "+custom_reward=\"" + rollout_str + "\""
                rollouts_to_process.append(item)
                
                experiment_name = f"trainer.experiment_name=qwen2.5-3b-base_meta_outer_{loop_num}_rollout{rollout_index}"

                checkpoints_dir = f"./checkpoints/verl_grpo_gsm8k_math_meta_llm_population_dlc/qwen2.5-3b-base_meta_outer_{loop_num}_rollout{rollout_index}/"
                print(f"Checking if checkpoints exist for rollout {rollout_index} in {checkpoints_dir}")

                merged_model_path = f"./outputs/meta_grpo_dlc/merged_model/meta_llm_rl_from_sft/meta_outerloop{loop_num}_rollout{rollout_index}"
                if os.path.exists(merged_model_path):
                    print(f"Merged model for rollout {rollout_index} already exists, skipping training.")
                    continue

                if os.path.exists(checkpoints_dir):
                    # already trained
                    print(f"Rollout {rollout_index} already trained, skipping reward calculation.")
                    continue
                
                # --- MODIFICATION START: 将 warm_start_param 加入到命令中 ---
                command = ["bash", REWARD_CALCULATION_SH, experiment_name, warm_start_param, custom_fn]
                # --- MODIFICATION END ---
                
                reward_tasks.append((command, None))

        if reward_tasks:
            parallel_task_runner(reward_tasks, GPUS_PER_REWARD_TASK, gpu_manager)
        else:
            print("No valid rollouts to calculate reward for. Skipping.")


        # --- Step 4 & 5: 并行Merge和Eval ---
        # ... (以下代码保持不变)
        print(f"\n--- [Step 4 & 5] Merging and Evaluating rollouts for iteration {loop_num} (Parallel) ---")
        eval_tasks = []
        merge_tasks = []
        for item in rollouts_to_process:
            rollout_index = item['rollout_index']
            checkpoints_dir = f"./checkpoints/verl_grpo_gsm8k_math_meta_llm_population_dlc/qwen2.5-3b-base_meta_outer_{loop_num}_rollout{rollout_index}/"
            print(f"Processing rollout {rollout_index} in {checkpoints_dir}")

            # Step 4 的命令
            merged_model_path = f"./outputs/meta_grpo_dlc/merged_model/meta_llm_rl_from_sft/meta_outerloop{loop_num}_rollout{rollout_index}"
            if not os.path.exists(merged_model_path):
                # check the latest checkpoint under checkpoints_dir
                checkpoints = os.listdir(checkpoints_dir)
                checkpoints = [os.path.join(checkpoints_dir, cp) for cp in checkpoints if cp.startswith('global_step')]
                checkpoints.sort(key=lambda x: int(x.split('_')[-1]))  # 按照global_step排序
                if checkpoints:
                    latest_checkpoint = checkpoints[-1] + "/actor"
                    print(f"Latest checkpoint found: {latest_checkpoint}")
                else:   
                    print(f"!!!!!! CRITICAL ERROR: No checkpoints found in {checkpoints_dir} !!!!!!")
                    continue

                print(f"Using latest checkpoint: {latest_checkpoint}")

                merge_cmd = ["bash", MERGE_ROLLOUT_SH, str(loop_num), str(rollout_index), latest_checkpoint]
                # run_command(merge_cmd) # Merge通常很快，可以串行执行，如果慢则也加入并行任务
                merge_tasks.append((merge_cmd, None))
                # 删除其他旧的 checkpoints（只保留最新的）
                for cp_dir in checkpoints[:-1]:  # 保留最后一个，删除前面所有
                    cp_actor_path = cp_dir
                    if os.path.exists(cp_actor_path):
                        print(f"Deleting old checkpoint: {cp_actor_path}")
                        try:
                            shutil.rmtree(cp_actor_path)  # 删除整个 actor 目录
                        except PermissionError:
                            print(f"Permission denied: {cp_actor_path}. Trying with sudo...")
                            import subprocess
                            subprocess.run(['sudo', 'rm', '-rf', cp_actor_path], check=True)
                        except Exception as e:
                            print(f"Failed to delete {cp_actor_path}: {e}")
            else:
                print(f"Model Merged for rollout {rollout_index} already exists at {merged_model_path}, skipping.")

            # Step 5 的命令
            eval_output_path = "./outputs/meta_grpo_dlc/" + f"meta_llm_outerloop{loop_num}_rollout{rollout_index}"
            if not os.path.exists(eval_output_path):
                eval_cmd = ["bash", EVAL_SH, str(loop_num), str(rollout_index)]
                # 假设eval.sh将reward打印到stdout
                eval_tasks.append((eval_cmd, None))
            else:
                print(f"Evaluation output for rollout {rollout_index} already exists at {eval_output_path}, skipping.")

        if merge_tasks:
            print(f"Merging rollout...")
            parallel_task_runner(merge_tasks, GPUS_PER_EVAL_TASK, gpu_manager)
        else:
            print("No rollouts to merge. Skipping.")

        if eval_tasks:
            print(f"Evaluating rollout...")
            run_parallel_ray_tasks(eval_tasks, GPUS_PER_EVAL_TASK, f"evaluate_reward_loop_{loop_num}")
            # parallel_task_runner(eval_tasks, GPUS_PER_EVAL_TASK, gpu_manager)
        else:
            print("No rollouts to evaluate. Skipping.")

        # --- Step 6: 模型更新 ---
        print(f"\n--- [Step 6/7] Updating main model for iteration {loop_num} ---")

        gpus = None # 初始化为None
        try:
            gpus = gpu_manager.acquire(GPUS_PER_UPDATE_TASK)
            while gpus is None:
                print("Waiting for GPUs to become available for model update...")
                time.sleep(10)
                gpus = gpu_manager.acquire(GPUS_PER_UPDATE_TASK)
            
            print(f"Acquired GPUs {gpus} for model update.") # 增加日志，方便调试

            next_loop_num = loop_num + 1
            # next_model_experiment_name = f"./checkpoints/verl_grpo_gsm8k_math_meta_llm_rl_from_sft_outer_loop_dlc/qwen2.5-05b-outer_loop_{next_loop_num}/global_step_1/actor"
            
            next_model_experiment_name = f"./outputs/meta_grpo_dlc/merged_model/meta_model/outer_loop_{next_loop_num}"
            if not os.path.exists(next_model_experiment_name):
                env = os.environ.copy()
                # env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        
                run_command([
                    "bash",
                    MODEL_UPDATE_SH,
                    str(next_loop_num),
                    current_model_path,
                    str(loop_num),
                ], env=env)
            else:
                print(f"Model update step for loop {loop_num} already done, skipping.")

        finally:
            if gpus: # 确保gpus确实被成功获取了再释放
                print(f"Releasing GPUs {gpus} from model update.") # 增加日志
                gpu_manager.release(gpus)   


        # --- Step 7: 最终Merge，生成新一轮的模型 ---
        print(f"\n--- [Step 7/7] Merging outer loop to create model for next iteration ---")
        new_model_path = f"./outputs/meta_grpo_dlc/merged_model/meta_model/outer_loop_{loop_num+1}" # 这是一个示例路径
        if not os.path.exists(new_model_path):
            run_command(["bash", MERGE_OUTER_SH, str(loop_num+1)])
        else:
            print(f"Outer loop model for next iteration already exists at {new_model_path}, skipping merge.")
        
        # 更新模型路径以备下一轮迭代使用
        if not os.path.exists(new_model_path):
             print(f"!!!!!! CRITICAL ERROR: New model for next iteration not found at {new_model_path} !!!!!!")
             break

        current_model_path = new_model_path

        print(f"\n{'='*30} COMPLETED GRPO ITERATION {loop_num} {'='*30}")
        print(f"New model for next iteration is at: {current_model_path}")

    print("\nAll GRPO iterations completed.")


if __name__ == "__main__":
    main()