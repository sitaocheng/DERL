import re
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams
from sympy import symbols, simplify

# Assuming verl.utils is available in your environment, otherwise comment out or mock
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

# Placeholder for verl.utils if not available
class MockVerlUtils:
    def last_boxed_only_string(self, text):
        match = re.search(r'\\boxed{(.*?)}', text, re.DOTALL)
        if match:
            return match.group(1)
        return None

    def remove_boxed(self, text):
        return text.replace('\\boxed{', '').replace('}', '')

# Use actual verl.utils if available, else use mock
try:
    from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
except ImportError:
    print("Verl utilities not found, using mock functions for math reward extraction.")
    mock_utils = MockVerlUtils()
    last_boxed_only_string = mock_utils.last_boxed_only_string
    remove_boxed = mock_utils.remove_boxed


tokenizer = AutoTokenizer.from_pretrained('/nas/shared/sys2/sitaocheng/model/Qwen2.5-3B')
mp.set_start_method('spawn', force=True) # 强制使用 'spawn' 启动方法

def extract_solution_gsm8k(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if solution is None:
        return ""
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def extract_solution_math(solution_str):
    solution_str = last_boxed_only_string(solution_str)
    if solution_str is None:
        return ""
    return remove_boxed(solution_str)

def extract_answer(generated_text):
    """
    从生成的文本中提取答案。
    假设答案由 '####' 或 '\boxed{}' 分割。
    """
    extracted_answer = ""
    if '####' in generated_text:
        extracted_answer = extract_solution_gsm8k(generated_text)
    elif '\\boxed' in generated_text:
        extracted_answer = extract_solution_math(generated_text)
    else:
        return generated_text
        
    return extracted_answer

def g1_outcome_reward(generated_text, ground_truth):
    """
    g1: 答案评价模型。抽取答案并与ground_truth比较。
    假设答案由 '####' 或 '\boxed{}' 分割。
    """
    extracted_answer = extract_answer(generated_text)
    extracted_gt = ground_truth

    return 1 if extracted_answer == extracted_gt else 0

def g2_format_reward(generated_text):
    """
    g2: 格式评价模型。检查答案是否包含 #### 或是否使用 \boxed{} 框出答案。
    """
    if '####' in generated_text or '\\boxed{' in generated_text:
        return 1
    else:
        return 0

def g3_length_limit_max_reward(generated_text, max_length=512):
    """
    g3: 长度限制模型。检查答案是否超过 max_length 个token。
    """
    tokens = tokenizer.tokenize(generated_text)
    if len(tokens) <= max_length:
        return 1
    else:
        return 0

def g4_length_limit_min_reward(generated_text, min_length=128):
    """
    g4: 长度限制模型。检查答案是否至少为 min_length 个token。
    """
    tokens = tokenizer.tokenize(generated_text)
    if len(tokens) >= min_length:
        return 1
    else:
        return 0


class RewardNode(nn.Module):
    def __init__(self):
        super(RewardNode, self).__init__()
        self.operations = ['add', 'sub', 'mul', 'div']
        self.op_map = {
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
            'div': lambda x, y: x / (y + 1e-6)
        }
        self.weights = nn.Parameter(torch.rand(len(self.operations))) # Initialize with random values

    def forward(self, x, y):
        # Calculate logits for each operation
        logits = self.weights
        
        # Apply Softmax to get probability distribution
        probs = torch.softmax(logits, dim=-1)
        
        # Sample an operation index from the probability distribution
        m = torch.distributions.Categorical(probs)
        op_idx = m.sample()
        
        # Get the sampled operation type and its corresponding function
        # Ensure op_idx is a scalar before converting to item()
        sampled_op_name = self.operations[op_idx.item()]
        sampled_op_func = self.op_map[sampled_op_name]
        
        # Execute the sampled operation
        output = sampled_op_func(x, y)
        
        # Return output value, log_prob, and the sampled operation name for symbolic representation
        return output, m.log_prob(op_idx), sampled_op_name



class VLLMLanguageModel:
    def __init__(self, model_name="Qwen/Qwen2.5-3B", port=8000, host="localhost", max_model_len=1024):
        """
        :param model_name
        :param port: VLLM服务器运行的端口。
        :param host: VLLM服务器的地址。
        :param max_model_len: 模型的最大上下文长度，用于SamplingParams
        """

        print(f"Initializing VLLM LLM with model: {model_name}. This requires significant GPU memory if not connecting to a remote server.")
        try:
            self.llm = LLM(
                    model=model_name, # 例如 "Qwen/Qwen2.5-3B-Instruct"
                    tokenizer=model_name,
                    tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1, # 使用所有可用GPU
                    max_model_len=max_model_len, # 设置最大模型上下文长度
                    enable_prefix_caching=True, # 启用前缀缓存以加速生成
                    gpu_memory_utilization=0.8
                )
            self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512) # 限制生成长度
            print("VLLM LLM initialized successfully.")
        except Exception as e:
            print(f"Error initializing VLLM LLM: {e}")
            print("Please ensure you have enough GPU memory and the model path is correct, or connect to a running VLLM API server.")
            self.llm = None # Indicate failure to initialize

    def generate_text(self, prompt):
        if self.llm is None:
            print("VLLM LLM not initialized. Returning empty string.")
            return ""
        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            if outputs and outputs[0].outputs:
                generated_text = outputs[0].outputs[0].text
                return generated_text
            else:
                return ""
        except Exception as e:
            print(f"Error during VLLM text generation: {e}")
            return ""

    def generate_text_batch(self, prompts): 
        if self.llm is None:
            print("VLLM LLM not initialized. Returning empty list.")
            return [""] * len(prompts) # Return list of empty strings for consistency
        try:
            # VLLM's generate method naturally takes a list of prompts
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            generated_texts = []
            # Map outputs back to the original prompt order if necessary,
            # though vLLM usually preserves order for single batch.
            # outputs is a list of RequestOutput objects.
            # Each RequestOutput has a list of CompletionOutput objects in its 'outputs' attribute.
            # We usually take the first CompletionOutput for each request.
            for output in outputs:
                if output.outputs:
                    generated_texts.append(output.outputs[0].text)
                else:
                    generated_texts.append("") # Append empty string if no generation
            return generated_texts
        except Exception as e:
            print(f"Error during VLLM text generation: {e}")
            return [""] * len(prompts) # Return empty list on error


def generate_training_data(samples, language_model, batch_size=128): # Added batch_size parameter
    data = []
    # Store prompts and ground truths to process them in batches
    all_prompts = [sample.get('prompt', '')[0].get('content', '') for sample in samples]
    all_ground_truths = [sample.get('reward_model', '').get('ground_truth', '') for sample in samples]

    num_samples = len(samples)
    for i in range(0, num_samples, batch_size):
        # Get batch of prompts
        batch_prompts = all_prompts[i:i + batch_size]
        batch_ground_truths = all_ground_truths[i:i + batch_size]

        # Generate texts for the entire batch
        batch_generated_texts = language_model.generate_text_batch(batch_prompts)

        # Process each generated text in the batch
        for j, generated_text in enumerate(batch_generated_texts):
            ground_truth = batch_ground_truths[j] # Get corresponding ground truth

            lm_performance = g1_outcome_reward(generated_text, ground_truth)

            # Calculate atomic reward function values
            g1_val = lm_performance # g1 is directly lm_performance
            g2_val = g2_format_reward(generated_text)
            g3_val = g3_length_limit_max_reward(generated_text)
            g4_val = g4_length_limit_min_reward(generated_text)
            
            data.append({
                'g1': g1_val,
                'g2': g2_val,
                'g3': g3_val,
                'g4': g4_val,
                'lm_performance': lm_performance # Actual performance of the language model
            })

    return data


class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.node1_1 = RewardNode()
        self.node1_2 = RewardNode()
        self.node2_1 = RewardNode() # Node 2.1 now accepts two inputs and produces a scalar

    def forward(self, g1_val, g2_val, g3_val, g4_val):
        # Ensure inputs are tensors and can compute gradients
        g1 = torch.tensor(g1_val, dtype=torch.float32)
        g2 = torch.tensor(g2_val, dtype=torch.float32)
        g3 = torch.tensor(g3_val, dtype=torch.float32)
        g4 = torch.tensor(g4_val, dtype=torch.float32)

        # Collect all nodes' log_prob and sampled operation names
        log_probs = []
        sampled_ops = {} # To store sampled operations for symbolic representation

        # Forward pass through the network, collecting log_probs and sampled ops
        # Make sure to unpack correctly: output, log_prob, op_name_str
        node1_1_output, log_prob_1_1, op_name_1_1 = self.node1_1(g1, g2)
        node1_2_output, log_prob_1_2, op_name_1_2 = self.node1_2(g3, g4)
        
        log_probs.extend([log_prob_1_1, log_prob_1_2])
        # Store the string directly, not a tensor
        sampled_ops['node1_1'] = op_name_1_1
        sampled_ops['node1_2'] = op_name_1_2

        final_reward_logit, log_prob_2_1, op_name_2_1 = self.node2_1(node1_1_output, node1_2_output)
        log_probs.append(log_prob_2_1)
        # Store the string directly, not a tensor
        sampled_ops['node2_1'] = op_name_2_1

        return final_reward_logit, torch.stack(log_probs), sampled_ops



def read_parquet(file_path):
    """
    读取Parquet文件并返回json格式的数据
    """
    import pandas as pd
    try:
        df = pd.read_parquet(file_path)
        return df.to_dict(orient='records')
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return []

def evaluate_reward_model(model, generated_text, ground_truth):
    g1_val = g1_outcome_reward(generated_text, ground_truth)
    g2_val = g2_format_reward(generated_text)
    g3_val = g3_length_limit_max_reward(generated_text)
    g4_val = g4_length_limit_min_reward(generated_text)

    with torch.no_grad():
        # 在评估时，我们通常不采样，而是选择概率最高的那个操作
        # 或者为了保持一致性，也可以选择采样，但对于确定性评估，选最高的更合理
        # 这里为了演示，我们依然执行 forward 获得一个 reward score，但其内部是采样的
        # 如果需要确定性评估，RewardNode需要一个额外的参数来控制是否采样
        predicted_reward_logit, _, _ = model(g1_val, g2_val, g3_val, g4_val)
        # 将logit转换为0/1预测以进行评估
        predicted_reward_score = torch.sigmoid(predicted_reward_logit).item()
        
    return 1 if predicted_reward_score > 0.5 else 0


# ### Reinforcement Learning Training Code

# Now for the RL training loop. We'll use the REINFORCE algorithm (also known as Monte Carlo Policy Gradient).

# Here's the core idea:
# 1.  **Generate Trajectories:** For each training sample, we'll perform a forward pass through the `RewardModel`. Since `RewardNode` now samples operations, each forward pass represents a "trajectory" of sampled operations. We'll collect the `log_prob` of each sampled operation.
# 2.  **Calculate Reward:** Based on the `lm_performance` and `predicted_reward_binary` (derived from the `RewardModel`'s output), we'll define a **scalar reward** for this specific trajectory.
# 3.  **Compute Loss:** The loss for REINFORCE is typically $-\sum \text{log_prob} \times \text{reward}$. This encourages sampled actions (operations) that lead to higher rewards.

# ###

def rollout_and_train_and_eval_rl():
    reward_model = RewardModel()
    optimizer = optim.Adam(reward_model.parameters(), lr=0.001) # 降低学习率，RL通常需要更小的学习率

    # model_name 应为你实际使用的Qwen模型路径或名称
    vllm_model_name = "merged_model/grpo_gsm8k_math_qwen2.5_3b" # 请替换为你的模型路径或名称
    num_training_samples = 500 # 减少样本数量以加快演示，实际应更多
    math_test = read_parquet("/mnt/workspace/sitaocheng/projects/dataset/math/test.parquet") # 读取测试数据
    gsm8k_test = read_parquet("/mnt/workspace/sitaocheng/projects/dataset/gsm8k/test.parquet") # 读取GSM8K测试数据
    print(math_test[:1]) # 打印前5条测试数据
    print(gsm8k_test[:1]) # 打印前5条GSM8K测试数据
    
    # Combined and shuffled samples for training and testing
    all_samples = random.sample(math_test, min(len(math_test), num_training_samples // 2)) + \
                  random.sample(gsm8k_test, min(len(gsm8k_test), num_training_samples // 2))
    random.shuffle(all_samples)

    # Simple split for demonstration
    train_size = int(len(all_samples) * 0.8)
    training_sample = all_samples[:train_size]
    testing_sample = all_samples[train_size:]

    # lm_generator = VLLMLanguageModel(model_name=vllm_model_name)
    
    # # 检查 vllm 是否成功初始化
    # if lm_generator.llm is None:
    #     print("VLLM language model failed to initialize. Exiting.")
    #     exit()

    # print("Generating initial training data (might take some time)...")
    # training_data = generate_training_data(training_sample, lm_generator, batch_size=16) # Using a batch size of 16
    # print("Initial training data generation complete.")

    # 训练循环
    num_epochs = 0 # 增加epoch数量，RL训练通常需要更多迭代
    for epoch in range(num_epochs):
        total_loss = 0
        total_rewards = 0
        
        # In RL, often you re-evaluate the environment (generate new data) per epoch
        # For simplicity and to avoid too many LLM calls, we'll reuse generated_text from initial generation
        # but in a more complex scenario, you might want to call lm_generator.generate_text within the loop.
        for i, sample_data in enumerate(training_data):
            g1 = sample_data['g1']
            g2 = sample_data['g2']
            g3 = sample_data['g3']
            g4 = sample_data['g4']
            lm_performance = sample_data['lm_performance']

            optimizer.zero_grad()

            # 前向传播，获取采样到的 reward_logit 和 log_probs
            predicted_reward_logit, log_probs_tensor, sampled_ops = reward_model(g1, g2, g3, g4)
            
            # 将评价模型的logit转换为0/1预测
            predicted_reward_score = torch.sigmoid(predicted_reward_logit)
            predicted_reward_binary = (predicted_reward_score > 0.5).float()

            # 定义强化学习的即时奖励 (R)
            # "如果评价模型的reward和模型的输出的performance恰好相同，则该样本为正例。如果两者不同，则为负例。"
            # 强化学习的奖励值可以是浮点数
            if predicted_reward_binary == lm_performance:
                reward = 1.0 # Positive reward
            else:
                reward = -1.0 # Negative reward (or 0.0 for no reward if no match)
            
            total_rewards += reward

            # REINFORCE 损失函数: - log_prob * reward
            # log_probs_tensor 包含了所有采样步骤的 log_prob
            # 我们需要对这些 log_prob 求和，然后乘以 reward
            # PyTorch的autograd会自动处理每个log_prob对对应权重梯度的贡献
            loss = -torch.sum(log_probs_tensor) * reward
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(training_data)
        avg_reward = total_rewards / len(training_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

    print("\n--- Model Evaluation (RL Trained Model) ---")
    correct_predictions = 0
    total_eval_samples = 0
    # For evaluation, we ideally want a deterministic reward model.
    # The current `evaluate_reward_model` still samples.
    # To make it deterministic for evaluation, you would need to modify RewardNode
    # to select the max-probability operation when in model.eval() mode or a separate eval function.
    # For now, we'll proceed with the current `evaluate_reward_model` which involves sampling
    # so evaluation will also have a stochastic component.

    for i, sample in enumerate(testing_sample[:100]):  # 只测试前100个样本
        prompt = sample['prompt'][0]['content']
        ground_truth = sample['reward_model']['ground_truth']
        gen_text = lm_generator.generate_text(prompt)
        lm_perf = g1_outcome_reward(gen_text, ground_truth) # 模拟语言模型性能
        
        # Evaluate the reward model's prediction
        # Note: evaluate_reward_model will call reward_model.forward() which still samples
        predicted_reward_binary = evaluate_reward_model(reward_model, gen_text, ground_truth)
        
        print(f"Sample {i+1}:")
        print(f"  Generated: {gen_text[:50]}...")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  LM Performance (g1): {lm_perf}")
        print(f"  Predicted Reward (Binary): {predicted_reward_binary}")
        if predicted_reward_binary == lm_perf:
            correct_predictions += 1
            print("  Match!")
        else:
            print("  Mismatch!")
        print("-" * 30)
        total_eval_samples += 1
    
    if total_eval_samples > 0:
        accuracy = correct_predictions / total_eval_samples
        print(f"Evaluation Accuracy (on {total_eval_samples} samples): {accuracy:.4f}")

    # save model to disk
    # Ensure the directory exists
    try:
        makedirs("/mnt/workspace/sitaocheng/projects/verl/checkpoints")
    except Exception as e:
        print(f"Could not create directory for checkpoints: {e}")

    torch.save(reward_model.state_dict(), "/mnt/workspace/sitaocheng/projects/verl/checkpoints/reward_model_rl_init.pt")
    print("RL trained Reward Model training complete.")


# The load_reward_model function would need significant changes to interpret the sampled model.
# Since the operations are now sampled, a fixed symbolic expression is no longer directly applicable.
# You could visualize the learned probabilities for each operation within each node.
# For simplicity, I'll keep the original load_reward_model, but note its limitation.
def load_reward_model():
    model = RewardModel()
    # Load the state dictionary for the RL-trained model
    model_path = "/mnt/workspace/sitaocheng/projects/verl/checkpoints/reward_model_rl.pt"
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded RL-trained model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please ensure the training script has been run and saved the model.")
        return
    except Exception as e:
        print(f"Error loading model state dictionary: {e}")
        return

    model.eval() # Set model to evaluation mode (important for consistent behavior, though sampling still occurs)

    # Define symbolic variables
    g1, g2, g3, g4 = symbols('g1 g2 g3 g4')

    # Define symbolic operations
    symbolic_op_map = {
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'mul': lambda x, y: x * y,
        'div': lambda x, y: x / (y + 1e-6) # Match the epsilon for division
    }

    # Perform three samplings
    num_samplings = 4
    print(f"\n--- Sampling {num_samplings} Reward Functions from the RL-Trained Model ---")

    # We need to provide dummy input values to run the forward pass and get sampled operations
    # The actual numerical values don't matter for reconstructing the symbolic expression,
    # only the sampled operation names.
    dummy_g1, dummy_g2, dummy_g3, dummy_g4 = 1.0, 1.0, 1.0, 1.0 

    for i in range(num_samplings):
        print(f"\n--- Sample {i+1} ---")
        
        # Perform a forward pass to get the sampled operations
        # We only care about `sampled_ops` for reconstructing the symbolic expression
        with torch.no_grad(): # No gradient calculation needed during sampling/inference
            _, _, sampled_ops = model(dummy_g1, dummy_g2, dummy_g3, dummy_g4)

        # Reconstruct the symbolic expression based on the sampled operations
        # Node 1.1: g1 and g2
        op_11 = sampled_ops['node1_1']
        term11 = symbolic_op_map[op_11](g1, g2)

        # Node 1.2: g3 and g4
        op_12 = sampled_ops['node1_2']
        term12 = symbolic_op_map[op_12](g3, g4)

        # Node 2.1: output of Node 1.1 and Node 1.2
        op_21 = sampled_ops['node2_1']
        final_term = symbolic_op_map[op_21](term11, term12)

        simplified_final_term = simplify(final_term)
        print(f"Sampled Operations: {sampled_ops}")
        print(f"Derived Reward Function: {simplified_final_term}")
        print("-" * 50)

    # Optionally, you can also print the learned probabilities for each operation within each node
    print("\n--- Learned Operation Probabilities (Softmax) ---")
    for name, param in model.named_parameters():
        if 'weights' in name:
            probs = torch.softmax(param, dim=-1).tolist()
            node_name = name.split('.')[0] # e.g., 'node1_1'
            operations = model.__getattr__(node_name).operations # Access operations list for the node
            op_probs = {op: prob for op, prob in zip(operations, probs)}
            print(f"  {name} (Node {node_name}): {op_probs}")


if __name__ == "__main__":
    # rollout_and_train_and_eval_rl()
    load_reward_model() # This will not work as expected for the RL trained model