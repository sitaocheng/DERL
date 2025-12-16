import re
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams
from sympy import symbols, simplify

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

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
        # 初始化权重参数，这些参数是可训练的
        self.w_add = nn.Parameter(torch.rand(1))
        self.w_sub = nn.Parameter(torch.rand(1))
        self.w_mul = nn.Parameter(torch.rand(1))
        self.w_div = nn.Parameter(torch.rand(1))

    def forward(self, g1, g2):
        # 按照图中的计算方式组合输入
        # 注意：这里假设 g1, g2 是标量，如果它们是批量的，需要调整维度
        term_add = self.w_add * (g1 + g2)
        term_sub = self.w_sub * (g1 - g2)
        term_mul = self.w_mul * (g1 * g2)
        # 避免除以零，可以添加一个小的epsilon或者进行条件判断
        term_div = self.w_div * (g1 / (g2 + 1e-6)) # Add a small epsilon to g2

        return term_add + term_sub + term_mul + term_div


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


def generate_training_data(samples, language_model):
    data = []
    for i, sample in enumerate(samples):
        prompt = sample.get('prompt', '')[0].get('content', '')  # 假设prompt在样本的第一个元素中
        ground_truth = sample.get('reward_model', '').get('ground_truth', '')

        generated_text = language_model.generate_text(prompt)
        lm_performance = g1_outcome_reward(generated_text, ground_truth)

        # 计算原子评价函数的值
        g1_val = lm_performance
        g2_val = g2_format_reward(generated_text)
        g3_val = g3_length_limit_max_reward(generated_text)
        g4_val = g4_length_limit_min_reward(generated_text)
        
        # 标签将在训练循环中动态生成
        data.append({
            'g1': g1_val,
            'g2': g2_val,
            'g3': g3_val,
            'g4': g4_val,
            'lm_performance': lm_performance # 语言模型的实际表现
        })

    return data


# 模拟一些ground truth文本
ground_truth_samples = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is a rapidly evolving field.",
    "PyTorch is a popular deep learning framework.",
    "Nature is beautiful."
]

class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.node1_1 = RewardNode()
        self.node1_2 = RewardNode()
        self.node2_1 = RewardNode() # Node 2.1 now accepts two inputs and produces a scalar
        # Node 2.1's output is essentially the logit
        # No final_linear or sigmoid here, let the criterion handle it

    def forward(self, g1_val, g2_val, g3_val, g4_val):
        g1 = torch.tensor(g1_val, dtype=torch.float32)
        g2 = torch.tensor(g2_val, dtype=torch.float32)
        g3 = torch.tensor(g3_val, dtype=torch.float32)
        g4 = torch.tensor(g4_val, dtype=torch.float32)

        node1_1_output = self.node1_1(g1, g2)
        node1_2_output = self.node1_2(g3, g4)

        final_reward_logit = self.node2_1(node1_1_output, node1_2_output)
        return final_reward_logit # Return logit, not sigmoid output

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
        predicted_reward_logit = model(g1_val, g2_val, g3_val, g4_val)
        predicted_reward_score = torch.sigmoid(predicted_reward_logit).item()
        
    return 1 if predicted_reward_score > 0.5 else 0


def rollout_and_train_and_eval():
    reward_model = RewardModel()
    optimizer = optim.Adam(reward_model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss() # 使用 BCEWithLogitsLoss

    # model_name 应为你实际使用的Qwen模型路径或名称
    vllm_model_name = "merged_model/grpo_gsm8k_math_qwen2.5_3b" # 请替换为你的模型路径或名称
    num_training_samples = 500 # 减少样本数量以加快演示，实际应更多
    math_test = read_parquet("/mnt/workspace/sitaocheng/projects/dataset/math/test.parquet") # 读取测试数据
    gsm8k_test = read_parquet("/mnt/workspace/sitaocheng/projects/dataset/gsm8k/test.parquet") # 读取GSM8K测试数据
    print(math_test[:1]) # 打印前5条测试数据
    print(gsm8k_test[:1]) # 打印前5条GSM8K测试数据
    training_sample = random.sample(math_test, num_training_samples) + random.sample(gsm8k_test, num_training_samples) # 从math和gsm8k中各取样本
    testing_sample = []
    training_prompts = [sample['prompt'][0]['content'] for sample in training_sample] # 提取训练样本的prompt内容
    for sample in gsm8k_test+math_test:
        if sample['prompt'][0]['content'] in training_prompts:
            continue
        testing_sample.append(sample)

    lm_generator = VLLMLanguageModel(model_name=vllm_model_name)
    training_data = generate_training_data(training_sample, lm_generator)

    # 检查 vllm 是否成功初始化
    if lm_generator.llm is None:
        print("VLLM language model failed to initialize. Exiting.")
        exit()

    # 训练循环
    num_epochs = 3
    for epoch in range(num_epochs):
        total_loss = 0
        for sample in training_data:
            g1 = sample['g1']
            g2 = sample['g2']
            g3 = sample['g3']
            g4 = sample['g4']
            lm_performance = sample['lm_performance']

            optimizer.zero_grad()

            # 得到评价模型的预测logit
            predicted_reward_logit = reward_model(g1, g2, g3, g4)
            
            # 将评价模型的logit转换为0/1预测（为了生成标签）
            # 这里是根据题目要求：“若最终大于0.5，则表示reward为1， 小于0.5则表示reward为0。”
            # 所以我们需要将 predicted_reward_logit 经过 sigmoid
            predicted_reward_score = torch.sigmoid(predicted_reward_logit)
            predicted_reward_binary = (predicted_reward_score > 0.5).float()

            # 根据题目要求构造标签
            # "如果评价模型的reward和模型的输出的performance恰好相同，则该样本为正例。如果两者不同，则为负例。"
            # 这里的 target 应该是 0 或 1
            # 如果 predicted_reward_binary == lm_performance，则 target = 1 (正例)
            # 否则 target = 0 (负例)
            target_label = (predicted_reward_binary == lm_performance).float()
            loss = criterion(predicted_reward_logit, torch.tensor(lm_performance, dtype=torch.float32).unsqueeze(0))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(training_data):.4f}")


    print("\n--- Model Evaluation ---")
    for i, sample in enumerate(testing_sample[:100]):  # 只测试前100个样本
        prompt = sample['prompt'][0]['content']
        ground_truth = sample['reward_model']['ground_truth']
        gen_text = lm_generator.generate_text(prompt)
        lm_perf = g1_outcome_reward(gen_text, ground_truth) # 模拟语言模型性能
        predicted_reward = evaluate_reward_model(reward_model, gen_text, ground_truth)
        print(f"Sample {i+1}:")
        print(f"  Generated: {gen_text[:50]}...")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  LM Performance (g1): {lm_perf}")
        print(f"  Predicted Reward: {predicted_reward}")
        print("-" * 30)

    # save model to disk
    torch.save(reward_model.state_dict(), "/mnt/workspace/sitaocheng/projects/verl/checkpoints/reward_model.pt")

def load_reward_model():
    model = RewardModel()
    model.load_state_dict(torch.load("/mnt/workspace/sitaocheng/projects/verl/checkpoints/reward_model.pt"))
    model.eval()

    # 定义符号变量
    g1, g2, g3, g4 = symbols('g1 g2 g3 g4')

    # 获取模型参数值
    def get_param(model, name):
        return float(model.state_dict()[name].item())

    # 获取各节点的权重
    w11_add = get_param(model, 'node1_1.w_add')
    w11_sub = get_param(model, 'node1_1.w_sub')
    w11_mul = get_param(model, 'node1_1.w_mul')
    w11_div = get_param(model, 'node1_1.w_div')

    w12_add = get_param(model, 'node1_2.w_add')
    w12_sub = get_param(model, 'node1_2.w_sub')
    w12_mul = get_param(model, 'node1_2.w_mul')
    w12_div = get_param(model, 'node1_2.w_div')

    w21_add = get_param(model, 'node2_1.w_add')
    w21_sub = get_param(model, 'node2_1.w_sub')
    w21_mul = get_param(model, 'node2_1.w_mul')
    w21_div = get_param(model, 'node2_1.w_div')


    # 构造中间节点输出
    term11 = w11_add*(g1 + g2) + w11_sub*(g1 - g2) + w11_mul*(g1*g2) + w11_div*(g1/(g2 + 1e-6))
    term12 = w12_add*(g3 + g4) + w12_sub*(g3 - g4) + w12_mul*(g3*g4) + w12_div*(g3/(g4 + 1e-6))

    # 最终输出
    final_term = (
        w21_add*(term11 + term12) +
        w21_sub*(term11 - term12) +
        w21_mul*(term11 * term12) +
        w21_div*(term11 / (term12 + 1e-6))
    )

    # 简化表达式（可选）
    simplified_final_term = simplify(final_term)
    print(simplified_final_term)

if __name__ == "__main__":
    rollout_and_train_and_eval()