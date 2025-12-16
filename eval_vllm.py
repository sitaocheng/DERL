import pandas as pd
from vllm import LLM, SamplingParams
import json
from verl.utils.reward_score.math import compute_score as math_compute_score
from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score
import torch.multiprocessing as mp
import argparse
import torch
import os
import time
import ray
from transformers import AutoTokenizer

mp.set_start_method('spawn', force=True)  # 使用 spawn 方法以支持多进程

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

def run_evaluation(llm, sampling_params, test_file_path: str, output_file_path='outputs/eval_math.json', batch_size=16, max_new_tokens: int = 1024):
    """
    使用VLLM部署模型并进行评估。

    Args:
        model_path (str): VLLM模型路径，例如 "mistralai/Mistral-7B-Instruct-v0.2"。
        test_file_path (str): 包含测试数据的parquet文件路径。
        max_new_tokens (int): 模型生成文本的最大token数量。
    """

    print(f"正在加载测试数据: {test_file_path}...")
    try:
        table = pd.read_parquet(test_file_path)
        test_data = table.to_dict(orient='records') 
        print(f"测试数据加载完成，共 {len(test_data)} 条记录。")
        print(f"测试数据示例: {test_data[0] if test_data else '无数据'}")
    except json.JSONDecodeError as e:
        print(f"加载JSON文件失败，请检查文件格式: {e}")
        return
    except Exception as e:
        print(f"加载测试文件失败: {e}")
        return

    results = []
    original_data_indices = [] # 记录原始数据索引，方便回溯
    total_score = 0

    print("正在准备模型输入...")
    # 分批处理数据
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        print(f"正在处理第 {i // batch_size + 1} 批数据（{i} - {min(i + batch_size, len(test_data))}）...")

        prompts = []
        ground_truths = []
        indices = []

        for j, item in enumerate(batch):
            try:
                prompt_content = item['prompt'][0]['content']
                ground_truth_content = item['reward_model']['ground_truth']

                messages = {"role": "user", "content": prompt_content}
                
                messages = tokenizer.apply_chat_template([messages], add_generation_prompt=True, tokenize=False)
                # print(messages)

                prompts.append(messages)
                ground_truths.append(ground_truth_content)
                indices.append(i + j)
            except (KeyError, TypeError) as e:
                print(f"跳过第 {i + j} 条数据（格式错误）: {e}")
                continue

        if not prompts:
            continue

        # 批量生成输出
        outputs = llm.generate(prompts, sampling_params)

        # 处理每个输出
        start_time = time.time()
        for idx, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            gt = ground_truths[idx]

            if "####" in generated_text.lower():
                score = gsm8k_compute_score(generated_text, gt)
            else:
                score = math_compute_score(generated_text, gt)

            total_score += score

            cur_time = time.time()

            if cur_time - start_time >= 60 * 60 * 0.25:
                print("exceed max training time!!! 15 MIN!!!")
                break

            results.append({
                "prompt": prompts[idx],
                "ground_truth": gt,
                "model_output": generated_text,
                "evaluation": score
            })


    avg_score = total_score / len(results) if results else 0
    print(f"评估完成，平均得分: {avg_score:.4f}")
    results.append({
        "average_score": avg_score,
        "total_records": len(results)
    })

    output_dir = "./outputs/meta_grpo_dlc_ppo_epoch_8/" + output_file_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将结果保存到JSON文件
    if "math" in test_file_path.lower():
        output_file_path = output_dir + "/math.json"
    elif "gsm8k" in test_file_path.lower():
        output_file_path = output_dir + "/gsm8k.json"
    elif 'aime' in test_file_path.lower():
        output_file_path = output_dir + "/aime24.json"
    elif 'amc' in test_file_path.lower():
        output_file_path = output_dir + "/amc.json"
        
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"评估结果已保存到 {output_file_path}")
    except Exception as e:
        print(f"保存结果到JSON文件失败: {e}")


def eval_file(file_name):
    """
    评估指定文件中的模型输出。

    Args:
        file_name (str): 包含模型输出和ground truth的JSON文件路径。
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"加载JSON文件失败，请检查文件格式: {e}")
        return
    except Exception as e:
        print(f"加载文件失败: {e}")
        return

    total_score = 0
    results = []
    for item in data:
        model_output = item.get("model_output", "")
        ground_truth = item.get("ground_truth", "")

        if "####" in model_output.lower():
            score = gsm8k_compute_score(model_output, ground_truth)
        else:
            score = math_compute_score(model_output, ground_truth)

        total_score += score

        results.append({
            "prompt": item.get("prompt", ""),
            "ground_truth": ground_truth,
            "model_output": model_output,
            "evaluation": score
        })

    avg_score = total_score / len(results) if results else 0
    print(f"评估完成，平均得分: {avg_score:.4f}")

    time.sleep(10)  # 确保文件写入完成
    output_file = file_name.replace(".json", "_eval.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"评估结果已保存到 {output_file}")


def rollout(llm, output_file_path, n=4):
    """
    执行模型的rollout操作并保存结果。

    Args:
        llm (LLM): VLLM模型实例。
        output_file_path (str): 输出文件路径。
    """


    prompts = """You are a code generation model. Your task is to generate a mathematical formula.
The formula should combine four variables:

- g1 (outcome_reward)

- g2 (format_reward)

- g3 (length_limit_max_reward)

- g4 (length_limit_min_reward)

The formula must be a valid mathematical expression using +, -, *, /, parentheses, and float numbers.

Your entire output must consist only of the expression. Do not include any additional text, explanations, or examples outside of the function definition.

Here are three examples of valid outputs:
0.5 * g1 + 0.5 * (g2 - g3 * 0.25 * g4)
0.3 * g1 + 0.5 * (g2 - g3 * g4) / (g2 + 1e-6)
0.6 * g2 + 0.5 * (g1 - g3) / (g4 + 1e-6)
Output ONE formula:
"""

    results = []
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, n=n, max_tokens=1024)
    output = llm.generate([prompts], sampling_params)
    
    for i, output_item in enumerate(output[0].outputs):
        generated_text = output_item.text.strip()
        results.append({
            "model_output": generated_text,
            "rollout_index": i + 1
        })
        print(f"Rollout {i + 1}: {generated_text}")

    # 将结果保存到JSON文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Rollout结果已保存到 {output_file_path}")
    except Exception as e:
        print(f"保存Rollout结果到JSON文件失败: {e}")



def compute_avg_score(math_output_file, gsm8k_output_file, loop_num, rollout_num):
    """
    计算数学公式输出和GSM8K输出的平均得分。

    Args:
        math_output_file (str): 数学公式输出文件路径。
        gsm8k_output_file (str): GSM8K输出文件路径。
    """

    with open(math_output_file, 'r', encoding='utf-8') as f:
        math_data = json.load(f)

    with open(gsm8k_output_file, 'r', encoding='utf-8') as f:
        gsm8k_data = json.load(f)

    math_score = math_data[-1]['average_score'] if math_data else 0
    gsm8k_score = gsm8k_data[-1]['average_score'] if gsm8k_data else 0
    # compute_avg_score = ((math_score*(len(math_data)-1)) + gsm8k_score*(len(gsm8k_data)-1)) / (len(math_data) + len(gsm8k_data) - 2)
    compute_avg_score = math_score

    print(f"MATH平均得分: {math_score:.4f}")
    print(f"GSM8K平均得分: {gsm8k_score:.4f}")  # 若用math，记得删掉gsm
    print(f"综合平均得分: {compute_avg_score:.4f}")

    with open(f"./outputs/meta_grpo_dlc_ppo_epoch_8/checked_rollouts/loop_{loop_num}_checked_with_reward.json", 'r', encoding='utf-8') as f:
        reward = json.load(f)
    
    for i in reward:
        if i['rollout_index'] == rollout_num:
            if i['reward'] != 0.0:
                print(f"Warning: Rollout {rollout_num} already has a reward value: {i['reward']}, rollout: {i['rollout']}")
            else:
                i['reward'] = compute_avg_score
            break

    with open(f"./outputs/meta_grpo_dlc_ppo_epoch_8/checked_rollouts/loop_{loop_num}_checked_with_reward.json", 'w', encoding='utf-8') as f:
        json.dump(reward, f, ensure_ascii=False, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation using VLLM model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the VLLM model.")
    parser.add_argument("--output_file_path", type=str, default="outputs/eval_math.json", help="Path to save the evaluation results.")
    parser.add_argument("--loop_num", type=int, required=True)
    parser.add_argument("--rollout_num", type=int, required=True)
    parser.add_argument("--testing", type=bool, required=True)
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()

    print(f"可用GPU数量: {num_gpus}")
    print(f"正在加载VLLM模型: {args.model_path}...")
    llm = LLM(model=args.model_path, tensor_parallel_size=num_gpus, max_num_batched_tokens=32768, enable_chunked_prefill=True, seed=42)
    print("模型加载完成。")

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024, stop=["<|im_end|>"])
    if args.testing:
        test_file_path = ["./dataset/math/test.parquet", "./dataset/gsm8k/test.parquet"]
    else:
        test_file_path = ["./dataset/math/val.parquet", "./dataset/gsm8k/val.parquet"]
    
    for file in test_file_path:
        run_evaluation(llm, sampling_params, file, output_file_path=args.output_file_path, batch_size=1024)

    time.sleep(10)
    print("compute_avg_score")
    if not ('amc' in args.output_file_path or 'aime' in args.output_file_path):
        compute_avg_score(
            math_output_file="./outputs/meta_grpo_dlc_ppo_epoch_8/" + args.output_file_path + "/math.json",
            gsm8k_output_file="./outputs/meta_grpo_dlc_ppo_epoch_8/" + args.output_file_path + "/gsm8k.json",
            loop_num=args.loop_num,
            rollout_num=args.rollout_num   
        )
