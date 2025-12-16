import json
import re
from typing import List, Optional

import torch
from vllm import LLM, SamplingParams
from vllm.sequence import SequenceGroupMetadata
from transformers import PreTrainedTokenizerBase
import torch.multiprocessing as mp
import argparse
import random

mp.set_start_method('spawn', force=True)  # 使用 spawn 方法以支持多进程


class AllowedVocabLogitsProcessor:
    """
    一个自定义的LogitsProcessor，用于将词汇表限制在允许的token ID集合中。
    
    它在每个生成步骤中被调用，将不在允许列表中的token的logit设置为负无穷，
    从而阻止模型生成这些token。
    """
    def __init__(self, allowed_token_ids: List[int]):
        """
        初始化LogitsProcessor。

        Args:
            allowed_token_ids (List[int]): 允许生成的token ID列表。
        """
        self.allowed_token_ids = set(allowed_token_ids)
        print(f"Logits processor initialized with {len(self.allowed_token_ids)} allowed tokens.")

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        """
        在每个生成步骤中修改logits。

        Args:
            token_ids (List[int]): 到目前为止已生成的token ID列表。
            logits (torch.Tensor): 模型为下一个token生成的原始logits。

        Returns:
            torch.Tensor: 修改后的logits。
        """
        # 创建一个与logits形状相同，填充为-inf的mask
        mask = torch.full_like(logits, -float('inf'))
        
        # 将允许的token位置的mask值设置为0
        allowed_ids_tensor = torch.tensor(list(self.allowed_token_ids), device=logits.device)
        mask[allowed_ids_tensor] = 0
        
        # 将mask应用到原始logits上
        return logits + mask

def get_allowed_token_ids(tokenizer: PreTrainedTokenizerBase) -> List[int]:
    """
    根据预定义的规则，从分词器中筛选出所有允许生成的token的ID。

    Args:
        tokenizer (PreTrainedTokenizerBase): vLLM使用的分词器实例。

    Returns:
        List[int]: 允许的token ID列表。
    """
    # 1. 定义我们希望模型生成的完整词汇（tokens）
    # 这包括Python关键字、函数名、变量名等
    allowed_full_tokens = {
        "+", "-", "*", "/", "(", ")",
        "g1", "g2", "g3", "g4"
    }
    # 为了保险起见，也加入一些可能被分词器拆分的子词
    allowed_full_tokens.update({"g", "1", "2", "3", "4"})

    # 2. 定义一个正则表达式，用于匹配由允许的字符组成的token


    
    # 这包括所有数字、运算符、括号、小数点、科学计数法'e'、冒号和空白字符
    allowed_chars_pattern = re.compile(r"^[+\-*/().:e0-9\s]+$")
    # allowed_chars_pattern = []
    allowed_token_ids = []
    vocab_size = tokenizer.vocab_size
    print(f"Checking vocabulary of size: {vocab_size}")

    for token_id in range(vocab_size):
        try:
            # 解码token ID为字符串，保留原始的空格等信息
            token_str = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except Exception:
            # 有些ID可能无法解码，直接跳过
            continue

        # 3. 进行筛选
        # 规则a: 如果解码后的字符串在我们的“完整词汇”列表中，则允许
        if token_str in allowed_full_tokens:
            allowed_token_ids.append(token_id)
            continue
        
        # 规则b: 如果解码后的字符串完全由“允许的字符”组成，则允许
        if allowed_chars_pattern.match(token_str):
            allowed_token_ids.append(token_id)
            continue
    
    # 4. 确保模型的特殊token（如句子结束符）也被加入允许列表，否则模型可能无法正常停止
    special_tokens_to_add = [
        tokenizer.eos_token_id, 
        tokenizer.bos_token_id, 
        tokenizer.pad_token_id, 
        tokenizer.unk_token_id
    ]
    for token_id in special_tokens_to_add:
        if token_id is not None and token_id not in allowed_token_ids:
            allowed_token_ids.append(token_id)
            print(f"Added special token ID: {token_id} ({tokenizer.decode([token_id])})")

    # 返回去重后的ID列表
    unique_allowed_ids = list(set(allowed_token_ids))
    print(f"Found {len(unique_allowed_ids)} unique allowed tokens in total.")
    return unique_allowed_ids


def rollout(llm: LLM, sampling_params: SamplingParams, prompts: List[str], output_file_path: str):
    """
    执行模型的rollout操作并保存结果。

    Args:
        llm (LLM): VLLM模型实例。
        sampling_params (SamplingParams): 包含logits处理器的采样参数。
        prompts (List[str]): 用于生成任务的提示列表。
        output_file_path (str): 输出JSON文件的路径。
    """
    print("\n--- Starting Rollout Generation ---")
    
    results = []
    outputs = llm.generate(prompts, sampling_params)

    for i, output_group in enumerate(outputs):
        for j, output in enumerate(output_group.outputs):
            generated_text = output.text.split("\n")[0]  # 只取第一行
            generated_text = generated_text.strip().strip(".")
            # match = re.search(r'<formula>(.*?)</formula>', generated_text, re.DOTALL)
            # if match:
            #     generated_text = match.group(1).strip()
            #     if "g1" not in generated_text or "g2" not in generated_text or "g3" not in generated_text or "g4" not in generated_text:
            #         print(f"Warning: Generated text does not contain all required variables: {generated_text}")
            #         continue
            # else:
            #     print(f"Warning: No <formula> tag found in output: {generated_text}")
            #     continue

            result_item = {
                "prompt_index": i,
                "rollout_index": j,
                "model_output": generated_text,
            }
            results.append(result_item)

            # 为了美观，给生成的代码加上缩进
            print("res = " + generated_text.replace("\n", "\n   "))

    # vLLM返回一个输出列表，对应每个prompt
    # for _ in range(0,50):
    #     random_num = random.randint(20, 100)
    #     prompt = prompts[0].replace("NUMBER", str(random_num))
    #     outputs = llm.generate([prompt], sampling_params)
    #     print(f"Generated {len(outputs)} outputs for prompt: {prompt}")

    #     for i, output_group in enumerate(outputs):
    #         for j, output in enumerate(output_group.outputs):
    #             generated_text = output.text.strip()

    #             # 提取result <formula>0.4*g1 + 0.3*g2 + 0.2*g3 - 0.1*g4</formula> --> 0.4*g1 + 0.3*g2 + 0.2*g3 - 0.1*g4
    #             match = re.search(r'<formula>(.*?)</formula>', generated_text, re.DOTALL)
    #             if match:
    #                 generated_text = match.group(1).strip()
    #                 if "g1" not in generated_text or "g2" not in generated_text or "g3" not in generated_text or "g4" not in generated_text:
    #                     print(f"Warning: Generated text does not contain all required variables: {generated_text}")
    #                     continue
    #             else:
    #                 print(f"Warning: No <formula> tag found in output: {generated_text}")
    #                 continue

    #             result_item = {
    #                 "prompt_index": i,
    #                 "rollout_index": j + 1,
    #                 "model_output": generated_text,
    #             }
    #             results.append(result_item)

    #             # 为了美观，给生成的代码加上缩进
    #             print("res = " + generated_text.replace("\n", "\n   "))

    # 将结果保存到JSON文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\nRollout结果已成功保存到 {output_file_path}")
    except Exception as e:
        print(f"\n保存Rollout结果到JSON文件失败: {e}")


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description="Generate mathematical formulas using a language model.")
    arg.add_argument("--model_path", type=str, required=True, help="Path to the meta-optimizer.")
    arg.add_argument("--output_file_path", type=str, required=True, default="outputs/llm_rollouts_new/sft_formulas_0.json", help="Path to save the generated formulas.")
    arg.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate for each prompt.")
    args = arg.parse_args()
    # --- 配置模型和参数 ---

    MODEL_PATH = args.model_path
    OUTPUT_FILE = args.output_file_path
    NUM_SAMPLES = args.num_samples

    # --- 初始化vLLM模型 ---
    print(f"Loading model: {MODEL_PATH}")
    # 如果GPU内存有限，可以设置 gpu_memory_utilization 来限制使用率
    llm = LLM(model=MODEL_PATH,
            trust_remote_code=True,         
            )
    tokenizer = llm.get_tokenizer()

    # --- 创建并配置Logits Processor ---
    print("\n--- Building Allowed Vocabulary ---")
    allowed_ids = get_allowed_token_ids(tokenizer)
    logits_processor = AllowedVocabLogitsProcessor(allowed_ids)

    # --- 设置采样参数 ---
    # 关键步骤：将我们的logits_processor实例传递给SamplingParams
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=1.0,
        n=NUM_SAMPLES,
        max_tokens=128,  # 公式通常不长，可以设置较小的最大长度
        logits_processors=[logits_processor],
    )

    prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nGenerate a mathematical formula.\nThe formula should combine four variables:\n- g1 (outcome_reward)\n- g2 (format_reward)\n- g3 (length_limit_max_reward)\n- g4 (length_limit_min_reward)\n\nThe formula must be a valid mathematical expression using +, -, *, /, parentheses, and float numbers.\nYour entire output must consist only of the expression. Do not include any additional text, explanations, or examples outside of the function definition.\nOutput ONE formula.\n<|im_end|>\n<|im_start|>assistant\n"

    rollout(
        llm=llm,
        sampling_params=sampling_params,
        prompts=[prompt],
        output_file_path=OUTPUT_FILE
    )