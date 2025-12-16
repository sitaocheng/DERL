# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer
# from difflib import SequenceMatcher


def extract_solution_gsm8k(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if solution is None:
        return ""
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def extract_solution_math(solution_str):
    try:
        solution_str = last_boxed_only_string(solution_str)
        if solution_str is None:
            return ""
        return remove_boxed(solution_str)
    except:
        return ""


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]
    return retval


def extract_answer(generated_text):
    extracted_answer = ""
    if '####' in generated_text:
        extracted_answer = extract_solution_gsm8k(generated_text)
    elif '\\boxed{' in generated_text:
        extracted_answer = extract_solution_math(generated_text)
    else:
        return generated_text
        
    return extracted_answer

def g1_outcome_reward(generated_text, ground_truth):
    extracted_answer = extract_answer(generated_text)
    extracted_gt = extract_answer(ground_truth)

    return 1+(1e-6) if extracted_answer == extracted_gt else 1e-6

def g2_format_reward(generated_text):
    if '\\boxed{' in generated_text:
        return 1+(1e-6)
    else:
        return 1e-6

def g3_cot_reward(generated_text, max_length=512):

    if not isinstance(generated_text, str):
        return 1e-6
        
    pattern1 = r'\b\d+\.\s+'
    matches1 = re.findall(pattern1, generated_text)
    has_numbered_list = len(matches1) >= 2 
    
    pattern2 = r'\bstep\s*\d+'  
    matches2 = re.findall(pattern2, generated_text, re.IGNORECASE)
    has_step = len(matches2) >= 2

    pattern3 = r'\bstep\s*[\d]+\s*[:.]'
    matches3 = re.findall(pattern3, generated_text, re.IGNORECASE)
    has_step_colon = len(matches3) >= 2

    if has_numbered_list or has_step or has_step_colon:
        return 1.0 + 1e-6
    else:
        return 1e-6

def g4_loose_outcome_reward(generated_text, ground_truth, min_length=125):
    if not isinstance(generated_text, str) or not isinstance(ground_truth, str):
        return 1e-6
    
    if len(generated_text) == 0 and len(ground_truth) == 0:
        return 1.0+(1e-6)
    if len(generated_text) == 0 or len(ground_truth) == 0:
        return 1e-6
    
    extracted_gt = extract_answer(ground_truth)
    if extracted_gt in generated_text:
        return 1 + 1e-6
    else:
        return 1e-6

def get_meta_reward(generated_text, ground_truth):
    g1_val = g1_outcome_reward(generated_text, ground_truth)
    g2_val = g2_format_reward(generated_text)
    g3_val = g3_cot_reward(generated_text)
    g4_val = g4_loose_outcome_reward(generated_text, ground_truth)

    return g1_val, g2_val, g3_val, g4_val