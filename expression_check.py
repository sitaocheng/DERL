import json
import ast
import argparse

def is_parentheses_balanced(expr):
    """检查括号是否匹配"""
    stack = 0
    for char in expr:
        if char == '(':
            stack += 1
        elif char == ')':
            stack -= 1
            if stack < 0:
                return False
    return stack == 0

def safe_eval_expression(expr, values):
    """
    安全地解析并计算数学表达式。
    仅允许数字、变量(g1,g2,g3,g4)、基本运算符和括号。
    """
    try:
        result = eval(expr, {"__builtins__": None}, values)

        if isinstance(result, (int, float)):
            return result
        else:
            raise ValueError("Expression did not evaluate to a number.")
    except Exception as e:
        # print(f"Error evaluating expression '{expr}': {e}")
        return None

def check_rollout(expr):
    """检查表达式是否合法并能求值"""
    if not is_parentheses_balanced(expr):
        return False
    
    if 'g1' not in expr and 'g2' not in expr and 'g3' not in expr and 'g4' not in expr:
        return False
    # 提供测试值
    test_values = {'g1': 1.0, 'g2': 1.0, 'g3': 1e-6, 'g4': 1.0+1e-6}
    result = safe_eval_expression(expr, test_values)
    return result is not None

# 主程序
def main(json_file_path, output_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    bad = 0
    for item in data:
        prompt_index = item.get("prompt_index")
        rollout_index = item.get("rollout_index")
        model_output = item.get("model_output", "").strip()

        if not model_output:
            reward = -1.0
        else:
            is_valid = check_rollout(model_output)
            reward = 0.0 if is_valid else -1.0

        if reward == -1.0:
            bad += 1
            # print(f"Invalid rollout at rollout_index {rollout_index}: {model_output}")

        results.append({
            "rollout": model_output,
            "rollout_index": rollout_index,
            "reward": reward
        })

        # 打印或记录结果（可选）
        print(f"rollout_index: {rollout_index}, reward: {reward}, model_output: {model_output}")
    
    print(f"Total invalid rollouts: {bad}")
        
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results

# 使用示例
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Check mathematical expressions in JSON file.")
    args.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    args.add_argument("--output_file", type=str, required=True, help="Path to save the results JSON file.")
    parsed_args = args.parse_args()

    rewards = main(parsed_args.input_file, parsed_args.output_file)