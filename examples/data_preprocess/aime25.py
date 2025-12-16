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
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/mnt/workspace/sitaocheng/projects/dataset/AIME_2025")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "opencompass/AIME2025"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    test_dataset_1 = datasets.load_dataset(data_source, 'AIME2025-I', trust_remote_code=True)['test']
    test_dataset_2 = datasets.load_dataset(data_source, 'AIME2025-II', trust_remote_code=True)['test']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")

            question = question + " " + instruction_following

            answer = example.pop("answer")
            solution = answer
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    # train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset_1 = test_dataset_1.map(function=make_map_fn("test-I"), with_indices=True)
    test_dataset_2 = test_dataset_2.map(function=make_map_fn("test-II"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset_1.to_parquet(os.path.join(local_dir, "test_1.parquet"))
    test_dataset_2.to_parquet(os.path.join(local_dir, "test_2.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
