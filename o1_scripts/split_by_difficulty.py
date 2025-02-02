import json
from tqdm import tqdm
import os
import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import random
from utils import extract_answer, get_example
from grader import grade_answer
from collections import defaultdict
from datasets import load_dataset
import os
import argparse
import json
import time
import os
import sys
import re
import random
import numpy as np

random.seed(42)
# 读取数据文件
def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# 划分数据集
def divide_data_by_difficulty(data, K):
    num_problems = len(data) // K
    problem_infos = []

    # 逐问题计算 acc
    for problem_index in tqdm(range(num_problems)):
        items = data[problem_index * K : (problem_index + 1) * K]
        real_answer = items[0]["ground_truth_answer"]
        answers = [item["answer"] for item in items]
        correctness = [int(grade_answer(answer, real_answer)) for answer in answers]
        acc = sum(correctness) / K

        problem_infos.append({
            "problem": items[0]["problem"],
            "items": items,
            "acc": acc
        })

    # 按 acc 排序
    problem_infos = sorted(problem_infos, key=lambda x: x["acc"])
    print(problem_infos[0]['acc'])
    print(problem_infos[-1]['acc'])

    # 划分子集 (按比例索引)
    num_problems = len(problem_infos)
    subsets = {
        "subset_0_40": problem_infos[: int(0.4 * num_problems)],
        "subset_30_70": problem_infos[int(0.3 * num_problems) : int(0.7 * num_problems)],
        "subset_60_100": problem_infos[int(0.6 * num_problems) :],
    }

    flattened_subsets = {
        subset_name: [item for problem in problems for item in problem["items"]]
        for subset_name, problems in subsets.items()
    }

    return flattened_subsets


# 保存划分后的数据
def save_subsets(subsets, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for subset_name, problems in subsets.items():
        output_path = os.path.join(output_dir, f"{subset_name}.json")
        with open(output_path, "w") as f:
            json.dump(problems, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=64)  # model path
    parser.add_argument("--model", type=str, default="QwQ")  # output dir
    parser.add_argument("--file_name", type=str, default="None")
    parser.add_argument("--dataset_type", type=str, default="sft")
    return parser.parse_args()
# 主流程
def main():
    

    model_names = {
        "QwQ": "Qwen/QwQ-32B-Preview",
        "Qwen7B": "Qwen/Qwen2.5-Math-7B-Instruct",
        "LLAMA70B": "meta-llama/Llama-3.1-70B-Instruct",
        "LLAMA8B": "meta-llama/Llama-3.1-8B-Instruct",
        "Marco":"AIDC-AI/Marco-o1"
    }

    args = parse_args()
    model = args.model
    model_path = model_names[model]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    file_name = args.file_name
    os.makedirs(f"./data/model_generated/{file_name}", exist_ok=True)
    K = args.K
    dataset_type = args.dataset_type
    input_path = f"./data/model_generated/{args.file_name}.json"
    output_dir = f"./data/model_generated/{args.file_name}/difficulty_divided"
    data = json.load(open(input_path,"r"))
    print("num data:",len(data))
    
    K = args.K

    data = load_data(input_path)
    subsets = divide_data_by_difficulty(data, K)
    save_subsets(subsets, output_dir)

    print(f"save to {output_dir}")

if __name__ == "__main__":
    main()
