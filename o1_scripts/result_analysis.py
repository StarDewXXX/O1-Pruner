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


def find_all_valid_equivalence_classes(objects, equivalence_func):
    # 字典用来记录等价类代表元素和等价类的所有元素
    equivalence_classes = {}

    for obj in objects:
        if obj == "None":
            continue
        found_class = False
        for rep in equivalence_classes.keys():
            # 判断当前对象是否和已有代表元素等价
            if obj == rep:
                equivalence_classes[rep].append(obj)
                found_class = True
                break
            elif equivalence_func(obj, rep):
                equivalence_classes[rep].append(obj)
                found_class = True
                break
        # 如果没有找到等价的代表元素，则将当前对象作为新的等价类的代表
        if not found_class:
            equivalence_classes[obj] = [obj]

    # 按等价类大小排序输出
    sorted_classes = sorted(equivalence_classes.values(), key=len, reverse=True)
    return sorted_classes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=64)  # model path
    parser.add_argument("--model", type=str, default="QwQ")  # output dir
    parser.add_argument("--file_name", type=str, default="None")
    return parser.parse_args()

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
os.makedirs(f"./data/model_evalution/{file_name}", exist_ok=True)
input_path =  f"./data/model_generated/{file_name}.json"
output_path = f"./data/model_evalution/{file_name}/majority.json"
K = args.K
K_values = [K]
data = json.load(open(input_path,"r"))
num_problems = len(data) // K

output_infos = []

solvable = 0
correct = 0
high_conf = 0
correct_high_conf = 0
total_tokens = 0

for problem_index in tqdm(range(num_problems)):

    items = data[problem_index*(K):(problem_index+1)*(K)] 

    real_answer = items[0]['ground_truth_answer']

    solutions = [item['solution'] for item in items]
    answers = [item['answer'] for item in items]

    # lengths = []
    for solution in solutions:
        num_tokens = len(tokenizer(solution)['input_ids'])
        total_tokens += num_tokens
    
    sorted_valid_answer_groups = find_all_valid_equivalence_classes(answers, grade_answer)
    
    if len(sorted_valid_answer_groups) > 0:

        if grade_answer(sorted_valid_answer_groups[0][0], real_answer):

            correct += 1

            # correct_solutions = []
            # for i in range(len(answers)):
            #     if grade_answer(answers[i], real_answer):
            #         correct_solutions.append(solutions[i])
            
            # for solution in correct_solutions:
            #     # print(solution)
            #     print("[real answer]:",real_answer)
                # input("continue?")


print(f"correct ratio:{correct/num_problems} avg_tokens:{total_tokens/(K*num_problems)}")
print(file_name)