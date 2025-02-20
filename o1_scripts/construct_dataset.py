import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import random
from utils import extract_answer
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

def format_data(item):
    new_item = {
        "messages":[
            {"role":"system", "content":f"You are a helpful assistant. You should think step-by-step and put your final answer within \\boxed{{}}."},
            {"role":"user", "content":item['problem']},
            {"role":"assistant", "content":item['solution']}
        ],
        "weight": item['weight']
    }
    return new_item


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=16)  # model path
    parser.add_argument("--model_path", type=str, default="Qwen/QwQ-32B-Preview")  # output dir
    parser.add_argument("--model_name", type=str, default="QwQ")
    parser.add_argument("--file_name", type=str, default="")
    parser.add_argument("--alpha", type=int, default="None")
    return parser.parse_args()

args = parse_args()
model_name = args.model_name
model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
alpha = args.alpha
file_name = args.file_name
input_path =  f"./data/model_generated/{file_name}.json"

K = args.K
data = json.load(open(input_path,"r"))[0:20]
print("num data:",len(data))

def generate_pruning_dataset(data):

    def get_token_length(solution):
        return len(tokenizer(solution)['input_ids'])
    num_problems = len(data) // K

    selected_items = []

    used_count_per_problem = 2

    lower_bound = -2
    upper_bound = 4

    weights = []

    for problem_index in tqdm(range(num_problems)):

        items = data[problem_index*(K):(problem_index+1)*(K)] 
        random.shuffle(items)

        problem = items[0]['problem']
        
        real_answer = items[0]['ground_truth_answer']
        
        answers = [item['answer'] for item in items]
        solutions = [item['solution'] for item in items]
        correctness = [grade_answer(answer, real_answer) for answer in answers]

        avg_length = sum([get_token_length(solution) for solution in solutions]) / len(solutions)
        avg_acc = sum([int(c) for c in correctness]) / len(correctness)

        for index in range(0, used_count_per_problem):
            solution = solutions[index]
            length = get_token_length(solution)
            length_term = (avg_length - length) / length
            acc_term = alpha * (int(correctness[index]) - avg_acc)
            weight = length_term + acc_term

            if weight <= lower_bound:
                weight = lower_bound
            if weight > upper_bound:
                weight = upper_bound

            print(f"length:{length} acc:{correctness[index]}, avg_length:{avg_length} avg_acc:{avg_acc}, weight:{weight}")
            selected_items.append(
                {
                    "problem": problem,
                    "solution": solution,
                    "weight": float(weight)
                }
            )
            weights.append(weight)
        
    mean_weight = np.mean(weights)
    std_weight = np.std(weights)

    for item in selected_items:
        # unnormlized_weight = item["weight"]
        item["weight"] = (item["weight"] - mean_weight) / std_weight
    
    print("mean_weight:",mean_weight)
    print("std_weight",std_weight)
    print("num selected:",len(selected_items))

    dataset_data = [format_data(item) for item in selected_items]

    save_dir = f"data/my_dataset/{model_name}-train-K-{K}-alpha-{alpha}-k-{used_count_per_problem}"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/raw.json"
    with open(filename,"w") as f:
        json.dump(dataset_data,f)
    dataset = load_dataset("json",data_files=filename)
    dataset.save_to_disk(f"{save_dir}")
    print(dataset)
    print(dataset['train'][0])
    print(save_dir)

generate_pruning_dataset(data)