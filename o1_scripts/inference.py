from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import random
from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import copy
from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import random
from utils import extract_answer, get_example
from grader import grade_answer
from collections import defaultdict
from datasets import load_dataset

import argparse
import json
import time
import os
import sys
import re
     
def prepare_prompts_for_solution(question, model, speed=None):
    
    if speed == None:
        speed_prompt = ""
    
    if speed == "very_fast":
        speed_prompt = "\nThis is an easy problem. Please solve it qucikly without any pause, check or reflection."
    if speed == "fast":
        speed_prompt = "\nBe confident. Solving this problem quickly with less pause, stop or reflection."
    if speed == "normal":
        speed_prompt = ""
    if speed == "slow":
        speed_prompt = "\nThis problem is hard. So you need to think rigorously and do more verifications and checks until you are absolutely confident about your answer."

    if model == "marco" or "marco" in model.lower():
        prompt = [
                {"role": "system", "content": f"You are a helpful assistant. You should think step-by-step and put your final answer within \\boxed{{}}.",},
                {"role":"user","content":f"Solve the problem: {question}{speed_prompt}"},
            ]

        return prompt

    if "qwq" in model or "qwen" in model.lower():

        prompt = [
                {"role": "system", "content": f"You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step and put your final answer within \\boxed{{}}.",},
                {"role":"user","content":f"Solve the problem: {question}{speed_prompt}"},
        ]
            
        return prompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--K", type=int, default=64)  # model path
    parser.add_argument("--dataset", type=str, default='math_train_hard')  # data path
    parser.add_argument("--model_name", type=str, default="QwQ")  # output dir
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--speed", type=str, default="normal")
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--save_output", type=bool, default=True)
    parser.add_argument("--max_tokens", type=int, default=8192)    
    
    return parser.parse_args()


dataset_paths = {
    "math_test": "./data/dataset/math_test.json",
    "math_train_hard": "./data/dataset/math_train_hard.json",
    "math_train": "./data/dataset/math_train.json",
    "gsm8k": "./data/dataset/gsm8k.json",
    "gaokao": "./data/dataset/gaokao.json"
}


args = parse_args()
speed = args.speed
K = args.K
model = args.model_name
dataset = args.dataset
num_samples = args.num_samples
n_gpus = args.n_gpus
max_tokens = args.max_tokens
input_path = dataset_paths[dataset]
model_path = args.model_path
save_output = args.save_output

print("INFERENCE:",K,model,dataset,"num:",num_samples)
llm = LLM(model=model_path,tensor_parallel_size=n_gpus,dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_path)

output_path = f"./data/model_generated/{model}_{dataset}_{max_tokens}_{speed}_K-{K}.json"

print(output_path)
if K == 1:
    sampling_params = SamplingParams(temperature=0,top_p=0.1,max_tokens=max_tokens)
else:
    sampling_params = SamplingParams(temperature=0.5,top_p=0.8,max_tokens=max_tokens)

greedy_sampling_params = SamplingParams(temperature=0,top_p=0.1,max_tokens=max_tokens)
data = json.load(open(input_path,"r"))
print("all num problem:",len(data))

random.seed(42)
random.shuffle(data)

data = data[0:num_samples]
print("used num problem:",len(data))
initial_data = [copy.deepcopy(item) for item in data]
data = [copy.deepcopy(item) for item in data for _ in range(K)]

prompts = [prepare_prompts_for_solution(item['problem'], model, speed=speed) for item in data]
prompts = tokenizer.apply_chat_template(prompts,add_generation_prompt=True,tokenize=False)
print(prompts[0])
outputs = llm.generate(prompts, sampling_params)
results = []

output_solutions = []
for output in outputs:
    context = output.prompt
    generated = output.outputs[0].text
    output_solutions.append(generated)

count = 0
correct = 0

for i in range(len(data)):
    item = data[i]
    answer = extract_answer(output_solutions[i], model_path)
    if grade_answer(answer, item['ground_truth_answer']):
        correct += 1
    count += 1

    item['solution'] = output_solutions[i]
    item['answer'] = answer
    
    results.append(item)

print(results[0])
print(results[0].keys)

if save_output:
    print("[Saved]")
    json.dump(results,open(output_path,"w"))
else:
    print("[Not saved]")
